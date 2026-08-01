"""MFCC front-end built entirely with tf.signal (no PyTorch/torchaudio).

Why this file exists
--------------------
The old front-end used torchaudio, which depends on PyTorch and cannot run on
the nRF5340. TensorFlow's `tf.signal` ops are the same family of ops used by
TFLite-Micro's `micro_speech` audio preprocessor, so a model trained on these
features has a clean path to the microcontroller.

What "MFCC" means, step by step (all of this is standard DSP)
-------------------------------------------------------------
Raw audio is just a long list of numbers (16000 per second). A neural net
struggles with that directly, so we squeeze it into a small "image" of the
sound. Five simple steps:

  1. FRAME    : chop the 2 s clip into short 30 ms windows, hop 20 ms.
                -> ~99 little frames.
  2. WINDOW   : multiply each frame by a Hann window (tapers the edges so the
                FFT does not see fake clicks at the frame boundaries).
  3. FFT      : for each frame, compute how much energy is at each frequency.
  4. MEL      : squash the many FFT bins into 40 "mel" bands. Mel spacing
                mimics human hearing (fine detail low, coarse detail high),
                then take log (ears perceive loudness logarithmically).
  5. DCT      : compress the 40 log-mel bands into 10 MFCC coefficients. The
                DCT keeps the overall shape and throws away redundancy.

Result: each 2 s clip -> a (frames x 10) grid of numbers. That grid is the
"image" we feed to the CNN.

All parameters come from config.yaml (`audio` + `mfcc_reference`) so training
and the on-device front-end stay in sync.
"""
from __future__ import annotations

import tensorflow as tf


class MFCCParams:
    """Plain container of MFCC settings, derived from config.yaml."""

    def __init__(self, cfg: dict):
        a, m = cfg["audio"], cfg["mfcc_reference"]
        self.sample_rate = int(a["sample_rate"])                      # 16000
        self.n_samples = int(self.sample_rate * a["clip_ms"] / 1000)  # 32000 (2 s)
        self.frame_length = int(self.sample_rate * m["frame_length_ms"] / 1000)  # 480
        self.frame_step = int(self.sample_rate * m["frame_stride_ms"] / 1000)    # 320
        self.fft_length = 512          # next power of two >= frame_length (fast FFT)
        self.num_mel_bins = int(m["num_filters"])          # 40
        self.num_mfcc = int(m["num_coefficients"])         # 10
        self.lower_hz = float(m["fmin_hz"])                # 20
        self.upper_hz = float(m["fmax_hz"])                # 4000

    @property
    def num_frames(self) -> int:
        return 1 + (self.n_samples - self.frame_length) // self.frame_step


def waveform_to_mfcc(waveform: tf.Tensor, p: MFCCParams) -> tf.Tensor:
    """(batch, n_samples) float32 in [-1, 1]  ->  (batch, frames, num_mfcc).

    Pure tf.signal. Every op here has an equivalent in the TFLite-Micro signal
    library used by micro_speech, so this is device-portable in spirit.
    """
    # 1. FRAME: (batch, frames, frame_length)
    frames = tf.signal.frame(
        waveform, frame_length=p.frame_length, frame_step=p.frame_step
    )

    # 2. WINDOW: taper each frame with a Hann window.
    window = tf.signal.hann_window(p.frame_length, periodic=True)
    frames = frames * window

    # 3. FFT: real FFT -> magnitude spectrum (batch, frames, fft_length/2 + 1).
    #    zero-padding to fft_length=512 is implicit in rfft's fft_length arg.
    spectrogram = tf.abs(tf.signal.rfft(frames, fft_length=[p.fft_length]))

    # 4. MEL: project the linear-frequency spectrum onto mel bands, then log.
    num_spectrogram_bins = p.fft_length // 2 + 1
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=p.num_mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=p.sample_rate,
        lower_edge_hertz=p.lower_hz,
        upper_edge_hertz=p.upper_hz,
    )
    mel = tf.matmul(tf.square(spectrogram), mel_matrix)   # power -> mel energy
    log_mel = tf.math.log(mel + 1e-6)

    # 5. DCT: log-mel -> MFCC, keep the first `num_mfcc` coefficients.
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)
    return mfcc[..., : p.num_mfcc]


class MFCCLayer(tf.keras.layers.Layer):
    """Keras layer wrapping `waveform_to_mfcc`, output shaped for a Conv2D.

    Input : (batch, n_samples) waveform.
    Output: (batch, frames, num_mfcc, 1) — the extra 1 is the image channel.
    """

    def __init__(self, cfg: dict, **kwargs):
        super().__init__(**kwargs)
        self.p = MFCCParams(cfg)

    def call(self, waveform: tf.Tensor) -> tf.Tensor:
        mfcc = waveform_to_mfcc(waveform, self.p)
        return tf.expand_dims(mfcc, axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.p.num_frames, self.p.num_mfcc, 1)
