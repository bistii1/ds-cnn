"""Representative dataset for int8 quantization — matches training preprocessing.

Share this with whoever quantizes the model. It reproduces the EXACT input
pipeline used in training so a quantized model calibrated with it is directly
comparable to ours:

    wav -> load/mono/pad-or-truncate -> MFCC (tf.signal) -> standardize -> yield

The MFCC settings come from `config.yaml` and the standardization constants
(mean/std) come from `kws_tf.json`, so this ALWAYS matches whatever model those
two files describe. Use the config.yaml + kws_tf.json that belong to the model
you are quantizing (old model = old settings, new model = new settings).

Usage as a library:

    from representative_dataset import make_representative_dataset
    rep = make_representative_dataset(wav_dir="calib_wavs", num_samples=300)
    converter.representative_dataset = rep

Usage as a script (quantizes a SavedModel end to end):

    python src/representative_dataset.py \
        --saved-model kws_tf_savedmodel \
        --json kws_tf.json \
        --wav-dir calib_wavs \
        --out kws_int8.tflite
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf

from config import load_config
from mfcc_tf import MFCCParams, waveform_to_mfcc


def _load_waveform(path: Path, n_samples: int) -> np.ndarray:
    """Load one wav as float32 mono, padded/truncated to a fixed length."""
    wav, _ = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:                      # stereo -> mono
        wav = wav.mean(axis=1)
    if len(wav) < n_samples:              # pad short clips with silence
        wav = np.pad(wav, (0, n_samples - len(wav)))
    else:                                 # truncate long clips
        wav = wav[:n_samples]
    return wav.astype(np.float32)


def make_representative_dataset(wav_dir: str | Path,
                                num_samples: int = 300,
                                json_path: str | Path = "kws_tf.json",
                                config_path: str | Path | None = None,
                                seed: int = 42):
    """Return a generator yielding standardized MFCC inputs, one clip at a time.

    Each yielded item is shaped (1, num_frames, num_mfcc, 1) float32 — the same
    shape the model was trained on.
    """
    cfg = load_config(config_path) if config_path else load_config()
    p = MFCCParams(cfg)

    meta = json.loads(Path(json_path).read_text())
    mean = float(meta["feature_norm"]["mean"])
    std = float(meta["feature_norm"]["std"])

    wavs = sorted(Path(wav_dir).rglob("*.wav"))
    if not wavs:
        raise SystemExit(f"[error] no .wav files found under {wav_dir}/")
    rng = np.random.default_rng(seed)
    if len(wavs) > num_samples:
        wavs = [wavs[i] for i in rng.choice(len(wavs), num_samples, replace=False)]

    def rep():
        for path in wavs:
            wav = _load_waveform(path, p.n_samples)
            feats = waveform_to_mfcc(tf.constant(wav[None, :]), p).numpy()  # (1, frames, coeff)
            feats = (feats - mean) / std                                    # same standardization
            feats = feats[..., None].astype(np.float32)                     # -> (1, frames, coeff, 1)
            yield [feats]

    return rep


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Quantize a SavedModel to int8 using the training preprocessing.")
    ap.add_argument("--saved-model", default="kws_tf_savedmodel",
                    help="Path to the float SavedModel directory.")
    ap.add_argument("--json", default="kws_tf.json",
                    help="Metadata file with feature_norm mean/std for this model.")
    ap.add_argument("--wav-dir", default="processed",
                    help="Folder of 16 kHz mono wavs used for calibration.")
    ap.add_argument("--config", default=None,
                    help="config.yaml to use (defaults to the project config).")
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--out", default="kws_int8.tflite")
    args = ap.parse_args()

    rep = make_representative_dataset(
        wav_dir=args.wav_dir, num_samples=args.num_samples,
        json_path=args.json, config_path=args.config,
    )

    conv = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    Path(args.out).write_bytes(conv.convert())
    print(f"[done] wrote {args.out}  ({Path(args.out).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
