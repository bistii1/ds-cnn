"""Train a simple TensorFlow CNN keyword spotter on the 30-word dataset.

Everything here is TensorFlow only (no PyTorch / torchaudio), so the trained
model and its MFCC front-end have a clean path to the nRF5340 via
TFLite-Micro / Edge Impulse.

Pipeline (read top to bottom):

    processed/<label>/*.wav   (16 kHz mono, 2 s)
            |
            v   [mfcc_tf.waveform_to_mfcc]   <- tf.signal, micro_speech-style
      MFCC features  (99 frames x 10 coeffs)   ... computed once and cached
            |
            v   [tiny SpecAugment: shift / mask / noise]   (training only)
            |
            v   [simple Conv2D CNN, this file]
      30 class scores (softmax)
            |
            v   [export]  kws_tf.keras, kws_tf_float.tflite, kws_tf_int8.tflite

Usage:
    python src/train_tf.py                          # MCU-safe DS-CNN default
    python src/train_tf.py --arch dscnn --width 128 --depth 6 --mixup 0.2
    python src/train_tf.py --rebuild-cache          # recompute MFCC cache
    python src/train_tf.py --smoke                   # 1 epoch on a tiny subset

The int8 .tflite is the artifact you flash / import into Edge Impulse.

--------------------------------------------------------------------------
Why this design (kept deliberately simple so it is easy to explain):

  * Features are plain MFCCs (see mfcc_tf.py) — the same DSP micro_speech uses.
  * Default model is a DS-CNN (depthwise + 1x1 convs): same idea as a CNN but
    much smaller, so it fits the nRF5340 after int8 quantization (~30-140 KB).
    A wide plain Conv2D can exceed the chip's 1 MB flash — the trainer blocks that.
  * Only ops that TFLite-Micro supports are used, so it runs on the MCU.
  * IMPORTANT FIX: BatchNormalization uses momentum=0.9 (not the Keras default
    0.99). With 0.99 the "moving" statistics used at inference lag far behind
    the real statistics, and across several BN layers the error compounds until
    inference accuracy collapses to random (~1/30). momentum=0.9 fixes this.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf

from config import PROJECT_ROOT, PROCESSED_DIR, load_config
from mfcc_tf import MFCCParams, waveform_to_mfcc

CACHE_PATH = PROJECT_ROOT / "mfcc_cache.npz"


# ---------------------------------------------------------------------------
# 1. Load audio and turn every clip into MFCC features (cached to disk)
# ---------------------------------------------------------------------------

def _list_clips(root: Path, labels: list[str]) -> tuple[list[Path], np.ndarray]:
    files: list[Path] = []
    ys: list[int] = []
    for i, label in enumerate(labels):
        for wav in sorted((root / label).glob("*.wav")):
            files.append(wav)
            ys.append(i)
    return files, np.array(ys, dtype=np.int64)


def _load_waveform(path: Path, n_samples: int) -> np.ndarray:
    wav, _ = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:                      # stereo -> mono
        wav = wav.mean(axis=1)
    if len(wav) < n_samples:              # pad short clips with silence
        wav = np.pad(wav, (0, n_samples - len(wav)))
    else:                                 # truncate long clips
        wav = wav[:n_samples]
    return wav


# ---------------------------------------------------------------------------
# Background-noise augmentation (waveform level, TRAIN clips only)
# ---------------------------------------------------------------------------
# Real ambient noise from the Google Speech Commands `_background_noise_` files
# is mixed into training clips at a random SNR. This happens here, offline,
# while the feature cache is built — so the noisy MFCCs are baked into the cache
# and Anvil can train straight from the cache on GPU without needing the audio.
# Only TRAIN clips are augmented; val/test stay clean for an honest metric.

def load_noise_bank(noise_dir: Path, sample_rate: int) -> list[np.ndarray]:
    """Load every background-noise wav into memory as float32 mono @ sample_rate."""
    if not noise_dir.is_dir():
        raise SystemExit(f"[error] noise dir not found: {noise_dir}")
    bank: list[np.ndarray] = []
    for wav in sorted(noise_dir.glob("*.wav")):
        y, sr = sf.read(str(wav), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != sample_rate:                       # crude resample (noise; quality uncritical)
            n_new = int(round(len(y) * sample_rate / sr))
            y = np.interp(np.linspace(0, len(y), n_new, endpoint=False),
                          np.arange(len(y)), y).astype(np.float32)
        bank.append(y.astype(np.float32))
    if not bank:
        raise SystemExit(f"[error] no .wav noise files in {noise_dir}")
    return bank


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add `noise` to `clean`, scaled to the requested signal-to-noise ratio (dB)."""
    clean_rms = float(np.sqrt(np.mean(clean ** 2)) + 1e-9)
    noise_rms = float(np.sqrt(np.mean(noise ** 2)) + 1e-9)
    snr = 10.0 ** (snr_db / 20.0)
    scale = clean_rms / (snr * noise_rms)
    mixed = clean + scale * noise
    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:                                   # avoid hard clipping after mixing
        mixed = mixed / peak
    return mixed.astype(np.float32)


def random_noisy(wav: np.ndarray, bank: list[np.ndarray],
                 snr_min: float, snr_max: float,
                 rng: np.random.Generator) -> np.ndarray:
    """One noisy copy: random noise file, random offset slice, random SNR."""
    noise = bank[int(rng.integers(len(bank)))]
    n = len(wav)
    if len(noise) > n:
        start = int(rng.integers(0, len(noise) - n + 1))
        seg = noise[start:start + n]
    else:
        seg = np.tile(noise, int(np.ceil(n / len(noise))))[:n]
    return mix_at_snr(wav, seg, float(rng.uniform(snr_min, snr_max)))


def build_feature_cache(root: Path, labels: list[str], p: MFCCParams, seed: int,
                        batch: int = 256, noise_aug: dict | None = None,
                        val_frac: float = 0.15, test_frac: float = 0.15
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute MFCC for every clip and return (X, y, split).

    X: (N, frames, num_mfcc) float32   y: (N,) int64   split: (N,) int8 {0,1,2}

    The train/val/test split is decided here at the CLIP level (0=train, 1=val,
    2=test). When `noise_aug` is set, each TRAIN clip also gets `copies` extra
    noisy versions (mixed real background noise) — never val/test — so there is
    no clip leakage across splits and the test metric stays clean.
    """
    files, y = _list_clips(root, labels)
    n = len(files)
    tr_clip, va_clip, te_clip = split_indices(n, seed, val_frac, test_frac)
    split_of = np.empty(n, dtype=np.int8)
    split_of[tr_clip] = 0
    split_of[va_clip] = 1
    split_of[te_clip] = 2

    copies = int(noise_aug["copies"]) if noise_aug else 0
    total = n + copies * len(tr_clip)
    X = np.empty((total, p.num_frames, p.num_mfcc), dtype=np.float32)
    Y = np.empty((total,), dtype=np.int64)
    S = np.empty((total,), dtype=np.int8)

    bank = load_noise_bank(Path(noise_aug["dir"]), p.sample_rate) if noise_aug else None
    rng = np.random.default_rng(noise_aug["seed"]) if noise_aug else None

    out = 0
    buf_w: list[np.ndarray] = []
    buf_l: list[int] = []
    buf_s: list[int] = []

    def flush() -> None:
        nonlocal out
        if not buf_w:
            return
        feats = waveform_to_mfcc(tf.constant(np.stack(buf_w)), p).numpy()
        m = len(feats)
        X[out:out + m] = feats
        Y[out:out + m] = buf_l
        S[out:out + m] = buf_s
        out += m
        buf_w.clear()
        buf_l.clear()
        buf_s.clear()

    print(f"[cache] computing MFCC for {n} clips"
          + (f" + {copies}x noise on {len(tr_clip)} train clips" if copies else "")
          + " ...")
    for i in range(n):
        wav = _load_waveform(files[i], p.n_samples)
        lbl = int(y[i])
        sp = int(split_of[i])
        buf_w.append(wav)
        buf_l.append(lbl)
        buf_s.append(sp)
        if sp == 0 and copies:                       # augment TRAIN clips only
            for _ in range(copies):
                buf_w.append(random_noisy(wav, bank, noise_aug["snr_min"],
                                          noise_aug["snr_max"], rng))
                buf_l.append(lbl)
                buf_s.append(0)
        if len(buf_w) >= batch:
            flush()
        if (i + 1) % 2000 == 0 or i + 1 == n:
            print(f"\r[cache]   {i + 1}/{n} clips -> {out} feats", end="", flush=True)
    flush()
    print()
    return X[:out], Y[:out], S[:out]


def load_features(root: Path, labels: list[str], p: MFCCParams, seed: int,
                  rebuild: bool = False, noise_aug: dict | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return (X, y, split). `split` is per-row {0,1,2} or None for old caches."""
    if CACHE_PATH.exists() and not rebuild:
        data = np.load(CACHE_PATH, allow_pickle=True)
        cached_labels = [str(x) for x in data["labels"]]
        matches = (cached_labels == labels
                   and int(data["n_frames"]) == p.num_frames
                   and int(data["X"].shape[2]) == p.num_mfcc)  # feature width from array
        if matches:
            split = data["split"] if "split" in data.files else None
            note = f"  noise_aug={data['noise_aug']}" if "noise_aug" in data.files else ""
            print(f"[cache] loaded {CACHE_PATH.name}  X={data['X'].shape}{note}")
            return data["X"], data["y"], split
        print("[cache] config changed (labels/frames/coeffs) — rebuilding")
    if not root.is_dir():
        raise SystemExit(
            f"[error] need wavs in {root}/ (or a matching mfcc_cache.npz)."
        )
    X, y, split = build_feature_cache(root, labels, p, seed, noise_aug=noise_aug)
    save_kwargs: dict = dict(X=X, y=y, labels=np.array(labels),
                             n_frames=p.num_frames, n_coeffs=p.num_mfcc, split=split)
    if noise_aug:
        save_kwargs["noise_aug"] = np.array(
            f"copies={noise_aug['copies']},snr={noise_aug['snr_min']}-{noise_aug['snr_max']}dB")
    np.savez(CACHE_PATH, **save_kwargs)
    print(f"[cache] saved {CACHE_PATH.name}  X={X.shape}")
    return X, y, split


def resolve_labels(root: Path, rebuild: bool = False) -> list[str]:
    """Prefer label folders under processed/; else reuse labels stored in the cache.

    Lets you train on Anvil from a zip that ships mfcc_cache.npz without the
    full 1.8 GB processed/ tree (same idea as the old --hf download shortcut).
    """
    if root.is_dir():
        labels = sorted(d.name for d in root.iterdir() if d.is_dir())
        if labels:
            return labels
    if CACHE_PATH.exists() and not rebuild:
        data = np.load(CACHE_PATH, allow_pickle=True)
        return [str(x) for x in data["labels"]]
    raise SystemExit(
        f"[error] no labels found. Put wav folders in {root}/ or include "
        f"{CACHE_PATH.name} in your upload zip."
    )


# ---------------------------------------------------------------------------
# 2. Split into train / val / test
# ---------------------------------------------------------------------------

def split_indices(n: int, seed: int, val_frac=0.15, test_frac=0.15):
    idx = np.random.default_rng(seed).permutation(n)
    n_test = int(test_frac * n)
    n_val = int(val_frac * n)
    return (idx[n_test + n_val:],            # train
            idx[n_test:n_test + n_val],       # val
            idx[:n_test])                     # test


# ---------------------------------------------------------------------------
# 3. Tiny "SpecAugment" — training-only data augmentation
# ---------------------------------------------------------------------------
# These three cheap tricks stop the model from memorizing exact clips and
# reliably add a couple of accuracy points. They ONLY touch training data, so
# they have zero effect on the model that ends up on the device.
#
#   * time shift : slide the word a few frames earlier/later (it never starts
#                  at exactly the same instant in real life).
#   * time mask  : blank out a short strip of frames (forces the model to use
#                  the whole word, not one lucky moment).
#   * noise      : add a little random jitter to the features.

def make_augmenter(n_frames: int, n_coeffs: int = 10, max_shift: int = 8,
                   max_time_mask: int = 16, max_freq_mask: int = 3,
                   noise_std: float = 0.12):
    def augment(x, y):
        # x: (frames, coeffs, 1)
        # -- time shift --
        shift = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
        x = tf.roll(x, shift, axis=0)

        # -- time mask: blank a short strip of frames --
        t0 = tf.random.uniform([], 0, n_frames, dtype=tf.int32)
        tw = tf.random.uniform([], 0, max_time_mask + 1, dtype=tf.int32)
        t_keep = tf.logical_or(tf.range(n_frames) < t0,
                               tf.range(n_frames) >= t0 + tw)
        x = x * tf.cast(t_keep, x.dtype)[:, None, None]

        # -- freq mask: blank a few MFCC coefficients --
        f0 = tf.random.uniform([], 0, n_coeffs, dtype=tf.int32)
        fw = tf.random.uniform([], 0, max_freq_mask + 1, dtype=tf.int32)
        f_keep = tf.logical_or(tf.range(n_coeffs) < f0,
                               tf.range(n_coeffs) >= f0 + fw)
        x = x * tf.cast(f_keep, x.dtype)[None, :, None]

        # -- small additive noise --
        x = x + tf.random.normal(tf.shape(x), stddev=noise_std, dtype=x.dtype)
        return x, y
    return augment


def make_dataset(X, y, batch_size, training, seed, augmenter=None,
                 mixup_alpha: float = 0.0, n_classes: int = 30):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
        if augmenter is not None:
            ds = ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)

    # Mixup blends two clips + their labels (train only). Helps generalization.
    if training and mixup_alpha > 0:
        alpha = float(mixup_alpha)

        def _mixup(x, y):
            batch = tf.shape(x)[0]
            # Beta(alpha, alpha) via two Gammas (no tensorflow_probability needed)
            g1 = tf.random.gamma([], alpha)
            g2 = tf.random.gamma([], alpha)
            lam = g1 / (g1 + g2)
            lam = tf.cast(tf.maximum(lam, 1.0 - lam), x.dtype)  # prefer >= 0.5
            idx = tf.random.shuffle(tf.range(batch))
            x2 = tf.gather(x, idx)
            y2 = tf.gather(y, idx)
            x = lam * x + (1.0 - lam) * x2
            y1 = tf.one_hot(y, n_classes, dtype=x.dtype)
            y2 = tf.one_hot(y2, n_classes, dtype=x.dtype)
            y = lam * y1 + (1.0 - lam) * y2
            return x, y

        ds = ds.map(_mixup, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# 4. The model — MCU-friendly classifiers (nRF5340 budget)
# ---------------------------------------------------------------------------
# nRF5340 app core: 1 MB flash + 512 KB RAM. After Zephyr + BLE + MFCC buffers,
# keep the int8 model roughly under ~200 KB. A plain wide Conv2D CNN blows past
# that; a DS-CNN (depthwise + 1x1) gives more accuracy per byte and is what
# TinyML / MLPerf Tiny / your old PyTorch trainer used.
#
# Rough int8 sizes (weights only):
#   cnn   width=64  depth=3  -> ~200-250 KB   (OK, tight)
#   cnn   width=128 depth=4  -> >1 MB         (WILL NOT FIT)
#   dscnn width=64  depth=4  -> ~30 KB        (comfortable)
#   dscnn width=128 depth=6  -> ~140 KB       (accuracy push, still OK)

# Soft ceiling on parameter count for --arch defaults (override with --allow-large).
MCU_MAX_PARAMS = 200_000


def build_cnn(input_shape: tuple[int, int, int], n_classes: int,
              width: int = 64, depth: int = 3,
              dropout: float = 0.3) -> tf.keras.Model:
    """Plain Conv2D CNN. Simple to explain; keep width<=64 for the nRF5340."""
    L = tf.keras.layers
    inp = tf.keras.Input(shape=input_shape, name="mfcc")

    def block(x, filters, pool):
        x = L.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = L.BatchNormalization(momentum=0.9)(x)   # <- key fix (was 0.99)
        x = L.ReLU()(x)
        return L.MaxPooling2D(pool)(x)

    plan = [
        (width, (2, 1)),
        (width * 2, (2, 2)),
        (width * 2, (2, 1)),
        (width * 2, (2, 1)),
    ]
    x = inp
    for filters, pool in plan[: max(2, min(depth, 4))]:
        x = block(x, filters, pool)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(dropout)(x)
    out = L.Dense(n_classes, activation="softmax", name="scores")(x)
    return tf.keras.Model(inp, out, name="kws_cnn")


def build_dscnn(input_shape: tuple[int, int, int], n_classes: int,
                width: int = 64, depth: int = 4,
                dropout: float = 0.2) -> tf.keras.Model:
    """Depthwise-separable CNN (TinyML / MLPerf Tiny family).

    Same idea as a normal CNN, but each block is:
      DepthwiseConv (looks at each channel alone) → 1x1 Conv (mixes channels)
    That cuts parameters a lot, so we can go wider/deeper and still flash on
    the nRF5340. Ops are TFLite-Micro / CMSIS-NN friendly.
    """
    L = tf.keras.layers
    inp = tf.keras.Input(shape=input_shape, name="mfcc")
    x = L.Conv2D(width, (10, 4), strides=(2, 2), padding="same",
                 use_bias=False)(inp)
    x = L.BatchNormalization(momentum=0.9)(x)
    x = L.ReLU()(x)
    for _ in range(max(1, depth)):
        x = L.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
        x = L.BatchNormalization(momentum=0.9)(x)
        x = L.ReLU()(x)
        x = L.Conv2D(width, 1, use_bias=False)(x)
        x = L.BatchNormalization(momentum=0.9)(x)
        x = L.ReLU()(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(dropout)(x)
    out = L.Dense(n_classes, activation="softmax", name="scores")(x)
    return tf.keras.Model(inp, out, name="kws_dscnn")


def build_model(arch: str, input_shape, n_classes: int, width: int, depth: int,
                dropout: float) -> tf.keras.Model:
    if arch == "dscnn":
        return build_dscnn(input_shape, n_classes, width=width, depth=depth,
                           dropout=dropout)
    if arch == "cnn":
        return build_cnn(input_shape, n_classes, width=width, depth=depth,
                         dropout=dropout)
    raise SystemExit(f"[error] unknown --arch {arch!r} (use cnn or dscnn)")


def assert_mcu_budget(model: tf.keras.Model, allow_large: bool) -> None:
    n = int(model.count_params())
    # int8 flatbuffer is roughly params bytes + graph overhead
    est_kb = n / 1024.0 * 1.2
    print(f"[model] params={n:,}  est_int8≈{est_kb:.0f} KB  "
          f"(nRF5340 budget ≈{MCU_MAX_PARAMS/1024*1.2:.0f} KB)")
    if n > MCU_MAX_PARAMS and not allow_large:
        raise SystemExit(
            f"[error] model too large for nRF5340 (params={n:,}). "
            f"Use --arch dscnn (recommended), keep --width<=128 --depth<=6, "
            f"or pass --allow-large to override."
        )


# ---------------------------------------------------------------------------
# 5. Export to TFLite (float + int8 for the MCU)
# ---------------------------------------------------------------------------

def export_saved_model(model: tf.keras.Model, out_dir: Path) -> Path:
    """Save the un-optimized float graph as a TF SavedModel (no quantization).

    This is the artifact a teammate should quantize FROM: it keeps the full
    float model + serving signature, so post-training quantization (or QAT)
    can be applied cleanly, instead of reverse-engineering a finished .tflite.
    """
    sm_dir = out_dir / "kws_tf_savedmodel"
    if hasattr(model, "export"):          # Keras 3 (TF >= 2.16)
        model.export(str(sm_dir))
    else:                                  # older Keras / TF 2.x fallback
        model.save(str(sm_dir), save_format="tf")
    print(f"[export] SavedModel -> {sm_dir.name}/  (float, pre-quantization)")
    return sm_dir


def export_tflite(model: tf.keras.Model, X_train: np.ndarray, out_dir: Path):
    # float32 tflite (sanity / desktop use)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    float_path = out_dir / "kws_tf_float.tflite"
    float_path.write_bytes(conv.convert())

    # int8 tflite — this is what runs on the nRF5340. Weights AND activations
    # are quantized to 8-bit ints using a small "representative" sample so the
    # converter can measure the real range of every tensor.
    def rep_dataset():
        for i in range(min(300, len(X_train))):
            yield [X_train[i:i + 1]]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    int8_path = out_dir / "kws_tf_int8.tflite"
    int8_path.write_bytes(conv.convert())

    print(f"[export] {float_path.name}  ({float_path.stat().st_size/1024:.0f} KB)")
    print(f"[export] {int8_path.name}   ({int8_path.stat().st_size/1024:.0f} KB)")
    return float_path, int8_path


# ---------------------------------------------------------------------------
# 6. Reporting helpers
# ---------------------------------------------------------------------------

def per_class_and_confusions(model, X, y, labels, top_k=12):
    pred = model.predict(X, batch_size=512, verbose=0).argmax(1)
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, pr in zip(y, pred):
        cm[t, pr] += 1
    print("[test] per-class accuracy:")
    for i, label in enumerate(labels):
        tot = cm[i].sum()
        acc = cm[i, i] / tot if tot else 0.0
        print(f"       {label:14s} {acc:.2f}  (n={tot})")
    pairs = [(cm[i, j], labels[i], labels[j])
             for i in range(n) for j in range(n) if i != j and cm[i, j] > 0]
    pairs.sort(reverse=True)
    print("[test] top confusions (true -> pred):")
    for c, a, b in pairs[:top_k]:
        print(f"       {a:14s} -> {b:14s} {c}")
    return cm


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Train TF CNN keyword spotter.")
    ap.add_argument("--data-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--arch", choices=("dscnn", "cnn"), default="dscnn",
                    help="dscnn = MCU-friendly (default); cnn = plain Conv2D.")
    ap.add_argument("--width", type=int, default=64,
                    help="Base channels/filters (dscnn: try 128 to push acc).")
    ap.add_argument("--depth", type=int, default=4,
                    help="Blocks (dscnn: 4–6; cnn: 3–4).")
    ap.add_argument("--dropout", type=float, default=None,
                    help="Default 0.2 for dscnn, 0.3 for cnn.")
    ap.add_argument("--mixup", type=float, default=0.0,
                    help="Mixup alpha (0=off; try 0.2 to push accuracy).")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="Softens hard labels (try 0.05).")
    ap.add_argument("--allow-large", action="store_true",
                    help="Allow models above the nRF5340 size budget.")
    ap.add_argument("--init", default=None,
                    help="Start from an existing .keras checkpoint (fine-tune).")
    ap.add_argument("--class-weight", action="store_true",
                    help="Up-weight rarer/harder classes (best with --mixup 0).")
    ap.add_argument("--no-augment", action="store_true",
                    help="Disable SpecAugment (train on raw features only).")
    ap.add_argument("--noise-aug", action="store_true",
                    help="Bake real background noise into TRAIN clips (needs --rebuild-cache).")
    ap.add_argument("--noise-copies", type=int, default=1,
                    help="Noisy copies per training clip (1 doubles the train set).")
    ap.add_argument("--snr-min", type=float, default=0.0,
                    help="Minimum mix SNR in dB (lower = noisier).")
    ap.add_argument("--snr-max", type=float, default=15.0,
                    help="Maximum mix SNR in dB.")
    ap.add_argument("--noise-dir",
                    default="downloads/speech_commands_v0.02/_background_noise_",
                    help="Folder of background-noise wav files.")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="Fast sanity run: tiny subset, 1 epoch, no export.")
    ap.add_argument("--out", default="kws_tf.keras")
    args = ap.parse_args()
    if args.dropout is None:
        args.dropout = 0.2 if args.arch == "dscnn" else 0.3

    cfg = load_config()
    seed = cfg["sampling"]["seed"]
    tf.random.set_seed(seed)
    np.random.seed(seed)
    p = MFCCParams(cfg)

    root = Path(args.data_dir)
    labels = resolve_labels(root, rebuild=args.rebuild_cache)
    print(f"[data] {len(labels)} classes: {', '.join(labels)}")

    # --- features ---
    noise_aug_cfg = None
    if args.noise_aug:
        noise_aug_cfg = {
            "dir": args.noise_dir, "copies": args.noise_copies,
            "snr_min": args.snr_min, "snr_max": args.snr_max, "seed": seed,
        }
        if not args.rebuild_cache:
            print("[warn] --noise-aug only applies while (re)building the cache; "
                  "add --rebuild-cache to bake noisy clips in.")
    X, y, split = load_features(root, labels, p, seed,
                                rebuild=args.rebuild_cache, noise_aug=noise_aug_cfg)

    # Split: prefer the clip-level split stored in the cache (keeps noisy copies
    # out of val/test); fall back to a random row split for older caches.
    if split is not None:
        tr_idx = np.where(split == 0)[0]
        va_idx = np.where(split == 1)[0]
        te_idx = np.where(split == 2)[0]
    else:
        tr_idx, va_idx, te_idx = split_indices(len(X), seed)

    # Standardize with train statistics (helps optimization + quantization).
    mean = X[tr_idx].mean().astype(np.float32)
    std = (X[tr_idx].std() + 1e-6).astype(np.float32)
    X = ((X - mean) / std).astype(np.float32)
    X = X[..., None]  # add channel dim -> (N, frames, coeffs, 1)

    if args.smoke:
        # keep it tiny so this finishes in ~1 min on a laptop CPU
        rng = np.random.default_rng(seed)
        tr_idx = rng.choice(tr_idx, size=min(2000, len(tr_idx)), replace=False)
        va_idx = va_idx[:1000]
        args.epochs = 1

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]
    Xte, yte = X[te_idx], y[te_idx]
    print(f"[data] train={len(Xtr)} val={len(Xva)} test={len(Xte)} "
          f"feat_shape={Xtr.shape[1:]}")

    # --- data pipelines (augment only the training set) ---
    augmenter = None if args.no_augment else make_augmenter(
        p.num_frames, n_coeffs=p.num_mfcc)
    use_mixup = args.mixup > 0 and not args.smoke
    train_ds = make_dataset(
        Xtr, ytr, args.batch_size, training=True, seed=seed,
        augmenter=augmenter, mixup_alpha=args.mixup if use_mixup else 0.0,
        n_classes=len(labels),
    )
    val_ds = make_dataset(Xva, yva, args.batch_size, training=False, seed=seed)

    # --- model ---
    if args.init:
        print(f"[model] fine-tuning from {args.init}")
        model = tf.keras.models.load_model(args.init)
        # architecture must match the checkpoint (same arch/width/depth)
    else:
        model = build_model(
            args.arch, Xtr.shape[1:], len(labels),
            width=args.width, depth=args.depth, dropout=args.dropout,
        )
    assert_mcu_budget(model, allow_large=args.allow_large or args.smoke)

    # Prefer categorical CE when mixup OR label_smoothing is on.
    # Some TF/Keras builds reject label_smoothing on SparseCategoricalCrossentropy.
    use_categorical = use_mixup or args.label_smoothing > 0

    # Class weights for hard words. Applied as per-example sample_weight so it
    # still works with one-hot labels / label_smoothing. Disabled with mixup.
    sample_weight_table = None
    if args.class_weight:
        if use_mixup:
            print("[warn] --class-weight ignored when mixup>0; use --mixup 0")
        else:
            counts = np.bincount(ytr, minlength=len(labels)).astype(np.float32)
            hard = {"left", "start", "six", "close", "play", "next", "eight"}
            inv = counts.sum() / (counts + 1e-6)
            for i, name in enumerate(labels):
                if name in hard:
                    inv[i] *= 1.35
            inv = (inv / inv.mean()).astype(np.float32)
            sample_weight_table = tf.constant(inv)
            print("[train] class_weight on; hard-class boost for:",
                  ", ".join(sorted(hard)))

    if use_categorical:
        loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=args.label_smoothing)
        if not use_mixup:
            if sample_weight_table is not None:
                wtab = sample_weight_table
                train_ds = train_ds.map(
                    lambda x, y: (x, tf.one_hot(y, len(labels)),
                                  tf.gather(wtab, y)),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
            else:
                train_ds = train_ds.map(
                    lambda x, y: (x, tf.one_hot(y, len(labels))),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
        val_ds = val_ds.map(
            lambda x, y: (x, tf.one_hot(y, len(labels))),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        Xte_eval, yte_eval = Xte, tf.keras.utils.to_categorical(yte, len(labels))
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        if sample_weight_table is not None:
            wtab = sample_weight_table
            train_ds = train_ds.map(
                lambda x, y: (x, y, tf.gather(wtab, y)),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        Xte_eval, yte_eval = Xte, yte

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=loss,
        metrics=["accuracy"],
    )
    model.summary()
    print(f"[train] arch={args.arch} width={args.width} depth={args.depth} "
          f"mixup={args.mixup} label_smoothing={args.label_smoothing} "
          f"init={args.init}")

    ckpt = Path(args.out)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt), monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=4, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=16, restore_best_weights=True),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # --- evaluate on held-out test set ---
    te_loss, te_acc = model.evaluate(Xte_eval, yte_eval, verbose=0)
    print(f"\n[test] accuracy = {te_acc:.4f}")

    if args.smoke:
        print("[smoke] OK — script runs end to end. Skipping export.")
        return

    cm = per_class_and_confusions(model, Xte, yte, labels)

    # --- export ---
    model.save(ckpt)
    sm_dir = export_saved_model(model, PROJECT_ROOT)  # float, for teammate's quantization
    float_path, int8_path = export_tflite(model, Xtr, PROJECT_ROOT)

    meta = {
        "labels": labels,
        "test_acc": float(te_acc),
        "feature_shape": list(Xtr.shape[1:]),
        "mfcc": {
            "sample_rate": p.sample_rate, "frame_length": p.frame_length,
            "frame_step": p.frame_step, "fft_length": p.fft_length,
            "num_mel_bins": p.num_mel_bins, "num_mfcc": p.num_mfcc,
            "lower_hz": p.lower_hz, "upper_hz": p.upper_hz,
        },
        "feature_norm": {"mean": float(mean), "std": float(std)},
        "train": {
            "arch": args.arch, "width": args.width, "depth": args.depth,
            "epochs": args.epochs, "mixup": args.mixup,
            "label_smoothing": args.label_smoothing,
        },
        "int8_tflite": int8_path.name,
        "saved_model_dir": sm_dir.name,
        "noise_aug": (f"copies={args.noise_copies},snr={args.snr_min}-{args.snr_max}dB"
                      if args.noise_aug else "none"),
        "n_params": int(model.count_params()),
    }
    Path(ckpt).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"[done] saved {ckpt.name}, {ckpt.with_suffix('.json').name}, "
          f"{sm_dir.name}/, {float_path.name}, {int8_path.name}")


if __name__ == "__main__":
    main()
