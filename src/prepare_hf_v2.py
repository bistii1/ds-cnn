"""Download snowballlab/30-keywords-v2 and write processed wav folders.

The Hugging Face set is already 16 kHz mono, fixed 1 s, 1,000 clips per class,
with an official 80/20 train/test split. This script writes:

    <out>/<label>/train_NNNN.wav
    <out>/<label>/val_NNNN.wav
    <out>/<label>/test_NNNN.wav

Official HF test stays test. A seeded slice of official train becomes val so
the trainer can keep a clean holdout without leaking test clips.

Usage:
    pip install datasets
    python src/prepare_hf_v2.py
    python src/prepare_hf_v2.py --out processed_v2
"""
from __future__ import annotations

import argparse
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from config import PROCESSED_DIR, load_config

HF_ID = "snowballlab/30-keywords-v2"
HF_CLASSES = [
    "answer", "back", "call", "close", "down", "eight", "five", "four",
    "go", "help", "left", "next", "nine", "no", "one", "open", "play",
    "previous", "record", "right", "search", "seven", "six", "start",
    "stop", "three", "two", "up", "yes", "zero",
]


def _decode_wav(ex: dict, n_samples: int, sample_rate: int) -> np.ndarray:
    audio = ex["audio"]
    raw = audio.get("bytes") if isinstance(audio, dict) else None
    if raw:
        y, sr = sf.read(io.BytesIO(raw), dtype="float32")
    else:
        path = audio["path"] if isinstance(audio, dict) else audio
        y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != sample_rate:
        n_new = int(round(len(y) * sample_rate / sr))
        y = np.interp(
            np.linspace(0, len(y), n_new, endpoint=False),
            np.arange(len(y)), y,
        ).astype(np.float32)
    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)))
    else:
        y = y[:n_samples]
    return y.astype(np.float32)


def _write_split(ds, split_name: str, out: Path, names: list[str],
                 n_samples: int, sample_rate: int, counters: dict[str, int]) -> int:
    written = 0
    for ex in ds:
        label = names[int(ex["label"])]
        dest_dir = out / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        counters[label] += 1
        path = dest_dir / f"{split_name}_{counters[label]:04d}.wav"
        y = _decode_wav(ex, n_samples, sample_rate)
        sf.write(str(path), y, sample_rate, subtype="PCM_16")
        written += 1
        if written % 2000 == 0:
            print(f"[{split_name}] wrote {written} clips", flush=True)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(PROCESSED_DIR),
                    help="Destination folder (default: processed/)")
    ap.add_argument("--hf-id", default=HF_ID)
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="Fraction of official HF train to hold out as val")
    args = ap.parse_args()

    cfg = load_config()
    sample_rate = int(cfg["audio"]["sample_rate"])
    n_samples = int(sample_rate * cfg["audio"]["clip_ms"] / 1000)
    seed = int(cfg["sampling"]["seed"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    print(f"[hf] loading {args.hf_id} ...")
    ds = load_dataset(args.hf_id)
    names = list(ds["train"].features["label"].names)
    missing = [c for c in HF_CLASSES if c not in names]
    extra = [c for c in names if c not in HF_CLASSES]
    if missing or extra:
        print(f"[warn] vocab mismatch  missing={missing} extra={extra}")
    print(f"[hf] classes={len(names)}  train={len(ds['train'])}  test={len(ds['test'])}")
    print(f"[out] {out}  clip={n_samples} samples @ {sample_rate} Hz")

    split = ds["train"].train_test_split(test_size=args.val_frac, seed=seed,
                                         stratify_by_column="label")
    train_ds, val_ds = split["train"], split["test"]
    test_ds = ds["test"]

    counters: dict[str, int] = defaultdict(int)
    n_train = _write_split(train_ds, "train", out, names, n_samples, sample_rate, counters)
    counters.clear()
    n_val = _write_split(val_ds, "val", out, names, n_samples, sample_rate, counters)
    counters.clear()
    n_test = _write_split(test_ds, "test", out, names, n_samples, sample_rate, counters)
    print(f"[done] train={n_train} val={n_val} test={n_test} -> {out}")


if __name__ == "__main__":
    main()
