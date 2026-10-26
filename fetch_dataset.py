"""Download the snowballlab/30-keywords dataset from HuggingFace and arrange it as
processed/<label>/*.wav (16 kHz mono) -- the layout src/train_tf.py expects -- so the QAT
fine-tune can run without Bisti's prebuilt mfcc_cache.npz.

Confirmed schema (HF datasets-server): features are `audio` (raw bytes) and `label` (ClassLabel,
30 classes). Splits: train 24000, test 6000. Label names already match the model's labels.

Usage (on Anvil):
    pip install --user datasets soundfile numpy
    python fetch_dataset.py                      # writes ./processed/<label>/*.wav
    python fetch_dataset.py --limit-per-label 5  # tiny subset to verify first
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import soundfile as sf


def decode_audio(v, target_sr: int) -> np.ndarray:
    """Decode an example's audio value (raw bytes, or an Audio dict) to float32 mono @ target_sr."""
    if isinstance(v, dict):
        if v.get("array") is not None:                 # HF Audio feature (already decoded)
            arr = np.asarray(v["array"], dtype=np.float32)
            sr = int(v.get("sampling_rate", target_sr))
        elif v.get("bytes") is not None:               # Audio(decode=False) -> {bytes, path}
            arr, sr = sf.read(io.BytesIO(v["bytes"]), dtype="float32")
        elif v.get("path"):
            arr, sr = sf.read(v["path"], dtype="float32")
        else:
            raise ValueError(f"unrecognized audio dict keys: {list(v)}")
    elif isinstance(v, (bytes, bytearray)):            # raw binary column (this dataset)
        arr, sr = sf.read(io.BytesIO(v), dtype="float32")
    else:
        raise ValueError(f"unrecognized audio value type: {type(v)}")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != target_sr:                                # crude resample (dataset is already 16 kHz)
        n = int(round(len(arr) * target_sr / sr))
        arr = np.interp(np.linspace(0, len(arr), n, endpoint=False),
                        np.arange(len(arr)), arr).astype(np.float32)
    return arr.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="snowballlab/30-keywords")
    ap.add_argument("--out", default="processed")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--limit-per-label", type=int, default=0, help="0 = all clips")
    args = ap.parse_args()

    from datasets import load_dataset, concatenate_datasets

    ds = load_dataset(args.dataset)
    print("[info] splits:", {k: len(ds[k]) for k in ds})
    data = concatenate_datasets([ds[s] for s in ds]) if len(ds) > 1 else ds[list(ds)[0]]
    print("[info] columns:", data.column_names)

    audio_col = "audio" if "audio" in data.column_names else data.column_names[0]
    label_col = next((c for c in data.column_names
                      if c.lower() in ("label", "labels", "keyword", "word", "class")), "label")
    names = getattr(data.features[label_col], "names", None)
    if names is not None:
        print(f"[info] {len(names)} labels:", names)

    def label_name(v):
        return names[v] if (names is not None and isinstance(v, int)) else str(v)

    out = Path(args.out)
    counts: dict[str, int] = {}
    for i, ex in enumerate(data):
        lbl = label_name(ex[label_col]).strip().replace(" ", "_")
        if args.limit_per_label and counts.get(lbl, 0) >= args.limit_per_label:
            continue
        try:
            wav = decode_audio(ex[audio_col], args.sr)
        except Exception as e:
            print(f"[warn] skip clip {i} ({lbl}): {e}")
            continue
        d = out / lbl
        d.mkdir(parents=True, exist_ok=True)
        sf.write(str(d / f"{counts.get(lbl, 0):05d}.wav"), wav, args.sr, subtype="PCM_16")
        counts[lbl] = counts.get(lbl, 0) + 1
        if (i + 1) % 2000 == 0:
            print(f"  wrote {i + 1} clips ...", flush=True)

    print("[done] per-label counts:")
    for k in sorted(counts):
        print(f"  {k:16s} {counts[k]}")
    print(f"[done] {sum(counts.values())} clips -> {out}/   ({len(counts)} labels)")


if __name__ == "__main__":
    main()
