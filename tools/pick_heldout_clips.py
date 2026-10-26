#!/usr/bin/env python3
"""Extract held-out raw wavs (train_tf's test split) as oracle source clips.

The oracle baseline uses training clips, and the highest-confidence words saturate the
int8 output (both chip and PC reach 127), which hides any rounding difference. This
extracts clips the model never trained on, including mid-confidence words (for example
search and five) that do not saturate, so the device-vs-PC gap stays visible.

It reproduces train_tf's clip-level split (same label order, same seed, same val/test
fractions) and copies, for each requested word, the first clips that land in the test
split. Copy the output to the host that builds the firmware, then build the oracle from
it with make_oracle_clips.py.

Usage:
    python tools/pick_heldout_clips.py --out heldout_clips \
        --words set_a_timer volume_down yes search five --per-word 1
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# config.py lives in src/ (train_tf.py imports it the same way when run from there)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import load_config, PROCESSED_DIR


def split_indices(n: int, seed: int, val_frac=0.15, test_frac=0.15):
    # identical to src/train_tf.py:split_indices
    idx = np.random.default_rng(seed).permutation(n)
    n_test = int(test_frac * n)
    n_val = int(val_frac * n)
    return (idx[n_test + n_val:], idx[n_test:n_test + n_val], idx[:n_test])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(PROCESSED_DIR))
    ap.add_argument("--out", default="heldout_clips")
    ap.add_argument("--words", nargs="+",
                    default=["set_a_timer", "volume_down", "yes", "search", "five"])
    ap.add_argument("--per-word", type=int, default=1)
    args = ap.parse_args()

    root = Path(args.root)
    seed = load_config()["sampling"]["seed"]

    # same ordering train_tf used: sorted label dirs, then sorted wavs per label
    labels = sorted(d.name for d in root.iterdir() if d.is_dir())
    files, ys = [], []
    for i, label in enumerate(labels):
        for wav in sorted((root / label).glob("*.wav")):
            files.append(wav); ys.append(i)
    ys = np.array(ys)
    n = len(files)
    _, _, test_idx = split_indices(n, seed)
    test_set = set(int(i) for i in test_idx)
    print(f"[split] n={n} clips, test split has {len(test_set)} clips (seed={seed})")

    out = Path(args.out); out.mkdir(exist_ok=True)
    picked = []
    for w in args.words:
        if w not in labels:
            print(f"[warn] label '{w}' not found; skipping"); continue
        c = labels.index(w)
        # global indices of this word that are in the TEST split
        word_test = [i for i in range(n) if ys[i] == c and i in test_set]
        if not word_test:
            print(f"[warn] no TEST-split clips for '{w}'"); continue
        for k, gi in enumerate(word_test[:args.per_word]):
            dst = out / f"{w}_{k}.wav"
            shutil.copy(files[gi], dst)
            picked.append((w, files[gi].name, dst.name))
            print(f"[pick] {w}: {files[gi].name}  (held-out) -> {dst}")

    print(f"\n[done] {len(picked)} clips in {out}/  (all from the test split, never trained on)")
    print("Copy this folder to the host that builds the firmware, then run make_oracle_clips.py on it.")


if __name__ == "__main__":
    main()
