"""Generate oracle_clips.h for firmware: one training clip per requested label.

Picks a random clip from the parquet dataset for each label, then writes a C
header with one int16[32000] array per word and a lookup table so the firmware
can select clips at runtime with the 'oracle' command.

Clips are taken verbatim from the training data (already loudness-normalised by
the training pipeline), so the on-device MFCC path sees exactly the input the
model was trained on — no additional normalisation applied.

Usage:
    python make_oracle_clips.py --labels set_a_timer volume_down yes no stop
    python make_oracle_clips.py --labels set_a_timer yes --seed 42   # reproducible
    python make_oracle_clips.py --labels set_a_timer yes --out /path/to/oracle_clips.h

After running, note the printed clip indices. To inspect a clip in browse_clips.py:
    python browse_clips.py
    → select the split tab, click the label, use the slider to jump to the index.

Then copy oracle_clips.h to your firmware src/ directory and rebuild:
    cp oracle_clips.h /path/to/nrfApps/tfml/src/oracle_clips.h
Or pass --out directly:
    python make_oracle_clips.py --labels yes no --out /path/to/nrfApps/tfml/src/oracle_clips.h
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

SCRIPT_DIR  = Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR / "data"
LABELS_JSON = SCRIPT_DIR / "kws_tf.json"

# Default output: ds-cnn repo root — copy to firmware src/ manually or pass --out.
DEFAULT_OUT = SCRIPT_DIR / "oracle_clips.h"

N = 32000   # 2 s at 16 kHz — must match RAW_SAMPLE_COUNT in firmware


def load_random_clip(
    label: str,
    label_idx: int,
    split: str,
    rng: random.Random,
) -> tuple[np.ndarray, float, int, int]:
    """Reservoir-sample one clip for label_idx.

    Returns (int16[N], dbfs, chosen_label_index, total_count).
    chosen_label_index is the 0-based position among all clips for this label
    in the given split — the same index browse_clips uses to navigate.
    """
    pattern = "test-*.parquet" if split == "test" else "train-*.parquet"
    shards = sorted(DATA_DIR.glob(pattern))
    if not shards:
        sys.exit(f"[error] no {split}-*.parquet files in {DATA_DIR}")

    # Phase 1: label-column scan only (fast) — reservoir-sample one position
    chosen_shard_idx: int | None = None
    chosen_row: int | None = None
    chosen_label_idx = 0
    total = 0

    for shard_idx, shard in enumerate(shards):
        col = pq.read_table(str(shard), columns=["label"])["label"].to_pylist()
        for row, lbl in enumerate(col):
            if lbl == label_idx:
                if rng.randint(0, total) == 0:   # prob 1/(total+1)
                    chosen_shard_idx = shard_idx
                    chosen_row = row
                    chosen_label_idx = total
                total += 1

    if total == 0:
        sys.exit(f"[error] no clips found for '{label}' in {split} split")

    # Phase 2: load audio for the chosen shard/row
    t = pq.read_table(str(shards[chosen_shard_idx]))
    wav_bytes = t["audio"][chosen_row].as_py()["bytes"]
    wav_f32, _sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if wav_f32.ndim > 1:
        wav_f32 = wav_f32.mean(axis=1)
    if len(wav_f32) < N:
        wav_f32 = np.pad(wav_f32, (0, N - len(wav_f32)))
    else:
        wav_f32 = wav_f32[:N]

    rms = float(np.sqrt(np.mean(wav_f32 ** 2)) + 1e-9)
    dbfs = 20.0 * np.log10(rms)
    int16 = np.clip(np.round(wav_f32 * 32768.0), -32768, 32767).astype(np.int16)
    return int16, dbfs, chosen_label_idx, total


def emit_array(name: str, data: np.ndarray, per_line: int = 16) -> str:
    rows = [
        "  " + ",".join(str(int(v)) for v in data[i : i + per_line]) + ","
        for i in range(0, N, per_line)
    ]
    return f"static const int16_t oracle_{name}[{N}] = {{\n" + "\n".join(rows) + "\n};"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--labels", nargs="+", required=True,
        help="Label names (e.g. set_a_timer volume_down yes no stop)",
    )
    ap.add_argument(
        "--split", default="train", choices=["train", "test"],
        help="Dataset split to draw from (default: train)",
    )
    ap.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (omit for a fresh random pick each run)",
    )
    ap.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"Output path for oracle_clips.h (default: {DEFAULT_OUT})",
    )
    args = ap.parse_args()

    meta = json.loads(LABELS_JSON.read_text())
    all_labels: list[str] = meta["labels"]

    for lbl in args.labels:
        if lbl not in all_labels:
            sys.exit(f"[error] '{lbl}' not in model labels.\nValid labels: {all_labels}")

    rng = random.Random(args.seed)

    print(f"[oracle] drawing from '{args.split}' split"
          + (f" (seed={args.seed})" if args.seed is not None else " (random)"))
    print()

    clips: list[tuple[str, np.ndarray]] = []
    for label in args.labels:
        label_idx = all_labels.index(label)
        int16, dbfs, chosen_idx, total = load_random_clip(label, label_idx, args.split, rng)
        clips.append((label, int16))
        print(f"  {label}: clip {chosen_idx} of {total - 1}  ({dbfs:.1f} dBFS)")
        print(f"    → browse_clips: '{args.split}' tab → '{label}' → index {chosen_idx}")

    print()

    lines = [
        "/* Auto-generated by make_oracle_clips.py in ds-cnn/. Do not edit by hand. */",
        "#pragma once",
        "#include <stdint.h>",
        "",
    ]
    for name, data in clips:
        lines.append(emit_array(name, data))
        lines.append("")
    lines += [
        "typedef struct { const char *name; const int16_t *pcm; } oracle_clip_t;",
        "static const oracle_clip_t oracle_clips[] = {",
    ]
    for name, _ in clips:
        lines.append(f'  {{ "{name}", oracle_{name} }},')
    lines += [
        "};",
        f"static const int oracle_clip_count = {len(clips)};",
        "",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))

    flash_kb = len(clips) * 64
    print(f"[oracle] wrote {args.out}")
    print(f"[oracle] {len(clips)} clip(s), {flash_kb} KB of flash")
    print()
    print("Next steps:")
    print("  1. Build and flash firmware")
    print("  2. On device: 'oracle' to list clips, 'oracle <name>' to select")
    print("  3. 'snapshot' to run the oracle through the full MFCC+inference pipeline")


if __name__ == "__main__":
    main()
