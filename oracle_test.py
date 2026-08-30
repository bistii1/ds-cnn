"""
Oracle test: feed a known training WAV through the MCU pipeline to verify
the model recognizes its own training data at full confidence.

If this passes (high confidence on device) but real speech doesn't → amplitude
mismatch between your live mic and the -20 dBFS training normalization.

Usage:
    # Generate a C header for a specific label (picks the first clip from parquet)
    python oracle_test.py --label yes --out oracle_pcm.h

    # Specify a WAV file directly
    python oracle_test.py --wav some_clip.wav --out oracle_pcm.h

    # Also compare amplitude vs a snapshot log
    python oracle_test.py --label yes --snapshot snapshot.txt --tflite kws_tf_int8.tflite

    # Pick a specific clip index (0 = first, 1 = second, etc.)
    python oracle_test.py --label yes --clip-index 3 --out oracle_pcm.h

After this script, in firmware (MLInferenceCtrl.cpp), add at the top:
    #include "oracle_pcm.h"   // <-- place oracle_pcm.h in src/

Then in thread_ml_continuous_consumer, right before compute_mfccs_incremental:
    memcpy(raw_window_buffer, oracle_pcm, sizeof(oracle_pcm));  // ORACLE TEST

That one line feeds the training WAV into the MFCC path. Remove it when done.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# --- resolve project paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
DATA_DIR = SCRIPT_DIR / "data"
sys.path.insert(0, str(SRC_DIR))

from config import load_config
from mfcc_tf import MFCCParams


N_SAMPLES = 32000   # matches firmware RAW_SAMPLE_COUNT (2 s @ 16 kHz)


def _wav_bytes_to_float(wav_bytes: bytes) -> np.ndarray:
    """Decode WAV bytes → float32 mono array, pad/truncate to N_SAMPLES."""
    wav, _sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if len(wav) < N_SAMPLES:
        wav = np.pad(wav, (0, N_SAMPLES - len(wav)))
    else:
        wav = wav[:N_SAMPLES]
    return wav


def load_from_parquet(label: str, clip_index: int = 0) -> tuple[np.ndarray, float, str]:
    """Find `clip_index`-th clip for `label` across all parquet shards.
    Returns (int16_array, rms_dbfs, source_path).
    """
    import pyarrow.parquet as pq

    meta = json.loads((SCRIPT_DIR / "kws_tf.json").read_text())
    labels = meta["labels"]
    if label not in labels:
        sys.exit(f"[error] '{label}' not in model labels: {labels}")
    target_idx = labels.index(label)

    shards = sorted(DATA_DIR.glob("*.parquet"))
    if not shards:
        sys.exit(f"[error] no .parquet files in {DATA_DIR}")

    found = 0
    for shard in shards:
        t = pq.read_table(str(shard))
        for row in range(len(t)):
            if int(t["label"][row].as_py()) == target_idx:
                if found == clip_index:
                    wav_bytes = t["audio"][row].as_py()["bytes"]
                    src = t["audio"][row].as_py().get("path", shard.name)
                    wav_f32 = _wav_bytes_to_float(wav_bytes)
                    rms = float(np.sqrt(np.mean(wav_f32 ** 2)) + 1e-9)
                    dbfs = 20.0 * np.log10(rms)
                    wav_int16 = np.clip(
                        np.round(wav_f32 * 32768.0), -32768, 32767
                    ).astype(np.int16)
                    return wav_int16, dbfs, str(src)
                found += 1

    sys.exit(f"[error] only {found} clip(s) found for '{label}' (requested index {clip_index})")


def load_from_wav(path: Path) -> tuple[np.ndarray, float]:
    """Load any WAV file directly."""
    wav, _sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if len(wav) < N_SAMPLES:
        wav = np.pad(wav, (0, N_SAMPLES - len(wav)))
    else:
        wav = wav[:N_SAMPLES]
    rms = float(np.sqrt(np.mean(wav ** 2)) + 1e-9)
    dbfs = 20.0 * np.log10(rms)
    wav_int16 = np.clip(np.round(wav * 32768.0), -32768, 32767).astype(np.int16)
    return wav_int16, dbfs


def write_c_header(int16_arr: np.ndarray, label: str, src_path: str,
                   dbfs: float, out_path: Path) -> None:
    n = len(int16_arr)
    src_name = Path(src_path).name
    lines = [
        f"/* Oracle test PCM — {label} — {src_name} — {dbfs:.1f} dBFS",
        f" * Feed into raw_window_buffer with:",
        f" *   memcpy(raw_window_buffer, oracle_pcm, sizeof(oracle_pcm));",
        f" * placed right before compute_mfccs_incremental() in MLInferenceCtrl.cpp",
        f" */",
        f"#pragma once",
        f"#include <stdint.h>",
        f"static const int16_t oracle_pcm[{n}] = {{",
    ]
    for i in range(0, n, 16):
        chunk = int16_arr[i : i + 16]
        lines.append("    " + ", ".join(str(v) for v in chunk) + ",")
    lines.append("};")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[oracle] wrote {out_path}  ({n} samples, {dbfs:.1f} dBFS)")


def rms_dbfs(int16_arr: np.ndarray) -> float:
    f = int16_arr.astype(np.float32) / 32768.0
    return 20.0 * float(np.log10(np.sqrt(np.mean(f ** 2)) + 1e-9))


def compare_amplitudes(training_int16: np.ndarray, snapshot_int16: np.ndarray) -> None:
    t_db = rms_dbfs(training_int16)
    s_db = rms_dbfs(snapshot_int16)
    delta = s_db - t_db
    print(f"\n[amplitude]")
    print(f"  training WAV  : {t_db:.1f} dBFS")
    print(f"  snapshot (mic): {s_db:.1f} dBFS")
    print(f"  delta         : {delta:+.1f} dB  (positive = mic is louder)")
    if abs(delta) < 3:
        print("  -> Amplitude OK (within 3 dB).")
    elif abs(delta) < 8:
        print("  -> Moderate mismatch. Unlikely to hurt much (log compression absorbs it),")
        print("     but worth investigating DMIC gain if accuracy is low.")
    else:
        print("  -> LARGE mismatch. This is likely hurting accuracy.")
        print("     Consider: re-normalizing training data OR adding gain normalization")
        print("     in the MFCC path (scale audio to target RMS before MFCC).")


def parse_pcm_from_log(log_text: str) -> np.ndarray:
    values = []
    for line in log_text.splitlines():
        if line.startswith("PCM,"):
            values.extend(int(v) for v in line[len("PCM,"):].split(","))
    return np.array(values, dtype=np.int16)


def run_tflite(int16_arr: np.ndarray, tflite_path: str, label: str) -> None:
    import tensorflow as tf
    from mfcc_arm import ArmMFCC

    cfg = load_config()
    p = MFCCParams(cfg)
    dsp_cfg = cfg.get("dsp", {})
    frontend = ArmMFCC(p, precision="f32", window=dsp_cfg.get("window", "hann"))

    wav_f32 = int16_arr.astype(np.float32) / 32768.0
    mfcc = frontend.one(wav_f32)   # (num_frames, num_coeffs), raw pre-norm

    meta_path = SCRIPT_DIR / "kws_tf.json"
    meta = json.loads(meta_path.read_text())
    mean = meta["feature_norm"]["mean"]
    std  = meta["feature_norm"]["std"]
    labels = meta["labels"]
    normalized = (mfcc - mean) / std

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_d  = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_scale, in_zp = in_d["quantization"]
    out_scale, out_zp = out_d["quantization"]

    q = np.round(normalized / in_scale + in_zp).clip(-128, 127).astype(np.int8)
    interp.set_tensor(in_d["index"], q.reshape(in_d["shape"]))
    interp.invoke()
    out = interp.get_tensor(out_d["index"])[0]
    pcts = (out.astype(np.float32) - out_zp) * out_scale * 100.0

    top5 = sorted(enumerate(pcts), key=lambda x: -x[1])[:5]
    print(f"\n[python model] top-5 predictions for '{label}' oracle:")
    for idx, pct in top5:
        marker = " <-- expected" if labels[idx] == label else ""
        print(f"  {labels[idx]:16s} {pct:.1f}%{marker}")
    expected_idx = labels.index(label) if label in labels else -1
    if expected_idx >= 0:
        print(f"  -> Python model gives {pcts[expected_idx]:.1f}% for '{label}'")
        if pcts[expected_idx] >= 75:
            print("     HIGH CONFIDENCE: Python pipeline is correct.")
            print("     If on-device confidence is low, the issue is amplitude (mic gain).")
        else:
            print("     LOW CONFIDENCE even in Python: something in the MFCC/norm/quant chain.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Oracle test: inject a training WAV into MCU firmware")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--label", help="Label name — pulls a clip from data/*.parquet")
    g.add_argument("--wav",   help="Direct path to any WAV file")
    ap.add_argument("--clip-index", type=int, default=0,
                    help="Which clip to use when --label is given (default: 0 = first)")
    ap.add_argument("--out",      default="oracle_pcm.h",
                    help="Output C header path (default: oracle_pcm.h)")
    ap.add_argument("--snapshot", help="Snapshot log file for amplitude comparison")
    ap.add_argument("--tflite",   help="kws_tf_int8.tflite — run Python model inference too")
    args = ap.parse_args()

    # --- load audio ---
    if args.wav:
        label = Path(args.wav).stem.split("_")[0]
        training_int16, dbfs = load_from_wav(Path(args.wav))
        src = args.wav
    else:
        label = args.label
        print(f"[oracle] scanning parquet shards for label='{label}' clip {args.clip_index}...")
        training_int16, dbfs, src = load_from_parquet(label, args.clip_index)
        print(f"[oracle] found: {src}")

    # --- write C header ---
    out_path = Path(args.out)
    write_c_header(training_int16, label, src, dbfs, out_path)

    print(f"\n[amplitude] training WAV RMS: {dbfs:.1f} dBFS  (training target: -20.0 dBFS)")
    print(f"[info] config.yaml target_dbfs=-20.0 — all training clips are loudness-normalized.")

    # --- amplitude comparison vs snapshot ---
    if args.snapshot:
        log_text = Path(args.snapshot).read_text()
        snap_int16 = parse_pcm_from_log(log_text)
        if len(snap_int16) == 0:
            print("[warn] no PCM lines found in snapshot log")
        else:
            print(f"[snapshot] parsed {len(snap_int16)} samples")
            compare_amplitudes(training_int16, snap_int16)

    # --- Python model inference on oracle data ---
    if args.tflite:
        run_tflite(training_int16, args.tflite, label)

    print(f"""
[next step — firmware oracle test]
1. Copy {out_path} into nrfApps/tfml/src/
2. In src/MLInferenceCtrl.cpp, add near the top (after the other includes):
       #include "oracle_pcm.h"
3. In thread_ml_continuous_consumer, right before the DSP call, add the memcpy
   AND switch from incremental to brute-force MFCC (CRITICAL — incremental only
   recomputes 12 of 99 frames so it sees mostly zeros):

       memcpy(raw_window_buffer, oracle_pcm, sizeof(oracle_pcm));  // ORACLE
       compute_mfccs(raw_window_buffer, input_tensor->data.int8);  // ORACLE brute-force
       // compute_mfccs_incremental(raw_window_buffer, input_tensor->data.int8);  // comment out

4. Build & flash. Run 'start'. The device should immediately detect '{label}' at high confidence.
5. To restore: remove the two ORACLE lines, uncomment compute_mfccs_incremental.
""")


if __name__ == "__main__":
    main()
