"""
Compares one on-device debug snapshot (captured via run_debug_snapshot() in
firmware) against the SAME pipeline recomputed in Python, using your
teammate's actual mfcc_arm.ArmMFCC reference implementation -- not a
reimplementation, so any divergence is a genuine firmware/training mismatch,
not a "two implementations disagree" artifact.

Usage:
    1. Trigger a snapshot on-device, capture the full serial log to a text
       file, e.g.:  minicom -C snapshot_yes.log   (or your terminal's logging)
    2. python compare_snapshot.py snapshot_yes.log --tflite kws_tf_int8.tflite

Requires: numpy, tensorflow (for tf.lite.Interpreter), plus your project's
own config.py / mfcc_tf.py / mfcc_arm.py importable (run this from the
trainer's src/ directory, or add it to sys.path).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_pcm(log_text: str) -> np.ndarray:
    """Extract PCM,... lines -> (32000,) int16 array."""
    values = []
    for line in log_text.splitlines():
        if line.startswith("PCM,"):
            values.extend(int(v) for v in line[len("PCM,"):].split(","))
    return np.array(values, dtype=np.int16)


def parse_firmware_mfcc(log_text: str) -> np.ndarray:
    """Extract MFCC,frame,c0,c1,...,c12 lines -> (num_frames, num_coeffs) float32."""
    rows = {}
    for line in log_text.splitlines():
        if line.startswith("MFCC,"):
            parts = line[len("MFCC,"):].split(",")
            frame_idx = int(parts[0])
            coeffs = [float(v) for v in parts[1:]]
            rows[frame_idx] = coeffs
    n_frames = max(rows.keys()) + 1
    n_coeffs = len(rows[0])
    out = np.zeros((n_frames, n_coeffs), dtype=np.float32)
    for i, vals in rows.items():
        out[i] = vals
    return out


def parse_firmware_output(log_text: str) -> dict:
    """Extract OUT,label,int8,pct lines -> {label: (int8_val, pct)}."""
    out = {}
    for line in log_text.splitlines():
        if line.startswith("OUT,") and "ERROR" not in line:
            _, label, i8, pct = line.split(",")
            out[label] = (int(i8), float(pct))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", help="Serial log captured from the on-device snapshot")
    ap.add_argument("--tflite", required=True, help="Path to the SAME .tflite firmware is running")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml used for training")
    ap.add_argument("--src", default="src", help="Path to the trainer's src/ dir (contains config.py, mfcc_tf.py, mfcc_arm.py)")
    args = ap.parse_args()

    log_text = Path(args.logfile).read_text()

    # --- 1. Parse what firmware captured/computed ---
    pcm = parse_pcm(log_text)
    fw_mfcc = parse_firmware_mfcc(log_text)
    fw_output = parse_firmware_output(log_text)

    # Parse INT8 dump line (first 20 input tensor values from firmware)
    fw_int8_sample = None
    for line in log_text.splitlines():
        if line.startswith("INT8,"):
            fw_int8_sample = [int(v) for v in line[len("INT8,"):].split(",")]
            break

    print(f"[parse] {len(pcm)} PCM samples, {fw_mfcc.shape} MFCC grid, "
          f"{len(fw_output)} output classes from firmware log")
    if fw_int8_sample:
        print(f"[int8 sample] Firmware first 20 input tensor values: {fw_int8_sample}")

    # --- 0. Amplitude check (training data is normalized to -20 dBFS) ---
    pcm_f32 = pcm.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(pcm_f32 ** 2)) + 1e-9)
    dbfs = 20.0 * np.log10(rms)
    TRAINING_TARGET_DBFS = -20.0
    delta = dbfs - TRAINING_TARGET_DBFS
    print(f"\n[amplitude] snapshot RMS: {dbfs:.1f} dBFS  (training target: {TRAINING_TARGET_DBFS:.1f} dBFS)")
    print(f"            delta: {delta:+.1f} dB", end="  ")
    if abs(delta) < 3:
        print("-> OK (within 3 dB)")
    elif abs(delta) < 8:
        print("-> moderate mismatch (may reduce confidence slightly)")
    else:
        print("-> LARGE mismatch — likely hurting accuracy")

    if len(pcm) == 0:
        sys.exit("No PCM data found -- did you paste the full snapshot log?")

    # --- 2. Recompute MFCC in Python using the REAL project reference ---
    # config.py / mfcc_tf.py / mfcc_arm.py live in src/, not next to config.yaml --
    # add src/ (not config.yaml's folder) to sys.path so these actually import.
    src_path = Path(args.src).resolve()
    if not src_path.exists():
        sys.exit(f"--src path '{src_path}' doesn't exist -- point it at your trainer's src/ directory")
    sys.path.insert(0, str(src_path))
    from config import load_config
    from mfcc_tf import MFCCParams
    from mfcc_arm import ArmMFCC

    cfg = load_config()
    p = MFCCParams(cfg)
    dsp_cfg = cfg.get("dsp", {})
    frontend = ArmMFCC(p, precision="f32", window=dsp_cfg.get("window", "hann"))

    wav_float = pcm.astype(np.float32) / 32768.0  # SAME convention as firmware's fix
    py_mfcc = frontend.one(wav_float)  # (num_frames, num_coeffs), raw pre-normalization

    print(f"[python] recomputed MFCC shape: {py_mfcc.shape}")

    # --- 3. Compare raw MFCC (pre-normalization) frame by frame ---
    if fw_mfcc.shape != py_mfcc.shape:
        print(f"[MISMATCH] shape differs: firmware={fw_mfcc.shape} python={py_mfcc.shape}")
    else:
        diff = np.abs(fw_mfcc - py_mfcc)
        print(f"\n[MFCC COMPARISON]")
        print(f"  max abs diff       : {diff.max():.6f}")
        print(f"  mean abs diff      : {diff.mean():.6f}")
        worst_frame, worst_coef = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"  worst mismatch at frame={worst_frame} coef={worst_coef}: "
              f"firmware={fw_mfcc[worst_frame, worst_coef]:.6f} "
              f"python={py_mfcc[worst_frame, worst_coef]:.6f}")
        if diff.max() < 1e-2:
            print("  -> MFCC stage matches. Bug (if any) is downstream: normalization/quant/model.")
        else:
            print("  -> MFCC stage itself diverges. Check window/frame-length/backend config match.")

    # --- 4. Recompute normalization + quantization + model inference ---
    meta = json.loads((Path(args.config).resolve().parent / "kws_tf.json").read_text())
    mean, std = meta["feature_norm"]["mean"], meta["feature_norm"]["std"]
    normalized = (py_mfcc - mean) / std

    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=args.tflite)
    interp.allocate_tensors()
    in_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]
    in_scale, in_zp = in_detail["quantization"]
    quantized = np.round(normalized / in_scale + in_zp).clip(-128, 127).astype(np.int8)

    print(f"[int8 sample] Python first 20 input tensor values: {list(quantized.flatten()[:20])}")

    interp.set_tensor(in_detail["index"], quantized.reshape(in_detail["shape"]))
    interp.invoke()
    py_out = interp.get_tensor(out_detail["index"])[0]
    out_scale, out_zp = out_detail["quantization"]
    py_pct = (py_out.astype(np.float32) - out_zp) * out_scale * 100.0

    print(f"\n[MODEL OUTPUT COMPARISON]")
    print(f"{'label':<16}{'firmware int8':<15}{'firmware %':<12}{'python int8':<13}{'python %'}")
    for i, label in enumerate(fw_output.keys()):
        fw_i8, fw_pct = fw_output[label]
        print(f"{label:<16}{fw_i8:<15}{fw_pct:<12.2f}{py_out[i]:<13}{py_pct[i]:.2f}")


if __name__ == "__main__":
    main()