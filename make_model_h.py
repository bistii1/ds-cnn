#!/usr/bin/env python3
"""Convert an int8 .tflite model into model.h for the tflm firmware.

The firmware (src/model.h, included by MLInferenceCtrl.cpp) expects these EXACT symbol names:
    alignas(16) const unsigned char kws_tf_int8_tflite[];
    unsigned int kws_tf_int8_tflite_len;

so this script always emits those names, regardless of the input filename. Drop the result into
the firmware's src/model.h, rebuild, and flash.

Usage:
    python make_model_h.py kws_tf_qat_int8.tflite -o model.h
    # then: cp model.h <thingy53-tflm>/src/model.h  &&  west build ... --pristine
"""
from __future__ import annotations

import argparse
from pathlib import Path

SYMBOL = "kws_tf_int8_tflite"   # must match the name used in the firmware's MLInferenceCtrl.cpp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tflite", help="path to the int8 .tflite (e.g. kws_tf_qat_int8.tflite)")
    ap.add_argument("-o", "--out", default="model.h", help="output header (default model.h)")
    ap.add_argument("--per-line", type=int, default=12, help="bytes per line (default 12)")
    args = ap.parse_args()

    data = Path(args.tflite).read_bytes()
    n = len(data)

    lines = [f"/* Auto-generated from {Path(args.tflite).name} by make_model_h.py. "
             f"Do not edit by hand. */",
             f"alignas(16) const unsigned char {SYMBOL}[] = {{"]
    for i in range(0, n, args.per_line):
        chunk = data[i:i + args.per_line]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    lines.append("};")
    lines.append(f"unsigned int {SYMBOL}_len = {n};")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"[done] wrote {args.out}  ({n} bytes, symbol '{SYMBOL}')")
    if n > 300 * 1024:
        print(f"[warn] model is {n/1024:.0f} KB - check it still fits the nRF5340 flash/arena")


if __name__ == "__main__":
    main()
