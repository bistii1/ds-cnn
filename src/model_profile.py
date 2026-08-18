#!/usr/bin/env python3
"""Per-layer profiler for the KWS model: params, MACs, and share of compute.

On a Cortex-M33 (nRF5340) inference latency is dominated by multiply-accumulate
(MAC) count, so ranking layers by MACs tells you where the time goes. This loads
a trained .keras model (or builds one from config/CLI) and prints a table sorted
by MACs, plus a rough int8 latency estimate.

Usage:
    python src/model_profile.py                       # loads kws_tf.keras
    python src/model_profile.py --model kws_tf_f32.keras
    python src/model_profile.py --build --width 128 --depth 8   # no checkpoint needed
    python src/model_profile.py --model kws_tf.keras --md ARCH_TABLE.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Very rough: MACs a Cortex-M33 @ 128 MHz sustains for int8 CMSIS-NN kernels.
# Real numbers depend on kernel/cache; treat the latency column as a relative guide.
DEFAULT_MACS_PER_SEC = 60e6


def _out_hw_c(layer):
    """Return (H, W, C) of a layer's output, ignoring the batch dim."""
    shp = layer.output.shape
    dims = [d for d in shp]
    dims = dims[1:]                      # drop batch
    if len(dims) == 3:
        return int(dims[0]), int(dims[1]), int(dims[2])
    if len(dims) == 1:
        return 1, 1, int(dims[0])
    return None


def layer_macs(layer) -> int:
    cls = layer.__class__.__name__
    w = layer.get_weights()
    out = _out_hw_c(layer)
    try:
        if cls == "Conv2D":
            kh, kw, cin, cout = w[0].shape
            oh, ow, _ = out
            return int(oh * ow * cout * kh * kw * cin)
        if cls == "DepthwiseConv2D":
            kh, kw, cin, mult = w[0].shape
            oh, ow, _ = out
            return int(oh * ow * cin * mult * kh * kw)
        if cls == "SeparableConv2D":
            # depthwise + pointwise
            dk = w[0].shape            # (kh,kw,cin,mult)
            pk = w[1].shape            # (1,1,cin*mult,cout)
            oh, ow, _ = out
            dw = oh * ow * dk[2] * dk[3] * dk[0] * dk[1]
            pw = oh * ow * pk[3] * pk[2]
            return int(dw + pw)
        if cls == "Dense":
            cin, cout = w[0].shape
            return int(cin * cout)
    except Exception:
        return 0
    return 0


def profile(model, macs_per_sec: float):
    rows = []
    total_macs = 0
    total_params = 0
    for layer in model.layers:
        macs = layer_macs(layer)
        params = int(layer.count_params())
        out = _out_hw_c(layer)
        out_str = "x".join(str(d) for d in out) if out else "-"
        rows.append([layer.name, layer.__class__.__name__, out_str, params, macs])
        total_macs += macs
        total_params += params
    for r in rows:
        r.append(100.0 * r[4] / total_macs if total_macs else 0.0)   # % of MACs
        r.append(r[4] / macs_per_sec * 1e3)                          # est ms
    return rows, total_macs, total_params


def fmt_table(rows, total_macs, total_params, macs_per_sec) -> str:
    rows_sorted = sorted(rows, key=lambda r: r[4], reverse=True)
    lines = []
    hdr = f"{'layer':22s} {'type':16s} {'output':12s} {'params':>10s} {'MACs':>13s} {'%MACs':>7s} {'~ms':>7s}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for name, cls, out, params, macs, pct, ms in rows_sorted:
        lines.append(f"{name:22s} {cls:16s} {out:12s} {params:>10,d} "
                     f"{macs:>13,d} {pct:>6.1f}% {ms:>6.2f}")
    lines.append("-" * len(hdr))
    total_ms = total_macs / macs_per_sec * 1e3
    lines.append(f"{'TOTAL':22s} {'':16s} {'':12s} {total_params:>10,d} "
                 f"{total_macs:>13,d} {100.0:>6.1f}% {total_ms:>6.2f}")
    lines.append("")
    lines.append(f"total params : {total_params:,}")
    lines.append(f"total MACs   : {total_macs:,}  "
                 f"(~{total_macs/1e6:.2f} M MAC / inference)")
    lines.append(f"est latency  : {total_ms:.1f} ms @ {macs_per_sec/1e6:.0f} M MAC/s "
                 f"(relative guide only; measure on device for real numbers)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-layer MAC/latency profiler.")
    ap.add_argument("--model", default=str(PROJECT_ROOT / "kws_tf.keras"),
                    help="Path to a trained .keras model.")
    ap.add_argument("--build", action="store_true",
                    help="Build a fresh model from --width/--depth instead of loading.")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--arch", choices=("dscnn", "cnn"), default="dscnn")
    ap.add_argument("--frames", type=int, default=99, help="MFCC frames (--build only).")
    ap.add_argument("--coeffs", type=int, default=13, help="MFCC coeffs (--build only).")
    ap.add_argument("--classes", type=int, default=30, help="Num labels (--build only).")
    ap.add_argument("--macs-per-sec", type=float, default=DEFAULT_MACS_PER_SEC,
                    help="Assumed device int8 MAC throughput for the ~ms column.")
    ap.add_argument("--md", default="", help="Also write the table to this markdown file.")
    args = ap.parse_args()

    import tensorflow as tf  # noqa: local import so --help works without TF

    if args.build:
        from train_tf import build_model
        model = build_model(args.arch, (args.frames, args.coeffs, 1),
                            args.classes, width=args.width, depth=args.depth,
                            dropout=0.2)
        print(f"[profile] built {args.arch} width={args.width} depth={args.depth}")
    else:
        mp = Path(args.model)
        if not mp.exists():
            raise SystemExit(f"[error] model not found: {mp} (or use --build)")
        model = tf.keras.models.load_model(mp)
        print(f"[profile] loaded {mp.name}")

    rows, total_macs, total_params = profile(model, args.macs_per_sec)
    table = fmt_table(rows, total_macs, total_params, args.macs_per_sec)
    print(table)

    if args.md:
        md_lines = ["```", table, "```", ""]
        Path(args.md).write_text("\n".join(md_lines))
        print(f"[profile] wrote {args.md}")


if __name__ == "__main__":
    main()
