#!/usr/bin/env python3
"""Export a trained .keras checkpoint to a version-agnostic weights bundle.

Why this exists:
    kws_tf.keras was saved by Keras 3 (TF 2.16+). The QAT fine-tune needs
    tensorflow-model-optimization, which only runs on Keras 2 (TF_USE_LEGACY_KERAS=1).
    Keras 2 cannot deserialize a Keras-3 .keras file, so `load_model()` crashes.

    This script runs under Keras 3, reads the model, and writes:
      <base>.weights.npz  - the raw weight arrays (model.get_weights()), plain numpy
      <base>.arch.json    - the architecture (arch/width/depth/dropout/...) read back
                            from the model, so the QAT side can rebuild it exactly.
    numpy arrays and a tiny JSON are understood by any Keras version, so the QAT run
    (Keras 2) rebuilds the same architecture and calls set_weights() on it.

Usage (run with Keras 3, i.e. WITHOUT TF_USE_LEGACY_KERAS):
    env -u TF_USE_LEGACY_KERAS python tools/export_weights.py kws_tf.keras
    # writes kws_tf.weights.npz and kws_tf.arch.json next to the input
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def infer_arch(model: tf.keras.Model) -> dict:
    """Read the architecture back out of a built model (no guessing)."""
    L = tf.keras.layers
    conv = [l for l in model.layers if isinstance(l, L.Conv2D)
            and not isinstance(l, L.DepthwiseConv2D)]
    depthwise = [l for l in model.layers if isinstance(l, L.DepthwiseConv2D)]
    dense = [l for l in model.layers if isinstance(l, L.Dense)]
    dropout = [l for l in model.layers if isinstance(l, L.Dropout)]

    in_shape = tuple(model.inputs[0].shape[1:])          # drop batch dim
    n_classes = int(model.outputs[0].shape[-1])
    arch = "dscnn" if depthwise else "cnn"
    width = int(conv[0].filters) if conv else int(dense[0].units)
    depth = len(depthwise) if depthwise else max(1, len(conv) - 1)
    drop = float(dropout[0].rate) if dropout else 0.2

    return {
        "arch": arch,
        "input_shape": [int(d) for d in in_shape],
        "n_classes": n_classes,
        "width": width,
        "depth": depth,
        "dropout": drop,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keras", help="path to the .keras checkpoint (e.g. kws_tf.keras)")
    args = ap.parse_args()

    src = Path(args.keras)
    base = src.with_suffix("")          # strip .keras
    npz_path = Path(f"{base}.weights.npz")
    json_path = Path(f"{base}.arch.json")

    print(f"[export] loading {src} (Keras {tf.keras.__version__ if hasattr(tf.keras, '__version__') else '?'})")
    model = tf.keras.models.load_model(src)

    weights = model.get_weights()
    np.savez(npz_path, *weights)
    arch = infer_arch(model)
    json_path.write_text(json.dumps(arch, indent=2))

    print(f"[export] wrote {npz_path}  ({len(weights)} weight arrays)")
    print(f"[export] wrote {json_path}  -> {arch}")


if __name__ == "__main__":
    main()
