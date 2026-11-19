#!/usr/bin/env python3
"""Compare PTQ vs QAT int8 accuracy on the held-out test split.

A confidence gain measured on training clips can be memorization. This evaluates both
int8 models on the clips the model never trained on (train_tf's internal test split,
split==2 in the cache), using train_tf's exact preprocessing (global standardization
from the TRAIN split, then int8-quantize with the model's own input scale).

If QAT's test accuracy holds up (and ideally its confidence rises on UNSEEN clips), the
result is real. If test accuracy drops while train accuracy stays high, that is
overfitting from the fine-tune.

Usage:
    python tools/eval_heldout.py \
        --cache mfcc_cache_sr16000_clip1000_stride40ms_c13.npz \
        --ptq runs/skip_frame_v2/kws_tf_int8.tflite \
        --qat runs/skip_frame_qat_v2/kws_tf_int8.tflite
"""
from __future__ import annotations

import argparse
import numpy as np
import tensorflow as tf

WORDS_OF_INTEREST = ["yes", "search", "five", "one", "zero"]


def run_tflite(path: str, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_class, confidence_of_pred) for every row of X (float, standardized)."""
    itp = tf.lite.Interpreter(model_path=path)
    itp.allocate_tensors()
    inp = itp.get_input_details()[0]
    out = itp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    preds = np.empty(len(X), dtype=np.int64)
    confs = np.empty(len(X), dtype=np.float32)
    for i, x in enumerate(X):
        xq = np.round(x / in_scale + in_zp).clip(-128, 127).astype(inp["dtype"])
        itp.set_tensor(inp["index"], xq.reshape(inp["shape"]))
        itp.invoke()
        o = itp.get_tensor(out["index"])[0].astype(np.float32)
        probs = (o - out_zp) * out_scale
        preds[i] = int(np.argmax(probs))
        confs[i] = float(probs[preds[i]])
    return preds, confs


def summarize(name, preds, confs, y, mask):
    acc = float((preds[mask] == y[mask]).mean())
    conf = float(confs[mask].mean())
    return f"{name:4s}  acc={acc*100:5.1f}%  mean_conf={conf*100:5.1f}%  (n={int(mask.sum())})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="MFCC cache written by train_tf.py")
    ap.add_argument("--ptq", required=True, help="int8 tflite without QAT")
    ap.add_argument("--qat", required=True, help="int8 tflite after QAT")
    args = ap.parse_args()

    data = np.load(args.cache, allow_pickle=True)
    if "split" not in data.files or data["split"] is None or data["split"].ndim == 0:
        raise SystemExit("[error] cache has no per-row split; rebuild it with the "
                         "current train_tf (it saves split=0/1/2). Cannot identify "
                         "the held-out set without it.")
    X = data["X"].astype(np.float32)          # (N, frames, coeffs) raw MFCC
    y = data["y"].astype(np.int64)
    split = data["split"].astype(np.int8)      # 0=train 1=val 2=test
    labels = [str(s) for s in data["labels"]] if "labels" in data.files else \
             [str(i) for i in range(int(y.max()) + 1)]

    # Match train_tf: one global mean/std, taken from the TRAIN split only
    tr = split == 0
    mean = X[tr].mean().astype(np.float32)
    std = (X[tr].std() + 1e-6).astype(np.float32)
    Xs = ((X - mean) / std).astype(np.float32)

    print(f"[data] N={len(y)}  train={int((split==0).sum())} "
          f"val={int((split==1).sum())} test={int((split==2).sum())}  "
          f"classes={len(labels)}")
    print(f"[norm] global mean={mean:.4f} std={std:.4f} (from train split)\n")

    test = split == 2
    for name, path in [("PTQ", args.ptq), ("QAT", args.qat)]:
        preds, confs = run_tflite(path, Xs)
        print(f"=== {name} ({path}) ===")
        print("  " + summarize("train", preds, confs, y, split == 0))
        print("  " + summarize("val  ", preds, confs, y, split == 1))
        print("  " + summarize("TEST ", preds, confs, y, test))
        # per-word on the held-out TEST split
        for w in WORDS_OF_INTEREST:
            if w not in labels:
                continue
            c = labels.index(w)
            m = test & (y == c)
            if m.sum() == 0:
                continue
            acc = float((preds[m] == y[m]).mean())
            print(f"    {w:12s} test acc={acc*100:5.1f}%  "
                  f"mean_conf(pred)={confs[m].mean()*100:5.1f}%  (n={int(m.sum())})")
        print()

    print("How to read it:")
    print("  - If QAT TEST acc >= PTQ TEST acc, QAT did not hurt generalization.")
    print("  - If QAT train acc is high but TEST acc drops vs PTQ, the fine-tune overfit.")


if __name__ == "__main__":
    main()
