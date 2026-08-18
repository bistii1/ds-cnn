# KWS DS-CNN — model architecture & latency notes

This document describes the keyword-spotting network trained by
`src/train_tf.py` (`build_dscnn`), and where the on-device latency actually goes.
For exact, per-layer numbers on any trained checkpoint, run the profiler:

```bash
python src/model_profile.py --model kws_tf.keras
# or, without a checkpoint, straight from the hyper-parameters:
python src/model_profile.py --build --width 128 --depth 8
```

## Input

- **Feature:** MFCC, shape `(frames, coeffs, 1)` = `(99, 13, 1)` for a 1 s clip
  at 16 kHz (30 ms window / 20 ms hop → 99 frames; 13 MFCCs).
- Features are produced by the DSP front-end selected in `config.yaml`
  (`dsp.backend`): `tf`, `arm_q15`, or `arm_f32`. The network is identical
  regardless of front-end — only the input *values* change.

## Topology (DS-CNN / depthwise-separable CNN)

```
Input (99, 13, 1)
  └─ Conv2D(width, kernel=10x4, stride=2x2, same)  ──► (50, 7, width)   "stem"
     BatchNorm → ReLU
  ── repeat `depth` times: ────────────────────────────────────────────
     DepthwiseConv2D(3x3, same)                      (50, 7, width)
     BatchNorm → ReLU
     Conv2D(width, kernel=1x1)  "pointwise"          (50, 7, width)
     BatchNorm → ReLU
  ─────────────────────────────────────────────────────────────────────
  GlobalAveragePooling2D                             (width,)
  Dropout
  Dense(n_classes, softmax)                          (n_classes,)
```

Key point: **only the stem strides** (2×2). Every depthwise/pointwise block after
it runs at the full `50×7` resolution, so per-block cost scales with `width`
(depthwise) and `width²` (pointwise).

## Cost model (per layer)

For output map `H×W` and channel count `C` (= `width`):

| Layer            | Params            | MACs / inference           |
|------------------|-------------------|----------------------------|
| Stem Conv 10×4   | `10·4·1·C`        | `H·W · C · (10·4·1)`       |
| DepthwiseConv 3×3| `3·3·C`           | `H·W · C · (3·3)`          |
| Pointwise 1×1    | `C·C`             | `H·W · C · C`              |
| Dense            | `C·n_classes`     | `C · n_classes`            |

(BatchNorm/ReLU/Pooling/Dropout add negligible MACs.)

## Worked example — `--width 128 --depth 8` (the ~95 % recipe)

All feature maps after the stem are `50×7`.

| Layer group          | MACs        | share  |
|----------------------|-------------|--------|
| Stem Conv 10×4       | ~1.79 M     | ~3.5 % |
| Depthwise (×8)       | ~3.23 M     | ~6.3 % |
| **Pointwise 1×1 (×8)** | **~45.9 M** | **~90 %** |
| Dense                | ~0.004 M    | <0.1 % |
| **Total**            | **~50.9 M** | 100 %  |

- **Total ≈ 50.9 M MAC/inference, ≈ 158 k parameters.**

### For `--width 64 --depth 4` (the earlier 89 % model, for comparison)

- Total ≈ **7.4 M MAC**, ≈ 25 k params. Pointwise 1×1 ≈ 77 %, stem ≈ 12 %.

## Latency hotspots (what to optimize first)

1. **Pointwise 1×1 convolutions dominate (~90 % of MACs).** They are the
   channel-mixing `C×C` matmul repeated at every spatial location and every
   block. This is the first place to cut latency:
   - lower `width` (cost ∝ `width²`),
   - lower `depth` (fewer blocks),
   - or add stride/pooling *inside* the blocks so later blocks run on a smaller
     `H×W` map.
2. **Stem Conv (10×4)** is the second cost (~3–12 % depending on width). Its cost
   is set by the input resolution (frames × coeffs); a lower audio **sample rate**
   or fewer MFCC frames shrinks it (see the 8 kHz experiment).
3. **Depthwise convs are cheap** (~6 %) — they are the whole point of DS-CNN.
4. **Dense/pooling/BN are negligible.**

On-device these map to CMSIS-NN int8 kernels; MAC count is the best first-order
proxy for cycles. Always confirm with a real on-device measurement — the `~ms`
column from `model_profile.py` is a *relative* guide, not a datasheet number.

## Sample-rate experiment (16 kHz vs 8 kHz)

Halving the sample rate to 8 kHz keeps the same `99×13` feature grid (the window
and hop are defined in ms), so the network MACs are unchanged — but the **DSP
front-end** does roughly half the FFT work, and the audio buffer is half the size.
The experiment therefore mostly moves *preprocessing* latency, not NN latency.
Run both and keep them side by side:

```bash
# 16 kHz (default)
python src/train_tf.py --rebuild-cache --width 128 --depth 8 \
    --tag 16khz --cache mfcc_cache_16khz.npz
# 8 kHz
python src/train_tf.py --rebuild-cache --width 128 --depth 8 \
    --sample-rate 8000 --tag 8khz --cache mfcc_cache_8khz.npz
```

Each run writes its own `kws_tf_<tag>.*` model, `kws_tf_<tag>.json` (accuracy +
config), and `training_curves_<tag>.png`, so nothing overwrites and you can
compare accuracy, model size, and (with on-device timing) latency directly.

## Convergence check

Every real training run writes `training_history[_tag].csv` and
`training_curves[_tag].png` (loss + accuracy, train vs val). Use them to confirm
the run converged: val loss should fall and flatten while val accuracy rises; if
val loss turns back up while train loss keeps dropping, it's overfitting (raise
dropout / mixup / label smoothing, or train fewer epochs).
