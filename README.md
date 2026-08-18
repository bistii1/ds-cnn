# DS-CNN keyword spotting

Depthwise-separable CNN (`build_dscnn` in `src/train_tf.py`) for 30-class
keyword spotting. This page is the model topology and where inference time
goes.

To rank layers on a trained checkpoint:

```bash
python src/model_profile.py --model kws_tf.keras
# or, without a checkpoint:
python src/model_profile.py --build --width 128 --depth 8
```

MAC count is the first-order proxy for on-device cycles (CMSIS-NN int8 on the
nRF5340). The profiler's `~ms` column is a *relative* ranking, not a datasheet
number — confirm with a real device measurement.

## Input

MFCC spectrogram `(frames, coeffs, 1)` = `(99, 13, 1)` for a 1 s clip at 16 kHz
(30 ms window / 20 ms hop → 99 frames; 13 coefficients).

The DSP front-end (`config.yaml` → `dsp.backend`: `tf`, `arm_q15`, or `arm_f32`)
only changes input *values*. The network is the same either way.

## Topology

Only the **stem** strides (`2×2`). Every depthwise/pointwise block after it
runs at full `50×7` resolution, so per-block cost scales with `width`
(depthwise) and `width²` (pointwise).

```
Input (99, 13, 1)
  └─ Conv2D(width, kernel=10×4, stride=2×2, same)  → (50, 7, width)   stem
     BatchNorm → ReLU
  ── repeat `depth` times ──────────────────────────────────────────
     DepthwiseConv2D(3×3, same)                      (50, 7, width)
     BatchNorm → ReLU
     Conv2D(width, kernel=1×1)  pointwise            (50, 7, width)
     BatchNorm → ReLU
  ──────────────────────────────────────────────────────────────────
  GlobalAveragePooling2D                             (width,)
  Dropout
  Dense(n_classes, softmax)                          (n_classes,)
```

## Where the latency is

For output map `H×W` and channel count `C` (= `width`):

| Layer            | Params       | MACs / inference        |
|------------------|--------------|-------------------------|
| Stem Conv 10×4   | `10·4·1·C`   | `H·W · C · 40`          |
| DepthwiseConv 3×3| `3·3·C`      | `H·W · C · 9`           |
| Pointwise 1×1    | `C·C`        | `H·W · C · C`           |
| Dense            | `C·n_classes`| `C · n_classes`         |

BatchNorm, ReLU, pooling, and dropout add negligible MACs.

### `--width 128 --depth 8` (~95 % recipe)

All maps after the stem are `50×7`. Total ≈ **50.9 M MAC / inference**, ≈ 158 k
parameters.

| Layer group            | MACs        | share   |
|------------------------|-------------|---------|
| Stem Conv 10×4         | ~1.79 M     | ~3.5 %  |
| Depthwise (×8)         | ~3.23 M     | ~6.3 %  |
| **Pointwise 1×1 (×8)** | **~45.9 M** | **~90 %** |
| Dense                  | ~0.004 M    | <0.1 %  |

For comparison, `--width 64 --depth 4` is ≈ 7.4 M MAC / ≈ 25 k params
(pointwise ≈ 77 %, stem ≈ 12 %).

### What to cut first

1. **Pointwise 1×1 convolutions (~90 % of MACs).** Channel-mixing `C×C` matmul
   at every spatial location, every block. Cost ∝ `width²`, so lowering
   `width` is the biggest lever; lowering `depth` drops whole blocks; adding
   stride/pooling *inside* the blocks shrinks `H×W` for later ones.
2. **Stem Conv 10×4 (~3–12 % depending on width).** Set by input resolution
   (frames × coeffs). Fewer MFCC frames shrinks it; halving the *sample rate*
   does **not**, because the 99×13 grid is defined in milliseconds — that
   mainly cuts DSP/FFT work, not NN MACs.
3. **Depthwise convs are cheap (~6 %).** That is the point of DS-CNN.
4. **Dense / pooling / BN are negligible.**
