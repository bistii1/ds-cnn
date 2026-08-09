# DS-CNN Keyword Spotting Model

TensorFlow DS-CNN artifacts for 30-keyword spotting (nRF5340 path).

| File | Description |
| --- | --- |
| `kws_tf_float.tflite` | Float TFLite model (~92% test acc) |
| `kws_tf.json` | Labels, MFCC params, DSP backend, feature norm, test accuracy |

quantize from `kws_tf_float.tflite` using metadata in `kws_tf.json`.

## Setup

```bash
pip install -r requirements.txt
```

`cmsisdsp` builds a native extension on install, so a C toolchain is required
(XCode Command Line Tools on macOS, `build-essential` on Linux). It is only needed
for the `arm_*` DSP backends below.

## DSP front-end (arm_math / CMSIS-DSP parity)

The MFCC front-end is selectable in [`config.yaml`](config.yaml) under `dsp.backend`:

| backend | implementation | use |
| --- | --- | --- |
| `tf` | `tf.signal` MFCC (`src/mfcc_tf.py`) | original float reference |
| `arm_q15` | CMSIS-DSP `arm_mfcc_q15` (`src/mfcc_arm.py`) | **fixed point, matches on-device `arm_math`; lowest preprocessing latency** |
| `arm_f32` | CMSIS-DSP `arm_mfcc_f32` | float parity/debug reference |

The `arm_*` backends use the official ARM `cmsisdsp` Python wrapper — the same C
source (`arm_math` / CMSIS-DSP) that runs on the nRF5340 — so training features are
mathematically identical to the on-device MFCC. This satisfies the hard rule that
*the DSP used to create training data must match the DSP used on device.*

### Switching to the fixed-point (arm_math) front-end

1. Set the backend in `config.yaml`:
   ```yaml
   dsp:
     backend: arm_q15
     window: hann        # must match the device dsp_constants.h
   ```
2. Retrain (this rebuilds the MFCC cache with the new backend and recomputes
   `feature_norm` mean/std automatically):
   ```bash
   python src/train_tf.py --rebuild-cache
   ```
   Switching backends changes the feature distribution, so a retrain is required;
   the old float weights and old mean/std do not transfer.
3. Generate the C tables for the firmware (same tables the trainer used):
   ```bash
   python src/gen_dsp_constants.py            # writes dsp_constants.h
   ```
   Hand `dsp_constants.h` to the firmware. It contains the DCT/mel/window tables for
   `arm_mfcc_init_q15`, the geometry `#define`s, and the `feature_norm` mean/std so
   the device applies the exact same standardization before the int8 model.
4. Re-quantize from the new float SavedModel using `src/representative_dataset.py`
   (it automatically uses the configured backend for calibration).

### On-device / training contract (coordinate with firmware)

- Q15 audio input = int16 PCM (`float [-1,1]` -> Q15). Match any DC-offset removal.
- Raw `arm_mfcc_q15` output is scaled to float by `dsp.q15_output_scale` (default
  `1/128`, i.e. q8.7) **before** `(x - mean) / std`. The device must use the same
  scale + mean/std (both emitted into `dsp_constants.h`).
- `window` in `config.yaml` must equal the window baked into `dsp_constants.h`.
