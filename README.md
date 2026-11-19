# DS-CNN Keyword Spotting — Model Variant Study

TensorFlow DS-CNN trained for 30-class keyword spotting, targeting the nRF5340 microcontroller via TFLite-Micro.

Current training set is [snowballlab/30-keywords-v2](https://huggingface.co/datasets/snowballlab/30-keywords-v2): 16 kHz mono, **1 s** clips, 1,000 per class. Classes drop `volume_up` / `volume_down` / `set_a_timer` and add `up` / `down` / `go`. The older 2 s runs below are kept for comparison.

---

## Results (v2, 1 s clips)

| Variant | Test Acc | Parameters | int8 Size | Input Frames | Width | Depth | Sample Rate |
|---|---|---|---|---|---|---|---|
| **baseline_v2** | **90.88%** | 120,862 | ~142 KB | **49** | 128 | 6 | 16 kHz |
| **skip_frame_v2** | **81.30%** | 25,502 | ~30 KB | **25** | 64 | 4 | 16 kHz |

Same architectures as the original baseline (128×6, 20 ms stride) and skip-frame (64×4, 40 ms stride). Shorter clips cut the MFCC grid from 99→49 frames (baseline) and 50→25 frames (skip-frame).

### Quantization-aware training (QAT), v2

QAT fine-tunes a trained float model with simulated int8 rounding, so the int8 model the nRF5340 runs loses less accuracy. All three start from the v2 float checkpoints.

| Run | Test Acc (int8 QAT) | Parameters | int8 Size | Input Frames | Coeffs | Width | Depth |
|---|---|---|---|---|---|---|---|
| **baseline_c48_qat_v2** | **93.78%** | 122,743 | 169 KB | 49 | **48** | 128 | 6 |
| baseline_qat_v2 | 90.78% | 122,743 | 169 KB | 49 | 13 | 128 | 6 |
| skip_frame_qat_v2 | 80.97% | 26,207 | 47 KB | 25 | 13 | 64 | 4 |

c48 keeps all 48 DCT coefficients (`--num-mfcc 48`) instead of 13. That is a full-rank rotation of the 48 log-mel bands, so it holds the same information as raw log-mel but runs on the same on-device MFCC front end. It is also about 3.4x slower on the device than baseline (larger input grid). Sizes are the real `.tflite` file sizes.

The firmware runs these models with a 1 s window. The firmware repo keeps ready-to-use headers for each one under `models/`.

---

## Results (original, 2 s clips)

| Variant | Test Acc | Parameters | int8 Size | Input Frames | Width | Depth | Sample Rate | Est. MCU Latency |
|---|---|---|---|---|---|---|---|---|
| baseline | **91.56%** | 120,862 | ~142 KB | 99 | 128 | 6 | 16 kHz | ~1016 ms |
| taper 64→128 | 90.13% | 67,230 | ~79 KB | 99 | 64→128 | 6 | 16 kHz | ~541 ms |
| sr8k | 88.44% | 25,502 | ~30 KB | 99 | 64 | 4 | **8 kHz** | ~196 ms† |
| skip frame | 87.69% | 25,502 | ~30 KB | **50** | 64 | 4 | 16 kHz | ~98 ms |
| w64 | 85.73% | 25,502 | ~30 KB | 99 | 64 | 4 | 16 kHz | ~196 ms |
| sr4k | 84.27% | 25,502 | ~30 KB | 99 | 64 | 4 | **4 kHz** | ~196 ms† |

int8 size = params × 1.2 / 1024 KB (weights + flatbuffer overhead).  
MCU latency = MAC-based estimate at 38M MACs/s, calibrated from a real nRF5340 measurement.  
† sr8k and sr4k save additional time in MFCC preprocessing (smaller FFT: 256 and 128 vs 512), not counted here.

---

## Model Descriptions

### Baseline
The reference model. Width 128 and depth 6 were chosen to maximize accuracy within the nRF5340 flash budget. All 120K parameters are active at 99 frames of temporal input. Highest accuracy (91.56%) at the cost of the longest inference time (~1 second on the MCU for the neural network alone).

### Taper 64→128
Channels ramp linearly from 64 (first depthwise block) to 128 (last): 64 → 77 → 90 → 102 → 115 → 128. Early layers handle low-level frequency patterns and do not need many channels; later layers must distinguish confusable words and benefit from more capacity. Result: 44% fewer parameters than baseline with only 1.43 percentage points of accuracy loss. Best efficiency of the six variants.

### sr8k (8 kHz downsampling)
Audio is resampled from 16 kHz to 8 kHz before computing MFCCs. This halves the FFT size (512 → 256) and cuts the Nyquist frequency to 4 kHz, removing high-frequency spectral content. The neural network input shape is unchanged (99 frames × 10 coefficients). At 88.44%, this is the strongest of the small models — suggesting that the 4–8 kHz band contributes relatively little to word discrimination for this vocabulary.

### Skip Frame
The MFCC frame stride is doubled from 20 ms to 40 ms, reducing the input from 99 frames to ~50 frames. The model architecture (width 64, depth 4) is identical to w64. Despite seeing half the temporal information, skip frame outperforms w64 by 1.96 percentage points. Explanation: 99 frames contains redundant temporal detail for 1-second keywords. The coarser input forces the model to learn more robust, speaker-independent features. It also roughly halves the MAC count, making it the fastest NN variant (~98 ms).

### w64 (Width-64)
Straightforward downsizing: width reduced from 128 → 64, depth from 6 → 4. Same parameter count as skip frame (25,502) but 1.96 points lower accuracy. This is the control for the skip frame experiment — it isolates the effect of reducing temporal resolution vs. reducing model capacity. Conclusion: temporal resolution is a safer axis to compress than channel width.

### sr4k (4 kHz downsampling)
Audio resampled from 16 kHz to 4 kHz before MFCCs. Nyquist drops to 2 kHz, upper mel filter capped at 1900 Hz. This is an extreme low-bandwidth test (phone-call quality). At 84.27%, it is the least accurate variant, indicating that the 2–4 kHz range carries meaningful phonetic information for this vocabulary. The FFT shrinks to 128 points, so preprocessing is the cheapest of all variants.

---

## Architecture: DS-CNN

All models use a Depthwise Separable CNN (DS-CNN), the same family used by MLPerf Tiny and Google micro_speech. Each block splits a convolution into two steps:

1. **Depthwise conv (3×3):** processes each channel independently — captures spatial patterns cheaply.
2. **Pointwise conv (1×1):** mixes information across channels — builds higher-level features.

Every model follows this structure:

```
Input (frames × MFCC coeffs, 13 by default)
  → Conv2D stem (10×4 kernel, stride 2×2, width channels)
  → N × [DepthwiseConv2D(3×3) → BN → ReLU → Conv2D(1×1) → BN → ReLU]
  → AveragePooling2D (full grid) → Flatten
  → Dropout(0.2)
  → Dense(30, softmax)
```

The global pool is written as a full-grid `AveragePooling2D` + `Flatten`. The math and weights are the same as `GlobalAveragePooling2D`, but it exports to the TFLite `AVERAGE_POOL_2D` op, which has a CMSIS-NN fast kernel. `GlobalAveragePooling2D` exports to `MEAN`, which uses a slow generic kernel (measured 29.5 ms on baseline and 126 ms on c48).

BatchNormalization uses momentum=0.9 (not the Keras default of 0.99). With 0.99, the moving statistics lag too far behind the real statistics across several BN layers, causing inference accuracy to collapse. 0.9 fixes this.

### Where the latency is

For output map H×W and channel count C (= width):

| Layer | Params | MACs / inference |
|---|---|---|
| Stem Conv 10×4 | 10·4·1·C | H·W · C · 40 |
| DepthwiseConv 3×3 | 3·3·C | H·W · C · 9 |
| Pointwise 1×1 | C·C | H·W · C · C |
| Dense | C·n_classes | C · n_classes |

All maps after the stem are 50×7 for the 2 s models (99 frames) and 25×7 for 1 s baseline_v2 (49 frames). Pointwise 1×1 convolutions dominate (~77–90% of MACs depending on width) because they perform a full C×C matrix multiply at every spatial location. Lowering `width` is the biggest latency lever. Lowering `depth` drops whole blocks. Halving the sample rate cuts FFT preprocessing but not NN MACs (the 99×10 feature grid is defined in milliseconds, not Hz).

To profile any configuration:

```bash
python src/model_profile.py --build --width 128 --depth 6
```

---

## Key Findings

**1. Taper is the most efficient design.**  
Taper 64→128 achieves 90.13% with 44% fewer parameters than baseline. The cost is 1.43 percentage points of accuracy and about half the flash size. Recommended for any deployment requiring accuracy above 90%.

**2. Skip frame beats w64 despite half the temporal input.**  
Skip frame and w64 have identical architectures and parameter counts (25,502). Skip frame scores 1.96 points higher. The 99-frame input contains redundant temporal information; halving the time resolution acts as implicit regularization and produces better generalization.

**3. 8 kHz is surprisingly competitive.**  
sr8k (88.44%) outperforms both skip frame and w64 despite using only half the audio bandwidth. For this vocabulary, the 4–8 kHz spectral band contributes less than might be expected. An 8 kHz microphone path would be a viable hardware simplification.

**4. 4 kHz is too lossy.**  
sr4k (84.27%) is the weakest small model. The 2–4 kHz range carries real phonetic information (fricatives, plosive bursts) that the model needs to separate similar-sounding words.

**5. Feature capacity matters more than temporal resolution.**  
Across all variants, reducing channel width hurts accuracy more than reducing frame count or bandwidth. Width should be the last axis to compress.

---

## Tradeoff Guide

| Goal | Recommended Variant |
|---|---|
| Maximum accuracy | baseline (91.56%, 142 KB) |
| Best accuracy-per-byte | taper 64→128 (90.13%, 79 KB) |
| Lowest NN inference time on MCU | skip frame (87.69%, 30 KB, ~98 ms) |
| Lowest total compute (NN + MFCC) | sr4k or sr8k (smaller FFT + same NN) |
| Simplest hardware (lower-cost mic) | sr8k (88.44%, standard 8 kHz audio path) |

---

## Repo Structure

```
ds-cnn/
├── src/                      trainer, MFCC front-ends, export helpers
│   ├── train_tf.py           train + export .keras / .tflite / .json
│   ├── mfcc.py               DSP backend selector (tf | arm_q15 | arm_f32)
│   ├── mfcc_arm.py           CMSIS-DSP MFCC (matches on-device arm_math)
│   ├── mfcc_tf.py            tf.signal MFCC reference
│   ├── gen_dsp_constants.py  writes dsp_constants.h for firmware
│   ├── model_profile.py      per-layer params / MACs
│   └── representative_dataset.py
├── config.yaml               audio, MFCC, DSP backend, 30-class vocab
├── compare_snapshot.py       device log vs Python pipeline
├── oracle_test.py            inject a known clip into firmware
├── make_model_h.py           int8 .tflite -> firmware model.h
├── tools/
│   ├── export_weights.py     Keras-version-free weights bundle (needed for QAT)
│   ├── inject_clip.py        stream a wav into the board over serial
│   ├── capture_snapshot.py   save one on-device snapshot log
│   ├── eval_heldout.py       PTQ vs QAT accuracy on the held-out split
│   └── pick_heldout_clips.py copy held-out wavs for oracle clips
├── slurm/                    SLURM batch scripts for Anvil HPC
├── anvil_*_v2.sbatch         Anvil jobs for QAT and v2 experiments
│
└── runs/                     trained variants (flash one folder at a time)
    ├── baseline_c48_qat_v2/  128×6, 1 s, 48 coeffs, QAT, 93.78%
    ├── baseline_qat_v2/      128×6, 1 s, QAT, 90.78%
    ├── skip_frame_qat_v2/    64×4, 1 s, 25 frames, QAT, 80.97%
    ├── baseline_v2/          128×6, 1 s, 49 frames, 90.88%  (float source for QAT)
    ├── skip_frame_v2/        64×4, 1 s, 25 frames, 81.30%  (float source for QAT)
    ├── baseline/             128×6, 2 s, 99 frames, 91.56%
    ├── taper64_128/          64→128 taper, 90.13%
    ├── sr8k/                 64×4, 8 kHz, 88.44%
    ├── skip_frame/           64×4, 50 frames, 87.69%
    ├── w64/                  64×4, 16 kHz, 85.73%
    └── sr4k/                 64×4, 4 kHz, 84.27%
```

Each `runs/<name>/` folder is a self-contained bundle:

| File | Use |
|---|---|
| `kws_tf_int8.tflite` | int8 model the nRF5340 runs |
| `kws_tf.json` | label order + feature_norm mean/std + MFCC params |
| `confusion_matrix.png` | per-class recall heatmap |
| `kws_tf.keras` / `kws_tf_savedmodel/` | float graph for QAT (v2 + original skip_frame) |

Flash one run at a time: convert that folder’s `.tflite` to `model.h`, and regenerate `dsp_constants.h` from that `kws_tf.json` if the sample rate, frame count, or feature_norm differ (sr4k, sr8k, skip_frame).

```bash
python src/gen_dsp_constants.py --json runs/sr8k/kws_tf.json --out dsp_constants.h
```

---

## Reproduction

v2 jobs were trained on Anvil (Purdue RCAC) using an H100 GPU, `cis260440-ai` allocation, partition `ai`.

```bash
# on Anvil, from ~/kws-v2/
python src/prepare_hf_v2.py          # login node, needs internet
sbatch slurm/train_baseline.slurm    # -> runs/baseline_v2
sbatch slurm/train_skip.slurm        # -> runs/skip_frame_v2
```

Feature caches are `mfcc_cache_sr16000_clip1000_stride20ms_c13.npz` (baseline) and `mfcc_cache_sr16000_clip1000_stride40ms_c13.npz` (skip-frame). The `_c13` suffix is the number of MFCC coefficients, so a `--num-mfcc 48` run gets its own `_c48` cache. Do not reuse the old 2 s caches.

### QAT

QAT needs `tensorflow-model-optimization`, which only works with Keras 2 (`TF_USE_LEGACY_KERAS=1`). The v2 checkpoints were saved by Keras 3, which Keras 2 cannot load. So first export a weights bundle under Keras 3, then fine-tune:

```bash
env -u TF_USE_LEGACY_KERAS python tools/export_weights.py runs/skip_frame_v2/kws_tf.keras
python src/train_tf.py --init runs/skip_frame_v2/kws_tf.keras --epochs 0 \
  --frame-stride-ms 40 --qat --run-name skip_frame_qat_v2
```

`--epochs 0` skips float training, and `--qat` adds the QAT fine-tune (`--qat-epochs`, default 5). The ready-made QAT jobs are `anvil_qat_baseline_v2.sbatch`, `anvil_qat_skip_v2.sbatch`, and `anvil_qat_c48_v2.sbatch`. c48 has no float checkpoint in `runs/`, so train it first with `anvil_train_baseline_c48_v2.sbatch` (writes `runs/baseline_c48_v2/`).
