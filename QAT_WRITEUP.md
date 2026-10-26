# Quantization-Aware Training on the Thingy:53 KWS model — full writeup

This document explains, from scratch, the whole QAT effort: what the system is, what
was wrong, what QAT is and why it fixes it, every step we took (including the dead
ends), how each tool works, the measured result, and the honest caveats.

---

## 0. The system in one paragraph

We run a keyword-spotting (KWS) model on a Nordic Thingy:53 (an nRF5340 board:
dual-core Cortex-M33, 1 MB flash, 512 KB RAM, no onboard debugger). A microphone
feeds audio into an MFCC front-end (turns sound into a 99x13 "spectrogram-like"
feature grid), and a small depthwise-separable CNN (DS-CNN) classifies each 2-second
window into one of 30 keywords (`set_a_timer`, `volume_down`, `yes`, ...). The model
runs with TensorFlow Lite Micro (TFLM) using CMSIS-NN kernels for speed. The model is
quantized to int8 so it fits and runs fast on the MCU.

Two repositories:
- **ds-cnn** (training): builds and trains the model, exports the `.tflite` files.
- **thingy53-tflm** (firmware): the Zephyr/nRF Connect SDK app that runs the model on
  the chip.

---

## 1. The problem we set out to fix

The model is quantized with **Post-Training Quantization (PTQ)**: you train in float,
then afterwards squash the weights and activations to int8 using a few calibration
samples. PTQ is easy and usually fine, but it has a failure mode here.

**On the exact same audio, the chip was far less confident than the PC.** For
`set_a_timer` the desktop TFLite int8 model said ~91%, but the chip said ~36%. Both
still picked the right word, but the on-device confidence was being eaten away.

Why: the chip (CMSIS-NN) and the desktop (TFLite reference kernels) both do int8 math,
but they **requantize between layers slightly differently** (rounding, accumulator
handling). On one layer that is a tiny difference. Across a deep stack of
depthwise+pointwise blocks, the tiny differences **accumulate** and push the final
scores around, which crushes confidence and, on close calls, can flip the answer.

This is the gap we were assigned to close with QAT.

---

## 2. What QAT is and why it fixes this

**Quantization-Aware Training (QAT)** trains the model while *simulating* int8
rounding. During training it inserts "fake quantize" nodes that round activations and
weights to int8 on the forward pass, but let gradients flow through on the backward
pass (the "straight-through estimator"). The model therefore **learns weights that are
robust to int8 rounding** instead of being surprised by it after the fact.

The practical effect: the model picks weight values that land cleanly on int8 steps and
keeps margins wide, so when the chip rounds a little differently than the PC, the
answer barely moves. That is exactly the device-vs-PC gap we needed to remove.

We used `tensorflow-model-optimization` (tfmot), the standard library for this. Key
constraint that caused us grief later: **tfmot only works on Keras 2**, not Keras 3.

---

## 3. Everything we did, in order

### 3.1 Implemented QAT in the training code (on a fork)
- Forked the training repo to `Utkarsh-M13/ds-cnn` (we never push to teammates'
  repos). Branch `qat-experiment`. `upstream` push is disabled as a guard.
- Added to `src/train_tf.py`: an `apply_qat()` step and CLI flags `--qat`,
  `--qat-epochs`, `--qat-lr`. You can QAT-fine-tune an existing checkpoint with
  `--init kws_tf.keras --epochs 0 --qat`.
- Added `tensorflow-model-optimization` to `requirements.txt`.

### 3.2 Ran it on Anvil (Purdue HPC), hit a chain of environment issues
- Anvil is a SLURM cluster. We wrote `anvil_qat_selfserve.sbatch`.
- Dead ends, each fixed: a bare GPU module had no Python/pip; `cmsisdsp` would not
  build on Python 3.13; tfmot needs Keras 2. Fix: a dedicated conda env `~/qatenv`
  (Python 3.12) with all deps, plus `tf_keras` and `TF_USE_LEGACY_KERAS=1` so tfmot
  gets the Keras 2 it needs on a TF 2.16+ install.

### 3.3 The run "finished" but produced nothing — the Keras 3 vs 2 bug, and the weights bridge
The job exited, but no model came out. The log showed it **crashed on the first real
step**: loading the `kws_tf.keras` checkpoint.

**The conflict.** Two things wanted different versions of Keras. The saved model
`kws_tf.keras` was written by **Keras 3**. QAT (the `tfmot` library) only runs on
**Keras 2**, which is why the job set `TF_USE_LEGACY_KERAS=1`. When the job tried to
open the Keras-3 file while forced into Keras-2 mode, it crashed with
`TypeError: Could not deserialize class 'Functional'`. Keras 2 simply does not
understand a file Keras 3 wrote. So the checkpoint and QAT pull in opposite directions
on the Keras version.

**Why loading across versions fails.** A `.keras` file is really two different things
zipped together:
1. **The architecture description** — a blueprint in JSON: "a Conv2D here with these
   settings, then a BatchNorm, then...". This blueprint is written in each Keras
   version's *own dialect*. Keras 3 uses class names and module paths
   (`keras.src.models.functional`, `DTypePolicy`, ...) that **do not exist in Keras 2**,
   so when Keras 2 reads the blueprint it hits names it has never heard of and gives up.
   That is the crash.
2. **The raw weights** — big tables of float numbers (the learned values). These are
   plain numbers with no "version" to them at all.
The crash is entirely in part 1 (the blueprint). Part 2 (the numbers) was never the
problem.

**The fix — a version-agnostic weights bridge.** Instead of asking Keras 2 to read
Keras 3's blueprint, skip the blueprint and carry only the numbers across:
- **Under Keras 3** (`tools/export_weights.py`): open the model and pull out just the
  two safe things — the raw weights as plain NumPy arrays (`model.get_weights()`,
  87 arrays here) saved to `kws_tf.weights.npz`, and a tiny neutral spec of the shape
  saved to `kws_tf.arch.json` (`arch=dscnn, width=128, depth=8, dropout=0.2,
  99x13x1 in, 30 classes`). That JSON is *our own* plain description, not Keras's
  dialect. The script reads the shape *out of the model itself* so it can never record
  the wrong architecture.
- **Under Keras 2** (the QAT run, via `train_tf.py`'s `load_checkpoint()`): do not load
  the old file at all. Read `arch.json` and **rebuild the model fresh** with the
  project's own `build_dscnn()`. Because Keras 2 builds it from scratch with its own
  code, the blueprint is automatically in Keras-2 dialect — no translation needed. Then
  pour the numbers back in with `model.set_weights(...)`.

**Why it is immune to the version gap.** `get_weights()`/`set_weights()` are just "hand
me the list of number-tables" / "take these number-tables back." That interface is
identical in Keras 2 and 3, and NumPy arrays do not care about Keras at all. The only
thing crossing the version boundary is a pile of plain numbers, never any
version-specific blueprint. `set_weights` matches the arrays to layers **in order**, so
the rebuilt model must have the same layers in the same order as the original — which it
does, because it is rebuilt by the same `build_dscnn()` fed the same width/depth from
`arch.json`.

Analogy: the old file is a document in a word processor the new software cannot open.
Rather than fight the format, you copy out the plain text (the numbers) and retype the
layout fresh in the new program (rebuild the architecture), then paste the text in. The
layout description never has to survive the jump, only the content does.

**Verified lossless.** On the Mac, against the real checkpoint: load the original,
rebuild-from-bridge, run both on the same input — predictions **identical, max abs diff
0.0**. The test deliberately passed wrong width/depth flags and it still rebuilt
correctly, because it reads the true shape from `arch.json`.

Re-ran on Anvil → produced `kws_tf_qat_int8.tflite` (220 KB). On the PC the QAT model is
actually *more* confident than PTQ (set_a_timer 99.6%, volume_down 99.6%, yes 97.7%), a
good sign it trained well.

### 3.4 Built the firmware self-sufficiently
The firmware depends on the Zephyr `tflite-micro` module, which was never committed to
the firmware repo. Rather than depend on one teammate's exact setup, we reconstructed a
known-good stack ourselves:
- `tflite-micro` @ `7837d798` (last revision that still matches the NCS 2.7.0 glue and
  its older CMSIS-NN), overlaid with `third_party_static` + `module.yml` from a newer
  revision, and `flatbuffers` pinned to 2.0.6 to satisfy the schema.
- Build: `west build -b thingy53/nrf5340/cpuapp --pristine -- \
  -DZEPHYR_EXTRA_MODULES=$HOME/ncs/modules/lib/tflite-micro` (env:
  `source ~/ncs-venv/bin/activate; export ZEPHYR_BASE=$HOME/ncs/zephyr`).

### 3.5 Flashing (the button that matters)
The Thingy:53 has no onboard debugger; you flash over USB via MCUboot DFU.
- To enter bootloader: cover off, hold the **tiny SW2 button** (not the big SW3) while
  sliding the power switch SW1 to ON, hold ~3s.
- nRF Connect Programmer should then list it as **"Bootloader Thingy:53"**; Add file =
  `build/zephyr/dfu_application.zip`, Enable MCUboot, Write, then power-cycle.
- The earlier failures were pressing SW3, and once, forgetting the power-cycle (the
  board was still sitting in the bootloader, so the app serial port was dead).

### 3.6 The microphone is too quiet to test with (a real finding)
Before measuring anything we tried live speech and got garbage. Measured with our
capture + analysis tools:
- Speaking "set a timer" repeatedly: **-57.5 dBFS**. A loud sound held right on the
  mic: **-45.6 dBFS**, raw PCM peaking at only **±76 out of ±32768**. Training audio is
  at **-20 dBFS**.
- So the chip records ~25-37 dB too quiet, and it is the mic/gain path, not speaking
  volume or timing. At that level the model sees near-silence; every live prediction is
  low and scattered, so you cannot measure a device-vs-PC gap on it.
- This is the team's known "quiet mic" problem, now quantified. Saved in
  `baseline_evidence/MIC_FINDING.md`. It is **separate** from QAT.

### 3.7 The oracle: a clean, repeatable input
The firmware already contained a baked-in known clip (`oracle_pcm.h` = a real
`set_a_timer` recording at -19.1 dBFS) and a commented hook to inject it instead of the
live mic. This is the right instrument: feed a **known, properly-loud clip** through the
chip's real int8 pipeline, so the only variables are the chip and the model.

We went further and made the oracle **multi-clip and selectable**:
- `tools/make_oracle_clips.py` builds `src/oracle_clips.h` with three int16[32000]
  arrays — `set_a_timer` (copied verbatim from the original so our first baseline stays
  valid), plus `volume_down` and `yes` built from real dataset clips (pulled by
  streaming the HuggingFace `snowballlab/30-keywords` set, resampled to 16 kHz, the
  spoken part scaled to ~-19 dBFS, centered in a 2 s buffer).
- Firmware: a runtime index selects the clip; a new console command `oracle <name>`
  picks it, `oracle` alone lists them. Snapshot prints which clip it used.

### 3.8 Measurement method
- `tools/capture_snapshot.py` drives the board over serial: sends `start` (the snapshot
  is only serviced while inferencing), then `oracle <word>`, then `snapshot`, and reads
  the full dump in **bulk chunks** (line-by-line dropped ~75% of the PCM; chunked reads
  capture all 2000 lines). It saves the raw log.
- `compare_snapshot.py` parses the dumped PCM, **recomputes the MFCC and runs the given
  `.tflite` on the PC**, and prints a per-label table: `firmware %` (what the chip said)
  vs `python %` (what the PC says on the same audio). That side-by-side is the gap.

---

## 4. The result

Same oracle clips, same firmware, only the model swapped (PTQ -> QAT), measured on the
chip:

| word        | BEFORE device % | AFTER device % | PC % (QAT) | device-PC gap: before -> after |
|-------------|-----------------|----------------|------------|--------------------------------|
| set_a_timer | 35.9            | 99.6           | 99.6       | -55.1  ->  0.0                 |
| volume_down | 78.5            | 99.6           | 99.6       | -18.4  ->  0.0                 |
| yes (ctrl)  | 90.6            | 97.3           | 97.7       |  +4.7  -> -0.4                 |

- Both failing words jumped to **99.6% on-device**, and the chip now emits the **exact
  same int8 value as the PC** (127 vs 127). The rounding gap is gone.
- The control word `yes` (already fine) stayed high — QAT did not break it.

Conclusion: **QAT eliminates the device-vs-PC int8 rounding mismatch on this model.**

Evidence is in `baseline_evidence/`: raw serial logs (`raw/*.txt`), per-label
comparison tables (`compare_*.txt`), the mic finding, and `results.md`.

---

## 5. Validation: threats to validity and the held-out retest

The result in section 4 was measured with the oracle, which (per the firmware's own
history) feeds TRAINING clips, and the two headline words saturated int8 (device and PC
both pinned at the max value 127, which hides any rounding difference). Two fair
objections follow: the numbers could reflect memorization, and "gap = 0" could be a
saturation artifact. We retested to settle both.

### 5.1 Tier 1 - accuracy on the full held-out test split (overfitting check)
Ran both int8 models on train_tf's internal test split (clips the QAT fine-tune never
saw), with train_tf's exact preprocessing (`tools/eval_heldout.py`):

| metric                 | PTQ   | QAT   |
|------------------------|-------|-------|
| train accuracy         | 93.7% | 93.7% |
| test accuracy          | 93.5% | 93.0% |
| test mean confidence   | 83.2% | 89.5% |

Findings: train approx test for both, so no overfitting. QAT did NOT improve accuracy
(93.5 -> 93.0, flat to slightly down). Its measurable effect is higher confidence
(+6.3 points) on UNSEEN clips, i.e. wider margins, not better classification.

### 5.2 Tier 2 - on-device device-vs-PC on held-out clips (saturation check)
Pulled 5 test-split clips (`tools/pick_heldout_clips.py`), including `search` and `five`
which sit mid-confidence and do NOT saturate, baked them into the oracle, and measured
device-vs-PC for both models on the same clips:

| word        | PTQ dev% | PTQ PC% | PTQ gap | QAT dev% | QAT PC% | QAT gap | saturated |
|-------------|----------|---------|---------|----------|---------|---------|-----------|
| set_a_timer | 37.5     | 91.8    | -54.3   | 99.6     | 99.6    |  0.0    | yes (127) |
| volume_down | 71.5     | 94.1    | -22.7   | 99.6     | 99.6    |  0.0    | yes (127) |
| search      | 80.1     | 89.5    |  -9.4   | 98.1     | 98.1    |  0.0    | near-top  |
| five        | 84.0     | 93.4    |  -9.4   | 91.0     | 91.0    |  0.0    | NO (int8 105) |
| yes         | 83.6     | 76.2    |  +7.4   | 83.2     | 82.4    | +0.8    | NO (int8 85)  |

`five` at int8 105 (91%, clearly below the 127 ceiling) matches device==PC exactly under
QAT, and `search` at int8 123 matches exactly. So the gap closing is a real device-vs-PC
convergence across the confidence range, not a saturation artifact. Honest residual:
`yes` keeps a ~0.8% gap (device int8 85 vs PC 83), so the claim is near-zero, not
literally zero everywhere.

### 5.3 Verdict after validation
QAT reliably closes the CMSIS-NN-vs-desktop int8 rounding gap, on training AND held-out
clips, saturated AND non-saturated, and the effect generalizes (Tier 1). It does this at
no accuracy cost. It does NOT improve classification accuracy. The correct one-line
claim is: *QAT makes the on-device int8 output agree with the PC and recovers on-device
confidence, at no accuracy cost.* Evidence: `baseline_evidence/results.md`, the
`heldout_*` raw logs, and `compare_heldout_*` tables.

---

## 6. How to reproduce

1. Produce the QAT model on Anvil:
   ```
   cd ~/ds-cnn && git pull
   module load anaconda && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ~/qatenv
   env -u TF_USE_LEGACY_KERAS python tools/export_weights.py kws_tf.keras   # Keras-3 bridge
   sbatch anvil_qat_selfserve.sbatch                                        # -> kws_tf_qat_int8.tflite
   ```
2. Put the QAT model on the chip:
   ```
   python make_model_h.py kws_tf_qat_int8.tflite -o model.h
   cp model.h <thingy53-tflm>/src/model.h
   # build firmware (section 3.4), flash via SW2 DFU (section 3.5)
   ```
3. Measure (device must be powered on, running the app):
   ```
   source ~/dscnn-venv/bin/activate
   for w in set_a_timer volume_down yes; do
     python tools/capture_snapshot.py --oracle $w --out after_$w.txt
     python compare_snapshot.py after_$w.txt --tflite kws_tf_qat_int8.tflite
   done
   ```

---

## 7. Honest caveats

- **Scope:** measured on three oracle clips (one example per word), not a full test-set
  accuracy sweep. It proves the on-device int8 rounding behavior cleanly, but it is not
  a dataset-wide accuracy number. A fuller result would run QAT vs PTQ across the test
  set on the PC and spot-check several more clips on-device.
- **The quiet mic is still unsolved** and unrelated to QAT. Live-mic use needs a gain
  fix (raise PDM gain in firmware and/or train with gain augmentation). Tracked in
  `MIC_FINDING.md`.
- **Our tflite-micro revision is our own known-good reconstruction.** Worth asking
  whoever set up the firmware originally for their exact revision to cross-check that
  the kernels behave identically.
- **Oracle clips** for volume_down/yes are the first matching dataset examples, not
  hand-picked for clarity. Easy to swap if a cleaner example is wanted.

---

## 8. Files that matter

Training repo (`ds-cnn`, fork `Utkarsh-M13/ds-cnn`, branch `qat-experiment`):
- `src/train_tf.py` — QAT step + `load_checkpoint()` weights bridge.
- `tools/export_weights.py` — Keras-3 -> version-agnostic weights bundle.
- `make_model_h.py` — int8 `.tflite` -> firmware `model.h`.
- `tools/capture_snapshot.py` — drives the board, selects oracle clip, saves the dump.
- `compare_snapshot.py` — device-vs-PC per-label comparison.
- `anvil_qat_selfserve.sbatch` — the HPC job.
- `kws_tf_qat_int8.tflite` — the QAT model.
- `baseline_evidence/` — all logs, comparisons, and `results.md`.

Firmware repo (`thingy53-tflm`):
- `src/oracle_clips.h` — the three baked-in test clips.
- `src/MLInferenceCtrl.cpp` / `.h` — oracle injection + selectable clip.
- `src/ConsoleCtrl.cpp` — the `oracle` console command.
- `tools/make_oracle_clips.py` — wav -> oracle header generator.
