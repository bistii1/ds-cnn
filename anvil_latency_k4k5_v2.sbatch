#!/bin/bash
#SBATCH --job-name=kws_latency_v2
#SBATCH --account=cis260440-ai
#SBATCH --partition=ai
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:12:00
#SBATCH --array=0-1
#SBATCH --output=latency_v2_%A_%a.log

# Low-latency candidates for the nRF5340. Each array task trains one model
# (float, 80 epochs, same recipe as runs/baseline_v2), then runs QAT and
# exports the int8 tflite.
#
#   task 0  K4  c48_skip_w64d4 : 48 coeffs, 40 ms stride (25 frames), width 64, depth 4
#   task 1  K5  base_w64d4     : 13 coeffs, 20 ms stride (49 frames), width 64, depth 4
#
# Estimated on-device NN time (latency ~ MACs / 39 M per s, measured on v2):
#   K4 ~6.6 M MACs -> ~170-250 ms,  K5 ~3.7 M MACs -> ~100-140 ms
#   (baseline_v2 492 ms, c48 1682 ms, skip_v2 72 ms)
#
# Comparisons this enables:
#   K4 vs skip_v2 : only the coefficient count differs (13 -> 48)
#   K5 vs baseline: only the network size differs (128x6 -> 64x4)
#   K4 vs K5      : coefficients + stride at the same small network
#
# Wall time: past v2 jobs used 3.5 min (c48 float), 5.5 min with a cache build,
# and ~2 min (QAT). One task does both, so ~8 min max; 12 min leaves margin.
# Check real usage:
#   sacct -u $USER -S 2026-10-20 -X --format=JobID,JobName%28,Partition,Elapsed,Timelimit,State
# Queue: ai partition on the cis260440-ai allocation (shorter waits, per the
# Nov 10 advice). Each account only accepts its own partition. Balances: `mybalance`.
# Fallback to the gpu queue:
#   sbatch -A cis260440-gpu -p gpu anvil_latency_k4k5_v2.sbatch
#
# Smoke test first (1 epoch on a tiny subset, no export, both tasks). It also
# builds the MFCC caches (~2 min for c48), so the full run reuses them:
#   sbatch --export=ALL,SMOKE=1 --time=00:06:00 anvil_latency_k4k5_v2.sbatch
#
# PREREQUISITE: processed/ holds the v2 wavs (bash jobs/prepare_hf_v2.sh on login).

set -euo pipefail

module load modtree/gpu
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/qatenv"

cd "$SLURM_SUBMIT_DIR"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) RUN=latency_k4_c48_skip_w64d4; FEAT=(--num-mfcc 48 --frame-stride-ms 40) ;;
  1) RUN=latency_k5_base_w64d4;     FEAT=(--num-mfcc 13 --frame-stride-ms 20) ;;
  *) echo "unknown array task ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac
echo "[job] task=${SLURM_ARRAY_TASK_ID:-0} run=$RUN feat=${FEAT[*]}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  env -u TF_USE_LEGACY_KERAS python src/train_tf.py \
    --arch dscnn --width 64 --depth 4 "${FEAT[@]}" \
    --smoke --run-name "${RUN}_smoke"
  exit 0
fi

# 1) Float training under Keras 3 (same recipe as runs/baseline_v2).
#    The MFCC cache is named by sample rate, clip, stride and coeff count, so it
#    is reused when it already exists and built once when it does not.
env -u TF_USE_LEGACY_KERAS python src/train_tf.py \
  --arch dscnn --width 64 --depth 4 "${FEAT[@]}" \
  --epochs 80 --mixup 0.2 --label-smoothing 0.05 \
  --run-name "$RUN"

# 2) QAT under Keras 2. Export a version-agnostic weights bundle first, as in
#    anvil_qat_skip_v2.sbatch, so the Keras 2 loader can rebuild the model.
env -u TF_USE_LEGACY_KERAS python tools/export_weights.py "runs/$RUN/kws_tf.keras"
export TF_USE_LEGACY_KERAS=1
python src/train_tf.py --init "runs/$RUN/kws_tf.keras" --epochs 0 \
  "${FEAT[@]}" --qat --run-name "${RUN}_qat"
