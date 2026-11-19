#!/bin/bash
#SBATCH --job-name=kws_qat_baseline_v2
#SBATCH --account=cis260440-gpu
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=qat_baseline_v2_%j.log

set -uo pipefail

module load modtree/gpu
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/qatenv"
export TF_USE_LEGACY_KERAS=1   # tensorflow-model-optimization (QAT) needs Keras 2

cd "$SLURM_SUBMIT_DIR"

# QAT of the v2 baseline model trained on the v2 dataset
# (runs/baseline_v2/kws_tf.keras): width 128, depth 6, 1 s clips / 49 frames.
# This is the high-accuracy model (90.88%, ~142 KB int8); the skip-frame job is
# the deployment candidate. Uses the DEFAULT 20 ms stride (no --frame-stride-ms),
# which gives 49 frames at a 1 s clip.
#
# PREREQUISITE: processed/ must hold the v2 wavs. Build them once on the LOGIN
# node (GPU nodes have no internet) before submitting:
#     bash jobs/prepare_hf_v2.sh        # writes processed/<label>/{train,val,test}_*.wav
#
# The v2 checkpoint is Keras 3; QAT runs under Keras 2. Make a version-agnostic
# weights bundle under Keras 3 first; load_checkpoint uses it when the direct
# Keras 2 load fails. (Verified locally: rebuilt model matches with 0 pred diff.)
env -u TF_USE_LEGACY_KERAS python tools/export_weights.py runs/baseline_v2/kws_tf.keras \
  || echo "[warn] export_weights skipped; will try a direct load under Keras 2"

# --init loads the float weights, --epochs 0 skips float retrain, QAT fine-tunes,
# exports int8. Default stride (20 ms -> 49 frames). --rebuild-cache builds the v2
# baseline MFCC cache (mfcc_cache_sr16000_clip1000_stride20ms_c13.npz).
python src/train_tf.py --init runs/baseline_v2/kws_tf.keras --epochs 0 \
  --qat --run-name baseline_qat_v2 --rebuild-cache
