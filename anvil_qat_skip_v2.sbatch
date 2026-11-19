#!/bin/bash
#SBATCH --job-name=kws_qat_skip_v2
#SBATCH --account=cis260440-gpu
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=qat_skip_v2_%j.log

set -uo pipefail

module load modtree/gpu
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/qatenv"
export TF_USE_LEGACY_KERAS=1   # tensorflow-model-optimization (QAT) needs Keras 2

cd "$SLURM_SUBMIT_DIR"

# QAT of the v2 skip-frame model trained on the v2 dataset
# (runs/skip_frame_v2/kws_tf.keras). The v2 set is 1 s clips / 25 frames and a
# different 30-class vocabulary (drops volume_up/volume_down/set_a_timer, adds
# up/down/go), so config.yaml already has clip_ms=1000 and the v2 labels.
#
# PREREQUISITE: processed/ must hold the v2 wavs. Build them once on the LOGIN
# node (GPU nodes have no internet) before submitting this job:
#     bash jobs/prepare_hf_v2.sh        # writes processed/<label>/{train,val,test}_*.wav
#
# The v2 checkpoint is Keras 3; QAT runs under Keras 2. Make a version-agnostic
# weights bundle under Keras 3 first; load_checkpoint uses it when the direct
# Keras 2 load of the .keras fails. (Verified locally: the rebuilt model matches
# the original with 0 prediction difference.)
env -u TF_USE_LEGACY_KERAS python tools/export_weights.py runs/skip_frame_v2/kws_tf.keras \
  || echo "[warn] export_weights skipped; will try a direct load under Keras 2"

# --init loads the float weights, --epochs 0 skips any float retrain, QAT fine-tunes,
# exports the int8 tflite. --frame-stride-ms 40 is the skip-frame stride (25 frames
# at 1 s). --rebuild-cache builds the v2 MFCC cache (mfcc_cache_sr16000_clip1000_stride40ms_c13.npz).
python src/train_tf.py --init runs/skip_frame_v2/kws_tf.keras --epochs 0 \
  --frame-stride-ms 40 --qat --run-name skip_frame_qat_v2 --rebuild-cache
