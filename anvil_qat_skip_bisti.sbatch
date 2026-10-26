#!/bin/bash
#SBATCH --job-name=kws_qat_skip_bisti
#SBATCH --account=cis260440-gpu
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=qat_skip_bisti_%j.log

set -uo pipefail

module load modtree/gpu
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/qatenv"
export TF_USE_LEGACY_KERAS=1   # tensorflow-model-optimization (QAT) needs Keras 2

cd "$SLURM_SUBMIT_DIR"

# QAT Bisti's actual skip-frame FLOAT model (runs/skip_frame/kws_tf.keras, from her
# "float" commit). No float retrain: load her weights, QAT fine-tune, export int8.
#
# Her checkpoint may be Keras 3; QAT runs under Keras 2. So first make a
# version-agnostic weights bundle under Keras 3 (best effort). load_checkpoint then
# loads her .keras directly if Keras 2, or falls back to that bundle if Keras 3.
env -u TF_USE_LEGACY_KERAS python tools/export_weights.py runs/skip_frame/kws_tf.keras \
  || echo "[warn] export_weights skipped; will try a direct load under Keras 2"

python src/train_tf.py --init runs/skip_frame/kws_tf.keras --epochs 0 \
  --frame-stride-ms 40 --qat --run-name skip_frame_qat_bisti --rebuild-cache
