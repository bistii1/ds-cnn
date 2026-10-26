#!/bin/bash
#SBATCH --job-name=kws_qat
#SBATCH --account=cis260440-gpu
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=qat_%j.log

set -uo pipefail
module load modtree/gpu
module load anaconda

# One-time env (safe to re-run): TensorFlow + QAT toolkit + deps
pip install --user "tensorflow[and-cuda]>=2.16" tensorflow-model-optimization numpy soundfile pyyaml

cd "$SLURM_SUBMIT_DIR"   # submit from inside the ds-cnn folder

# DEFAULT: uses mfcc_cache.npz if it is in this folder (fast path, Bisti's cache).
python src/train_tf.py --init kws_tf.keras --epochs 0 --qat

# NO CACHE? You need the processed/ wav folders present, plus: pip install --user cmsisdsp scipy
# then comment the line above and use:
#   python src/train_tf.py --init kws_tf.keras --epochs 0 --qat --rebuild-cache
