#!/bin/bash
#SBATCH --job-name=kws_taper
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/taper_%j.out
#SBATCH --error=logs/taper_%j.err

module load anaconda/2021.05-py38
conda activate tf2-gpu

cd "$SLURM_SUBMIT_DIR"

# Taper 64→128: starts narrow (fast early layers) and widens for the final layers
# depth=6 blocks ramp: 64, 77, 90, 102, 115, 128 channels
# Reuses mfcc_cache_sr16000_stride20ms.npz — no --rebuild-cache needed
python src/train_tf.py \
  --arch dscnn \
  --width 64 \
  --width-taper 128 \
  --depth 6 \
  --epochs 80 \
  --mixup 0.2 \
  --run-name taper64_128

echo "[job] taper done — exit $?"
