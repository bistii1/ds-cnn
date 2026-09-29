#!/bin/bash
#SBATCH --job-name=kws_w64
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/w64_%j.out
#SBATCH --error=logs/w64_%j.err

module load anaconda/2021.05-py38
conda activate tf2-gpu

cd "$SLURM_SUBMIT_DIR"

# Width-64: dscnn width=64 depth=4 at 16 kHz, 99 frames
# Smaller than baseline — should be faster on MCU, lower accuracy
# Reuses mfcc_cache_sr16000_stride20ms.npz — no --rebuild-cache needed
python src/train_tf.py \
  --arch dscnn \
  --width 64 \
  --depth 4 \
  --epochs 80 \
  --mixup 0.2 \
  --run-name w64

echo "[job] w64 done — exit $?"
