#!/bin/bash
#SBATCH --job-name=kws_sr8k
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sr8k_%j.out
#SBATCH --error=logs/sr8k_%j.err

module load anaconda/2021.05-py38
conda activate tf2-gpu

cd "$SLURM_SUBMIT_DIR"

# 8 kHz model: audio is resampled from 16 kHz → 8 kHz before MFCC.
# Nyquist = 4 kHz, so upper_hz is capped to 3800 Hz.
# frame_length=240, fft_length=256, 99 frames (same temporal density).
# Tests how much high-frequency content (4–8 kHz) contributes to accuracy.
python src/train_tf.py \
  --arch dscnn \
  --width 64 \
  --depth 4 \
  --sample-rate 8000 \
  --epochs 80 \
  --mixup 0.2 \
  --run-name sr8k \
  --rebuild-cache

echo "[job] sr8k done — exit $?"
