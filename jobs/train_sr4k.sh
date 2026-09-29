#!/bin/bash
#SBATCH --job-name=kws_sr4k
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sr4k_%j.out
#SBATCH --error=logs/sr4k_%j.err

module load anaconda/2021.05-py38
conda activate tf2-gpu

cd "$SLURM_SUBMIT_DIR"

# 4 kHz model: audio is resampled from 16 kHz → 4 kHz before MFCC.
# Nyquist = 2 kHz, so upper_hz is capped to 1900 Hz.
# frame_length=120, fft_length=128, ~98 frames.
# Extreme low-bandwidth case — can you still recognize words at phone-quality?
# Also cuts MCU compute: smaller FFT, same architecture.
python src/train_tf.py \
  --arch dscnn \
  --width 64 \
  --depth 4 \
  --sample-rate 4000 \
  --epochs 80 \
  --mixup 0.2 \
  --run-name sr4k \
  --rebuild-cache

echo "[job] sr4k done — exit $?"
