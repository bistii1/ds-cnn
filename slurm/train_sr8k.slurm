#!/bin/bash
#SBATCH --job-name=kws_sr8k
#SBATCH --account=cis220051
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sr8k_%j.out
#SBATCH --error=logs/sr8k_%j.err
##SBATCH --mail-user=bpotdar@purdue.edu
##SBATCH --mail-type=END,FAIL

module load conda
source activate ~/envs/kws-nrf5340

cd ~/kws-nrf5340-dataset

# 8 kHz model: 16 kHz audio resampled to 8 kHz before MFCC
# Nyquist = 4 kHz, upper_hz auto-capped to 3800 Hz
# FFT shrinks: 512 → 256. Still 99 frames. Tests how much 4–8 kHz band matters.
# Builds mfcc_cache_sr8000_stride20ms.npz
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
