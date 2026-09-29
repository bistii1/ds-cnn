#!/bin/bash
#SBATCH --job-name=kws_sr4k
#SBATCH --account=cis220051
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sr4k_%j.out
#SBATCH --error=logs/sr4k_%j.err
##SBATCH --mail-user=bpotdar@purdue.edu
##SBATCH --mail-type=END,FAIL

module load conda
source activate ~/envs/kws-nrf5340

cd ~/kws-nrf5340-dataset

# 4 kHz model: 16 kHz audio resampled to 4 kHz before MFCC
# Nyquist = 2 kHz, upper_hz auto-capped to 1900 Hz (phone-call quality)
# FFT shrinks: 512 → 128. ~98 frames. Extreme low-bandwidth test.
# Builds mfcc_cache_sr4000_stride20ms.npz
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
