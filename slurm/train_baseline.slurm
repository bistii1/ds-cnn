#!/bin/bash
#SBATCH --job-name=kws_baseline
#SBATCH --account=cis220051        # your Anvil allocation (check: sacctmgr show user $USER withassoc format=account -p)
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
##SBATCH --mail-user=bpotdar@purdue.edu   # uncomment for email on start/end
##SBATCH --mail-type=END,FAIL

module load conda
source activate ~/envs/kws-nrf5340

cd ~/kws-nrf5340-dataset

# OG / baseline: dscnn width=128 depth=6 at 16 kHz, 20 ms stride (99 frames)
# Reuses mfcc_cache_sr16000_stride20ms.npz (renamed from mfcc_cache.npz before submitting)
python src/train_tf.py \
  --arch dscnn \
  --width 128 \
  --depth 6 \
  --epochs 80 \
  --mixup 0.2 \
  --run-name baseline

echo "[job] baseline done — exit $?"
