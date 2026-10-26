#!/bin/bash
#SBATCH --job-name=kws_skip_v2
#SBATCH --account=cis260440-ai
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/skip_%j.out
#SBATCH --error=logs/skip_%j.err
##SBATCH --mail-user=bpotdar@purdue.edu
##SBATCH --mail-type=END,FAIL

module load conda
source activate ~/envs/kws-nrf5340

cd ~/kws-v2

# skip_frame_v2: stride 20ms → 40ms on 1 s clips (~25 frames)
# Needs its own cache: mfcc_cache_sr16000_clip1000_stride40ms.npz
python src/train_tf.py \
  --arch dscnn \
  --width 64 \
  --depth 4 \
  --frame-stride-ms 40 \
  --epochs 80 \
  --mixup 0.2 \
  --run-name skip_frame_v2 \
  --rebuild-cache

echo "[job] skip-frame done — exit $?"
