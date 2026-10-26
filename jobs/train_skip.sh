#!/bin/bash
#SBATCH --job-name=kws_skip
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/skip_%j.out
#SBATCH --error=logs/skip_%j.err

module load anaconda/2021.05-py38
conda activate tf2-gpu

cd "$SLURM_SUBMIT_DIR"

# skip_frame_v2: stride 20ms → 40ms on 1 s clips (~25 frames).
# Requires: python src/prepare_hf_v2.py
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
