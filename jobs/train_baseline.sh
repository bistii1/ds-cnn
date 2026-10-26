#!/bin/bash
#SBATCH --job-name=kws_baseline
#SBATCH --account=YOUR_ACCOUNT       # <-- replace with your Anvil allocation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

# ---- environment -----------------------------------------------------------
module load anaconda/2021.05-py38     # adjust to your Anvil module name
conda activate tf2-gpu                # adjust to your conda env name

cd "$SLURM_SUBMIT_DIR"

# ---- run -------------------------------------------------------------------
# baseline_v2: dscnn 128x6 on snowballlab/30-keywords-v2 (1 s -> 49 frames)
# Requires: python src/prepare_hf_v2.py   (writes processed/)
python src/train_tf.py \
  --arch dscnn \
  --width 128 \
  --depth 6 \
  --epochs 80 \
  --mixup 0.2 \
  --run-name baseline_v2 \
  --rebuild-cache

echo "[job] baseline done — exit $?"
