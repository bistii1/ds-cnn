#!/bin/bash
#SBATCH --job-name=kws_w64
#SBATCH --account=cis220051
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/w64_%j.out
#SBATCH --error=logs/w64_%j.err
##SBATCH --mail-user=bpotdar@purdue.edu
##SBATCH --mail-type=END,FAIL

module load conda
source activate ~/envs/kws-nrf5340

cd ~/kws-nrf5340-dataset

# Width-64: smaller model — fewer params, faster on MCU, slightly lower accuracy
# Reuses mfcc_cache_sr16000_stride20ms.npz
python src/train_tf.py \
  --arch dscnn \
  --width 64 \
  --depth 4 \
  --epochs 80 \
  --mixup 0.2 \
  --run-name w64

echo "[job] w64 done — exit $?"
