#!/bin/bash
#SBATCH --job-name=kws_w32
#SBATCH --account=cis260440-ai
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/w32_%j.out
#SBATCH --error=logs/w32_%j.err
#SBATCH --mail-user=bpotdar@purdue.edu
#SBATCH --mail-type=END,FAIL

module load conda
source activate ~/envs/kws-nrf5340

cd ~/kws-nrf5340-dataset

# Width-32: extreme compression — ~8K params, ~10 KB int8
# Reuses mfcc_cache_sr16000_stride20ms.npz (no --rebuild-cache needed)
python src/train_tf.py \
  --arch dscnn \
  --width 32 \
  --depth 4 \
  --epochs 80 \
  --mixup 0.2 \
  --run-name w32

echo "[job] w32 done — exit $?"
