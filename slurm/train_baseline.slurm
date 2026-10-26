#!/bin/bash
#SBATCH --job-name=kws_base_v2
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

# baseline_v2: dscnn 128x6 on snowballlab/30-keywords-v2 (1 s -> 49 frames)
# Prepare wavs on a login node first:  python src/prepare_hf_v2.py
python src/train_tf.py \
  --arch dscnn \
  --width 128 \
  --depth 6 \
  --epochs 80 \
  --mixup 0.2 \
  --run-name baseline_v2 \
  --rebuild-cache

echo "[job] baseline done — exit $?"
