#!/bin/bash
#SBATCH --job-name=kws_baseline_aug_v2
#SBATCH --account=cis260440-gpu
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=baseline_aug_v2_%j.log

set -uo pipefail

module load modtree/gpu
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/qatenv"

cd "$SLURM_SUBMIT_DIR"

# Experiment: does stronger augmentation push baseline accuracy past 90.88%?
# Trains the baseline arch (width 128, depth 6) FROM SCRATCH on the v2 data, with
# the same recipe as baseline_v2 EXCEPT augmentation is turned up. This is a clean A/B
# vs runs/baseline_v2 (same data, same arch, only augmentation changed).
#
# Augmentation turned up from the defaults:
#   time shift   8  -> 12
#   time mask   16  -> 22   (on 49 frames)
#   freq mask    3  ->  4   (on 13 MFCC coeffs)
#   feat noise 0.12 -> 0.15
#   mixup      0.2  -> 0.3
# label smoothing stays 0.05.
#
# PREREQUISITE: processed/ holds the v2 wavs (bash jobs/prepare_hf_v2.sh on login).
#
# Optional fast sanity check before the full run (1 epoch, tiny subset):
#   python src/train_tf.py --arch dscnn --width 128 --depth 6 --smoke --rebuild-cache
python src/train_tf.py \
  --arch dscnn --width 128 --depth 6 \
  --epochs 80 --mixup 0.3 --label-smoothing 0.05 \
  --aug-shift 12 --aug-time-mask 22 --aug-freq-mask 4 --aug-noise-std 0.15 \
  --run-name baseline_aug_v2 --rebuild-cache
