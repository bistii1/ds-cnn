#!/bin/bash
#SBATCH --job-name=kws_baseline_c48_v2
#SBATCH --account=cis260440-gpu
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=baseline_c48_v2_%j.log

set -uo pipefail

module load modtree/gpu
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/qatenv"

cd "$SLURM_SUBMIT_DIR"

# Experiment: do richer spectral features help? Trains the baseline arch (width 128, depth 6) FROM SCRATCH on
# the v2 data with the SAME recipe as runs/baseline_v2, changing only the feature
# width: keep all 48 DCT coefficients instead of 13 (--num-mfcc 48).
#
# Why 48 DCT instead of raw log-mel: the on-device front-end is CMSIS arm_mfcc_f32,
# which computes mel -> log -> DCT in one call. Keeping all 48 DCT coefficients is a
# full-rank rotation of the 48 log-mel bands, so it carries the same information as
# raw log-mel while staying on the exact front-end the device runs. True raw log-mel
# would need a new device front-end path; this tests the same question safely.
#
# Clean A/B vs runs/baseline_v2 (13 coeffs, 90.88%): same data, arch, and recipe,
# only the feature width differs. The cache is namespaced by coeff count
# (mfcc_cache_..._c48.npz), so it will not clash with the 13-coeff caches.
#
# PREREQUISITE: processed/ holds the v2 wavs (bash jobs/prepare_hf_v2.sh on login).
#
# Optional fast sanity check first:
#   python src/train_tf.py --arch dscnn --width 128 --depth 6 --num-mfcc 48 --smoke --rebuild-cache
python src/train_tf.py \
  --arch dscnn --width 128 --depth 6 \
  --num-mfcc 48 \
  --epochs 80 --mixup 0.2 --label-smoothing 0.05 \
  --run-name baseline_c48_v2 --rebuild-cache
