#!/bin/bash
# Anvil batch job: retrain the KWS DS-CNN on one A100 GPU.
#
# Usage:
#   1) edit the two placeholders below (account + email)
#   2) sbatch train_gpu.sh
#   3) watch:  squeue --me   and   tail -f logs/train_<jobid>.out
#
# This run uses the arm_f32 (float CMSIS-DSP) front-end set in config.yaml and the
# larger network (width 128 / depth 8 + mixup + label smoothing) that matched the
# original ~95% model, so it's an apples-to-apples comparison. Adjust the python
# line for other experiments (see the sample-rate note at the bottom).

#SBATCH -A cis260440-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=04:00:00
#SBATCH -J kws_train
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@purdue.edu

cd "$SLURM_SUBMIT_DIR"
module load conda
source activate ~/envs/kws-nrf5340
nvidia-smi

# arm_f32 DSP is set in config.yaml. --rebuild-cache is included because switching
# the DSP backend (or sample rate) changes the features; after the first run the
# cache is reused automatically, so you can drop the flag on repeat runs.
python src/train_tf.py \
  --rebuild-cache \
  --arch dscnn --width 128 --depth 8 \
  --mixup 0.2 --label-smoothing 0.05 \
  --epochs 100

# --- Later: sample-rate experiment (16 kHz vs 8 kHz), kept as separate artifacts ---
# 16 kHz (default):
#   python src/train_tf.py --rebuild-cache --width 128 --depth 8 \
#     --tag 16khz --cache mfcc_cache_16khz.npz
# 8 kHz:
#   python src/train_tf.py --rebuild-cache --width 128 --depth 8 \
#     --sample-rate 8000 --tag 8khz --cache mfcc_cache_8khz.npz
# Each --tag writes its own kws_tf_<tag>.* model + training_curves_<tag>.png so
# nothing overwrites, ready for a side-by-side latency comparison.
