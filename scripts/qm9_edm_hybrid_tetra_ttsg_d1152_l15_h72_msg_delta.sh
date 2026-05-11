#!/bin/bash
# Tetra T,T,S_g baseline with Platonic QM9 defaults and max current Triton S_g.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --job-name=qm9_ttsg1152
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --mem=64000M
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_qm9_ttsg1152.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
export HYBRID_CONFIG_JSON="${HYBRID_CONFIG_JSON:-$REPO_ROOT/configs/qm9_hybrid/tetra_ttsg_d1152_l15_h72_msg.json}"
export D_MODEL="${D_MODEL:-1152}"
export N_HEADS="${N_HEADS:-72}"
export N_LAYERS="${N_LAYERS:-15}"
export COORD_HEAD_MODE="${COORD_HEAD_MODE:-group_vector}"
export BATCH_SIZE="${BATCH_SIZE:-96}"
export SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-96}"
export LR="${LR:-2e-4}"
export WARMUP_STEPS="${WARMUP_STEPS:-5000}"
export SKIP_LOSS_THRESHOLD="${SKIP_LOSS_THRESHOLD:-0}"
export SKIP_GRAD_NORM_THRESHOLD="${SKIP_GRAD_NORM_THRESHOLD:-0}"
export MAX_CONSECUTIVE_SKIPPED_UPDATES="${MAX_CONSECUTIVE_SKIPPED_UPDATES:-0}"
export FLOAT32_MATMUL_PRECISION="${FLOAT32_MATMUL_PRECISION:-medium}"
export EDM_LOSS_REDUCTION="${EDM_LOSS_REDUCTION:-sample_mean}"
export ROTATION_AUGMENTATION_MODE="${ROTATION_AUGMENTATION_MODE:-per_sample}"
export RUN_SLUG="${RUN_SLUG:-matterformer_qm9_tetra_ttsg_d1152_l15_h72_msg_stage1_b96_flash_samplemean_permolrot}"
export WANDB_GROUP="${WANDB_GROUP:-qm9_tetra_ttsg_d1152_l15_h72_msg_stage1_b96_flash_samplemean_permolrot}"

exec "$REPO_ROOT/scripts/qm9_edm_hybrid_delta.sh"
