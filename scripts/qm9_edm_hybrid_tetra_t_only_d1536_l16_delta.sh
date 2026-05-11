#!/bin/bash
# Tetra-only Platonic baseline, parameter-matched to roughly 32M+ params.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --job-name=qm9_t1536_l16
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --mem=64000M
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_qm9_t1536_l16.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
export HYBRID_CONFIG_JSON="${HYBRID_CONFIG_JSON:-$REPO_ROOT/configs/qm9_hybrid/tetra_t_only_d1536_l16.json}"
export D_MODEL="${D_MODEL:-1536}"
export N_HEADS="${N_HEADS:-12}"
export N_LAYERS="${N_LAYERS:-16}"
export COORD_HEAD_MODE="${COORD_HEAD_MODE:-group_vector}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export LR="${LR:-2e-4}"
export WARMUP_STEPS="${WARMUP_STEPS:-5000}"
export RUN_SLUG="${RUN_SLUG:-matterformer_qm9_tetra_t_only_d1536_l16_lr2e4_w5k}"
export WANDB_GROUP="${WANDB_GROUP:-qm9_tetra_t_only_d1536_l16_lr2e4_w5k}"

exec "$REPO_ROOT/scripts/qm9_edm_hybrid_delta.sh"
