#!/bin/bash
# Tetra-only Platonic baseline using the public Platonic QM9 model defaults.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --job-name=qm9_t1152_l14
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --mem=64000M
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_qm9_t1152_l14.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
export HYBRID_CONFIG_JSON="${HYBRID_CONFIG_JSON:-$REPO_ROOT/configs/qm9_hybrid/tetra_t_only_d1152_l14_h72.json}"
export D_MODEL="${D_MODEL:-1152}"
export N_HEADS="${N_HEADS:-72}"
export N_LAYERS="${N_LAYERS:-14}"
export COORD_HEAD_MODE="${COORD_HEAD_MODE:-group_vector}"
export BATCH_SIZE="${BATCH_SIZE:-96}"
export LR="${LR:-5e-4}"
export WARMUP_STEPS="${WARMUP_STEPS:-0}"
export SKIP_LOSS_THRESHOLD="${SKIP_LOSS_THRESHOLD:-100.0}"
export RUN_SLUG="${RUN_SLUG:-matterformer_qm9_tetra_t_only_d1152_l14_h72_platonicio}"
export WANDB_GROUP="${WANDB_GROUP:-qm9_tetra_t_only_d1152_l14_h72_platonicio}"

exec "$REPO_ROOT/scripts/qm9_edm_hybrid_delta.sh"
