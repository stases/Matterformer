#!/bin/bash
# Scalar-only SIT baseline: one local simplicial layer plus one scalar global layer per block.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --job-name=qm9_sit_scalar
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --mem=64000M
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_qm9_sit_scalar.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
export HYBRID_CONFIG_JSON="${HYBRID_CONFIG_JSON:-$REPO_ROOT/configs/qm9_hybrid/scalar_sit_d768_l8.json}"
export D_MODEL="${D_MODEL:-768}"
export N_HEADS="${N_HEADS:-12}"
export N_LAYERS="${N_LAYERS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export MAX_STEPS="${MAX_STEPS:-250000}"
export SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-64}"
export RUN_SLUG="${RUN_SLUG:-matterformer_qm9_scalar_sit_d768_l8}"
export WANDB_GROUP="${WANDB_GROUP:-qm9_hybrid_sit_l8}"

exec "$REPO_ROOT/scripts/qm9_edm_hybrid_delta.sh"
