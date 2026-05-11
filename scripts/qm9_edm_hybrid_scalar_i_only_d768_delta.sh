#!/bin/bash
# Scalar-only global baseline: all-to-all scalar MHA blocks only.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --job-name=qm9_i_scalar
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --mem=64000M
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_qm9_i_scalar.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
export HYBRID_CONFIG_JSON="${HYBRID_CONFIG_JSON:-$REPO_ROOT/configs/qm9_hybrid/scalar_i_only_d768_l8.json}"
export D_MODEL="${D_MODEL:-768}"
export N_HEADS="${N_HEADS:-12}"
export N_LAYERS="${N_LAYERS:-8}"
export COORD_HEAD_MODE="${COORD_HEAD_MODE:-equivariant}"
export RUN_SLUG="${RUN_SLUG:-matterformer_qm9_scalar_i_only_d768_l8}"
export WANDB_GROUP="${WANDB_GROUP:-qm9_scalar_i_only_l8}"

exec "$REPO_ROOT/scripts/qm9_edm_hybrid_delta.sh"
