#!/bin/bash
# Tetra local/global baseline: S_g + T repeated.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --job-name=qm9_sgt_tetra
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --mem=64000M
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_qm9_sgt_tetra.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
export HYBRID_CONFIG_JSON="${HYBRID_CONFIG_JSON:-$REPO_ROOT/configs/qm9_hybrid/tetra_sgt_d768_l8.json}"
export D_MODEL="${D_MODEL:-768}"
export N_HEADS="${N_HEADS:-12}"
export N_LAYERS="${N_LAYERS:-8}"
export COORD_HEAD_MODE="${COORD_HEAD_MODE:-group_vector}"
export RUN_SLUG="${RUN_SLUG:-matterformer_qm9_tetra_sgt_d768_l8}"
export WANDB_GROUP="${WANDB_GROUP:-qm9_tetra_sgt_l8}"

exec "$REPO_ROOT/scripts/qm9_edm_hybrid_delta.sh"
