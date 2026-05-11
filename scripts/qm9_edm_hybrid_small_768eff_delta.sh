#!/bin/bash
# Small 768-effective-width hybrid scout run for QM9 EDM.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --job-name=qm9_hyb_768
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=64000M
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_qm9_hyb_768.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
export HYBRID_CONFIG_JSON="${HYBRID_CONFIG_JSON:-$REPO_ROOT/configs/qm9_hybrid/hybrid_small_768eff.json}"
export D_MODEL="${D_MODEL:-384}"
export N_HEADS="${N_HEADS:-12}"
export N_LAYERS="${N_LAYERS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-96}"
export MAX_STEPS="${MAX_STEPS:-100000}"
export SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-96}"
export RUN_SLUG="${RUN_SLUG:-matterformer_qm9_hybrid_small_768eff}"
export WANDB_GROUP="${WANDB_GROUP:-qm9_hybrid_scout}"

exec "$REPO_ROOT/scripts/qm9_edm_hybrid_delta.sh"
