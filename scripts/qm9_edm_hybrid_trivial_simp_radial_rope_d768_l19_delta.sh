#!/bin/bash
# Scalar-only alternating I/S run: compact simplicial content + radial u/v bias.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --job-name=qm9_i_s_rad
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --mem=64000M
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_qm9_i_s_rad.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
export HYBRID_CONFIG_JSON="${HYBRID_CONFIG_JSON:-$REPO_ROOT/configs/qm9_hybrid/trivial_simp_radial_rope_d768_l19.json}"
export D_MODEL="${D_MODEL:-768}"
export N_HEADS="${N_HEADS:-12}"
export N_LAYERS="${N_LAYERS:-19}"
export BATCH_SIZE="${BATCH_SIZE:-96}"
export MAX_STEPS="${MAX_STEPS:-250000}"
export SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-64}"
export BF16="${BF16:-0}"
export RUN_SLUG="${RUN_SLUG:-matterformer_qm9_trivial_simp_radial_rope_d768_l19}"
export WANDB_GROUP="${WANDB_GROUP:-qm9_hybrid_trivial_simp_radial_rope_l19}"

exec "$REPO_ROOT/scripts/qm9_edm_hybrid_delta.sh"
