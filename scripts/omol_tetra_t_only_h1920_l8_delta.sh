#!/bin/bash
# OMol 4M pure tetra-global run: T only.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --job-name=omol_T_h1920
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=5-00:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_omol_T_h1920.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
export HYBRID_CONFIG_JSON="${HYBRID_CONFIG_JSON:-$REPO_ROOT/configs/omol/tetra_t_only_h1920_l8_pt2.json}"
export D_MODEL="${D_MODEL:-1920}"
export N_HEADS="${N_HEADS:-60}"
export RUN_SLUG="${RUN_SLUG:-matterformer_omol4m_T_h1920_l8_ffn2_rs2_ema999_w5k_wd1e5}"
export WANDB_PROJECT="${WANDB_PROJECT:-matterformer_omol_4m}"
export WANDB_GROUP="${WANDB_GROUP:-matterformer_omol4m_h1920_l8_pt2}"

exec "$REPO_ROOT/scripts/omol_forcefield_delta.sh"
