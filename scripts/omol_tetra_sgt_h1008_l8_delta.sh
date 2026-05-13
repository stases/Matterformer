#!/bin/bash
# OMol 4M tetra local/global run: S_g + T.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --job-name=omol_SgT_h1008
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=5-00:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_omol_SgT_h1008.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
export HYBRID_CONFIG_JSON="${HYBRID_CONFIG_JSON:-$REPO_ROOT/configs/omol/tetra_sgt_h1008_l8_pt2.json}"
export D_MODEL="${D_MODEL:-1008}"
export N_HEADS="${N_HEADS:-36}"
export RUN_SLUG="${RUN_SLUG:-matterformer_omol4m_SgT_h1008_l8_ffn2_rs2_ema999_w5k_wd1e5}"
export WANDB_PROJECT="${WANDB_PROJECT:-matterformer_omol_4m}"
export WANDB_GROUP="${WANDB_GROUP:-matterformer_omol4m_h1008_l8_pt2}"
export MAX_ATOMS_PER_BATCH="${MAX_ATOMS_PER_BATCH:-3000}"
export MAX_ATOMS_PER_BATCH_VAL="${MAX_ATOMS_PER_BATCH_VAL:-3000}"
export MAX_EDGES_PER_BATCH="${MAX_EDGES_PER_BATCH:-500000}"
export MAX_EDGES_PER_BATCH_VAL="${MAX_EDGES_PER_BATCH_VAL:-500000}"

exec "$REPO_ROOT/scripts/omol_forcefield_delta.sh"
