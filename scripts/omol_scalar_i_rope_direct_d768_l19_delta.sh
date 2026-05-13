#!/bin/bash
# OMol 4M scalar global Matterformer run: I/MHA only with non-equivariant direct 3D force head.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --job-name=omol_I_dir_d768
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=5-00:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_omol_I_dir_d768.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
export HYBRID_CONFIG_JSON="${HYBRID_CONFIG_JSON:-$REPO_ROOT/configs/omol/scalar_i_rope_d768_l19.json}"
export D_MODEL="${D_MODEL:-768}"
export N_HEADS="${N_HEADS:-12}"
export N_LAYERS="${N_LAYERS:-19}"
export MLP_RATIO="${MLP_RATIO:-4.0}"
export FORCE_HEAD_MODE="${FORCE_HEAD_MODE:-direct}"
export RUN_SLUG="${RUN_SLUG:-matterformer_omol4m_I_rope_direct3d_d768_l19_mlp4_b64_a4k_e175k_ema999_w5k_wd1e5}"
export WANDB_PROJECT="${WANDB_PROJECT:-matterformer_omol_4m}"
export WANDB_GROUP="${WANDB_GROUP:-matterformer_omol4m_I_rope_direct3d_d768_l19}"

export BATCH_SIZE="${BATCH_SIZE:-64}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
export MAX_ATOMS_PER_BATCH="${MAX_ATOMS_PER_BATCH:-4000}"
export MAX_ATOMS_PER_BATCH_VAL="${MAX_ATOMS_PER_BATCH_VAL:-4000}"
export MAX_EDGES_PER_BATCH="${MAX_EDGES_PER_BATCH:-175000}"
export MAX_EDGES_PER_BATCH_VAL="${MAX_EDGES_PER_BATCH_VAL:-175000}"

exec "$REPO_ROOT/scripts/omol_forcefield_delta.sh"
