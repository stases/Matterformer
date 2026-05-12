#!/bin/bash
# OMol 4M scalar I+S run: RoPE MHA plus Triton simplicial, direct 3D force head.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --job-name=omol_IS_dir_d768
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=5-00:00:00
#SBATCH --mem=224000M
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_omol_IS_dir_d768.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
export HYBRID_CONFIG_JSON="${HYBRID_CONFIG_JSON:-$REPO_ROOT/configs/omol/scalar_is_triton_d768_l19.json}"
export D_MODEL="${D_MODEL:-768}"
export N_HEADS="${N_HEADS:-12}"
export N_LAYERS="${N_LAYERS:-19}"
export MLP_RATIO="${MLP_RATIO:-4.0}"
export FORCE_HEAD_MODE="${FORCE_HEAD_MODE:-direct}"
export LR="${LR:-3e-4}"
export RUN_SLUG="${RUN_SLUG:-matterformer_omol4m_IS_triton_direct3d_d768_l19_k16_angle16_b32_a2k_e100k_lr3e4_ema999_w5k_wd1e5}"
export WANDB_PROJECT="${WANDB_PROJECT:-matterformer_omol_4m}"
export WANDB_GROUP="${WANDB_GROUP:-matterformer_omol4m_IS_triton_direct3d_d768_l19}"

export BATCH_SIZE="${BATCH_SIZE:-32}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
export MAX_ATOMS_PER_BATCH="${MAX_ATOMS_PER_BATCH:-2000}"
export MAX_ATOMS_PER_BATCH_VAL="${MAX_ATOMS_PER_BATCH_VAL:-2000}"
export MAX_EDGES_PER_BATCH="${MAX_EDGES_PER_BATCH:-100000}"
export MAX_EDGES_PER_BATCH_VAL="${MAX_EDGES_PER_BATCH_VAL:-100000}"

exec "$REPO_ROOT/scripts/omol_forcefield_delta.sh"
