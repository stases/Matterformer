#!/bin/bash
# OMol 4M FairChem AllScAIP-sm direct-force run with LAE/ERoPE encodings enabled.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --job-name=omol_allscaip_sm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=5-00:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_omol_allscaip_sm.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
export FAIRCHEM_SRC="${FAIRCHEM_SRC:-/home/thadziv/GitHub/fairchem/src}"
export MODEL_BACKEND="${MODEL_BACKEND:-allscaip_direct}"
ALLSCAIP_BASELINE_CONFIG="${ALLSCAIP_BASELINE_CONFIG:-$REPO_ROOT/configs/omol/allscaip_omol4m_sm_nenosi_direct.json}"

export RUN_SLUG="${RUN_SLUG:-matterformer_omol4m_allscaip_sm_nenosi_direct_h512_l6_b96_a400_lr8e4_baseline}"
export WANDB_PROJECT="${WANDB_PROJECT:-matterformer_omol_4m}"
export WANDB_GROUP="${WANDB_GROUP:-matterformer_omol4m_allscaip_sm_nenosi_direct_baseline}"

export BATCH_SIZE="${BATCH_SIZE:-96}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-96}"
export MAX_ATOMS_PER_BATCH="${MAX_ATOMS_PER_BATCH:-400}"
export MAX_ATOMS_PER_BATCH_VAL="${MAX_ATOMS_PER_BATCH_VAL:-400}"
export MAX_EDGES_PER_BATCH="${MAX_EDGES_PER_BATCH:-2000000}"
export MAX_EDGES_PER_BATCH_VAL="${MAX_EDGES_PER_BATCH_VAL:-2000000}"

export ALLSCAIP_CONFIG_JSON="${ALLSCAIP_CONFIG_JSON:-$ALLSCAIP_BASELINE_CONFIG}"
export ALLSCAIP_STRICT_CONFIG_JSON="${ALLSCAIP_STRICT_CONFIG_JSON:-$ALLSCAIP_BASELINE_CONFIG}"
export ALLSCAIP_HIDDEN_SIZE="${ALLSCAIP_HIDDEN_SIZE:-512}"
export ALLSCAIP_NUM_LAYERS="${ALLSCAIP_NUM_LAYERS:-6}"
export ALLSCAIP_ATTEN_NUM_HEADS="${ALLSCAIP_ATTEN_NUM_HEADS:-8}"
export ALLSCAIP_MAX_ATOMS="${ALLSCAIP_MAX_ATOMS:-400}"
export ALLSCAIP_MAX_BATCH_SIZE="${ALLSCAIP_MAX_BATCH_SIZE:-96}"
export ALLSCAIP_MAX_RADIUS="${ALLSCAIP_MAX_RADIUS:-6.0}"
export ALLSCAIP_KNN_K="${ALLSCAIP_KNN_K:-30}"
export ALLSCAIP_KNN_PAD_SIZE="${ALLSCAIP_KNN_PAD_SIZE:-50}"
export ALLSCAIP_ATTEN_NAME="${ALLSCAIP_ATTEN_NAME:-memory_efficient}"
export ALLSCAIP_COMPILE="${ALLSCAIP_COMPILE:-1}"
export ALLSCAIP_USE_PADDING="${ALLSCAIP_USE_PADDING:-1}"
export ALLSCAIP_USE_CHUNKED_GRAPH="${ALLSCAIP_USE_CHUNKED_GRAPH:-0}"
export ALLSCAIP_PREPROCESS_ON_CPU="${ALLSCAIP_PREPROCESS_ON_CPU:-0}"

export MAX_EPOCHS="${MAX_EPOCHS:-80}"
export LR="${LR:-8e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}"
export WARMUP_STEPS="${WARMUP_STEPS:-2000}"
export TRAIN_AUGMENTATION="${TRAIN_AUGMENTATION:-off}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-100}"
export EMA_DECAY="${EMA_DECAY:-0}"
export EMA_WARMUP_STEPS="${EMA_WARMUP_STEPS:-0}"
export BF16="${BF16:-0}"
export FLOAT32_MATMUL_PRECISION="${FLOAT32_MATMUL_PRECISION:-high}"
export VAL_ESTIMATE_EVERY_STEPS="${VAL_ESTIMATE_EVERY_STEPS:-0}"
export FULL_VAL_EVERY_STEPS="${FULL_VAL_EVERY_STEPS:-2000}"

exec "$REPO_ROOT/scripts/omol_forcefield_delta.sh"
