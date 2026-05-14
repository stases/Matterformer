#!/bin/bash
# Matterformer PT2 exact-style OMol run with a 16-layer tetra trunk.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --job-name=mf_pt2_l16
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=5-00:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_mf_pt2_l16.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"

export HYBRID_CONFIG_JSON="${HYBRID_CONFIG_JSON:-$REPO_ROOT/configs/omol/tetra_t_only_h1920_l16_pt2_exact_sin_layerscale.json}"
export MODEL_BACKEND="${MODEL_BACKEND:-matterformer}"
export OMOL_RUNTIME_MODE="${OMOL_RUNTIME_MODE:-internal_flat_tetra}"

export D_MODEL="${D_MODEL:-1920}"
export N_HEADS="${N_HEADS:-60}"
export N_LAYERS="${N_LAYERS:-16}"
export MLP_RATIO="${MLP_RATIO:-2.0}"
export DROPOUT="${DROPOUT:-0.0}"
export ATTN_DROPOUT="${ATTN_DROPOUT:-0.0}"
export CHGSPIN_MODE="${CHGSPIN_MODE:-add}"
export CHGSPIN_EMB_DIM="${CHGSPIN_EMB_DIM:-160}"
export READOUT_HEAD_MODE="${READOUT_HEAD_MODE:-platonic}"
export READOUT_ACTIVATION="${READOUT_ACTIVATION:-sin}"

export BATCH_SIZE="${BATCH_SIZE:-64}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
export MAX_GRAPHS_PER_BATCH="${MAX_GRAPHS_PER_BATCH:-999999}"
export MAX_GRAPHS_PER_BATCH_VAL="${MAX_GRAPHS_PER_BATCH_VAL:-999999}"
export MAX_ATOMS_PER_BATCH="${MAX_ATOMS_PER_BATCH:-12000}"
export MAX_ATOMS_PER_BATCH_VAL="${MAX_ATOMS_PER_BATCH_VAL:-12000}"
export MAX_EDGES_PER_BATCH="${MAX_EDGES_PER_BATCH:-2400000}"
export MAX_EDGES_PER_BATCH_VAL="${MAX_EDGES_PER_BATCH_VAL:-2400000}"
export NUM_WORKERS="${NUM_WORKERS:-16}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
export BATCHING_MODE="${BATCHING_MODE:-random}"
export VALIDATION_MODE="${VALIDATION_MODE:-heldout}"

export MAX_EPOCHS="${MAX_EPOCHS:-40}"
export MAX_STEPS="${MAX_STEPS:-0}"
export LR="${LR:-2.5e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-2000}"
export NORMALIZER_RMSD="${NORMALIZER_RMSD:-1.433569}"
export ENERGY_WEIGHT="${ENERGY_WEIGHT:-10}"
export FORCE_WEIGHT="${FORCE_WEIGHT:-10}"
export TRAIN_AUGMENTATION="${TRAIN_AUGMENTATION:-o3}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1}"
export SKIP_LOSS_ABOVE="${SKIP_LOSS_ABOVE:-1000}"
export EMA_DECAY="${EMA_DECAY:-0.99}"
export EMA_WARMUP_STEPS="${EMA_WARMUP_STEPS:-2000}"
export BF16="${BF16:-0}"
export FLOAT32_MATMUL_PRECISION="${FLOAT32_MATMUL_PRECISION:-high}"
export COMPILE="${COMPILE:-1}"
export COMPILE_MODE="${COMPILE_MODE:-default}"

export FLOPS_COEF="${FLOPS_COEF:-72}"
export FULL_VAL_EVERY_STEPS="${FULL_VAL_EVERY_STEPS:-5000}"
export LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-500}"
export LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-500}"
export PROFILE_STEPS="${PROFILE_STEPS:-0}"

export RUN_SLUG="${RUN_SLUG:-matterformer-pt2-h1920-l16-ls1e-4-ffn2-actsin-ractsin-rs2.0-ema0.99-wd1e-5-lr2.5e-4-40ep-n12000-flat-maxg999999-platonichead-skip1000}"
export WANDB_PROJECT="${WANDB_PROJECT:-matterformer_omol_4m}"
export WANDB_GROUP="${WANDB_GROUP:-matterformer_pt2_exact_config_l16_maxg999999_platonichead_lr2p5e-4_skip1000}"

exec "$REPO_ROOT/scripts/omol_forcefield_delta.sh"
