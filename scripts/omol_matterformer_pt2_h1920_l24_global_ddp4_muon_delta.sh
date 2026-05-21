#!/bin/bash
# Matterformer h1920/dpf160/l24 all-global FlashAttention run with Muon,
# launched as 4-process DDP on one Delta node.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:4
#SBATCH --job-name=mf_l24_global_muon
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=7-00:00:00
#SBATCH --mem=256G
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_mf_l24_global_muon.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export HYBRID_CONFIG_JSON="${HYBRID_CONFIG_JSON:-$REPO_ROOT/configs/omol/tetra_t_only_h1920_l24_pt2_goodrun_qkn_uk_csfilm_platonic_input.json}"
export MODEL_BACKEND="${MODEL_BACKEND:-matterformer}"
export OMOL_RUNTIME_MODE="${OMOL_RUNTIME_MODE:-internal_flat_tetra}"

export D_MODEL="${D_MODEL:-1920}"
export MAX_ATOMIC_NUMBER="${MAX_ATOMIC_NUMBER:-99}"
export N_HEADS="${N_HEADS:-12}"
export N_LAYERS="${N_LAYERS:-24}"
export MLP_RATIO="${MLP_RATIO:-4.0}"
export DROPOUT="${DROPOUT:-0.0}"
export ATTN_DROPOUT="${ATTN_DROPOUT:-0.0}"
export CHGSPIN_MODE="${CHGSPIN_MODE:-add}"
export CHGSPIN_EMB_DIM="${CHGSPIN_EMB_DIM:-160}"
export READOUT_HEAD_MODE="${READOUT_HEAD_MODE:-platonic}"
export TETRA_READOUT_MODE="${TETRA_READOUT_MODE:-platonic}"
export READOUT_ACTIVATION="${READOUT_ACTIVATION:-gelu}"
export PLATONIC_INPUT_CONDITIONING="${PLATONIC_INPUT_CONDITIONING:-1}"
export FORCE_ZERO_MEAN="${FORCE_ZERO_MEAN:-0}"
export ROPE_FP64="${ROPE_FP64:-1}"
export READOUT_DISABLE_TF32="${READOUT_DISABLE_TF32:-1}"

export BATCH_SIZE="${BATCH_SIZE:-16}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
export MAX_GRAPHS_PER_BATCH="${MAX_GRAPHS_PER_BATCH:-999999}"
export MAX_GRAPHS_PER_BATCH_VAL="${MAX_GRAPHS_PER_BATCH_VAL:-999999}"
export MAX_ATOMS_PER_BATCH="${MAX_ATOMS_PER_BATCH:-3000}"
export MAX_ATOMS_PER_BATCH_VAL="${MAX_ATOMS_PER_BATCH_VAL:-3000}"
export MAX_EDGES_PER_BATCH="${MAX_EDGES_PER_BATCH:-600000}"
export MAX_EDGES_PER_BATCH_VAL="${MAX_EDGES_PER_BATCH_VAL:-600000}"
export NUM_WORKERS="${NUM_WORKERS:-16}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
export PIN_MEMORY="${PIN_MEMORY:-1}"
export BATCHING_MODE="${BATCHING_MODE:-random}"
export BUCKET_WINDOW_SIZE="${BUCKET_WINDOW_SIZE:-4096}"
export BUCKET_SHUFFLE_GROUPS="${BUCKET_SHUFFLE_GROUPS:-8}"
export VALIDATION_MODE="${VALIDATION_MODE:-heldout}"

export MAX_EPOCHS="${MAX_EPOCHS:-60}"
export MAX_STEPS="${MAX_STEPS:-0}"
export LR="${LR:-3e-4}"
export LR_MIN="${LR_MIN:-1e-6}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-8}"
export WARMUP_STEPS="${WARMUP_STEPS:-6000}"
export NORMALIZER_RMSD="${NORMALIZER_RMSD:-1.433569}"
export ENERGY_WEIGHT="${ENERGY_WEIGHT:-10}"
export FORCE_WEIGHT="${FORCE_WEIGHT:-20}"
export ENERGY_LOSS="${ENERGY_LOSS:-per_atom_mae}"
export FORCE_LOSS="${FORCE_LOSS:-l2norm}"
export TRAIN_AUGMENTATION="${TRAIN_AUGMENTATION:-o3}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1}"
export SKIP_LOSS_ABOVE="${SKIP_LOSS_ABOVE:-1000}"
export EMA_DECAY="${EMA_DECAY:-0.99}"
export EMA_WARMUP_STEPS="${EMA_WARMUP_STEPS:-2000}"
export BF16="${BF16:-0}"
export FLOAT32_MATMUL_PRECISION="${FLOAT32_MATMUL_PRECISION:-high}"
export COMPILE="${COMPILE:-1}"
export COMPILE_MODE="${COMPILE_MODE:-default}"
export COMPILE_SCOPE="${COMPILE_SCOPE:-trunk_flat}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-8}"

# train_omol_forcefield.py uses cosine warmup/annealing unconditionally.
export OPTIMIZER="${OPTIMIZER:-muon}"
export MUON_LR="${MUON_LR:-0.01}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0}"
export MUON_ADAM_LR="${MUON_ADAM_LR:-3e-4}"
export MUON_ADAM_WEIGHT_DECAY="${MUON_ADAM_WEIGHT_DECAY:-1e-8}"
export MUON_ADAM_BETA1="${MUON_ADAM_BETA1:-0.9}"
export MUON_ADAM_BETA2="${MUON_ADAM_BETA2:-0.999}"
export MUON_ADAM_EPS="${MUON_ADAM_EPS:-1e-8}"
export MUON_HIDDEN_ONLY="${MUON_HIDDEN_ONLY:-1}"
export MUON_MIN_NDIM="${MUON_MIN_NDIM:-2}"
export MUON_PLATONIC_KERNEL_VIEW="${MUON_PLATONIC_KERNEL_VIEW:-conv}"
export MUON_EXCLUDE_NAME_FRAGMENTS="${MUON_EXCLUDE_NAME_FRAGMENTS:-embed,embedding,head,readout,rope,freq,attn.}"

export FLOPS_COEF="${FLOPS_COEF:-108}"
export FULL_VAL_EVERY_STEPS="${FULL_VAL_EVERY_STEPS:-5000}"
export LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-500}"
export LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-500}"
export VAL_ESTIMATE_EVERY_STEPS="${VAL_ESTIMATE_EVERY_STEPS:-0}"
export VAL_ESTIMATE_BATCHES="${VAL_ESTIMATE_BATCHES:-16}"
export PROFILE_STEPS="${PROFILE_STEPS:-0}"
export PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS:-5}"
export GPU_METRICS_INTERVAL="${GPU_METRICS_INTERVAL:-120}"

export RUN_SLUG="${RUN_SLUG:-mf_pt2_h1920_l24_global_ddp4_muon0p01_adam3e-4_warm6k_ropefp64}"
export WANDB_PROJECT="${WANDB_PROJECT:-matterformer_omol_4m}"
export WANDB_GROUP="${WANDB_GROUP:-matterformer_pt2_h1920_l24_global_ddp4_muon0p01_adam3e-4_warm6k_fw20}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-matterformer-pt2-h1920-l24-global-ddp4-muon0p01-adam3e-4-warm6k}"

if command -v research-run-start >/dev/null 2>&1; then
  research-run-start || true
fi
trap 'status=$?; if command -v research-run-finish >/dev/null 2>&1; then research-run-finish "$status" || true; fi; exit "$status"' EXIT

"$REPO_ROOT/scripts/omol_forcefield_delta.sh"
