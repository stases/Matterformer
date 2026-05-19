#!/bin/bash
# Matterformer-side Platoformer good-run parity architecture, widened to
# tetra_dim_per_frame=256 (h3072), trained with 3-rank DDP and Muon.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:3
#SBATCH --job-name=mf_h3072_good_ddp3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=7-00:00:00
#SBATCH --mem=192G
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_mf_h3072_good_ddp3.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
export HYBRID_CONFIG_JSON="${HYBRID_CONFIG_JSON:-$REPO_ROOT/configs/omol/tetra_t_only_h3072_l16_pt2_goodrun_qkn_uk_csfilm.json}"
export MODEL_BACKEND="${MODEL_BACKEND:-matterformer}"
export OMOL_RUNTIME_MODE="${OMOL_RUNTIME_MODE:-internal_flat_tetra}"

export D_MODEL="${D_MODEL:-3072}"
export N_HEADS="${N_HEADS:-12}"
export N_LAYERS="${N_LAYERS:-16}"
export MLP_RATIO="${MLP_RATIO:-4.0}"
export DROPOUT="${DROPOUT:-0.0}"
export ATTN_DROPOUT="${ATTN_DROPOUT:-0.0}"
export CHGSPIN_MODE="${CHGSPIN_MODE:-add}"
export CHGSPIN_EMB_DIM="${CHGSPIN_EMB_DIM:-256}"
export READOUT_HEAD_MODE="${READOUT_HEAD_MODE:-platonic}"
export READOUT_ACTIVATION="${READOUT_ACTIVATION:-}"

export BATCH_SIZE="${BATCH_SIZE:-16}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
export MAX_GRAPHS_PER_BATCH="${MAX_GRAPHS_PER_BATCH:-999999}"
export MAX_GRAPHS_PER_BATCH_VAL="${MAX_GRAPHS_PER_BATCH_VAL:-999999}"
# Per-rank budget. This matches the pasted Platonic run's 3000 atoms/GPU.
# On 3 GPUs the effective global atom budget is about 9k atoms/update.
export MAX_ATOMS_PER_BATCH="${MAX_ATOMS_PER_BATCH:-3000}"
export MAX_ATOMS_PER_BATCH_VAL="${MAX_ATOMS_PER_BATCH_VAL:-3000}"
export MAX_EDGES_PER_BATCH="${MAX_EDGES_PER_BATCH:-600000}"
export MAX_EDGES_PER_BATCH_VAL="${MAX_EDGES_PER_BATCH_VAL:-600000}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
export PIN_MEMORY="${PIN_MEMORY:-1}"
export BATCHING_MODE="${BATCHING_MODE:-random}"
export BUCKET_WINDOW_SIZE="${BUCKET_WINDOW_SIZE:-4096}"
export BUCKET_SHUFFLE_GROUPS="${BUCKET_SHUFFLE_GROUPS:-8}"
export VALIDATION_MODE="${VALIDATION_MODE:-heldout}"

export MAX_EPOCHS="${MAX_EPOCHS:-60}"
export MAX_STEPS="${MAX_STEPS:-0}"
export LR="${LR:-2.5e-4}"
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

export OPTIMIZER="${OPTIMIZER:-muon}"
export MUON_LR="${MUON_LR:-0.015}"
export MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
export MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0}"
export MUON_ADAM_LR="${MUON_ADAM_LR:-2.5e-4}"
export MUON_ADAM_WEIGHT_DECAY="${MUON_ADAM_WEIGHT_DECAY:-1e-8}"
export MUON_ADAM_BETA1="${MUON_ADAM_BETA1:-0.9}"
export MUON_ADAM_BETA2="${MUON_ADAM_BETA2:-0.999}"
export MUON_ADAM_EPS="${MUON_ADAM_EPS:-1e-8}"
export MUON_HIDDEN_ONLY="${MUON_HIDDEN_ONLY:-1}"
export MUON_MIN_NDIM="${MUON_MIN_NDIM:-2}"
export MUON_PLATONIC_KERNEL_VIEW="${MUON_PLATONIC_KERNEL_VIEW:-conv}"
export MUON_EXCLUDE_NAME_FRAGMENTS="${MUON_EXCLUDE_NAME_FRAGMENTS:-embed,embedding,head,readout,rope,freq,attn.}"

export FLOPS_COEF="${FLOPS_COEF:-72}"
export FULL_VAL_EVERY_STEPS="${FULL_VAL_EVERY_STEPS:-5000}"
export LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-500}"
export LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-500}"
export VAL_ESTIMATE_EVERY_STEPS="${VAL_ESTIMATE_EVERY_STEPS:-0}"
export VAL_ESTIMATE_BATCHES="${VAL_ESTIMATE_BATCHES:-16}"
export PROFILE_STEPS="${PROFILE_STEPS:-0}"
export PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS:-5}"

export RUN_SLUG="${RUN_SLUG:-matterformer-pt2-h3072-dpf256-l16-ls1e-4-ffn4-actsin-ractsin-rs2.0-qkn-uk-csFiLM-flash-ddp3-ema0.99-wd1e-8-adamlr2.5e-4-lrmin1e-6-muonffnconv0.015-mwd0-warm6k-auxadamw-fw20-60ep-n3000pergpu-flat-maxg999999-platonichead-trunkcompile}"
export WANDB_PROJECT="${WANDB_PROJECT:-matterformer_omol_4m}"
export WANDB_GROUP="${WANDB_GROUP:-matterformer_pt2_h3072_dpf256_l16_goodrun_qkn_uk_csfilm_ddp3_muon0p015_warm6k_adamlr2p5e-4_fw20_60ep_n3000pergpu}"

if command -v research-run-start >/dev/null 2>&1; then
  research-run-start || true
fi
trap 'status=$?; if command -v research-run-finish >/dev/null 2>&1; then research-run-finish "$status" || true; fi; exit "$status"' EXIT

"$REPO_ROOT/scripts/omol_forcefield_delta.sh"
