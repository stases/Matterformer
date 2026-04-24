#!/bin/bash
# BWDB stage-1 CSP run on delta using Triton simplicial attention with geometry bias disabled.
# Mirrors the current 65M-stage launcher but forces lattice_repr=y1.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --job-name=bwdb_s1_65m_y1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --mem=64000M
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_bwdb_s1_65m_y1.out

set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
export PYTHON_BIN="${PYTHON_BIN:-/home/thadziv/GitHub/erwin/erwin/bin/python}"

export D_MODEL="${D_MODEL:-512}"
export N_LAYERS="${N_LAYERS:-12}"
export N_HEADS="${N_HEADS:-8}"
export MLP_RATIO="${MLP_RATIO:-4.0}"
export ATTN_TYPE="${ATTN_TYPE:-simplicial}"
export SIMPLICIAL_GEOM_MODE="${SIMPLICIAL_GEOM_MODE:-factorized}"
export SIMPLICIAL_IMPL="${SIMPLICIAL_IMPL:-triton}"
export SIMPLICIAL_PRECISION="${SIMPLICIAL_PRECISION:-bf16_tc}"
export DISABLE_GEOMETRY_BIAS="${DISABLE_GEOMETRY_BIAS:-1}"
export LATTICE_REPR="${LATTICE_REPR:-y1}"
export PBC_RADIUS="${PBC_RADIUS:-1}"

export LR="${LR:-8e-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-10000}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export MAX_STEPS="${MAX_STEPS:-1500000}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-12}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export SEED="${SEED:-0}"

export LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-50}"
export VAL_ESTIMATE_EVERY_STEPS="${VAL_ESTIMATE_EVERY_STEPS:-500}"
export VAL_ESTIMATE_BATCHES="${VAL_ESTIMATE_BATCHES:-16}"
export FULL_VAL_EVERY_STEPS="${FULL_VAL_EVERY_STEPS:-5000}"

export PSEUDO_MATCH_EVERY_STEPS="${PSEUDO_MATCH_EVERY_STEPS:-10000}"
export PSEUDO_MATCH_NUM_SAMPLES="${PSEUDO_MATCH_NUM_SAMPLES:-1024}"
export PSEUDO_MATCH_BATCH_SIZE="${PSEUDO_MATCH_BATCH_SIZE:-128}"
export PSEUDO_MATCH_STOL="${PSEUDO_MATCH_STOL:-0.30}"
export PSEUDO_MATCH_LTOL="${PSEUDO_MATCH_LTOL:-0.20}"
export PSEUDO_MATCH_ANGLE_TOL="${PSEUDO_MATCH_ANGLE_TOL:-10.0}"

export EMA_DECAY="${EMA_DECAY:-0.9999}"
export EMA_USE_FOR_SAMPLING="${EMA_USE_FOR_SAMPLING:-1}"
export BF16="${BF16:-1}"
export GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.5}"

export SAMPLE_NUM_STEPS="${SAMPLE_NUM_STEPS:-100}"
export SAMPLE_SIGMA_MIN="${SAMPLE_SIGMA_MIN:-0.002}"
export SAMPLE_SIGMA_MAX="${SAMPLE_SIGMA_MAX:-10}"
export SAMPLE_RHO="${SAMPLE_RHO:-7}"
export SAMPLE_S_CHURN="${SAMPLE_S_CHURN:-30}"
export SAMPLE_S_MIN="${SAMPLE_S_MIN:-0}"
export SAMPLE_S_MAX="${SAMPLE_S_MAX:-inf}"
export SAMPLE_S_NOISE="${SAMPLE_S_NOISE:-1.003}"

export P_MEAN="${P_MEAN:--1.2}"
export P_STD="${P_STD:-1.2}"
export SIGMA_DATA_COORD="${SIGMA_DATA_COORD:-0.5}"
export SIGMA_DATA_LATTICE="${SIGMA_DATA_LATTICE:-0.5}"
export COORD_WEIGHT="${COORD_WEIGHT:-15.0}"
export LATTICE_WEIGHT="${LATTICE_WEIGHT:-1.0}"
export ALIGN_GLOBAL_SHIFT="${ALIGN_GLOBAL_SHIFT:-1}"
export SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-1}"

export DATA_DIR="${DATA_DIR:-/home/thadziv/GitHub/Matterformer/data/mofs/bwdb}"

export RUN_SLUG="${RUN_SLUG:-bwdb_stage1_csp_65m_triton_simp_nogb_y1}"
export WANDB_PROJECT="${WANDB_PROJECT:-Matterformer_BWDB_stage1}"
export WANDB_GROUP="${WANDB_GROUP:-bwdb_stage1_csp_65m_triton_simp_nogb_y1}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_SLUG}_${SLURM_JOB_ID:-slurm}}"
export WANDB_MODE="${WANDB_MODE:-online}"

export GPU_METRICS_INTERVAL="${GPU_METRICS_INTERVAL:-60}"

exec /home/thadziv/matterformer_jobs/mof_stage1_edm.sh
