#!/bin/bash
# Generic QM9 EDM hybrid launcher for the delta partition.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --job-name=qm9_hybrid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --mem=64000M
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_qm9_hybrid.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
PYTHON_BIN="${PYTHON_BIN:-/home/thadziv/GitHub/erwin/erwin/bin/python}"
SLURM_RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"

HYBRID_CONFIG_JSON="${HYBRID_CONFIG_JSON:-$REPO_ROOT/configs/qm9_hybrid/hybrid_v1_1152eff.json}"
D_MODEL="${D_MODEL:-576}"
N_HEADS="${N_HEADS:-12}"
N_LAYERS="${N_LAYERS:-4}"
MLP_RATIO="${MLP_RATIO:-4.0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-250000}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"
TRAIN_AUGM="${TRAIN_AUGM:-1}"

SIGMA_DATA="${SIGMA_DATA:-1.0}"
P_MEAN="${P_MEAN:--1.2}"
P_STD="${P_STD:-1.2}"
ATOM_FEATURE_SCALE="${ATOM_FEATURE_SCALE:-4.0}"
CHARGE_FEATURE_SCALE="${CHARGE_FEATURE_SCALE:-8.0}"
MAX_LOSS_WEIGHT="${MAX_LOSS_WEIGHT:-1000.0}"
NOISE_CONDITIONING="${NOISE_CONDITIONING:-concat}"
COORD_HEAD_MODE="${COORD_HEAD_MODE:-direct}"
USE_CHARGES="${USE_CHARGES:-1}"
NORM_AFFINE_WHEN_NO_ADALN="${NORM_AFFINE_WHEN_NO_ADALN:-1}"
USE_FINAL_NORM="${USE_FINAL_NORM:-1}"

SAMPLE_NUM_STEPS="${SAMPLE_NUM_STEPS:-50}"
SAMPLE_SIGMA_MIN="${SAMPLE_SIGMA_MIN:-0.002}"
SAMPLE_SIGMA_MAX="${SAMPLE_SIGMA_MAX:-1}"
SAMPLE_RHO="${SAMPLE_RHO:-7}"
SAMPLE_S_CHURN="${SAMPLE_S_CHURN:-10}"
SAMPLE_S_MIN="${SAMPLE_S_MIN:-0}"
SAMPLE_S_MAX="${SAMPLE_S_MAX:-inf}"
SAMPLE_S_NOISE="${SAMPLE_S_NOISE:-1}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-64}"

LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-50}"
VAL_ESTIMATE_EVERY_STEPS="${VAL_ESTIMATE_EVERY_STEPS:-500}"
VAL_ESTIMATE_BATCHES="${VAL_ESTIMATE_BATCHES:-16}"
FULL_VAL_EVERY_STEPS="${FULL_VAL_EVERY_STEPS:-5000}"
APPROX_METRICS_EVERY_STEPS="${APPROX_METRICS_EVERY_STEPS:-5000}"
APPROX_METRICS_NUM_MOLECULES="${APPROX_METRICS_NUM_MOLECULES:-256}"
PRECISE_METRICS_EVERY_STEPS="${PRECISE_METRICS_EVERY_STEPS:-50000}"
PRECISE_METRICS_NUM_MOLECULES="${PRECISE_METRICS_NUM_MOLECULES:-10000}"

EMA_DECAY="${EMA_DECAY:-0.999}"
EMA_USE_FOR_SAMPLING="${EMA_USE_FOR_SAMPLING:-1}"
BF16="${BF16:-0}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.5}"
SKIP_LOSS_THRESHOLD="${SKIP_LOSS_THRESHOLD:-20.0}"
SKIP_GRAD_NORM_THRESHOLD="${SKIP_GRAD_NORM_THRESHOLD:-5000.0}"
MAX_CONSECUTIVE_SKIPPED_UPDATES="${MAX_CONSECUTIVE_SKIPPED_UPDATES:-1000}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-1}"
BEST_CHECKPOINT_SELECTOR="${BEST_CHECKPOINT_SELECTOR:-precise_composite}"
CHECKPOINT_MOLECULE_STABILITY_WEIGHT="${CHECKPOINT_MOLECULE_STABILITY_WEIGHT:-0.60}"
CHECKPOINT_VALIDITY_WEIGHT="${CHECKPOINT_VALIDITY_WEIGHT:-0.30}"
CHECKPOINT_UNIQUENESS_WEIGHT="${CHECKPOINT_UNIQUENESS_WEIGHT:-0.10}"

DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/qm9}"
RUN_SLUG="${RUN_SLUG:-matterformer_qm9_hybrid_v1_1152eff}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/thadziv/matterformer_runs/qm9_hybrid}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/${RUN_SLUG}_${SLURM_RUN_ID}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$OUTPUT_DIR/checkpoints/best.pt}"
WANDB_PROJECT="${WANDB_PROJECT:-Matterformer_QM9_hybrid}"
WANDB_GROUP="${WANDB_GROUP:-qm9_hybrid_delta}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_SLUG}_${SLURM_RUN_ID}}"
WANDB_DIR="${WANDB_DIR:-$OUTPUT_DIR/wandb}"
WANDB_MODE="${WANDB_MODE:-online}"
GPU_METRICS_INTERVAL="${GPU_METRICS_INTERVAL:-120}"
GPU_METRICS_LOG="${GPU_METRICS_LOG:-$OUTPUT_DIR/gpu_metrics.csv}"

cd "$REPO_ROOT"
mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$(dirname "$CHECKPOINT_PATH")" "$WANDB_DIR" /home/thadziv/matterformer_jobs/job_outputs

PROCESSED_CACHE="${PROCESSED_CACHE:-$DATA_DIR/processed_qm9_data.pkl}"
CACHE_CANDIDATES=(
  "/home/thadziv/GitHub/ECM/data/QM9/processed_qm9_data.pkl"
  "/home/thadziv/GitHub/pc-gen/datasets/qm9/processed_qm9_data.pkl"
  "/home/thadziv/GitHub/ponita-torch/datasets/qm9/processed_qm9_data.pkl"
)
if [ ! -f "$PROCESSED_CACHE" ]; then
  for candidate in "${CACHE_CANDIDATES[@]}"; do
    if [ -f "$candidate" ]; then
      ln -sfn "$candidate" "$PROCESSED_CACHE"
      break
    fi
  done
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[error] PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [ ! -f "$HYBRID_CONFIG_JSON" ]; then
  echo "[error] HYBRID_CONFIG_JSON not found: $HYBRID_CONFIG_JSON" >&2
  exit 1
fi

REQUIRED_IMPORTS="import torch, wandb, numpy, pandas, requests, tqdm; import rdkit"
if ! "$PYTHON_BIN" -c "$REQUIRED_IMPORTS" >/dev/null 2>&1; then
  echo "[error] PYTHON_BIN is missing required QM9 EDM deps: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_LOG_PID=""
cleanup() {
  if [ -n "$GPU_LOG_PID" ]; then
    kill "$GPU_LOG_PID" >/dev/null 2>&1 || true
    wait "$GPU_LOG_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "gpu_snapshot_start:"
  nvidia-smi || true
  if [ "$GPU_METRICS_INTERVAL" != "0" ]; then
    nvidia-smi \
      --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw \
      --format=csv \
      -l "$GPU_METRICS_INTERVAL" \
      >"$GPU_METRICS_LOG" 2>&1 &
    GPU_LOG_PID=$!
  fi
fi

EXTRA_ARGS=()
[ "$TRAIN_AUGM" = "1" ] && EXTRA_ARGS+=(--train-augm) || EXTRA_ARGS+=(--no-train-augm)
[ "$USE_CHARGES" = "1" ] && EXTRA_ARGS+=(--use-charges) || EXTRA_ARGS+=(--no-use-charges)
[ "$NORM_AFFINE_WHEN_NO_ADALN" = "1" ] && EXTRA_ARGS+=(--norm-affine-when-no-adaln) || EXTRA_ARGS+=(--no-norm-affine-when-no-adaln)
[ "$USE_FINAL_NORM" = "1" ] && EXTRA_ARGS+=(--use-final-norm) || EXTRA_ARGS+=(--no-use-final-norm)
[ "$EMA_USE_FOR_SAMPLING" = "1" ] && EXTRA_ARGS+=(--ema-use-for-sampling) || EXTRA_ARGS+=(--no-ema-use-for-sampling)
[ "$BF16" = "1" ] && EXTRA_ARGS+=(--bf16) || EXTRA_ARGS+=(--no-bf16)
[ "$SAVE_CHECKPOINT" = "1" ] && EXTRA_ARGS+=(--save-checkpoint) || EXTRA_ARGS+=(--no-save-checkpoint)

echo "============================================================"
echo "Matterformer QM9 Hybrid EDM"
echo "start_time:             $(date -Iseconds)"
echo "host:                   $(hostname)"
echo "repo_root:              $REPO_ROOT"
echo "python_bin:             $PYTHON_BIN"
echo "data_dir:               $DATA_DIR"
echo "processed_cache:        $PROCESSED_CACHE"
echo "hybrid_config_json:     $HYBRID_CONFIG_JSON"
echo "d_model:                $D_MODEL"
echo "n_heads:                $N_HEADS"
echo "n_layers:               $N_LAYERS"
echo "batch_size:             $BATCH_SIZE"
echo "max_steps:              $MAX_STEPS"
echo "lr:                     $LR"
echo "weight_decay:           $WEIGHT_DECAY"
echo "noise_conditioning:     $NOISE_CONDITIONING"
echo "coord_head_mode:        $COORD_HEAD_MODE"
echo "sample_num_steps:       $SAMPLE_NUM_STEPS"
echo "sample_sigma_max:       $SAMPLE_SIGMA_MAX"
echo "sample_s_churn:         $SAMPLE_S_CHURN"
echo "bf16:                   $BF16"
echo "skip_loss_threshold:    $SKIP_LOSS_THRESHOLD"
echo "skip_grad_norm_thresh:  $SKIP_GRAD_NORM_THRESHOLD"
echo "output_dir:             $OUTPUT_DIR"
echo "wandb_project:          $WANDB_PROJECT"
echo "wandb_group:            $WANDB_GROUP"
echo "wandb_run_name:         $WANDB_RUN_NAME"
echo "wandb_mode:             $WANDB_MODE"
echo "============================================================"

"$PYTHON_BIN" scripts/train_qm9_edm.py \
  --data-dir "$DATA_DIR" \
  --output "$CHECKPOINT_PATH" \
  --attn-type hybrid \
  --hybrid-config-json "$HYBRID_CONFIG_JSON" \
  --d-model "$D_MODEL" \
  --n-heads "$N_HEADS" \
  --n-layers "$N_LAYERS" \
  --mlp-ratio "$MLP_RATIO" \
  --dropout 0.0 \
  --attn-dropout 0.0 \
  --max-steps "$MAX_STEPS" \
  --warmup-steps "$WARMUP_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --seed "$SEED" \
  --sigma-data "$SIGMA_DATA" \
  --p-mean "$P_MEAN" \
  --p-std "$P_STD" \
  --atom-feature-scale "$ATOM_FEATURE_SCALE" \
  --charge-feature-scale "$CHARGE_FEATURE_SCALE" \
  --max-loss-weight "$MAX_LOSS_WEIGHT" \
  --noise-conditioning "$NOISE_CONDITIONING" \
  --coord-embed-mode none \
  --coord-head-mode "$COORD_HEAD_MODE" \
  --sample-num-steps "$SAMPLE_NUM_STEPS" \
  --sample-sigma-min "$SAMPLE_SIGMA_MIN" \
  --sample-sigma-max "$SAMPLE_SIGMA_MAX" \
  --sample-rho "$SAMPLE_RHO" \
  --sample-s-churn "$SAMPLE_S_CHURN" \
  --sample-s-min "$SAMPLE_S_MIN" \
  --sample-s-max "$SAMPLE_S_MAX" \
  --sample-s-noise "$SAMPLE_S_NOISE" \
  --sample-batch-size "$SAMPLE_BATCH_SIZE" \
  --log-every-steps "$LOG_EVERY_STEPS" \
  --val-estimate-every-steps "$VAL_ESTIMATE_EVERY_STEPS" \
  --val-estimate-batches "$VAL_ESTIMATE_BATCHES" \
  --full-val-every-steps "$FULL_VAL_EVERY_STEPS" \
  --approx-metrics-every-steps "$APPROX_METRICS_EVERY_STEPS" \
  --approx-metrics-num-molecules "$APPROX_METRICS_NUM_MOLECULES" \
  --precise-metrics-every-steps "$PRECISE_METRICS_EVERY_STEPS" \
  --precise-metrics-num-molecules "$PRECISE_METRICS_NUM_MOLECULES" \
  --ema-decay "$EMA_DECAY" \
  --grad-clip-norm "$GRAD_CLIP_NORM" \
  --skip-loss-threshold "$SKIP_LOSS_THRESHOLD" \
  --skip-grad-norm-threshold "$SKIP_GRAD_NORM_THRESHOLD" \
  --max-consecutive-skipped-updates "$MAX_CONSECUTIVE_SKIPPED_UPDATES" \
  --best-checkpoint-selector "$BEST_CHECKPOINT_SELECTOR" \
  --checkpoint-molecule-stability-weight "$CHECKPOINT_MOLECULE_STABILITY_WEIGHT" \
  --checkpoint-validity-weight "$CHECKPOINT_VALIDITY_WEIGHT" \
  --checkpoint-uniqueness-weight "$CHECKPOINT_UNIQUENESS_WEIGHT" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-group "$WANDB_GROUP" \
  --wandb-name "$WANDB_RUN_NAME" \
  --wandb-dir "$WANDB_DIR" \
  --wandb-mode "$WANDB_MODE" \
  "${EXTRA_ARGS[@]}"
