#!/bin/bash
# Generic single-GPU Matterformer OMol direct-force launcher for delta.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
PYTHON_BIN="${PYTHON_BIN:-/home/thadziv/GitHub/erwin/erwin/bin/python}"
SLURM_RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"

OMOL_DATA_ROOT="${OMOL_DATA_ROOT:-/home/ebekker/data/omol/open_mol}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$OMOL_DATA_ROOT/train_4M}"
VAL_DATA_PATH="${VAL_DATA_PATH:-$OMOL_DATA_ROOT/val}"
VALIDATION_MODE="${VALIDATION_MODE:-heldout}"
ELEMENT_REFS_JSON="${ELEMENT_REFS_JSON:-$REPO_ROOT/configs/omol/element_refs.json}"
HYBRID_CONFIG_JSON="${HYBRID_CONFIG_JSON:-$REPO_ROOT/configs/omol/tetra_t_only_h1920_l8_pt2.json}"

D_MODEL="${D_MODEL:-1920}"
N_HEADS="${N_HEADS:-60}"
N_LAYERS="${N_LAYERS:-8}"
MLP_RATIO="${MLP_RATIO:-2.0}"
DROPOUT="${DROPOUT:-0.0}"
ATTN_DROPOUT="${ATTN_DROPOUT:-0.0}"
CHGSPIN_MODE="${CHGSPIN_MODE:-add}"
PAIR_HIDDEN_DIM="${PAIR_HIDDEN_DIM:-128}"
PAIR_N_RBF="${PAIR_N_RBF:-16}"
PAIR_RBF_MAX="${PAIR_RBF_MAX:-6.0}"

BATCH_SIZE="${BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
MAX_ATOMS_PER_BATCH="${MAX_ATOMS_PER_BATCH:-12000}"
MAX_ATOMS_PER_BATCH_VAL="${MAX_ATOMS_PER_BATCH_VAL:-12000}"
MAX_EDGES_PER_BATCH="${MAX_EDGES_PER_BATCH:-2000000}"
MAX_EDGES_PER_BATCH_VAL="${MAX_EDGES_PER_BATCH_VAL:-2000000}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"

MAX_EPOCHS="${MAX_EPOCHS:-20}"
MAX_STEPS="${MAX_STEPS:-0}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
NORMALIZER_RMSD="${NORMALIZER_RMSD:-1.433569}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-10}"
FORCE_WEIGHT="${FORCE_WEIGHT:-10}"
TRAIN_AUGMENTATION="${TRAIN_AUGMENTATION:-o3}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1}"
EMA_DECAY="${EMA_DECAY:-0.999}"
EMA_WARMUP_STEPS="${EMA_WARMUP_STEPS:-5000}"
BF16="${BF16:-0}"
FLOAT32_MATMUL_PRECISION="${FLOAT32_MATMUL_PRECISION:-highest}"

LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-10}"
PROFILE_STEPS="${PROFILE_STEPS:-0}"
PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS:-5}"
VAL_ESTIMATE_EVERY_STEPS="${VAL_ESTIMATE_EVERY_STEPS:-0}"
VAL_ESTIMATE_BATCHES="${VAL_ESTIMATE_BATCHES:-16}"
FULL_VAL_EVERY_STEPS="${FULL_VAL_EVERY_STEPS:-5000}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-500}"
LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-500}"

SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-1}"
RUN_SLUG="${RUN_SLUG:-matterformer_omol4m_tetra_t_h1920_l8_pt2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/thadziv/matterformer_runs/omol_4m}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/${RUN_SLUG}_${SLURM_RUN_ID}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$OUTPUT_DIR/checkpoints/best.pt}"
WANDB_PROJECT="${WANDB_PROJECT:-matterformer_omol_4m}"
WANDB_GROUP="${WANDB_GROUP:-matterformer_omol4m_h1920_l8_pt2}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_SLUG}_${SLURM_RUN_ID}}"
WANDB_DIR="${WANDB_DIR:-$OUTPUT_DIR/wandb}"
WANDB_MODE="${WANDB_MODE:-online}"
GPU_METRICS_INTERVAL="${GPU_METRICS_INTERVAL:-120}"
GPU_METRICS_LOG="${GPU_METRICS_LOG:-$OUTPUT_DIR/gpu_metrics.csv}"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR" "$(dirname "$CHECKPOINT_PATH")" "$WANDB_DIR" /home/thadziv/matterformer_jobs/job_outputs

if [ -n "${FAIRCHEM_SRC:-}" ]; then
  export PYTHONPATH="$FAIRCHEM_SRC:$REPO_ROOT/src:${PYTHONPATH:-}"
else
  export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[error] PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [ ! -d "$TRAIN_DATA_PATH" ]; then
  echo "[error] TRAIN_DATA_PATH does not exist: $TRAIN_DATA_PATH" >&2
  exit 1
fi
if [ ! -d "$VAL_DATA_PATH" ]; then
  echo "[error] VAL_DATA_PATH does not exist: $VAL_DATA_PATH" >&2
  exit 1
fi
if [ ! -f "$HYBRID_CONFIG_JSON" ]; then
  echo "[error] HYBRID_CONFIG_JSON not found: $HYBRID_CONFIG_JSON" >&2
  exit 1
fi
if [ ! -f "$ELEMENT_REFS_JSON" ]; then
  echo "[error] ELEMENT_REFS_JSON not found: $ELEMENT_REFS_JSON" >&2
  exit 1
fi

REQUIRED_IMPORTS="import torch, wandb, ase, ase_db_backends; import ase.db"
if ! "$PYTHON_BIN" -c "$REQUIRED_IMPORTS" >/dev/null 2>&1; then
  echo "[error] PYTHON_BIN cannot import OMol runtime deps (torch, wandb, ase, ase-db-backends):" >&2
  echo "        $PYTHON_BIN" >&2
  echo "        Install the missing package in that environment or set PYTHON_BIN to one that has it." >&2
  exit 1
fi

if ! "$PYTHON_BIN" - "$TRAIN_DATA_PATH" "$VAL_DATA_PATH" <<'PY' >/dev/null 2>&1
import sys
from pathlib import Path

import ase.db
import ase_db_backends  # noqa: F401

for root in sys.argv[1:]:
    shards = sorted(path for path in Path(root).iterdir() if path.name.endswith(".aselmdb"))
    if not shards:
        raise SystemExit(f"No .aselmdb shards found under {root}")
    db = ase.db.connect(str(shards[0]), readonly=True, use_lock_file=False)
    ids = list(getattr(db, "ids", []))
    if not ids:
        ids = [row.id for row in db.select(limit=1)]
    if not ids:
        raise SystemExit(f"No rows found in {shards[0]}")
    row = db._get_row(ids[0])
    _ = row.energy
    _ = row.forces
PY
then
  echo "[error] Could not read OMol ASE DB shards with $PYTHON_BIN" >&2
  echo "        train_data_path: $TRAIN_DATA_PATH" >&2
  echo "        val_data_path:   $VAL_DATA_PATH" >&2
  exit 1
fi

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
[ "$BF16" = "1" ] && EXTRA_ARGS+=(--bf16) || EXTRA_ARGS+=(--no-bf16)
[ "$SAVE_CHECKPOINT" = "1" ] && EXTRA_ARGS+=(--save-checkpoint) || EXTRA_ARGS+=(--no-save-checkpoint)
[ -n "${RESUME_CHECKPOINT:-}" ] && EXTRA_ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
[ -n "${WARM_START_CHECKPOINT:-}" ] && EXTRA_ARGS+=(--warm-start-checkpoint "$WARM_START_CHECKPOINT")

echo "============================================================"
echo "Matterformer OMol Direct Force"
echo "start_time:              $(date -Iseconds)"
echo "host:                    $(hostname)"
echo "repo_root:               $REPO_ROOT"
echo "python_bin:              $PYTHON_BIN"
echo "train_data_path:         $TRAIN_DATA_PATH"
echo "val_data_path:           $VAL_DATA_PATH"
echo "hybrid_config_json:      $HYBRID_CONFIG_JSON"
echo "d_model:                 $D_MODEL"
echo "n_heads:                 $N_HEADS"
echo "n_layers:                $N_LAYERS"
echo "mlp_ratio:               $MLP_RATIO"
echo "batch_size:              $BATCH_SIZE"
echo "max_atoms_per_batch:     $MAX_ATOMS_PER_BATCH"
echo "max_edges_per_batch:     $MAX_EDGES_PER_BATCH"
echo "max_epochs:              $MAX_EPOCHS"
echo "max_steps:               $MAX_STEPS (0 means epoch-limited)"
echo "lr:                      $LR"
echo "weight_decay:            $WEIGHT_DECAY"
echo "warmup_steps:            $WARMUP_STEPS"
echo "normalizer_rmsd:         $NORMALIZER_RMSD"
echo "train_augmentation:      $TRAIN_AUGMENTATION"
echo "bf16:                    $BF16"
echo "float32_matmul_prec:     $FLOAT32_MATMUL_PRECISION"
echo "grad_clip_norm:          $GRAD_CLIP_NORM"
echo "ema_decay:               $EMA_DECAY"
echo "ema_warmup_steps:        $EMA_WARMUP_STEPS"
echo "profile_steps:           $PROFILE_STEPS"
echo "profile_warmup_steps:    $PROFILE_WARMUP_STEPS"
echo "output_dir:              $OUTPUT_DIR"
echo "wandb_project:           $WANDB_PROJECT"
echo "wandb_group:             $WANDB_GROUP"
echo "wandb_run_name:          $WANDB_RUN_NAME"
echo "wandb_mode:              $WANDB_MODE"
echo "============================================================"

"$PYTHON_BIN" scripts/train_omol_forcefield.py \
  --train-data-path "$TRAIN_DATA_PATH" \
  --val-data-path "$VAL_DATA_PATH" \
  --validation-mode "$VALIDATION_MODE" \
  --output "$CHECKPOINT_PATH" \
  --element-refs-json "$ELEMENT_REFS_JSON" \
  --hybrid-config-json "$HYBRID_CONFIG_JSON" \
  --d-model "$D_MODEL" \
  --n-heads "$N_HEADS" \
  --n-layers "$N_LAYERS" \
  --mlp-ratio "$MLP_RATIO" \
  --dropout "$DROPOUT" \
  --attn-dropout "$ATTN_DROPOUT" \
  --chgspin-mode "$CHGSPIN_MODE" \
  --pair-hidden-dim "$PAIR_HIDDEN_DIM" \
  --pair-n-rbf "$PAIR_N_RBF" \
  --pair-rbf-max "$PAIR_RBF_MAX" \
  --batch-size "$BATCH_SIZE" \
  --val-batch-size "$VAL_BATCH_SIZE" \
  --max-atoms-per-batch "$MAX_ATOMS_PER_BATCH" \
  --max-atoms-per-batch-val "$MAX_ATOMS_PER_BATCH_VAL" \
  --max-edges-per-batch "$MAX_EDGES_PER_BATCH" \
  --max-edges-per-batch-val "$MAX_EDGES_PER_BATCH_VAL" \
  --num-workers "$NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR" \
  --max-epochs "$MAX_EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --warmup-steps "$WARMUP_STEPS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --normalizer-rmsd "$NORMALIZER_RMSD" \
  --energy-weight "$ENERGY_WEIGHT" \
  --force-weight "$FORCE_WEIGHT" \
  --train-augmentation "$TRAIN_AUGMENTATION" \
  --grad-clip-norm "$GRAD_CLIP_NORM" \
  --ema-decay "$EMA_DECAY" \
  --ema-warmup-steps "$EMA_WARMUP_STEPS" \
  --float32-matmul-precision "$FLOAT32_MATMUL_PRECISION" \
  --log-every-steps "$LOG_EVERY_STEPS" \
  --profile-steps "$PROFILE_STEPS" \
  --profile-warmup-steps "$PROFILE_WARMUP_STEPS" \
  --val-estimate-every-steps "$VAL_ESTIMATE_EVERY_STEPS" \
  --val-estimate-batches "$VAL_ESTIMATE_BATCHES" \
  --full-val-every-steps "$FULL_VAL_EVERY_STEPS" \
  --limit-val-batches "$LIMIT_VAL_BATCHES" \
  --limit-test-batches "$LIMIT_TEST_BATCHES" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-name "$WANDB_RUN_NAME" \
  --wandb-group "$WANDB_GROUP" \
  --wandb-dir "$WANDB_DIR" \
  --wandb-mode "$WANDB_MODE" \
  "${EXTRA_ARGS[@]}"
