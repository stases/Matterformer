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
MODEL_BACKEND="${MODEL_BACKEND:-matterformer}"

D_MODEL="${D_MODEL:-1920}"
N_HEADS="${N_HEADS:-60}"
N_LAYERS="${N_LAYERS:-8}"
MLP_RATIO="${MLP_RATIO:-2.0}"
DROPOUT="${DROPOUT:-0.0}"
ATTN_DROPOUT="${ATTN_DROPOUT:-0.0}"
CHGSPIN_MODE="${CHGSPIN_MODE:-add}"
CHGSPIN_EMB_DIM="${CHGSPIN_EMB_DIM:-}"
PAIR_HIDDEN_DIM="${PAIR_HIDDEN_DIM:-128}"
PAIR_N_RBF="${PAIR_N_RBF:-16}"
PAIR_RBF_MAX="${PAIR_RBF_MAX:-6.0}"
FORCE_HEAD_MODE="${FORCE_HEAD_MODE:-auto}"
READOUT_HEAD_MODE="${READOUT_HEAD_MODE:-dense}"
READOUT_ACTIVATION="${READOUT_ACTIVATION:-}"

BATCH_SIZE="${BATCH_SIZE:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
MAX_GRAPHS_PER_BATCH="${MAX_GRAPHS_PER_BATCH:-}"
MAX_GRAPHS_PER_BATCH_VAL="${MAX_GRAPHS_PER_BATCH_VAL:-}"
MAX_ATOMS_PER_BATCH="${MAX_ATOMS_PER_BATCH:-12000}"
MAX_ATOMS_PER_BATCH_VAL="${MAX_ATOMS_PER_BATCH_VAL:-12000}"
MAX_EDGES_PER_BATCH="${MAX_EDGES_PER_BATCH:-2000000}"
MAX_EDGES_PER_BATCH_VAL="${MAX_EDGES_PER_BATCH_VAL:-2000000}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
BATCHING_MODE="${BATCHING_MODE:-random}"
BUCKET_WINDOW_SIZE="${BUCKET_WINDOW_SIZE:-4096}"
BUCKET_SHUFFLE_GROUPS="${BUCKET_SHUFFLE_GROUPS:-8}"
OMOL_RUNTIME_MODE="${OMOL_RUNTIME_MODE:-padded}"

ALLSCAIP_CONFIG_JSON="${ALLSCAIP_CONFIG_JSON:-}"
ALLSCAIP_STRICT_CONFIG_JSON="${ALLSCAIP_STRICT_CONFIG_JSON:-}"
ALLSCAIP_HIDDEN_SIZE="${ALLSCAIP_HIDDEN_SIZE:-128}"
ALLSCAIP_NUM_LAYERS="${ALLSCAIP_NUM_LAYERS:-5}"
ALLSCAIP_ATTEN_NUM_HEADS="${ALLSCAIP_ATTEN_NUM_HEADS:-2}"
ALLSCAIP_MAX_ATOMS="${ALLSCAIP_MAX_ATOMS:-$MAX_ATOMS_PER_BATCH}"
ALLSCAIP_MAX_BATCH_SIZE="${ALLSCAIP_MAX_BATCH_SIZE:-$BATCH_SIZE}"
ALLSCAIP_MAX_RADIUS="${ALLSCAIP_MAX_RADIUS:-6.0}"
ALLSCAIP_KNN_K="${ALLSCAIP_KNN_K:-30}"
ALLSCAIP_KNN_PAD_SIZE="${ALLSCAIP_KNN_PAD_SIZE:-30}"
ALLSCAIP_ATTEN_NAME="${ALLSCAIP_ATTEN_NAME:-memory_efficient}"
ALLSCAIP_FREQUENCY_LIST="${ALLSCAIP_FREQUENCY_LIST:-}"
ALLSCAIP_COMPILE="${ALLSCAIP_COMPILE:-0}"
ALLSCAIP_USE_PADDING="${ALLSCAIP_USE_PADDING:-1}"
ALLSCAIP_USE_CHUNKED_GRAPH="${ALLSCAIP_USE_CHUNKED_GRAPH:-0}"
ALLSCAIP_GRAPH_CHUNK_SIZE="${ALLSCAIP_GRAPH_CHUNK_SIZE:-512}"
ALLSCAIP_PREPROCESS_ON_CPU="${ALLSCAIP_PREPROCESS_ON_CPU:-0}"

MAX_EPOCHS="${MAX_EPOCHS:-20}"
MAX_STEPS="${MAX_STEPS:-0}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
OPTIMIZER="${OPTIMIZER:-adamw}"
MUON_LR="${MUON_LR:-0.02}"
MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0}"
MUON_ADAM_LR="${MUON_ADAM_LR:-}"
MUON_ADAM_WEIGHT_DECAY="${MUON_ADAM_WEIGHT_DECAY:-}"
MUON_ADAM_BETA1="${MUON_ADAM_BETA1:-0.9}"
MUON_ADAM_BETA2="${MUON_ADAM_BETA2:-0.95}"
MUON_ADAM_EPS="${MUON_ADAM_EPS:-1e-10}"
MUON_HIDDEN_ONLY="${MUON_HIDDEN_ONLY:-1}"
MUON_MIN_NDIM="${MUON_MIN_NDIM:-2}"
MUON_EXCLUDE_NAME_FRAGMENTS="${MUON_EXCLUDE_NAME_FRAGMENTS:-embed,embedding,head,readout,rope,freq}"
MUON_PLATONIC_KERNEL_VIEW="${MUON_PLATONIC_KERNEL_VIEW:-slice}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
NORMALIZER_RMSD="${NORMALIZER_RMSD:-1.433569}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-10}"
FORCE_WEIGHT="${FORCE_WEIGHT:-10}"
TRAIN_AUGMENTATION="${TRAIN_AUGMENTATION:-o3}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1}"
SKIP_LOSS_ABOVE="${SKIP_LOSS_ABOVE:-0}"
EMA_DECAY="${EMA_DECAY:-0.999}"
EMA_WARMUP_STEPS="${EMA_WARMUP_STEPS:-5000}"
BF16="${BF16:-0}"
FLOAT32_MATMUL_PRECISION="${FLOAT32_MATMUL_PRECISION:-highest}"
COMPILE="${COMPILE:-0}"
COMPILE_MODE="${COMPILE_MODE:-default}"

LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-10}"
FLOPS_COEF="${FLOPS_COEF:-72.0}"
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

CUDA_PIP_LIB_DIRS="$("$PYTHON_BIN" - <<'PY'
from pathlib import Path
import site

roots = []
for root in site.getsitepackages():
    nvidia_root = Path(root) / "nvidia"
    if nvidia_root.is_dir():
        roots.extend(str(path) for path in sorted(nvidia_root.glob("*/lib")) if path.is_dir())
print(":".join(roots))
PY
)"
if [ -n "$CUDA_PIP_LIB_DIRS" ]; then
  export LD_LIBRARY_PATH="$CUDA_PIP_LIB_DIRS:${LD_LIBRARY_PATH:-}"
fi

if [ ! -d "$TRAIN_DATA_PATH" ]; then
  echo "[error] TRAIN_DATA_PATH does not exist: $TRAIN_DATA_PATH" >&2
  exit 1
fi
if [ ! -d "$VAL_DATA_PATH" ]; then
  echo "[error] VAL_DATA_PATH does not exist: $VAL_DATA_PATH" >&2
  exit 1
fi
if [ "$MODEL_BACKEND" = "matterformer" ] && [ ! -f "$HYBRID_CONFIG_JSON" ]; then
  echo "[error] HYBRID_CONFIG_JSON not found: $HYBRID_CONFIG_JSON" >&2
  exit 1
fi
if [ ! -f "$ELEMENT_REFS_JSON" ]; then
  echo "[error] ELEMENT_REFS_JSON not found: $ELEMENT_REFS_JSON" >&2
  exit 1
fi

REQUIRED_IMPORTS="import torch, wandb, ase, ase_db_backends; import ase.db"
if [ "$MODEL_BACKEND" = "allscaip_direct" ]; then
  REQUIRED_IMPORTS="$REQUIRED_IMPORTS; from fairchem.core.models.allscaip.AllScAIP import AllScAIPBackbone"
fi
if ! "$PYTHON_BIN" -c "$REQUIRED_IMPORTS" >/dev/null 2>&1; then
  echo "[error] PYTHON_BIN cannot import required OMol runtime deps for MODEL_BACKEND=$MODEL_BACKEND:" >&2
  echo "        $PYTHON_BIN" >&2
  echo "        Install the missing package in that environment or set PYTHON_BIN to one that has it." >&2
  [ "$MODEL_BACKEND" = "allscaip_direct" ] && echo "        For AllScAIP, set FAIRCHEM_SRC=/path/to/fairchem if using a source checkout." >&2
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
[ "$COMPILE" = "1" ] && EXTRA_ARGS+=(--compile) || EXTRA_ARGS+=(--no-compile)
[ "$SAVE_CHECKPOINT" = "1" ] && EXTRA_ARGS+=(--save-checkpoint) || EXTRA_ARGS+=(--no-save-checkpoint)
[ "$ALLSCAIP_COMPILE" = "1" ] && EXTRA_ARGS+=(--allscaip-compile) || EXTRA_ARGS+=(--no-allscaip-compile)
[ "$ALLSCAIP_USE_PADDING" = "1" ] && EXTRA_ARGS+=(--allscaip-use-padding) || EXTRA_ARGS+=(--no-allscaip-use-padding)
[ "$ALLSCAIP_USE_CHUNKED_GRAPH" = "1" ] && EXTRA_ARGS+=(--allscaip-use-chunked-graph) || EXTRA_ARGS+=(--no-allscaip-use-chunked-graph)
[ "$ALLSCAIP_PREPROCESS_ON_CPU" = "1" ] && EXTRA_ARGS+=(--allscaip-preprocess-on-cpu) || EXTRA_ARGS+=(--no-allscaip-preprocess-on-cpu)
[ -n "$CHGSPIN_EMB_DIM" ] && EXTRA_ARGS+=(--chgspin-emb-dim "$CHGSPIN_EMB_DIM")
[ -n "$READOUT_ACTIVATION" ] && EXTRA_ARGS+=(--readout-activation "$READOUT_ACTIVATION")
[ -n "$MAX_GRAPHS_PER_BATCH" ] && EXTRA_ARGS+=(--max-graphs-per-batch "$MAX_GRAPHS_PER_BATCH")
[ -n "$MAX_GRAPHS_PER_BATCH_VAL" ] && EXTRA_ARGS+=(--max-graphs-per-batch-val "$MAX_GRAPHS_PER_BATCH_VAL")
[ -n "$ALLSCAIP_CONFIG_JSON" ] && EXTRA_ARGS+=(--allscaip-config-json "$ALLSCAIP_CONFIG_JSON")
[ -n "$ALLSCAIP_STRICT_CONFIG_JSON" ] && EXTRA_ARGS+=(--allscaip-strict-config-json "$ALLSCAIP_STRICT_CONFIG_JSON")
[ -n "$ALLSCAIP_FREQUENCY_LIST" ] && EXTRA_ARGS+=(--allscaip-frequency-list "$ALLSCAIP_FREQUENCY_LIST")
[ "$MUON_HIDDEN_ONLY" = "1" ] && EXTRA_ARGS+=(--muon-hidden-only) || EXTRA_ARGS+=(--no-muon-hidden-only)
[ -n "$MUON_ADAM_LR" ] && EXTRA_ARGS+=(--muon-adam-lr "$MUON_ADAM_LR")
[ -n "$MUON_ADAM_WEIGHT_DECAY" ] && EXTRA_ARGS+=(--muon-adam-weight-decay "$MUON_ADAM_WEIGHT_DECAY")
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
echo "model_backend:           $MODEL_BACKEND"
echo "hybrid_config_json:      $HYBRID_CONFIG_JSON"
echo "d_model:                 $D_MODEL"
echo "n_heads:                 $N_HEADS"
echo "n_layers:                $N_LAYERS"
echo "mlp_ratio:               $MLP_RATIO"
echo "chgspin_mode:            $CHGSPIN_MODE"
echo "chgspin_emb_dim:         ${CHGSPIN_EMB_DIM:-null}"
echo "batch_size:              $BATCH_SIZE"
echo "max_graphs_per_batch:    ${MAX_GRAPHS_PER_BATCH:-null}"
echo "max_graphs_per_batch_val:${MAX_GRAPHS_PER_BATCH_VAL:-null}"
echo "max_atoms_per_batch:     $MAX_ATOMS_PER_BATCH"
echo "max_edges_per_batch:     $MAX_EDGES_PER_BATCH"
echo "batching_mode:           $BATCHING_MODE"
echo "bucket_window_size:      $BUCKET_WINDOW_SIZE"
echo "bucket_shuffle_groups:   $BUCKET_SHUFFLE_GROUPS"
echo "omol_runtime_mode:       $OMOL_RUNTIME_MODE"
echo "max_epochs:              $MAX_EPOCHS"
echo "max_steps:               $MAX_STEPS (0 means epoch-limited)"
echo "lr:                      $LR"
echo "weight_decay:            $WEIGHT_DECAY"
echo "optimizer:               $OPTIMIZER"
echo "muon_lr:                 $MUON_LR"
echo "muon_momentum:           $MUON_MOMENTUM"
echo "muon_weight_decay:       $MUON_WEIGHT_DECAY"
echo "muon_adam_lr:            ${MUON_ADAM_LR:-default_lr}"
echo "muon_adam_weight_decay:  ${MUON_ADAM_WEIGHT_DECAY:-default_weight_decay}"
echo "muon_hidden_only:        $MUON_HIDDEN_ONLY"
echo "muon_min_ndim:           $MUON_MIN_NDIM"
echo "muon_platonic_view:      $MUON_PLATONIC_KERNEL_VIEW"
echo "warmup_steps:            $WARMUP_STEPS"
echo "normalizer_rmsd:         $NORMALIZER_RMSD"
echo "train_augmentation:      $TRAIN_AUGMENTATION"
echo "force_head_mode:        $FORCE_HEAD_MODE"
echo "readout_head_mode:      $READOUT_HEAD_MODE"
echo "readout_activation:     ${READOUT_ACTIVATION:-auto}"
echo "allscaip_config_json:   $ALLSCAIP_CONFIG_JSON"
echo "allscaip_strict_json:   $ALLSCAIP_STRICT_CONFIG_JSON"
echo "allscaip_hidden_size:    $ALLSCAIP_HIDDEN_SIZE"
echo "allscaip_num_layers:     $ALLSCAIP_NUM_LAYERS"
echo "allscaip_atten_heads:    $ALLSCAIP_ATTEN_NUM_HEADS"
echo "allscaip_max_atoms:      $ALLSCAIP_MAX_ATOMS"
echo "allscaip_max_batch_size: $ALLSCAIP_MAX_BATCH_SIZE"
echo "allscaip_knn_pad_size:  $ALLSCAIP_KNN_PAD_SIZE"
echo "allscaip_atten_name:    $ALLSCAIP_ATTEN_NAME"
echo "allscaip_compile:        $ALLSCAIP_COMPILE"
echo "bf16:                    $BF16"
echo "float32_matmul_prec:     $FLOAT32_MATMUL_PRECISION"
echo "compile:                 $COMPILE"
echo "compile_mode:            $COMPILE_MODE"
echo "grad_clip_norm:          $GRAD_CLIP_NORM"
echo "skip_loss_above:         $SKIP_LOSS_ABOVE"
echo "ema_decay:               $EMA_DECAY"
echo "ema_warmup_steps:        $EMA_WARMUP_STEPS"
echo "flops_coef:              $FLOPS_COEF"
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
  --model-backend "$MODEL_BACKEND" \
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
  --force-head-mode "$FORCE_HEAD_MODE" \
  --readout-head-mode "$READOUT_HEAD_MODE" \
  --omol-runtime-mode "$OMOL_RUNTIME_MODE" \
  --allscaip-hidden-size "$ALLSCAIP_HIDDEN_SIZE" \
  --allscaip-num-layers "$ALLSCAIP_NUM_LAYERS" \
  --allscaip-atten-num-heads "$ALLSCAIP_ATTEN_NUM_HEADS" \
  --allscaip-max-atoms "$ALLSCAIP_MAX_ATOMS" \
  --allscaip-max-batch-size "$ALLSCAIP_MAX_BATCH_SIZE" \
  --allscaip-max-radius "$ALLSCAIP_MAX_RADIUS" \
  --allscaip-knn-k "$ALLSCAIP_KNN_K" \
  --allscaip-knn-pad-size "$ALLSCAIP_KNN_PAD_SIZE" \
  --allscaip-atten-name "$ALLSCAIP_ATTEN_NAME" \
  --allscaip-graph-chunk-size "$ALLSCAIP_GRAPH_CHUNK_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --val-batch-size "$VAL_BATCH_SIZE" \
  --max-atoms-per-batch "$MAX_ATOMS_PER_BATCH" \
  --max-atoms-per-batch-val "$MAX_ATOMS_PER_BATCH_VAL" \
  --max-edges-per-batch "$MAX_EDGES_PER_BATCH" \
  --max-edges-per-batch-val "$MAX_EDGES_PER_BATCH_VAL" \
  --batching-mode "$BATCHING_MODE" \
  --bucket-window-size "$BUCKET_WINDOW_SIZE" \
  --bucket-shuffle-groups "$BUCKET_SHUFFLE_GROUPS" \
  --num-workers "$NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR" \
  --max-epochs "$MAX_EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --warmup-steps "$WARMUP_STEPS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --optimizer "$OPTIMIZER" \
  --muon-lr "$MUON_LR" \
  --muon-momentum "$MUON_MOMENTUM" \
  --muon-weight-decay "$MUON_WEIGHT_DECAY" \
  --muon-adam-beta1 "$MUON_ADAM_BETA1" \
  --muon-adam-beta2 "$MUON_ADAM_BETA2" \
  --muon-adam-eps "$MUON_ADAM_EPS" \
  --muon-min-ndim "$MUON_MIN_NDIM" \
  --muon-exclude-name-fragments "$MUON_EXCLUDE_NAME_FRAGMENTS" \
  --muon-platonic-kernel-view "$MUON_PLATONIC_KERNEL_VIEW" \
  --normalizer-rmsd "$NORMALIZER_RMSD" \
  --energy-weight "$ENERGY_WEIGHT" \
  --force-weight "$FORCE_WEIGHT" \
  --train-augmentation "$TRAIN_AUGMENTATION" \
  --grad-clip-norm "$GRAD_CLIP_NORM" \
  --skip-loss-above "$SKIP_LOSS_ABOVE" \
  --ema-decay "$EMA_DECAY" \
  --ema-warmup-steps "$EMA_WARMUP_STEPS" \
  --float32-matmul-precision "$FLOAT32_MATMUL_PRECISION" \
  --compile-mode "$COMPILE_MODE" \
  --log-every-steps "$LOG_EVERY_STEPS" \
  --flops-coef "$FLOPS_COEF" \
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
