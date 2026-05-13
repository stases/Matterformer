#!/bin/bash
# Short detailed profiler for the submitted OMol scalar I+S direct-head run.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --job-name=prof_IS_dir_d768
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_profile_omol_IS_dir_d768.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
PYTHON_BIN="${PYTHON_BIN:-/home/thadziv/GitHub/erwin/erwin/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/thadziv/matterformer_profiles}"
PROFILE_NAME="${PROFILE_NAME:-omol_IS_triton_direct_d768_l19_${SLURM_JOB_ID:-manual}}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$PROFILE_NAME}"

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/home/ebekker/data/omol/open_mol/train_4M}"
VAL_DATA_PATH="${VAL_DATA_PATH:-/home/ebekker/data/omol/open_mol/val}"

mkdir -p "$OUTPUT_DIR" /home/thadziv/matterformer_jobs/job_outputs
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export HYBRID_CONFIG_JSON="${HYBRID_CONFIG_JSON:-$REPO_ROOT/configs/omol/scalar_is_triton_d768_l19.json}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[error] PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c "import torch, ase, ase_db_backends; import ase.db" >/dev/null 2>&1; then
  echo "[error] PYTHON_BIN cannot import OMol runtime deps: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/profile_omol_forcefield_detail.py \
  --train-data-path "$TRAIN_DATA_PATH" \
  --val-data-path "$VAL_DATA_PATH" \
  --validation-mode heldout \
  --hybrid-config-json "$HYBRID_CONFIG_JSON" \
  --element-refs-json "$REPO_ROOT/configs/omol/element_refs.json" \
  --d-model "${D_MODEL:-768}" \
  --n-heads "${N_HEADS:-12}" \
  --n-layers "${N_LAYERS:-19}" \
  --mlp-ratio "${MLP_RATIO:-4.0}" \
  --force-head-mode "${FORCE_HEAD_MODE:-direct}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --max-atoms-per-batch "${MAX_ATOMS_PER_BATCH:-2000}" \
  --max-edges-per-batch "${MAX_EDGES_PER_BATCH:-100000}" \
  --num-workers "${NUM_WORKERS:-16}" \
  --prefetch-factor "${PREFETCH_FACTOR:-4}" \
  --lr "${LR:-3e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-5}" \
  --normalizer-rmsd "${NORMALIZER_RMSD:-1.433569}" \
  --energy-weight "${ENERGY_WEIGHT:-10}" \
  --force-weight "${FORCE_WEIGHT:-10}" \
  --train-augmentation "${TRAIN_AUGMENTATION:-o3}" \
  --float32-matmul-precision "${FLOAT32_MATMUL_PRECISION:-highest}" \
  --warmup-steps "${PROFILE_WARMUP_STEPS:-3}" \
  --profile-steps "${PROFILE_STEPS:-4}" \
  --row-limit "${PROFILE_ROW_LIMIT:-120}" \
  --output-dir "$OUTPUT_DIR" \
  --export-chrome-trace
