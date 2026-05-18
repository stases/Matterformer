#!/bin/bash
# Profile h1536 OMol flash baseline vs alternating global/radius-sparse eSEN-local architecture.

#SBATCH --partition=all6000
#SBATCH --account=all6000users
#SBATCH --gres=gpu:1
#SBATCH --job-name=mf_h1536_prof
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_mf_h1536_prof.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
PYTHON_BIN="${PYTHON_BIN:-/home/thadziv/GitHub/erwin/erwin/bin/python}"
SLURM_RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/thadziv/matterformer_runs/benchmarks/omol_h1536_flash_vs_rsparse_${SLURM_RUN_ID}}"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR" /home/thadziv/matterformer_jobs/job_outputs

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MATTERFORMER_PLATONIC_TRITON_DELTA="${MATTERFORMER_PLATONIC_TRITON_DELTA:-triton_vector}"
export MATTERFORMER_PLATONIC_TRITON_SPLIT_HEAD_DIM="${MATTERFORMER_PLATONIC_TRITON_SPLIT_HEAD_DIM:-auto}"
export MATTERFORMER_PLATONIC_RADIUS_TRITON_BWD_NUM_STAGES="${MATTERFORMER_PLATONIC_RADIUS_TRITON_BWD_NUM_STAGES:-2}"

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

echo "============================================================"
echo "Matterformer h1536 flash vs radius-sparse OMol profile"
echo "start_time:       $(date -Iseconds)"
echo "host:             $(hostname)"
echo "repo_root:        $REPO_ROOT"
echo "python_bin:       $PYTHON_BIN"
echo "output_dir:       $OUTPUT_DIR"
echo "git_commit:       $(git rev-parse HEAD)"
echo "============================================================"
nvidia-smi || true

"$PYTHON_BIN" scripts/profile_omol_architectures.py \
  --baseline-config "$REPO_ROOT/configs/omol/tetra_t_only_h1536_l16_pt2_goodrun_qkn_uk_csfilm.json" \
  --sparse-config "$REPO_ROOT/configs/omol/tetra_t_only_h1536_l16_pt2_goodrun_qkn_uk_csfilm_radius_sparse_r4_every2.json" \
  --train-data-path /home/ebekker/data/omol/open_mol/train_4M \
  --val-data-path /home/ebekker/data/omol/open_mol/val \
  --element-refs-json "$REPO_ROOT/configs/omol/element_refs.json" \
  --d-model 1536 \
  --n-heads 12 \
  --n-layers 16 \
  --mlp-ratio 4.0 \
  --chgspin-mode add \
  --chgspin-emb-dim 128 \
  --readout-activation gelu \
  --batch-size 16 \
  --max-graphs-per-batch 999999 \
  --max-atoms-per-batch 12000 \
  --max-edges-per-batch 2400000 \
  --num-workers 4 \
  --prefetch-factor 2 \
  --batching-mode random \
  --bucket-window-size 4096 \
  --bucket-shuffle-groups 8 \
  --energy-weight 10 \
  --force-weight 20 \
  --energy-loss per_atom_mae \
  --force-loss l2norm \
  --train-augmentation o3 \
  --float32-matmul-precision high \
  --compile-mode default \
  --full-warmup "${FULL_WARMUP:-1}" \
  --full-repeats "${FULL_REPEATS:-3}" \
  --layer-warmup "${LAYER_WARMUP:-1}" \
  --layer-repeats "${LAYER_REPEATS:-2}" \
  --output-dir "$OUTPUT_DIR/full_model"

"$PYTHON_BIN" scripts/benchmark_omol_radius_sparse_vs_flash.py \
  --data-path /home/ebekker/data/omol/open_mol/train_4M \
  --max-batch-size 999999 \
  --max-atoms 12000 \
  --max-edges 2400000 \
  --batching-mode random \
  --bucket-window-size 4096 \
  --bucket-shuffle-groups 8 \
  --radii 3.0 4.0 5.0 6.0 \
  --radius 4.0 \
  --heads 12 \
  --head-dim 128 \
  --heads-per-frame 1 \
  --num-rbf 4 \
  --block-m 16 \
  --block-n 32 \
  --warmup "${ATTN_WARMUP:-3}" \
  --repeat "${ATTN_REPEAT:-5}" \
  --precision tf32x3 \
  --matmul-precision high \
  --output-json "$OUTPUT_DIR/attention_layer.json"

echo "full_model_json:     $OUTPUT_DIR/full_model/profile_results.json"
echo "layer_forward_csv:   $OUTPUT_DIR/full_model/layer_forward.csv"
echo "attention_layer_json:$OUTPUT_DIR/attention_layer.json"
