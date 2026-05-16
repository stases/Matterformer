#!/bin/bash
# Platonic RBF/type local attention parity + benchmark on one Delta GPU.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --job-name=rbf_type_bench
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --mem=64000M
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_rbf_type_bench.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
PYTHON_BIN="${PYTHON_BIN:-/home/thadziv/GitHub/erwin/erwin/bin/python}"
RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/oracle_context/rbf_type_bias_delta_${RUN_ID}}"

mkdir -p "$OUTPUT_DIR" /home/thadziv/matterformer_jobs/job_outputs
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "============================================================"
echo "Platonic RBF/type local attention benchmark"
echo "start_time: $(date -Iseconds)"
echo "host: $(hostname)"
echo "repo_root: $REPO_ROOT"
echo "python_bin: $PYTHON_BIN"
echo "output_dir: $OUTPUT_DIR"
echo "============================================================"

git rev-parse HEAD
git status --short
nvidia-smi || true

"$PYTHON_BIN" - <<'PY'
import torch
import triton
from matterformer.models.platonic.triton_attention import TRITON_PLATONIC_ATTENTION_AVAILABLE
from matterformer.models.platonic.layers import flash_attn_varlen_func
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "cuda", torch.version.cuda)
print("triton", triton.__version__, "platonic_triton_available", TRITON_PLATONIC_ATTENTION_AVAILABLE)
print("flash_attn_available", flash_attn_varlen_func is not None)
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

"$PYTHON_BIN" scripts/benchmark_platonic_rbf_type_bias.py \
  --hybrid-config-json "${HYBRID_CONFIG_JSON:-configs/omol/tetra_t_only_h1920_l16_pt2_exact_sin_layerscale.json}" \
  --d-model "${D_MODEL:-1920}" \
  --n-heads "${N_HEADS:-60}" \
  --n-layers "${N_LAYERS:-16}" \
  --mlp-ratio "${MLP_RATIO:-2.0}" \
  --total-atoms "${TOTAL_ATOMS:-12000}" \
  --num-graphs "${NUM_GRAPHS:-218}" \
  --max-atoms-per-graph "${MAX_ATOMS_PER_GRAPH:-220}" \
  --attn-tokens "${ATTN_TOKENS:-4096}" \
  --attn-graphs "${ATTN_GRAPHS:-64}" \
  --warmup "${WARMUP:-2}" \
  --repeats "${REPEATS:-3}" \
  --compile-scope ${COMPILE_SCOPE:-none trunk_flat} \
  --full-variants ${FULL_VARIANTS:-flash_fast triton_plain local_rbf_type_every4 local_rbf_type_all} \
  --matmul-precision "${MATMUL_PRECISION:-high}" \
  --output-json "$OUTPUT_DIR/results.json"

echo "finished_time: $(date -Iseconds)"
echo "results_json: $OUTPUT_DIR/results.json"
echo "results_md: $OUTPUT_DIR/results.md"
