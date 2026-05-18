#!/bin/bash
# Benchmark radius-local Platonic attention references and available Triton paths on one Delta GPU.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --job-name=radius_sparse_bench
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_radius_sparse_bench.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
PYTHON_BIN="${PYTHON_BIN:-/home/thadziv/GitHub/erwin/erwin/bin/python}"
RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/thadziv/matterformer_runs/benchmarks/radius_sparse_attention_delta_${RUN_ID}}"

mkdir -p "$OUTPUT_DIR" /home/thadziv/matterformer_jobs/job_outputs
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
echo "Matterformer radius-sparse Platonic attention benchmark"
echo "start_time:      $(date -Iseconds)"
echo "host:            $(hostname)"
echo "repo_root:       $REPO_ROOT"
echo "python_bin:      $PYTHON_BIN"
echo "output_dir:      $OUTPUT_DIR"
echo "tokens/segments: ${BENCH_TOKENS:-4096}/${BENCH_SEGMENTS:-64}"
echo "heads/head_dim:  ${BENCH_HEADS:-60}/${BENCH_HEAD_DIM:-32}"
echo "cutoff/spread:   ${BENCH_CUTOFF:-1.5}/${BENCH_POSITION_SPREAD:-12.0}"
echo "warmup/repeat:   ${BENCH_WARMUP:-3}/${BENCH_REPEAT:-5}"
echo "============================================================"

git rev-parse HEAD
git status --short
nvidia-smi || true

if command -v research-run-start >/dev/null 2>&1; then
  research-run-start || true
fi
trap 'if command -v research-run-finish >/dev/null 2>&1; then research-run-finish $? || true; fi' EXIT

"$PYTHON_BIN" - <<'PY'
import torch
try:
    import triton
    triton_version = triton.__version__
except Exception as exc:
    triton_version = f"unavailable: {exc}"
from matterformer.models.platonic.triton_attention import TRITON_PLATONIC_ATTENTION_AVAILABLE

print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "cuda", torch.version.cuda)
print("triton", triton_version, "platonic_triton_available", TRITON_PLATONIC_ATTENTION_AVAILABLE)
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

"$PYTHON_BIN" scripts/benchmark_radius_sparse_attention.py \
  --device cuda \
  --tokens "${BENCH_TOKENS:-4096}" \
  --segments "${BENCH_SEGMENTS:-64}" \
  --heads "${BENCH_HEADS:-60}" \
  --heads-per-frame "${BENCH_HEADS_PER_FRAME:-5}" \
  --head-dim "${BENCH_HEAD_DIM:-32}" \
  --cutoff "${BENCH_CUTOFF:-1.5}" \
  --position-spread "${BENCH_POSITION_SPREAD:-12.0}" \
  --block-m "${BENCH_BLOCK_M:-16}" \
  --block-n "${BENCH_BLOCK_N:-32}" \
  --warmup "${BENCH_WARMUP:-3}" \
  --repeat "${BENCH_REPEAT:-5}" \
  --precision "${BENCH_PRECISION:-tf32x3}" \
  --matmul-precision "${BENCH_MATMUL_PRECISION:-high}" \
  --output-json "$OUTPUT_DIR/results.json"

echo "finished_time: $(date -Iseconds)"
echo "results_json:  $OUTPUT_DIR/results.json"
