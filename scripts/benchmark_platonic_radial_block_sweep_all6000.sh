#!/bin/bash
# Benchmark the h1920 radial-r2 Platonic attention block-size sweep on one all6000 GPU.

#SBATCH --partition=all6000
#SBATCH --account=all6000users
#SBATCH --gres=gpu:1
#SBATCH --job-name=mf_r2_blk_bench
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --mem=64G
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_mf_r2_blk_bench.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
PYTHON_BIN="${PYTHON_BIN:-/home/thadziv/GitHub/erwin/erwin/bin/python}"
SLURM_RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/thadziv/matterformer_runs/benchmarks/plato_radial_r2_block_sweep_${SLURM_RUN_ID}}"

BLOCK_SIZES="${BLOCK_SIZES:-16x32,16x16,8x32,8x16,4x16}"
BENCH_WARMUP="${BENCH_WARMUP:-3}"
BENCH_REPEAT="${BENCH_REPEAT:-10}"
BENCH_TOKENS="${BENCH_TOKENS:-12000}"
BENCH_SEGMENTS="${BENCH_SEGMENTS:-218}"
BENCH_MAX_LEN="${BENCH_MAX_LEN:-283}"
BENCH_HEADS="${BENCH_HEADS:-12}"
BENCH_HEAD_DIM="${BENCH_HEAD_DIM:-160}"
BENCH_HEADS_PER_FRAME="${BENCH_HEADS_PER_FRAME:-1}"
BENCH_PRECISION="${BENCH_PRECISION:-tf32x3}"
BENCH_BWD_MODE="${BENCH_BWD_MODE:-auto}"
BENCH_DELTA_MODE="${BENCH_DELTA_MODE:-triton_vector}"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR" /home/thadziv/matterformer_jobs/job_outputs

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

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
echo "Matterformer radial-r2 Platonic block-size benchmark"
echo "start_time:       $(date -Iseconds)"
echo "host:             $(hostname)"
echo "repo_root:        $REPO_ROOT"
echo "python_bin:       $PYTHON_BIN"
echo "output_dir:       $OUTPUT_DIR"
echo "block_sizes:      $BLOCK_SIZES"
echo "warmup/repeat:    $BENCH_WARMUP/$BENCH_REPEAT"
echo "tokens/segments:  $BENCH_TOKENS/$BENCH_SEGMENTS"
echo "max_len:          $BENCH_MAX_LEN"
echo "heads/head_dim:   $BENCH_HEADS/$BENCH_HEAD_DIM"
echo "heads_per_frame:  $BENCH_HEADS_PER_FRAME"
echo "precision:        $BENCH_PRECISION"
echo "bwd/delta mode:   $BENCH_BWD_MODE/$BENCH_DELTA_MODE"
echo "============================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

"$PYTHON_BIN" scripts/benchmark_platonic_radial_block_sweep.py \
  --block-sizes "$BLOCK_SIZES" \
  --warmup "$BENCH_WARMUP" \
  --repeat "$BENCH_REPEAT" \
  --tokens "$BENCH_TOKENS" \
  --segments "$BENCH_SEGMENTS" \
  --max-len "$BENCH_MAX_LEN" \
  --heads "$BENCH_HEADS" \
  --head-dim "$BENCH_HEAD_DIM" \
  --heads-per-frame "$BENCH_HEADS_PER_FRAME" \
  --precision "$BENCH_PRECISION" \
  --bwd-mode "$BENCH_BWD_MODE" \
  --delta-mode "$BENCH_DELTA_MODE" \
  --output-json "$OUTPUT_DIR/results.json" \
  --output-csv "$OUTPUT_DIR/results.csv"

echo "results_json: $OUTPUT_DIR/results.json"
echo "results_csv:  $OUTPUT_DIR/results.csv"
