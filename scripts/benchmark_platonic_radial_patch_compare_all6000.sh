#!/bin/bash
# Compare current and proposed patched Platonic radial-r2 Triton attention kernels.

#SBATCH --partition=all6000
#SBATCH --account=all6000users
#SBATCH --gres=gpu:1
#SBATCH --job-name=mf_r2_patch_cmp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --mem=64G
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_mf_r2_patch_cmp.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
PYTHON_BIN="${PYTHON_BIN:-/home/thadziv/GitHub/erwin/erwin/bin/python}"
SLURM_RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/thadziv/matterformer_runs/benchmarks/plato_radial_patch_compare_${SLURM_RUN_ID}}"

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
echo "Matterformer radial-r2 patched/current Triton comparison"
echo "start_time:       $(date -Iseconds)"
echo "host:             $(hostname)"
echo "repo_root:        $REPO_ROOT"
echo "python_bin:       $PYTHON_BIN"
echo "output_dir:       $OUTPUT_DIR"
echo "patched_file:     ${PATCHED_TRITON_FILE:-$REPO_ROOT/patches/triton_attention_patched.py}"
echo "============================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

"$PYTHON_BIN" scripts/benchmark_platonic_radial_patch_compare.py \
  --patched-file "${PATCHED_TRITON_FILE:-$REPO_ROOT/patches/triton_attention_patched.py}" \
  --output-json "$OUTPUT_DIR/results.json" \
  --output-csv "$OUTPUT_DIR/results.csv"

echo "results_json: $OUTPUT_DIR/results.json"
echo "results_csv:  $OUTPUT_DIR/results.csv"
