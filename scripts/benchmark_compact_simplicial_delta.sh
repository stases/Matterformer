#!/bin/bash
# Compact kNN simplicial Triton parity + benchmark on one Delta GPU.

#SBATCH --partition=delta
#SBATCH --account=deltausers
#SBATCH --gres=gpu:1
#SBATCH --job-name=simp_knn_bench
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --mem=32000M
#SBATCH --output=/home/thadziv/matterformer_jobs/job_outputs/slurm_output_%A_simp_knn_bench.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/thadziv/GitHub/Matterformer}"
PYTHON_BIN="${PYTHON_BIN:-/home/thadziv/GitHub/erwin/erwin/bin/python}"
RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/thadziv/matterformer_runs/compact_simplicial_bench/${RUN_ID}}"

mkdir -p "$OUTPUT_DIR" /home/thadziv/matterformer_jobs/job_outputs
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "============================================================"
echo "Compact kNN simplicial Triton benchmark"
echo "start_time: $(date -Iseconds)"
echo "host: $(hostname)"
echo "repo_root: $REPO_ROOT"
echo "python_bin: $PYTHON_BIN"
echo "output_dir: $OUTPUT_DIR"
echo "============================================================"

nvidia-smi || true
"$PYTHON_BIN" - <<'PY'
import torch
import triton
from matterformer.models.triton_compact_simplicial_attention import TRITON_COMPACT_SIMPLICIAL_AVAILABLE
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "cuda", torch.version.cuda)
print("triton", triton.__version__, "compact_available", TRITON_COMPACT_SIMPLICIAL_AVAILABLE)
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

echo "running CUDA parity smoke test"
"$PYTHON_BIN" - <<'PY'
import torch
from matterformer.models import CompactSimplicialBias, compact_simplicial_attention_torch, compact_simplicial_attention_triton

torch.manual_seed(123)
device = torch.device("cuda")
B, H, N, K, D, R = 2, 3, 9, 8, 16, 16
kwargs = {"device": device, "dtype": torch.float32, "requires_grad": True}
base = [torch.randn(B, H, N, D, **kwargs) for _ in range(5)]
neighbor_idx = ((torch.arange(N, device=device)[:, None] + torch.arange(K, device=device)[None, :]) % N)
neighbor_idx = neighbor_idx.expand(B, -1, -1).contiguous()
neighbor_mask = (torch.rand(B, N, K, device=device) > 0.15).contiguous()
neighbor_mask[..., 0] = True
neighbor_mask[0, 0, :] = False
bias = CompactSimplicialBias(
    u=torch.randn(B, H, N, K, **kwargs),
    v=torch.randn(B, H, N, K, **kwargs),
    gate=torch.randn(B, H, N, **kwargs),
    angle_left=torch.randn(B, H, N, K, R, **kwargs),
    angle_right=torch.randn(B, H, N, K, R, **kwargs),
    angle_gate=torch.randn(B, H, N, **kwargs),
    message_left=torch.randn(B, H, N, K, R, **kwargs),
    message_right=torch.randn(B, H, N, K, R, **kwargs),
    message_basis=torch.randn(H, R, D, **kwargs),
)

def clone_tensors(xs):
    return [x.detach().clone().requires_grad_(True) for x in xs]

def clone_bias(b):
    def c(x):
        return x.detach().clone().requires_grad_(True) if x is not None else None
    return CompactSimplicialBias(
        u=c(b.u), v=c(b.v), gate=c(b.gate),
        angle_left=c(b.angle_left), angle_right=c(b.angle_right), angle_gate=c(b.angle_gate),
        message_left=c(b.message_left), message_right=c(b.message_right), message_basis=c(b.message_basis),
    )

def bias_tensors(b):
    return [x for x in (b.u, b.v, b.gate, b.angle_left, b.angle_right, b.angle_gate, b.message_left, b.message_right, b.message_basis) if x is not None]

ref_tensors = clone_tensors(base)
tri_tensors = clone_tensors(base)
ref_bias = clone_bias(bias)
tri_bias = clone_bias(bias)
ref = compact_simplicial_attention_torch(*ref_tensors, neighbor_idx=neighbor_idx, neighbor_mask=neighbor_mask, bias=ref_bias)
tri = compact_simplicial_attention_triton(*tri_tensors, neighbor_idx=neighbor_idx, neighbor_mask=neighbor_mask, bias=tri_bias, precision="ieee_fp32", strict=True)
grad = torch.randn_like(ref)
ref.backward(grad)
tri.backward(grad)
torch.cuda.synchronize()
max_abs = (ref.float() - tri.float()).abs().max().item()
grad_pairs = [(a.grad, b.grad) for a, b in zip(tri_tensors, ref_tensors)]
grad_pairs += [(a.grad, b.grad) for a, b in zip(bias_tensors(tri_bias), bias_tensors(ref_bias))]
max_grad = max((a.float() - b.float()).abs().max().item() for a, b in grad_pairs if a is not None and b is not None)
print(f"parity max_abs={max_abs:.6g} max_grad_abs={max_grad:.6g}")
assert max_abs < 5e-4, max_abs
assert max_grad < 1e-3, max_grad
PY

echo "benchmark: fp32/tf32, radial+angle bias, message off"
"$PYTHON_BIN" scripts/benchmark_compact_simplicial_attention.py \
  --name qm9_fp32_radial_angle \
  --batch-size 64 \
  --tokens 32 \
  --heads 12 \
  --head-dim 64 \
  --neighbors 32 \
  --angle-rank 32 \
  --dtype fp32 \
  --precision tf32 \
  --gate-mode random \
  --warmup 8 \
  --iters 30 \
  --json-out "$OUTPUT_DIR/qm9_fp32_radial_angle.json"

echo "benchmark: bf16/tensorcore, radial+angle bias, message off"
"$PYTHON_BIN" scripts/benchmark_compact_simplicial_attention.py \
  --name qm9_bf16_radial_angle \
  --batch-size 64 \
  --tokens 32 \
  --heads 12 \
  --head-dim 64 \
  --neighbors 32 \
  --angle-rank 32 \
  --dtype bf16 \
  --precision bf16_tc \
  --gate-mode random \
  --warmup 8 \
  --iters 30 \
  --json-out "$OUTPUT_DIR/qm9_bf16_radial_angle.json"

echo "benchmark: bf16/tensorcore, radial+angle bias, value-side message on"
"$PYTHON_BIN" scripts/benchmark_compact_simplicial_attention.py \
  --name qm9_bf16_radial_angle_message \
  --batch-size 64 \
  --tokens 32 \
  --heads 12 \
  --head-dim 64 \
  --neighbors 32 \
  --angle-rank 32 \
  --message \
  --message-rank 16 \
  --dtype bf16 \
  --precision bf16_tc \
  --gate-mode random \
  --warmup 8 \
  --iters 30 \
  --json-out "$OUTPUT_DIR/qm9_bf16_radial_angle_message.json"

echo "end_time: $(date -Iseconds)"
echo "json_outputs:"
ls -lh "$OUTPUT_DIR"
