# Matterformer Hybrid Triton Implementation Request

## Question

How should we implement efficient Triton kernels for the Matterformer hybrid
stack, especially the new compact kNN simplicial local layer, while preserving
the current Python/PyTorch API and numerical behavior?

Please review the included source files and propose a concrete implementation
plan with kernel boundaries, tensor layouts, forward/backward formulas, masking
rules, precision choices, tests, and benchmark targets. If any layer should not
receive custom Triton because PyTorch SDPA/FlashAttention is already the right
backend, say so explicitly.

## Current Hybrid Layer Types

The hybrid stack has three layer types:

1. `S`: local compact kNN simplicial attention.
2. `I`: scalar global invariant/trivial MHA.
3. `T`: tetrahedral Platonic global attention.

Only the older dense simplicial attention path currently has real custom Triton
kernels. The new hybrid layer types are not custom Triton yet.

## Current Triton Status

### 1. `S` local compact kNN simplicial layer

Main files:

- `src/matterformer/models/hybrid.py`
- `tests/test_hybrid.py`

Current status:

- `SimplicialLocalLayer` calls `CompactSimplicialAttention`.
- The config may say `"backend": "triton_knn"`.
- However, `compact_simplicial_attention_triton(...)` currently delegates to
  `compact_simplicial_attention_torch(...)`.
- The kNN graph is built with Torch `topk`, Torch gather, and Torch RBF.

This is the most important missing Triton work.

The compact local attention computes over neighbor pairs:

```text
q:              [B, H, N, D]
k1, v1, k2, v2: [B, H, N, D]
neighbor_idx:   [B, N, K]
neighbor_mask:  [B, N, K]
```

Per query atom `i`, head `h`, and neighbor pair `(j, k)`:

```text
score[i,j,k] =
    dot(q_i, k1_j * k2_k)
  + gate_i * (u_ij + v_ik)
  + angle_gate_i * dot(angle_left_ij, angle_right_ik) / sqrt(R)
```

Then:

```text
p[i,j,k] = softmax_{j,k}(score[i,j,k])
out_i = sum_{j,k} p[i,j,k] * (v1_j * v2_k)
```

Optional value-side message residual:

```text
coeff_i,r = sum_{j,k} p[i,j,k] * message_left_ij,r * message_right_ik,r / sqrt(Rm)
out_i += sum_r coeff_i,r * message_basis[h,r,:]
```

What needs implementation:

- A real Triton forward kernel for compact kNN simplicial attention.
- A real Triton backward kernel or a recommended autograd decomposition.
- Support for masks and padded neighbor entries.
- Support for geometry bias tensors:
  - `u`, `v`, `gate`
  - `angle_left`, `angle_right`, `angle_gate`
  - optional `message_left`, `message_right`, `message_basis`
- Good behavior for QM9 default shapes:
  - `B = 64` or `96`
  - `N <= ~30`
  - `K = 32`
  - `H = 12`
  - `D = 64`
  - angle rank `R = 32`
- A path that also scales to larger GEOM/MOF systems where `N` is much larger.
- Precision modes comparable to existing dense Triton:
  - `bf16_tc`
  - `tf32`
  - `ieee_fp32`
- Tests comparing forward/backward to `compact_simplicial_attention_torch`.
- Benchmarks against the current Torch compact reference.

Open design questions:

- Should kNN construction remain Torch `topk` initially, or should we also
  implement a Triton kNN/RBF/gather kernel?
- Should the first Triton target assume `dropout=0`, as our current QM9 runs do?
- Is it better to fuse geometry-bias evaluation into attention, or keep the
  current MLP-produced bias tensors as kernel inputs?
- Should the kernel tile over `K x K`, over query atoms, or over both query and
  neighbor-pair tiles?

### 2. `I` scalar global invariant/trivial layer

Main files:

- `src/matterformer/models/hybrid.py`
- `src/matterformer/models/regular_attention.py`
- `src/matterformer/models/transformer.py`

Current status:

- `TrivialGlobalLayer` wraps `AdaLNBlock(attn_type="mha")`.
- Without MHA RoPE, it uses `torch.nn.MultiheadAttention`.
- With MHA RoPE, it uses `RotaryMultiheadAttention`, which calls
  `torch.nn.functional.scaled_dot_product_attention`.
- Geometry bias is built in Torch by `GeometryBiasBuilder`.
- `src/matterformer/models/triton_regular_attention.py` only contains
  `TRITON_REGULAR_ATTENTION_AVAILABLE = False`.

This layer is not custom Triton. It may already use PyTorch FlashAttention or
memory-efficient SDPA internally on CUDA.

What needs implementation or a decision:

- Decide whether custom Triton is worth it for `I`.
- If yes, propose a fused global attention kernel that supports:
  - MHA RoPE positions
  - optional constant keys (`mha_rope_use_key=False`)
  - RoPE on values and inverse-rotation
  - additive per-head geometry bias `[B,H,T,T]`
  - key padding masks
- If no, recommend keeping PyTorch SDPA and only optimizing bias/RoPE
  preparation if profiling shows it matters.

### 3. `T` tetra Platonic global layer

Main files:

- `src/matterformer/models/hybrid.py`
- `src/matterformer/models/platonic/layers.py`
- `src/matterformer/models/platonic/linear.py`
- `src/matterformer/models/platonic/rope.py`
- `src/matterformer/models/platonic/groups.py`

Current status:

- `TetraPlatonicGlobalLayer` wraps `PlatonicBlock`.
- `PlatonicLinear` constructs a dense group-convolution-tied weight with Torch
  indexing, then calls `F.linear`.
- `PlatonicRoPE` uses Torch `einsum`, `cos`, `sin`, and reshape operations.
- `PlatonicAttention` calls `torch.nn.functional.scaled_dot_product_attention`.

This layer is not custom Triton. Attention may use PyTorch SDPA/FlashAttention
internally, but the group linears and group RoPE are standard Torch ops.

What needs implementation or a decision:

- Decide whether `T` should get custom Triton now or after `S`.
- Potential kernels:
  - group-constrained linear without materializing the expanded dense weight
  - fused Platonic RoPE over group frames
  - possibly fused QKV projection plus group RoPE
- Decide whether attention itself should remain PyTorch SDPA.

## Existing Real Triton Reference

Main files:

- `src/matterformer/models/simplicial_attention.py`
- `src/matterformer/models/triton_simplicial_attention.py`
- `tests/test_simplicial_attention.py`
- `scripts/benchmark_simplicial_attention.py`
- `scripts/validate_simplicial_attention.py`
- `scripts/diagnose_simplicial_triton_parity.py`

This is the older dense all-token simplicial attention path, not the hybrid
compact kNN `S` layer. It has useful patterns for:

- Triton availability checks.
- Precision-mode handling.
- Autograd wrapper structure.
- Optional low-rank geometry residuals.
- CUDA-only parity tests.
- Benchmark reporting.

Please use it as a design reference, but do not assume it already covers compact
kNN attention.

## Recommended Priority

1. Implement real Triton for hybrid compact kNN `S`.
2. Keep `I` on PyTorch SDPA unless profiling shows RoPE/bias overhead dominates.
3. Keep `T` on PyTorch SDPA initially; consider Triton group linears/RoPE later.
4. Add CUDA parity tests and benchmark scripts for compact kNN `S`.
5. Only after `S` is stable, consider fused kernels for `I` and `T`.

## Constraints

- Preserve the public caller API where possible:
  `compact_simplicial_attention_triton(...)` should become the real backend.
- Keep the Torch reference path as the source of truth.
- Avoid requiring Triton for CPU tests.
- Keep behavior deterministic enough for parity tolerances.
- Current QM9 experiments typically use dropout `0.0`; first Triton version may
  reject training dropout if that simplifies implementation.

## Deliverable Requested From The LLM

Please return:

1. A precise implementation plan.
2. Pseudocode for the compact kNN Triton forward and backward.
3. Recommended tensor layouts and tiling choices.
4. A list of edge cases and unsupported modes for v1.
5. Test additions and benchmark commands.
6. Any source-level API changes you recommend before implementation.
