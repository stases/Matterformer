# Platonic Fourier and Mixed Backend Design Notes

## Purpose

This note explains the current Matterformer Platonic backend split for the tetrahedral OMol model. It is meant to disambiguate three separate pieces of work that are easy to conflate:

- finite-group Fourier linears for `PlatonicLinear`
- exact attention plumbing fastpaths for RoPE and constant keys
- Triton attention kernels and radial geometric attention bias

The current best speed-oriented configuration is a mixed backend: FlashAttention for attention, spatial Platonic linears for attention projections, and direct Fourier Platonic linears for FFN up/down projections.

## Key Distinction: Two Backend Axes

There are two independent backend concepts.

`attention_backend` controls the attention kernel after q/k/v have been produced:

- `flash`: use FlashAttention when available.
- `sdpa`: use PyTorch scaled dot-product attention.
- `triton`: use the custom Platonic Triton attention path.

Geometric attention bias is selected independently with `attention_bias.kind`, for example
`radial_rbf`, `radial_r2`, `radial_slope`, or `rbf_type_enveloped`.

`linear_backend`, `attention_linear_backend`, and `ffn_linear_backend` control how Platonic group-convolution linear maps are evaluated:

- `spatial`: original dense flattened group-convolution implementation.
- `fourier`: correctness scaffold. Keeps `.kernel [G, Cout, Cin]` and `.bias [Cout]`, converts the spatial kernel into Fourier blocks each forward.
- `fourier_direct`: production-oriented Fourier parameterization with direct Fourier block parameters.

These axes compose. For example, a block can use FlashAttention for attention while using Fourier-direct linears only in the FFN.

## Current Best Mixed Backend

The measured mixed configuration is:

```json
{
  "tetra": {
    "attention_backend": "flash",
    "linear_backend": "spatial",
    "attention_linear_backend": "spatial",
    "ffn_linear_backend": "fourier_direct",
    "rope_cache": true,
    "constant_key_fastpath": true,
    "fused_qv": true
  }
}
```

This means:

- q/v/out attention projections remain spatial `PlatonicLinear`.
- the attention operation remains FlashAttention.
- FFN up/down projections use `fourier_direct`.
- RoPE cos/sin are computed once per attention call and reused.
- constant keys are constructed directly from cached RoPE factors.
- q and v are projected by one fused `PlatonicLinear(d_model, 2 * d_model)` when `use_key=false`.

The runnable config is:

`configs/omol/tetra_t_only_h1920_l16_pt2_exact_sin_layerscale_mixedffn.json`

## Fourier Linears

### Spatial Backend

The original `PlatonicLinear` stores a group-convolution kernel:

```text
kernel: [G, Cout, Cin]
bias:   [Cout]
```

For the tetrahedral group, `G = 12`. The spatial convention is:

```text
y_g = sum_h K[h^{-1} g] x_h
```

This is implemented by expanding the group-convolution kernel into a dense flattened weight and calling `F.linear`.

### Fourier Scaffold Backend

`linear_backend="fourier"` is a correctness scaffold. It keeps the same `.kernel` and `.bias` state dict as the spatial backend, computes the tetra Fourier block weights inside forward, applies the block-diagonal Fourier linear map, then transforms back to spatial group coordinates.

The tetra regular representation decomposes as:

```text
regular(A4) = 1D + 2D + 3 * 3D
```

The implementation uses a real orthonormal basis `Q [12, 12]` and real blocks:

- low block: merged padded 1D + 2D block, shape `3Cout x 3Cin`
- 3D block: shape `3Cout x 3Cin`, applied across the three 3D irrep-coordinate batches

The convention-sensitive 3D formula is:

```python
w3[n, q, o, i] = sum_g group.elements[g, q, n] * kernel[g, o, i]
```

The scaffold is useful for parity and checkpoint compatibility, but it rebuilds Fourier weights every forward and is not the performance target.

### Fourier Direct Backend

`linear_backend="fourier_direct"` uses direct Fourier parameters:

```text
w1:    [Cout, Cin]
w2_re: [Cout, Cin]
w2_im: [Cout, Cin]
w3:    [3 * Cout, 3 * Cin]
bias:  [Cout]
```

This preserves the same parameter count as the spatial kernel:

```text
1*Cout*Cin + 2*Cout*Cin + 9*Cout*Cin = 12*Cout*Cin
```

It avoids per-forward spatial-kernel conversion and is the backend used for mixed FFNs.

## Why Fourier Is Applied Only To FFNs For Now

Microbenchmarks showed that exact Fourier linears do not uniformly beat dense spatial linears under TF32.

For q/v/out shapes:

```text
Cin=160, Cout=160
```

spatial dense GEMMs are extremely efficient on tensor cores. Fourier does fewer theoretical multiplies, but the work is split into smaller matmuls plus fixed group transforms and layout movement.

For FFN shapes:

```text
FFN up:   Cin=160, Cout=320
FFN down: Cin=320, Cout=160
```

compiled `fourier_direct` starts to win. This is why the current mixed backend keeps attention projections spatial and uses Fourier only for FFN up/down.

## Attention Plumbing Fastpaths

These changes are exact model-preserving optimizations.

### RoPE Cache

Before the fastpath, `PlatonicAttention` recomputed the same RoPE angles, cosines, and sines separately for q, k, v, and inverse value transport.

Now `PlatonicRoPE` exposes:

```python
cos, sin = rope.cos_sin(pos, dtype=x.dtype, device=x.device)
x = rope.apply_from_cos_sin(x, cos, sin)
```

This lets one attention call reuse the same `cos` and `sin` tensors for q, k, v, and inverse value transport.

### Constant-Key Fastpath

When `use_key=false`, keys are deterministic all-ones before RoPE. RoPE applied to the pair `(1, 1)` gives:

```text
k0 = cos - sin
k1 = sin + cos
```

So the implementation constructs roped constant keys directly:

```python
k = rope.constant_key_from_cos_sin(cos, sin)
```

This avoids allocating `torch.ones_like(q)` and running a full RoPE application for keys.

### Fused q/v Projection

When `use_key=false`, the attention block only needs learned q and v projections. The optional `fused_qv=True` path replaces:

```text
q = q_proj(x)
v = v_proj(x)
```

with:

```text
qv = qv_proj(x)
q, v = split(qv)
```

This is exact because concatenating two equivariant outputs is still equivariant. It is currently opt-in for checkpoint compatibility.

## Triton Attention Work

Triton attention work is separate from Fourier FFN work.

The Triton path changes the attention kernel itself:

```json
{
  "tetra": {
    "attention_backend": "triton"
  }
}
```

or, with geometric bias:

```json
{
  "tetra": {
    "attention_backend": "triton",
    "attention_bias": {
      "kind": "radial_r2",
      "zero_init": true
    }
  }
}
```

These modes can combine with the Fourier FFN mixed backend, but they answer a different question:

- Fourier-direct FFN is a speed optimization for group-convolution linears.
- Triton radial attention is a modeling and memory/layout path for geometric attention bias.

The current profile artifacts show FlashAttention remains the speed baseline for no-bias attention. Triton radial modes are more expensive, so they should be evaluated as quality/geometry experiments, not as pure speed wins.

## Full-Model Benchmark Results

The full-model benchmark script is:

`scripts/benchmark_omol_mixed_backend.py`

Benchmark setup:

- GPU: NVIDIA RTX 6000 Ada
- model: h1920, 16 layers, 60 heads, tetra-only, platonic readout
- runtime: `internal_flat_tetra`
- synthetic OMol-shaped batch: 12000 atoms, 218 molecules, max segment 163, sum_n2 1.158M
- precision: TF32 high
- attention: FlashAttention

Variants:

- `recompute`: old RoPE recompute, separate q/v, spatial FFN
- `attention_fast`: RoPE cache, constant-key fastpath, fused q/v, spatial FFN
- `mixed`: attention_fast plus `ffn_linear_backend="fourier_direct"`

Compiled trunk-flat results:

| variant | forward | fwd+bwd | speedup vs recompute | max memory |
|---|---:|---:|---:|---:|
| recompute | 186.6 ms | 606.0 ms | 1.00x | 25.99 GB |
| attention_fast | 179.9 ms | 579.2 ms | 1.05x | 25.91 GB |
| mixed | 164.5 ms | 492.0 ms | 1.23x | 25.09 GB |

Whole-model compile results were similar:

| variant | forward | fwd+bwd | speedup vs recompute | max memory |
|---|---:|---:|---:|---:|
| recompute | 183.0 ms | 606.8 ms | 1.00x | 25.82 GB |
| attention_fast | 175.8 ms | 579.1 ms | 1.05x | 25.74 GB |
| mixed | 159.6 ms | 489.4 ms | 1.24x | 25.08 GB |

Eager results were different:

- attention_fast helped modestly
- mixed lost the speed back
- therefore mixed FFN should be used with `torch.compile`

## Parity

Parity tests cover:

- spatial vs Fourier scaffold forward/backward
- Fourier direct forward/input-gradient parity after spatial-to-Fourier conversion
- Fourier equivariance
- dense `get_weight()` parity for the scaffold
- RoPE cached helpers vs old RoPE forward
- constant-key fastpath vs RoPE-applied ones
- separate q/v vs fused q/v for spatial, Fourier scaffold, and Fourier direct
- OMol construction/forward/backward smoke tests

Full-model benchmark parity:

- attention_fast is mathematically exact; observed differences are at FlashAttention/TF32 roundoff scale
- mixed has small FP-level drift from spatial-to-Fourier direct conversion and different matmul order, around `1e-4` energy max and `2e-5` force max in the 12k-atom benchmark

## Practical Guidance

For pure speed testing on the current h1920/l16 tetra OMol run:

```bash
HYBRID_CONFIG_JSON=configs/omol/tetra_t_only_h1920_l16_pt2_exact_sin_layerscale_mixedffn.json
OMOL_RUNTIME_MODE=internal_flat_tetra
COMPILE=1
COMPILE_SCOPE=trunk_flat
FLOAT32_MATMUL_PRECISION=high
```

For quality experiments with radial attention:

1. Start from the best radial/Triton config from the attention-kernel work.
2. Add:

```json
{
  "linear_backend": "spatial",
  "attention_linear_backend": "spatial",
  "ffn_linear_backend": "fourier_direct",
  "rope_cache": true,
  "constant_key_fastpath": true,
  "fused_qv": true
}
```

3. Compare against Flash mixed, not against the old recompute baseline.

## Open Issues

- Muon optimizer grouping needs explicit review for `fourier_direct` parameters because FFN parameters become `w1`, `w2_re`, `w2_im`, and `w3` instead of `.kernel`.
- The direct Fourier backend is performance-oriented but changes parameter names, so checkpoint conversion requires deliberate handling.
- Triton radial attention is still a quality/geometry bet; current speed profiles are slower than FlashAttention.
- A dedicated Fourier FFN could further improve speed by fusing the fixed inverse-Fourier -> activation -> Fourier sandwich.
- Framewise-shared and separable group linears remain promising cheaper equivariant families to test after the mixed backend baseline.
