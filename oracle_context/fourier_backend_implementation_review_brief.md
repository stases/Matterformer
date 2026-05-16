# Fourier PlatonicLinear Implementation Review Brief

## Current Goal

Review the first implementation of an opt-in Fourier backend for `PlatonicLinear` and explain why it is not faster in the training-relevant setup. The implementation is mathematically equivalent to the current spatial group convolution and keeps the same `kernel` / `bias` parameters for checkpoint compatibility.

The next LLM should answer:

1. Is the Fourier implementation mathematically correct?
2. Is any obvious implementation mistake making it slower?
3. Given our usual OMol training setup, what should be changed to make Fourier linears actually faster?
4. Should we keep this backend only as a correctness scaffold and move to direct Fourier parameters, fusion, or Triton?

## Implementation Summary

Implemented files:

- `src/matterformer/models/platonic/linear.py`
- `src/matterformer/models/platonic/layers.py`
- `src/matterformer/models/hybrid.py`
- `src/matterformer/models/omol.py`
- `src/matterformer/models/qm9.py`
- `tests/test_platonic_fourier_linear.py`
- `scripts/benchmark_platonic_fourier_linear.py`

API:

```python
PlatonicLinear(..., linear_backend="spatial")  # default, old path
PlatonicLinear(..., linear_backend="fourier")  # new tetra-only path
```

State dict compatibility:

- Parameters remain `kernel [G, Cout, Cin]` and optional `bias [Cout]`.
- Fourier data are non-persistent buffers.
- `get_weight()` remains available and matches the old dense spatial weight.

Fourier path:

- Builds tetra real Fourier basis `Q [12, 12]` from irreps `1 + 2 + 3 + 3 + 3`.
- Builds A4/V4 quotient exponents from the Cayley table, not hard-coded element indices.
- Converts the spatial kernel to Fourier block weights each forward:
  - low merged block: `3*Cout x 3*Cin`
  - 3D block: `3*Cout x 3*Cin`, applied independently over the 3 irrep-coordinate batches
- Uses the convention-sensitive formula:

```python
w3[n, q, o, i] = sum_g group.elements[g, q, n] * kernel[g, o, i]
```

## Usual OMol Run Context

The representative production run is the PT2-style 16-layer tetra-only OMol model:

- Script: `scripts/omol_matterformer_pt2_l16_delta.sh`
- Wrapper: `scripts/omol_forcefield_delta.sh`
- Python: `/home/thadziv/GitHub/erwin/erwin/bin/python`
- Training partition normally: `delta`, account `deltausers`
- Benchmark partition used for this check: `all6000`, account `all6000users`
- Runtime mode: `internal_flat_tetra`
- Model backend: `matterformer`
- Compile: `true`, `compile_mode="default"`
- Precision: `bf16=false`, `float32_matmul_precision="high"` so TF32 is enabled for matmul
- Attention backend: `flash`
- Optimizer in the real run shown by user: `muon`
- Muon is applied mainly to FFN Platonic kernels, not attention projections, because the run excludes names containing `attn.`

Important model dimensions:

- `d_model = 1920`
- tetra group order `G = 12`
- `tetra_dim_per_frame = 160`
- `heads_per_frame = 5`
- total heads `60`
- `head_dim = 32`
- `num_blocks = 16`
- `ffn_mult = 2`
- `use_key = false`
- `rope_on_values = true`
- input lift: `scalar_copy`
- readout: `platonic_ffn`
- batch settings:
  - `batch_size = 64`
  - `max_atoms_per_batch = 12000`
  - `max_graphs_per_batch = 999999`

Approximate token interpretation:

- In `internal_flat_tetra`, one token is one atom in the flattened batch.
- The effective tokens per forward step are capped by `max_atoms_per_batch`, usually up to about `12000`.
- A single `PlatonicLinear` sees shape roughly `[tokens, 12*C]`.

Important linear shapes per tetra block:

- q projection: `Cin=160, Cout=160`
- v projection: `Cin=160, Cout=160`
- out projection: `Cin=160, Cout=160`
- no k projection because `use_key=false`
- FFN up: `Cin=160, Cout=320`
- FFN down: `Cin=320, Cout=160`

Per block, linears are the main target:

- q/v/out: `3 * d_model^2`
- FFN: `d * 2d + 2d * d = 4d_model^2`
- total about `7d_model^2` multiply terms per atom per block before attention

## GPU Environment Used For Check

Interactive SLURM request that worked:

```bash
srun --partition=all6000 --account=all6000users --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:30:00 --pty bash -l
```

Node/GPU:

- Node: `ivi-cn032`
- GPU: `NVIDIA RTX 6000 Ada Generation`
- Memory: about `49 GB`
- Python: `/home/thadziv/GitHub/erwin/erwin/bin/python`
- Torch: `2.5.0+cu124`
- CUDA: `12.4`

The default base conda environment is CPU-only PyTorch and should not be used for GPU timing.

## Correctness Findings

With TF32 disabled:

```python
torch.backends.cuda.matmul.allow_tf32 = False
```

Forward parity:

| shape | max abs error | mean abs error |
|---|---:|---:|
| tokens=64, Cin=4, Cout=7 | `9.536743e-07` | `1.152335e-07` |
| tokens=512, Cin=160, Cout=160 | `5.245209e-06` | `5.052745e-07` |
| tokens=512, Cin=160, Cout=320 | `3.337860e-06` | `3.743341e-07` |
| tokens=512, Cin=320, Cout=160 | `4.291534e-06` | `4.867893e-07` |

Backward parity on `Cin=16, Cout=24, tokens=128`:

| gradient | max abs error | mean abs error |
|---|---:|---:|
| input grad | `9.313226e-09` | `1.548801e-09` |
| kernel grad | `4.172325e-07` | `7.904399e-08` |
| bias grad | `0.000000e+00` | `0.000000e+00` |

Local CPU tests also passed:

```bash
pytest tests/test_platonic_fourier_linear.py -q
# 5 passed, 1 skipped

pytest tests/test_platonic_fourier_linear.py tests/test_hybrid.py tests/test_omol_models.py tests/test_qm9_models.py -q
# 86 passed, 13 skipped
```

Note: `/home/thadziv/GitHub/erwin/erwin/bin/python` did not have `pytest` installed on the GPU node, so GPU pytest was not run. Direct GPU parity scripts were run instead.

## Speed Findings

### Strict FP32 / TF32 Off

With TF32 disabled, Fourier is faster for larger shapes, especially FFN up/down.

| tokens | Cin | Cout | spatial | Fourier | speedup |
|---:|---:|---:|---:|---:|---:|
| 512 | 160 | 160 | 0.133 ms | 0.289 ms | 0.46x |
| 512 | 160 | 320 | 0.339 ms | 0.298 ms | 1.14x |
| 512 | 320 | 160 | 0.341 ms | 0.293 ms | 1.16x |
| 2048 | 160 | 160 | 0.596 ms | 0.463 ms | 1.29x |
| 2048 | 160 | 320 | 1.730 ms | 0.766 ms | 2.26x |
| 2048 | 320 | 160 | 1.668 ms | 0.778 ms | 2.14x |
| 8192 | 160 | 160 | 2.856 ms | 1.688 ms | 1.69x |
| 8192 | 160 | 320 | 4.941 ms | 2.889 ms | 1.71x |
| 8192 | 320 | 160 | 5.741 ms | 2.841 ms | 2.02x |
| 16384 | 160 | 160 | 4.985 ms | 3.375 ms | 1.48x |
| 16384 | 160 | 320 | 10.113 ms | 5.848 ms | 1.73x |
| 16384 | 320 | 160 | 9.931 ms | 5.430 ms | 1.83x |

### Training-Relevant Matmul Precision High / TF32 On

With:

```python
torch.set_float32_matmul_precision("high")
```

PyTorch enables TF32 for matmul. This is what the OMol script normally does through `FLOAT32_MATMUL_PRECISION=high`.

Under this setting, dense spatial cuBLAS is much faster and Fourier is mostly slower:

| tokens | Cin | Cout | spatial | Fourier | speedup |
|---:|---:|---:|---:|---:|---:|
| 512 | 160 | 160 | 0.066 ms | 0.295 ms | 0.22x |
| 512 | 160 | 320 | 0.121 ms | 0.294 ms | 0.41x |
| 512 | 320 | 160 | 0.132 ms | 0.296 ms | 0.45x |
| 2048 | 160 | 160 | 0.154 ms | 0.299 ms | 0.51x |
| 2048 | 160 | 320 | 0.306 ms | 0.346 ms | 0.88x |
| 2048 | 320 | 160 | 0.359 ms | 0.423 ms | 0.85x |
| 8192 | 160 | 160 | 0.806 ms | 1.230 ms | 0.66x |
| 8192 | 160 | 320 | 1.855 ms | 2.254 ms | 0.82x |
| 8192 | 320 | 160 | 1.972 ms | 1.863 ms | 1.06x |
| 16384 | 160 | 160 | 1.913 ms | 2.728 ms | 0.70x |
| 16384 | 160 | 320 | 3.787 ms | 4.603 ms | 0.82x |
| 16384 | 320 | 160 | 3.538 ms | 3.960 ms | 0.89x |

The max errors with TF32 on were around `1.8e-3` to `3.3e-3`, which is expected because TF32 changes matmul precision.

## Likely Reasons Fourier Is Not Fast Enough

This backend reduces arithmetic in theory, but the current PyTorch implementation has overhead that dense cuBLAS avoids:

1. Dense spatial path is one large cuBLAS GEMM. With TF32 enabled, that GEMM is extremely fast.
2. Fourier path performs multiple operations per forward:
   - group transform `Q.T x`
   - spatial-kernel to Fourier-weight conversion
   - low-block matmul
   - repeated 3D block matmul
   - inverse transform `Q y`
   - extra reshapes/einsums/dtype conversions
3. Fourier block weights are rebuilt every forward from the spatial kernel for checkpoint compatibility. That is good for v1 correctness but bad for speed.
4. The low block is padded to `3C x 3C`, so some multiplies hit structural zeros.
5. Small fixed `G=12` transforms are not fused, so launch/memory overhead matters.
6. The benchmark did not use `torch.compile`; the training script does. The next LLM should assess whether compile can fuse enough of this path, but the current implementation may still resist fusion because of dynamic weight construction and multiple matmul calls.
7. Muon optimizer uses a special `platonic_conv` view on spatial kernels. Direct Fourier parameters would need optimizer handling if the parameterization changes later.

My current interpretation: the implementation appears correct. The speed problem is likely not a math-convention bug; it is that this conservative spatial-kernel Fourier backend is an unfused scaffold. It can beat dense spatial only when TF32 is disabled, but our actual training uses TF32.

## Questions For The Next LLM

Please inspect the included code and answer:

1. Is the `w3 = einsum("gqn,goi->nqoi", rho3, kernel)` convention correct for the repo’s `kernel_group = inverse(input_group) * output_group`? Parity tests suggest yes, but verify.
2. Can the current PyTorch Fourier backend be made faster without changing parameters?
   - precompute/cached Fourier weights?
   - use `torch.compile` friendly structure?
   - replace `einsum` group transforms with explicit small matmuls?
   - batch the low/3D matmuls differently?
3. Is direct Fourier parameterization required for speed?
4. Should we add a fused Triton kernel for:
   - spatial-to-Fourier transform
   - block matmul
   - inverse transform
   - bias
5. How should this interact with Muon’s current `platonic_conv` view?
6. Given `tokens ~ 12000`, `C=160`, `Cff=320`, and TF32 enabled, what implementation is likely to beat dense cuBLAS?

## Recommendation So Far

Keep this Fourier backend as a correctness scaffold, but do not flip production configs yet. The next implementation should target the training-relevant TF32 case explicitly. A real speed win likely needs direct Fourier parameters, cached/compiled Fourier weights, or a fused Triton/cuBLAS design rather than rebuilding Fourier blocks from spatial kernels every forward.
