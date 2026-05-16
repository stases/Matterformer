# Fourier Direct / Mixed Backend Findings For Oracle Review

## Why This Bundle Exists

We implemented the first Fourier `PlatonicLinear` backend and then tested the oracle's diagnosis. The result is nuanced:

- The original scaffold Fourier backend is mathematically correct.
- A layout-only optimization helped a little, but did not make the scaffold win under TF32.
- A new direct Fourier parameter backend removes per-forward spatial-kernel conversion overhead.
- Direct Fourier is still slower for `160 -> 160` attention projections.
- Direct Fourier becomes useful for compiled FFN-shaped linears at the production token scale.

The next oracle should review the implementation and tell us how to turn this into a real full-model speedup.

## Current Implementation State

`PlatonicLinear` now supports:

```python
PlatonicLinear(..., linear_backend="spatial")
PlatonicLinear(..., linear_backend="fourier")
PlatonicLinear(..., linear_backend="fourier_direct")
```

Backend meanings:

- `spatial`: original dense spatial group-convolution path.
- `fourier`: correctness scaffold. Keeps spatial `kernel [G, Cout, Cin]`, converts it to Fourier block weights every forward.
- `fourier_direct`: direct Fourier parameters:
  - `w1 [Cout, Cin]`
  - `w2_re [Cout, Cin]`
  - `w2_im [Cout, Cin]`
  - `w3 [3*Cout, 3*Cin]`
  - optional `bias [Cout]`

`fourier_direct` preserves the same parameter count as the spatial kernel:

```text
w1:    1 * Cout * Cin
w2:    2 * Cout * Cin
w3:    9 * Cout * Cin
total: 12 * Cout * Cin
```

There is a conversion helper:

```python
layer.set_spatial_parameters_(spatial.kernel, spatial.bias)
```

This lets tests and benchmarks initialize direct Fourier params from an equivalent spatial kernel.

## Important API Addition

`PlatonicBlock` now supports split linear backend controls:

```python
PlatonicBlock(
    ...,
    linear_backend="spatial",
    attention_linear_backend="spatial",
    ffn_linear_backend="fourier_direct",
)
```

`HybridConfig.tetra` now has defaults:

```json
{
  "linear_backend": "spatial",
  "attention_linear_backend": null,
  "ffn_linear_backend": null
}
```

Recommended next experiment config:

```json
"tetra": {
  "group": "tetrahedron",
  "group_order": 12,
  "heads_per_frame": 5,
  "rope_sigma": 2.0,
  "learned_freqs": true,
  "freq_init": "random",
  "use_key": false,
  "rope_on_values": true,
  "attention_backend": "flash",
  "activation": "sin",
  "ffn_mult": 2,
  "layer_scale_init_value": 0.0001,
  "linear_backend": "spatial",
  "attention_linear_backend": "spatial",
  "ffn_linear_backend": "fourier_direct"
}
```

Do not set all linears to Fourier. The benchmark says this is worse.

## Usual OMol Run Shape

Representative production setup:

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
- `attention_backend = flash`
- `runtime_mode = internal_flat_tetra`
- `compile = true`
- `float32_matmul_precision = high`
- TF32 enabled in training-relevant benchmarks
- `max_atoms_per_batch = 12000`

In `internal_flat_tetra`, one benchmark token is one atom in the flattened batch. Production batches are capped at about 12000 atoms.

Per tetra block:

- q projection: `Cin=160, Cout=160`
- v projection: `Cin=160, Cout=160`
- out projection: `Cin=160, Cout=160`
- no k projection because `use_key=false`
- FFN up: `Cin=160, Cout=320`
- FFN down: `Cin=320, Cout=160`

## GPU Environment

Interactive request used:

```bash
srun --partition=all6000 --account=all6000users --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:30:00 --pty bash -l
```

Environment:

- Node: `ivi-cn032`
- GPU: `NVIDIA RTX 6000 Ada Generation`, about 49 GB
- Python: `/home/thadziv/GitHub/erwin/erwin/bin/python`
- Torch: `2.5.0+cu124`
- CUDA: `12.4`

The base conda environment is CPU-only PyTorch and should not be used for GPU tests.

## Correctness Results

After the layout patch:

GPU parity with TF32 disabled:

| shape | max abs error | mean abs error |
|---|---:|---:|
| `tokens=512, 160 -> 160` | `5.006790e-06` | `5.045532e-07` |
| `tokens=512, 160 -> 320` | `3.576279e-06` | `3.747453e-07` |
| `tokens=512, 320 -> 160` | `4.053116e-06` | `4.866940e-07` |

Backward parity on moderate shape:

| gradient | max abs error | mean abs error |
|---|---:|---:|
| input grad | `9.313226e-09` | `1.634853e-09` |
| kernel grad | `4.768372e-07` | `6.789732e-08` |
| bias grad | `9.536743e-07` | `3.973643e-08` |

Local targeted test suite:

```bash
pytest tests/test_platonic_fourier_linear.py tests/test_hybrid.py tests/test_omol_models.py tests/test_qm9_models.py -q
```

Result after `fourier_direct` and split backend controls:

```text
91 passed, 13 skipped, 26 warnings
```

## Speed Results: Layout-Optimized Scaffold

After replacing general `einsum` group transforms with explicit `matmul` and contiguous 2D GEMMs, scaffold results at TF32 on:

| tokens | shape | spatial | scaffold Fourier | speedup |
|---:|---|---:|---:|---:|
| 12000 | `160 -> 160` | 1.003 ms | 2.004 ms | 0.50x |
| 12000 | `160 -> 320` | 2.213 ms | 3.271 ms | 0.68x |
| 12000 | `320 -> 160` | 1.995 ms | 2.965 ms | 0.67x |

Forward+backward, TF32 on:

| tokens | shape | spatial | scaffold Fourier | speedup |
|---:|---|---:|---:|---:|
| 12000 | `160 -> 160` | 4.618 ms | 6.522 ms | 0.71x |
| 12000 | `160 -> 320` | 9.181 ms | 9.991 ms | 0.92x |
| 12000 | `320 -> 160` | 8.199 ms | 9.862 ms | 0.83x |

Conclusion: layout patch helps only marginally; scaffold is still not a production speed backend.

## Speed Results: `fourier_direct`

Eager forward+backward at `tokens=12000`, TF32 on:

| shape | spatial | scaffold | direct | direct speedup |
|---|---:|---:|---:|---:|
| `160 -> 160` | 4.569 ms | 6.513 ms | 6.361 ms | 0.72x |
| `160 -> 320` | 9.185 ms | 9.971 ms | 9.824 ms | 0.93x |
| `320 -> 160` | 8.127 ms | 9.855 ms | 9.527 ms | 0.85x |

Eager direct is better than scaffold, but not enough.

## Speed Results: Compiled Forward

At `tokens=12000`, TF32 on, `torch.compile(mode="default")`, forward only:

| shape | compiled spatial | compiled scaffold | compiled direct |
|---|---:|---:|---:|
| `160 -> 160` | 0.923 ms | 1.534 ms | 1.506 ms |
| `160 -> 320` | 2.491 ms | 2.456 ms | 2.422 ms |
| `320 -> 160` | 2.569 ms | 2.288 ms | 2.424 ms |

Compiled forward suggests Fourier can help FFN-shaped linears, but not attention-shaped `160 -> 160`.

## Speed Results: Compiled Forward+Backward

Most relevant microbenchmark:

At `tokens=12000`, TF32 on, `torch.compile(mode="default")`, forward+backward:

| shape | compiled spatial | compiled direct | direct speedup |
|---|---:|---:|---:|
| `160 -> 160` q/v/out | 3.846 ms | 4.569 ms | 0.84x |
| `160 -> 320` FFN up | 8.780 ms | 7.856 ms | 1.12x |
| `320 -> 160` FFN down | 7.867 ms | 6.418 ms | 1.23x |

This is the first promising signal.

Estimated compiled linear-stack effect per tetra block:

```text
all spatial:
  3 * 3.846 + 8.780 + 7.867 = 28.185 ms

attention spatial + FFN direct:
  3 * 3.846 + 7.856 + 6.418 = 25.812 ms
```

That is about `1.09x` faster for the PlatonicLinear stack alone, before full-model effects.

## Diagnosis So Far

1. The Fourier math is correct.
2. The scaffold backend is useful for parity but should not be used for production speed.
3. Direct Fourier params remove conversion overhead but still suffer transform/layout overhead.
4. Dense spatial TF32 cuBLAS is extremely fast for `160 -> 160`, so attention q/v/out should remain spatial for now.
5. FFN up/down are the only currently promising target for `fourier_direct`.
6. `torch.compile` materially improves the Fourier path and changes conclusions; eager-only benchmarks are misleading.

## Questions For Oracle

Please review the included code and answer:

1. Is `fourier_direct` parameterization correct and complete?
2. Is the direct parameter initialization from a spatial kernel correct?
3. Is the mixed backend plan sound: spatial attention projections, `fourier_direct` FFN?
4. What is the next best optimization?
   - hard-code tetra transform sums instead of small matmul?
   - custom Triton transform kernels around cuBLAS matmuls?
   - fuse low/3D block preparation?
   - avoid transforms across consecutive FFN linears somehow?
5. How should Muon handle `fourier_direct` params?
   - Current Muon special view expects spatial `PlatonicLinear.kernel`.
   - Direct backend has `w1`, `w2_re`, `w2_im`, `w3` instead.
   - Production run uses `muon_hidden_only=true` and applies Muon mostly to FFN Platonic kernels.
6. Should we benchmark a full compiled `PlatonicBlock` next, or implement a fused transform first?

## Recommended Next Experiment

Do not enable Fourier globally.

Run a short compiled OMol or block-level benchmark with:

```json
"tetra": {
  "linear_backend": "spatial",
  "attention_linear_backend": "spatial",
  "ffn_linear_backend": "fourier_direct"
}
```

Compare against all-spatial under the same settings:

- `compile=true`
- `float32_matmul_precision=high`
- `tokens ~= 12000`
- forward+backward, not just forward
- ideally include optimizer step or at least assess Muon compatibility

Current best guess: mixed mode may give a modest full-model speedup, but only if Muon and compile work cleanly with the new direct Fourier parameters.
