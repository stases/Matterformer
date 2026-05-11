# Compact kNN Simplicial Triton Benchmark

Run date: 2026-05-11

Slurm job: `166521`

Node/GPU: `ivi-cn033`, NVIDIA RTX 6000 Ada Generation

Environment:

- Python: `/home/thadziv/GitHub/erwin/erwin/bin/python`
- PyTorch: `2.5.0+cu124`
- Triton: `3.1.0`
- CUDA runtime reported by PyTorch: `12.4`

## Shape

The benchmark uses QM9-like compact simplicial tensors:

```text
B = 64
H = 12
N = 32
K = 32
D = 64
angle_rank = 32
```

Biases are enabled in the usual compact local form:

- radial `u/v` bias
- spherical low-rank angle bias
- random gates to exercise the active bias path
- optional value-side low-rank message path

The benchmark compares:

- Torch reference compact path
- Triton compact kNN path

## Correctness Smoke

The Delta job first ran a strict CUDA parity smoke test with `precision="ieee_fp32"`:

```text
max_abs = 2.38419e-06
max_grad_abs = 2.86102e-05
```

## Results

### fp32 / tf32, radial + angle, message off

```text
Torch  forward:  6.991 ms
Triton forward:  0.318 ms
Speedup:        22.02x

Torch  fwd+bwd: 21.919 ms
Triton fwd+bwd:  2.818 ms
Speedup:         7.78x

Torch  fwd+bwd peak delta: 1936.00 MiB
Triton fwd+bwd peak delta:  240.19 MiB
Memory ratio:                8.06x lower

max_abs_diff:      0.0170531
max_grad_abs_diff: 0.104345
```

The larger diff here is expected from `tf32`; the strict parity check above uses
`ieee_fp32`.

### bf16 / tensor core, radial + angle, message off

```text
Torch  forward:  5.852 ms
Triton forward:  0.245 ms
Speedup:        23.85x

Torch  fwd+bwd: 16.640 ms
Triton fwd+bwd:  3.007 ms
Speedup:         5.53x

Torch  fwd+bwd peak delta: 1399.19 MiB
Triton fwd+bwd peak delta:  270.14 MiB
Memory ratio:                5.18x lower

max_abs_diff:      0.03125
max_grad_abs_diff: 0.25
```

### bf16 / tensor core, radial + angle, message on

```text
Torch  forward:  6.717 ms
Triton forward:  0.413 ms
Speedup:        16.26x

Torch  fwd+bwd: 19.250 ms
Triton fwd+bwd:  4.197 ms
Speedup:         4.59x

Torch  fwd+bwd peak delta: 1544.69 MiB
Triton fwd+bwd peak delta:  366.19 MiB
Memory ratio:                4.22x lower

max_abs_diff:      0.0625
max_grad_abs_diff: 0.5
```

## Notes

- Forward-only Triton peak delta prints as `0.00 MiB` in these runs because the
  CUDA allocator reuses cached blocks and the live extra allocation is below the
  useful reporting granularity in this benchmark.  The forward+backward peak
  delta is the more useful memory number.
- The benchmark does not include kNN construction, RBF construction, spherical
  basis construction, or bias MLPs.  It measures the compact simplicial
  attention kernel boundary.
- During GPU validation, two real kernel issues were found and fixed:
  - `BLOCK_K` now has a minimum of 16 because `tl.dot` requires non-batch matrix
    dimensions of at least 16.
  - bf16 backward uses fp32 gradient accumulation buffers because Triton 3.1
    does not support `atomic_add` into bf16 buffers.
