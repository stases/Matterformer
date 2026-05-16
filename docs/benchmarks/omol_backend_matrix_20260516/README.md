# OMol Backend Matrix Benchmark, 2026-05-16

This folder preserves the OMol full-model backend benchmark used to choose the
current tetra/Platonic backend recommendation.

## Setup

- GPU: NVIDIA RTX 6000 Ada Generation
- Torch: 2.5.0+cu124
- CUDA: 12.4
- Model shape: OMol internal flat tetra, `d_model=1920`, `n_layers=16`, `n_heads=60`
- Synthetic batch shape: `12000` atoms, `218` graphs, max graph size `163`, `sum_n2=1158000`
- Metric: forward-only time, forward+backward time, and peak allocated CUDA memory
- Loss used for timing: synthetic `energy.square().mean() + forces.square().mean()`
- No optimizer step or dataloader time is included.

## Raw Results

- `full_all_backends_high_108.json`: complete `fp32_high` matrix, 108 rows
- `full_all_backends_high_108.csv`: same complete table in CSV form
- `full_all_backends_high_108.md`: auto-generated top-row summary
- `full_all_backends_all_precisions_432_r1_fixedref.json`: corrected precision sweep, stopped at 419/432 rows
- `full_all_backends_all_precisions_432_r1_fixedref.csv`: same partial precision sweep in CSV form
- `full_all_backends_all_precisions_432_r1_fixedref.md`: auto-generated top-row summary

The precision sweep was intentionally stopped early after the useful signal was clear.
The missing tail is AMP bf16 compiled radial attention combinations. The completed
rows already cover all practical Flash/Triton non-radial choices and all fp32/TF32
precision modes.

## Recommendation

Use this as the production-speed default candidate:

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

Run it with:

- `compile=true`
- `compile_scope=trunk_flat` or equivalent trunk compilation
- `float32_matmul_precision="high"`

This keeps attention projections spatial and uses Fourier only for the FFN linears.
It is the best speedup that still keeps numerical drift close to the spatial model.

## Key Results

From the complete `fp32_high` matrix:

| config | fwd+bwd ms | forward ms | mem GB | force max abs | energy max abs | interpretation |
|---|---:|---:|---:|---:|---:|---|
| Flash + spatial baseline | 608.3 | 187.4 | 25.85 | 1.55e-05 | 3.62e-05 | compiled baseline |
| Flash + spatial_fast | 577.6 | 178.1 | 25.77 | 1.45e-05 | 4.45e-05 | RoPE cache/fused qv helps |
| Flash + FFN fourier_direct | 488.1 | 158.9 | 24.95 | 1.89e-05 | 6.96e-05 | recommended safe speed path |
| Flash + all_fourier_direct | 468.5 | 161.5 | 24.56 | 2.72e-05 | 5.70e-03 | fastest fp32 row, but larger energy drift |
| Triton bf16compat + FFN fourier_direct | 515.8 | 174.9 | 24.36 | 1.80e-05 | 7.30e-05 | viable, slower than Flash here |
| Triton radial_rbf8 + FFN fourier_direct | 618.0 | 183.8 | 25.73 | 1.86e-05 | 7.87e-05 | useful only if radial bias quality pays for it |

From the precision sweep:

| config | fwd+bwd ms | mem GB | force max abs | energy max abs | interpretation |
|---|---:|---:|---:|---:|---|
| AMP bf16 + compiled Flash + all_fourier_direct | 249.0 | 13.25 | 2.59e-04 | 3.49e-02 | fastest absolute, too much drift for default |
| AMP bf16 + compiled Flash + FFN fourier_direct | 307.2 | 18.03 | 1.83e-04 | 8.79e-04 | fast and memory-light, needs quality validation |
| AMP bf16 + compiled Flash + spatial_fast | 342.3 | 18.42 | 8.31e-05 | 3.17e-04 | safer AMP candidate, still nontrivial drift |
| fp32_medium + compiled Flash + FFN fourier_direct | 482.2 | 24.95 | 1.80e-05 | 6.03e-05 | tied with fp32_high, no clear advantage |
| fp32_high + compiled Flash + FFN fourier_direct | 483.7 | 24.95 | 2.03e-05 | 7.78e-05 | stable default candidate |

## Interpretation

`spatial_fast` is an exact plumbing improvement: RoPE cos/sin cache, constant-key
fastpath, and fused q/v projection. It improves the compiled Flash baseline from
about `608 ms` to `578 ms` fwd+bwd.

`ffn_fourier_direct` is the useful Fourier path. With compiled Flash attention,
it improves fwd+bwd from about `578 ms` to `488 ms` while slightly reducing memory.
This is the main result.

`all_fourier_direct` is faster in micro/full-model timing but changes attention
projection numerics more strongly. The `fp32_high` force delta is still small in
absolute terms, but the energy delta is about `5.7e-03` against the spatial
reference in this synthetic parity check. Treat it as an experiment, not a default.

AMP bf16 is dramatically faster and lower memory, but the parity drift is much
larger. It needs actual training/validation before use. Do not infer quality from
the speed table alone.

SDPA should not be used for this OMol flat tetra shape. Its backward pass is around
`5.2-5.6 s` per measured row at 12k atoms, even when compiled.

Triton radial attention is not a speed win here. It should be used only when the
radial bias improves model quality enough to pay for the slower attention path.

## Known Failures

Five AMP bf16 + Triton radial + all-Fourier rows failed with a Triton compilation
error around `tl.atomic_add` in the radial backward kernel. These are recorded in
the precision sweep JSON/CSV. The practical Flash and non-radial Triton rows ran.

## Reproduction Commands

Complete `fp32_high` matrix:

```bash
srun --partition=delta --account=deltausers --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=03:00:00 \
  env PYTHONPATH=src /home/thadziv/GitHub/erwin/erwin/bin/python \
  scripts/benchmark_omol_backend_matrix.py \
  --total-atoms 12000 \
  --num-graphs 218 \
  --max-atoms-per-graph 220 \
  --warmup 1 \
  --repeats 2 \
  --attention-modes sdpa flash triton_tf32 triton_tf32x3 triton_ieee triton_bf16compat triton_radial_r2 triton_radial_slope triton_radial_rbf8 \
  --linear-modes spatial_recompute spatial_fast ffn_fourier_scaffold ffn_fourier_direct all_fourier_scaffold all_fourier_direct \
  --compile-scopes none trunk_flat \
  --precision-modes fp32_high \
  --output-prefix oracle_context/omol_backend_matrix_20260516/full_all_backends_high_108
```

Corrected precision sweep:

```bash
srun --partition=delta --account=deltausers --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=03:00:00 \
  env PYTHONPATH=src /home/thadziv/GitHub/erwin/erwin/bin/python \
  scripts/benchmark_omol_backend_matrix.py \
  --total-atoms 12000 \
  --num-graphs 218 \
  --max-atoms-per-graph 220 \
  --warmup 1 \
  --repeats 1 \
  --attention-modes sdpa flash triton_tf32 triton_tf32x3 triton_ieee triton_bf16compat triton_radial_r2 triton_radial_slope triton_radial_rbf8 \
  --linear-modes spatial_recompute spatial_fast ffn_fourier_scaffold ffn_fourier_direct all_fourier_scaffold all_fourier_direct \
  --compile-scopes none trunk_flat \
  --precision-modes fp32_highest fp32_high fp32_medium ampbf16_high \
  --output-prefix oracle_context/omol_backend_matrix_20260516/full_all_backends_all_precisions_432_r1_fixedref
```
