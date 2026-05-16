# OMol Backend Matrix Benchmark

## Shape

```json
{
  "atoms": 512.0,
  "graphs": 20.0,
  "max_atoms": 80.0,
  "padded_slots": 1600.0,
  "sum_n2": 27490.0
}
```

## Fastest fwd+bwd Rows

| rank | attention | linear | compile | precision | fwd+bwd ms | forward ms | mem GB | force max abs |
|---:|---|---|---|---|---:|---:|---:|---:|
| 1 | triton_tf32x3 | ffn_fourier_direct | none | fp32_high | 78.573 | 20.938 | 2.058 | 1.543e-05 |
| 2 | flash | ffn_fourier_direct | none | fp32_high | 81.884 | 21.228 | 2.010 | 1.465e-05 |
| 3 | flash | spatial_recompute | none | fp32_high | 116.553 | 22.200 | 3.072 | 7.276e-12 |
| 4 | triton_tf32x3 | spatial_recompute | none | fp32_high | 782.787 | 21.997 | 3.138 | 1.294e-05 |
