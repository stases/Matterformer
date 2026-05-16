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
| 1 | flash | all_fourier_direct | none | fp32_high | 134.861 | 27.373 | 1.366 | 2.094e-05 |
