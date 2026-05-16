# OMol Backend Matrix Benchmark

## Shape

```json
{
  "atoms": 12000.0,
  "graphs": 218.0,
  "max_atoms": 163.0,
  "padded_slots": 35534.0,
  "sum_n2": 1158000.0
}
```

## Fastest fwd+bwd Rows

| rank | attention | linear | compile | precision | fwd+bwd ms | forward ms | mem GB | force max abs |
|---:|---|---|---|---|---:|---:|---:|---:|
| 1 | flash | all_fourier_direct | trunk_flat | fp32_high | 468.474 | 161.514 | 24.559 | 2.716e-05 |
| 2 | flash | all_fourier_scaffold | trunk_flat | fp32_high | 470.454 | 161.743 | 24.659 | 2.716e-05 |
| 3 | flash | ffn_fourier_scaffold | trunk_flat | fp32_high | 487.934 | 155.586 | 25.004 | 2.000e-05 |
| 4 | flash | ffn_fourier_direct | trunk_flat | fp32_high | 488.119 | 158.922 | 24.947 | 1.889e-05 |
| 5 | triton_tf32 | all_fourier_direct | trunk_flat | fp32_high | 494.607 | 176.639 | 25.343 | 2.898e-05 |
| 6 | triton_tf32 | all_fourier_scaffold | trunk_flat | fp32_high | 496.193 | 177.848 | 25.445 | 2.527e-05 |
| 7 | triton_bf16compat | all_fourier_direct | trunk_flat | fp32_high | 498.804 | 184.427 | 23.970 | 2.395e-05 |
| 8 | triton_bf16compat | all_fourier_scaffold | trunk_flat | fp32_high | 500.745 | 185.412 | 24.070 | 2.484e-05 |
| 9 | triton_bf16compat | ffn_fourier_direct | trunk_flat | fp32_high | 515.808 | 174.942 | 24.358 | 1.796e-05 |
| 10 | triton_bf16compat | ffn_fourier_scaffold | trunk_flat | fp32_high | 516.364 | 174.967 | 24.416 | 1.693e-05 |
| 11 | triton_tf32 | ffn_fourier_direct | trunk_flat | fp32_high | 522.093 | 168.376 | 25.733 | 1.816e-05 |
| 12 | triton_tf32 | ffn_fourier_scaffold | trunk_flat | fp32_high | 522.869 | 169.418 | 25.789 | 1.796e-05 |
| 13 | triton_tf32x3 | all_fourier_direct | trunk_flat | fp32_high | 523.026 | 180.648 | 25.343 | 2.721e-05 |
| 14 | triton_tf32x3 | all_fourier_scaffold | trunk_flat | fp32_high | 524.563 | 182.006 | 25.445 | 2.747e-05 |
| 15 | triton_radial_r2 | all_fourier_direct | trunk_flat | fp32_high | 540.735 | 182.360 | 25.341 | 2.734e-05 |
| 16 | triton_radial_slope | all_fourier_direct | trunk_flat | fp32_high | 541.636 | 182.799 | 25.341 | 2.594e-05 |
| 17 | triton_radial_r2 | all_fourier_scaffold | trunk_flat | fp32_high | 542.395 | 183.149 | 25.442 | 2.589e-05 |
| 18 | triton_radial_slope | all_fourier_scaffold | trunk_flat | fp32_high | 542.472 | 183.198 | 25.442 | 2.738e-05 |
| 19 | triton_tf32x3 | ffn_fourier_scaffold | trunk_flat | fp32_high | 557.603 | 176.606 | 25.789 | 1.787e-05 |
| 20 | triton_tf32x3 | ffn_fourier_direct | trunk_flat | fp32_high | 558.398 | 176.266 | 25.733 | 1.823e-05 |
