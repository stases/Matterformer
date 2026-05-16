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
| 1 | triton_tf32 | all_fourier_scaffold | none | fp32_highest | 1168.303 | 430.940 | 27.511 | 2.382e-05 |
| 2 | triton_tf32 | all_fourier_direct | none | fp32_highest | 1168.895 | 429.618 | 27.407 | 2.383e-05 |
| 3 | triton_bf16compat | all_fourier_scaffold | none | fp32_highest | 1173.975 | 438.926 | 26.136 | 2.375e-05 |
| 4 | flash | all_fourier_direct | none | fp32_highest | 1177.809 | 431.517 | 26.030 | 2.382e-05 |
| 5 | flash | all_fourier_scaffold | none | fp32_highest | 1179.096 | 434.909 | 26.133 | 2.382e-05 |
| 6 | triton_tf32x3 | all_fourier_scaffold | none | fp32_highest | 1199.423 | 439.692 | 27.511 | 2.382e-05 |
| 7 | triton_tf32x3 | all_fourier_direct | none | fp32_highest | 1199.981 | 434.773 | 27.407 | 2.383e-05 |
| 8 | triton_ieee | all_fourier_direct | none | fp32_highest | 1244.248 | 448.984 | 27.407 | 2.382e-05 |
| 9 | triton_ieee | all_fourier_scaffold | none | fp32_highest | 1245.700 | 449.574 | 27.511 | 2.382e-05 |
| 10 | flash | ffn_fourier_direct | none | fp32_highest | 1270.114 | 475.521 | 26.417 | 2.375e-05 |
| 11 | flash | ffn_fourier_scaffold | none | fp32_highest | 1271.313 | 477.088 | 26.477 | 2.378e-05 |
| 12 | triton_bf16compat | ffn_fourier_direct | none | fp32_highest | 1288.514 | 485.162 | 26.419 | 2.372e-05 |
| 13 | triton_bf16compat | ffn_fourier_scaffold | none | fp32_highest | 1288.701 | 480.635 | 26.479 | 2.373e-05 |
| 14 | triton_tf32 | ffn_fourier_direct | none | fp32_highest | 1288.998 | 474.034 | 27.794 | 2.382e-05 |
| 15 | triton_tf32 | ffn_fourier_scaffold | none | fp32_highest | 1289.257 | 478.511 | 27.853 | 2.382e-05 |
| 16 | triton_tf32x3 | ffn_fourier_scaffold | none | fp32_highest | 1331.355 | 488.639 | 27.853 | 2.383e-05 |
| 17 | triton_tf32x3 | ffn_fourier_direct | none | fp32_highest | 1333.289 | 483.813 | 27.794 | 2.384e-05 |
| 18 | triton_ieee | ffn_fourier_direct | none | fp32_highest | 1365.793 | 495.102 | 27.794 | 2.384e-05 |
| 19 | triton_ieee | ffn_fourier_scaffold | none | fp32_highest | 1367.385 | 495.540 | 27.853 | 2.384e-05 |
| 20 | flash | spatial_fast | none | fp32_highest | 1480.486 | 536.831 | 27.240 | 2.378e-05 |
