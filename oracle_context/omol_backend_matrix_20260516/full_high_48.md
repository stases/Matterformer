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
| 1 | flash | ffn_fourier_direct | trunk_flat | fp32_high | 491.074 | 160.642 | 24.947 | 2.008e-05 |
| 2 | triton_bf16compat | ffn_fourier_direct | trunk_flat | fp32_high | 520.533 | 174.357 | 24.358 | 1.775e-05 |
| 3 | triton_tf32x3 | ffn_fourier_direct | trunk_flat | fp32_high | 559.375 | 175.459 | 25.732 | 1.919e-05 |
| 4 | flash | spatial_fast | trunk_flat | fp32_high | 576.310 | 177.651 | 25.770 | 1.509e-05 |
| 5 | triton_radial_r2 | ffn_fourier_direct | trunk_flat | fp32_high | 582.111 | 179.450 | 25.730 | 1.998e-05 |
| 6 | triton_radial_slope | ffn_fourier_direct | trunk_flat | fp32_high | 583.239 | 179.678 | 25.730 | 1.850e-05 |
| 7 | triton_bf16compat | spatial_fast | trunk_flat | fp32_high | 605.606 | 193.610 | 25.180 | 1.789e-05 |
| 8 | flash | spatial_recompute | trunk_flat | fp32_high | 608.682 | 186.683 | 25.853 | 1.540e-05 |
| 9 | triton_radial_rbf8 | ffn_fourier_direct | trunk_flat | fp32_high | 625.272 | 187.190 | 25.730 | 1.850e-05 |
| 10 | triton_bf16compat | spatial_recompute | trunk_flat | fp32_high | 637.825 | 199.027 | 25.866 | 1.742e-05 |
| 11 | triton_tf32x3 | spatial_fast | trunk_flat | fp32_high | 648.204 | 195.947 | 26.554 | 1.729e-05 |
| 12 | triton_radial_r2 | spatial_fast | trunk_flat | fp32_high | 674.807 | 197.945 | 26.551 | 2.055e-05 |
| 13 | triton_radial_slope | spatial_fast | trunk_flat | fp32_high | 678.401 | 199.791 | 26.551 | 2.048e-05 |
| 14 | triton_tf32x3 | spatial_recompute | trunk_flat | fp32_high | 680.033 | 201.490 | 27.238 | 2.051e-05 |
| 15 | triton_radial_r2 | spatial_recompute | trunk_flat | fp32_high | 705.945 | 204.300 | 27.237 | 1.971e-05 |
| 16 | triton_radial_slope | spatial_recompute | trunk_flat | fp32_high | 710.425 | 204.030 | 27.237 | 2.047e-05 |
| 17 | triton_radial_rbf8 | spatial_fast | trunk_flat | fp32_high | 718.061 | 204.619 | 26.551 | 2.033e-05 |
| 18 | triton_radial_rbf8 | spatial_recompute | trunk_flat | fp32_high | 754.054 | 211.624 | 27.237 | 2.008e-05 |
| 19 | flash | spatial_fast | none | fp32_high | 808.579 | 273.011 | 27.240 | 1.723e-05 |
| 20 | triton_bf16compat | spatial_fast | none | fp32_high | 811.085 | 276.897 | 27.242 | 1.544e-05 |

## Failed Rows

- `flash` / `all_fourier_direct` / `none` / `fp32_high`: ValueError: q, v, and qv projections must use the same linear backend
- `triton_tf32x3` / `all_fourier_direct` / `none` / `fp32_high`: ValueError: q, v, and qv projections must use the same linear backend
- `triton_bf16compat` / `all_fourier_direct` / `none` / `fp32_high`: ValueError: q, v, and qv projections must use the same linear backend
- `triton_radial_r2` / `all_fourier_direct` / `none` / `fp32_high`: ValueError: q, v, and qv projections must use the same linear backend
- `triton_radial_slope` / `all_fourier_direct` / `none` / `fp32_high`: ValueError: q, v, and qv projections must use the same linear backend
- `triton_radial_rbf8` / `all_fourier_direct` / `none` / `fp32_high`: ValueError: q, v, and qv projections must use the same linear backend
- `flash` / `all_fourier_direct` / `trunk_flat` / `fp32_high`: ValueError: q, v, and qv projections must use the same linear backend
- `triton_tf32x3` / `all_fourier_direct` / `trunk_flat` / `fp32_high`: ValueError: q, v, and qv projections must use the same linear backend
- `triton_bf16compat` / `all_fourier_direct` / `trunk_flat` / `fp32_high`: ValueError: q, v, and qv projections must use the same linear backend
- `triton_radial_r2` / `all_fourier_direct` / `trunk_flat` / `fp32_high`: ValueError: q, v, and qv projections must use the same linear backend
- `triton_radial_slope` / `all_fourier_direct` / `trunk_flat` / `fp32_high`: ValueError: q, v, and qv projections must use the same linear backend
- `triton_radial_rbf8` / `all_fourier_direct` / `trunk_flat` / `fp32_high`: ValueError: q, v, and qv projections must use the same linear backend
