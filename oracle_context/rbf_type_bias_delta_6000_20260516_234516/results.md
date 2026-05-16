# Platonic RBF/Type Bias Benchmark

## Attention Parity
- `triton_available`: `True`
- `forward_max_abs`: `7.152557373046875e-07`
- `dq_max_abs`: `1.6689300537109375e-06`
- `dk_max_abs`: `0.0008774995803833008`
- `dv_max_abs`: `0.0008929967880249023`
- `drbf_weight_max_abs`: `5.7220458984375e-06`
- `dtype_bias_max_abs`: `1.2218952178955078e-06`
- `zero_init_vs_plain_triton_forward_max_abs`: `2.384185791015625e-07`

## Attention Microbenchmark
| name | mean ms | median ms | peak GiB |
|---|---:|---:|---:|
| attention_flash_varlen_bf16 | 2.380 | 2.398 | 0.486 |
| attention_triton_plain_tf32x3 | 1.753 | 1.737 | 0.427 |
| attention_triton_rbf_type4_zero_tf32x3 | 2.658 | 2.646 | 0.427 |
| attention_triton_rbf_type4_nonzero_tf32x3 | 2.636 | 2.627 | 0.427 |

## Full Model Fwd+Bwd
| variant | compile | mean ms | median ms | peak GiB | energy max diff | force max diff |
|---|---|---:|---:|---:|---:|---:|
| flash_fast | none | 360.663 | 360.693 | 28.818 | 2.491e-05 | 1.568e-05 |
| triton_plain | none | 372.672 | 372.835 | 29.507 | 5.925e-05 | 2.002e-05 |
| local_rbf_type_every4 | none | 372.721 | 373.083 | 28.991 | 4.590e-05 | 1.925e-05 |
| local_rbf_type_all | none | 411.313 | 410.865 | 29.509 | 5.925e-05 | 1.921e-05 |
| flash_fast | trunk_flat | 265.037 | 262.814 | 28.084 | 4.148e-05 | 1.682e-05 |
| triton_plain | trunk_flat | 296.773 | 296.062 | 28.736 | 5.639e-05 | 2.167e-05 |
| local_rbf_type_every4 | trunk_flat | 283.846 | 282.252 | 27.704 | 4.876e-05 | 1.862e-05 |
| local_rbf_type_all | trunk_flat | 330.455 | 329.985 | 28.741 | 5.126e-05 | 2.114e-05 |
