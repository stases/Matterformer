# Platonic RBF/Type Bias Benchmark

## Attention Parity
- `ieee`: `{'precision': 'ieee', 'triton_available': True, 'forward_max_abs': 5.960464477539062e-07, 'dq_max_abs': 7.152557373046875e-07, 'dk_max_abs': 0.0008777379989624023, 'dv_max_abs': 0.0008928775787353516, 'drbf_weight_max_abs': 2.5033950805664062e-06, 'dtype_bias_max_abs': 1.2218952178955078e-06, 'zero_init_vs_plain_triton_forward_max_abs': 3.5762786865234375e-07}`
- `tf32x3`: `{'precision': 'tf32x3', 'triton_available': True, 'forward_max_abs': 7.152557373046875e-07, 'dq_max_abs': 1.6689300537109375e-06, 'dk_max_abs': 0.0008774995803833008, 'dv_max_abs': 0.0008929967880249023, 'drbf_weight_max_abs': 4.76837158203125e-06, 'dtype_bias_max_abs': 1.2218952178955078e-06, 'zero_init_vs_plain_triton_forward_max_abs': 2.384185791015625e-07}`

## Attention Microbenchmark
| name | mean ms | median ms | peak GiB |
|---|---:|---:|---:|
| attention_flash_varlen_bf16 | 2.378 | 2.367 | 0.262 |
| attention_triton_plain_tf32x3 | 1.803 | 1.803 | 0.235 |
| attention_triton_rbf_type4_zero_tf32x3 | 1.773 | 1.762 | 0.235 |
| attention_triton_rbf_type4_nonzero_tf32x3 | 1.765 | 1.763 | 0.235 |

## Full Model Fwd+Bwd
| variant | compile | mean ms | median ms | peak GiB | energy max diff | force max diff |
|---|---|---:|---:|---:|---:|---:|
