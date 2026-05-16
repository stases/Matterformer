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
| 1 | flash | all_fourier_scaffold | trunk_flat | ampbf16_high | 248.889 | 83.924 | 13.253 | 2.360e-04 |
| 2 | flash | all_fourier_direct | trunk_flat | ampbf16_high | 248.991 | 82.322 | 13.250 | 2.594e-04 |
| 3 | triton_tf32x3 | all_fourier_scaffold | trunk_flat | ampbf16_high | 255.656 | 89.871 | 12.931 | 2.646e-04 |
| 4 | triton_bf16compat | all_fourier_scaffold | trunk_flat | ampbf16_high | 255.897 | 89.509 | 12.931 | 2.642e-04 |
| 5 | triton_bf16compat | all_fourier_direct | trunk_flat | ampbf16_high | 255.941 | 88.463 | 12.929 | 2.589e-04 |
| 6 | triton_tf32 | all_fourier_direct | trunk_flat | ampbf16_high | 256.656 | 88.948 | 12.929 | 2.584e-04 |
| 7 | triton_tf32 | all_fourier_scaffold | trunk_flat | ampbf16_high | 256.758 | 89.880 | 12.931 | 2.642e-04 |
| 8 | triton_ieee | all_fourier_direct | trunk_flat | ampbf16_high | 259.283 | 89.734 | 12.929 | 2.589e-04 |
| 9 | triton_tf32x3 | all_fourier_direct | trunk_flat | ampbf16_high | 259.447 | 89.869 | 12.929 | 2.589e-04 |
| 10 | triton_ieee | all_fourier_scaffold | trunk_flat | ampbf16_high | 259.817 | 90.968 | 12.931 | 2.642e-04 |
| 11 | flash | ffn_fourier_direct | trunk_flat | ampbf16_high | 307.202 | 87.254 | 18.032 | 1.826e-04 |
| 12 | flash | ffn_fourier_scaffold | trunk_flat | ampbf16_high | 307.260 | 88.255 | 18.030 | 1.845e-04 |
| 13 | flash | spatial_fast | trunk_flat | ampbf16_high | 342.321 | 87.404 | 18.418 | 8.309e-05 |
| 14 | triton_bf16compat | ffn_fourier_direct | trunk_flat | ampbf16_high | 349.487 | 111.241 | 18.098 | 1.626e-04 |
| 15 | triton_bf16compat | ffn_fourier_scaffold | trunk_flat | ampbf16_high | 350.867 | 111.507 | 18.097 | 1.435e-04 |
| 16 | triton_tf32 | ffn_fourier_direct | trunk_flat | ampbf16_high | 351.750 | 105.652 | 19.466 | 1.616e-04 |
| 17 | triton_tf32 | ffn_fourier_scaffold | trunk_flat | ampbf16_high | 352.999 | 105.582 | 19.466 | 1.612e-04 |
| 18 | flash | spatial_recompute | trunk_flat | ampbf16_high | 362.202 | 93.765 | 18.503 | 8.297e-05 |
| 19 | triton_tf32x3 | ffn_fourier_direct | trunk_flat | ampbf16_high | 387.656 | 114.262 | 19.466 | 1.659e-04 |
| 20 | triton_tf32x3 | ffn_fourier_scaffold | trunk_flat | ampbf16_high | 388.509 | 112.918 | 19.466 | 1.554e-04 |

## Failed Rows

- `triton_radial_r2` / `all_fourier_scaffold` / `none` / `ampbf16_high`: CompilationError: at 121:12:
                if diag_zero:
                    bias = tl.where(offs_m[:, None] == n[None, :], 0.0, bias)
                scores += bias
            scores = tl.where(m_mask[:, None] & n_mask[None, :], scores, -float("inf"))
            p = tl.exp(scores - lse[:, None])
            p = tl.where(m_mask[:, None] & n_mask[None, :], p, 0.0)
            dp = tl.dot(dout, tl.trans(v), input_precision=input_precision).to(tl.float32)
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(k.dtype), k, input_precision=input_precision) * scale
            dk = tl.dot(tl.trans(ds.to(q.dtype)), q, input_precision=input_precision) * scale
            dv = tl.dot(tl.trans(p.to(dout.dtype)), dout, input_precision=input_precision)
            tl.atomic_add(
            ^
- `triton_radial_r2` / `all_fourier_direct` / `none` / `ampbf16_high`: CompilationError: at 121:12:
                if diag_zero:
                    bias = tl.where(offs_m[:, None] == n[None, :], 0.0, bias)
                scores += bias
            scores = tl.where(m_mask[:, None] & n_mask[None, :], scores, -float("inf"))
            p = tl.exp(scores - lse[:, None])
            p = tl.where(m_mask[:, None] & n_mask[None, :], p, 0.0)
            dp = tl.dot(dout, tl.trans(v), input_precision=input_precision).to(tl.float32)
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(k.dtype), k, input_precision=input_precision) * scale
            dk = tl.dot(tl.trans(ds.to(q.dtype)), q, input_precision=input_precision) * scale
            dv = tl.dot(tl.trans(p.to(dout.dtype)), dout, input_precision=input_precision)
            tl.atomic_add(
            ^
- `triton_radial_slope` / `all_fourier_scaffold` / `none` / `ampbf16_high`: CompilationError: at 121:12:
                if diag_zero:
                    bias = tl.where(offs_m[:, None] == n[None, :], 0.0, bias)
                scores += bias
            scores = tl.where(m_mask[:, None] & n_mask[None, :], scores, -float("inf"))
            p = tl.exp(scores - lse[:, None])
            p = tl.where(m_mask[:, None] & n_mask[None, :], p, 0.0)
            dp = tl.dot(dout, tl.trans(v), input_precision=input_precision).to(tl.float32)
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(k.dtype), k, input_precision=input_precision) * scale
            dk = tl.dot(tl.trans(ds.to(q.dtype)), q, input_precision=input_precision) * scale
            dv = tl.dot(tl.trans(p.to(dout.dtype)), dout, input_precision=input_precision)
            tl.atomic_add(
            ^
- `triton_radial_slope` / `all_fourier_direct` / `none` / `ampbf16_high`: CompilationError: at 121:12:
                if diag_zero:
                    bias = tl.where(offs_m[:, None] == n[None, :], 0.0, bias)
                scores += bias
            scores = tl.where(m_mask[:, None] & n_mask[None, :], scores, -float("inf"))
            p = tl.exp(scores - lse[:, None])
            p = tl.where(m_mask[:, None] & n_mask[None, :], p, 0.0)
            dp = tl.dot(dout, tl.trans(v), input_precision=input_precision).to(tl.float32)
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(k.dtype), k, input_precision=input_precision) * scale
            dk = tl.dot(tl.trans(ds.to(q.dtype)), q, input_precision=input_precision) * scale
            dv = tl.dot(tl.trans(p.to(dout.dtype)), dout, input_precision=input_precision)
            tl.atomic_add(
            ^
- `triton_radial_r2` / `all_fourier_scaffold` / `trunk_flat` / `ampbf16_high`: CompilationError: at 121:12:
                if diag_zero:
                    bias = tl.where(offs_m[:, None] == n[None, :], 0.0, bias)
                scores += bias
            scores = tl.where(m_mask[:, None] & n_mask[None, :], scores, -float("inf"))
            p = tl.exp(scores - lse[:, None])
            p = tl.where(m_mask[:, None] & n_mask[None, :], p, 0.0)
            dp = tl.dot(dout, tl.trans(v), input_precision=input_precision).to(tl.float32)
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(k.dtype), k, input_precision=input_precision) * scale
            dk = tl.dot(tl.trans(ds.to(q.dtype)), q, input_precision=input_precision) * scale
            dv = tl.dot(tl.trans(p.to(dout.dtype)), dout, input_precision=input_precision)
            tl.atomic_add(
            ^
