from __future__ import annotations

import argparse
import time

import torch

from matterformer.models.platonic.layers import PlatonicAttention, flash_attn_varlen_func


def _build_cu_seqlens(total_tokens: int, segment_len: int, device: torch.device) -> tuple[torch.Tensor, int]:
    lengths: list[int] = []
    remaining = int(total_tokens)
    while remaining > 0:
        length = min(int(segment_len), remaining)
        lengths.append(length)
        remaining -= length
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(torch.tensor(lengths, dtype=torch.int32, device=device), dim=0)
    return cu, max(lengths) if lengths else 0


def _make_attention(
    args: argparse.Namespace,
    *,
    rope_cache: bool,
    constant_key_fastpath: bool,
    fused_qv: bool = False,
) -> PlatonicAttention:
    return PlatonicAttention(
        d_model=args.d_model,
        num_heads=args.num_heads,
        solid_name="tetrahedron",
        dropout=0.0,
        rope_sigma=args.rope_sigma,
        learned_freqs=True,
        freq_init=args.freq_init,
        use_key=args.use_key,
        rope_on_values=args.rope_on_values,
        attention_backend=args.attention_backend,
        linear_backend=args.linear_backend,
        rope_cache=rope_cache,
        constant_key_fastpath=constant_key_fastpath,
        fused_qv=fused_qv,
    )


def _time_forward(
    fn,
    module: PlatonicAttention,
    x: torch.Tensor,
    pos: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn(x, pos)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            fn(x, pos)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / float(repeats)


def _time_forward_backward(
    fn,
    module: PlatonicAttention,
    x: torch.Tensor,
    pos: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
) -> float:
    x = x.detach().clone().requires_grad_(True)
    for _ in range(warmup):
        module.zero_grad(set_to_none=True)
        x.grad = None
        fn(x, pos).square().mean().backward()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        module.zero_grad(set_to_none=True)
        x.grad = None
        fn(x, pos).square().mean().backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / float(repeats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark cached RoPE/constant-key fastpath for PlatonicAttention.")
    parser.add_argument("--total-tokens", type=int, default=12000)
    parser.add_argument("--segment-len", type=int, default=300)
    parser.add_argument("--d-model", type=int, default=1920)
    parser.add_argument("--num-heads", type=int, default=60)
    parser.add_argument("--rope-sigma", type=float, default=2.0)
    parser.add_argument("--freq-init", choices=["spiral", "random"], default="random")
    parser.add_argument("--use-key", action="store_true")
    parser.add_argument("--no-rope-on-values", dest="rope_on_values", action="store_false")
    parser.set_defaults(rope_on_values=True)
    parser.add_argument("--attention-backend", choices=["sdpa", "flash"], default="flash")
    parser.add_argument("--linear-backend", choices=["spatial", "fourier", "fourier_direct"], default="spatial")
    parser.add_argument("--mode", choices=["forward", "backward", "both"], default="both")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--matmul-precision", choices=["default", "highest", "high", "medium"], default="high")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    if args.matmul_precision != "default":
        torch.set_float32_matmul_precision(args.matmul_precision)

    device = torch.device("cuda")
    torch.manual_seed(0)
    recompute = _make_attention(args, rope_cache=False, constant_key_fastpath=False).to(device)
    cached = _make_attention(args, rope_cache=True, constant_key_fastpath=True).to(device)
    cached.load_state_dict(recompute.state_dict())
    fused = None
    if not args.use_key:
        fused = _make_attention(args, rope_cache=True, constant_key_fastpath=True, fused_qv=True).to(device)
        fused.load_state_dict(cached.state_dict(), strict=False)
        if cached.q_proj is None or cached.v_proj is None:
            raise RuntimeError("Cached reference attention unexpectedly has no separate q/v projections")
        fused.set_fused_qv_from_separate_(cached.q_proj, cached.v_proj)
    recompute.eval()
    cached.eval()
    if fused is not None:
        fused.eval()

    x = torch.randn(args.total_tokens, args.d_model, device=device)
    pos = torch.randn(args.total_tokens, 3, device=device)
    cu_seqlens, max_seqlen = _build_cu_seqlens(args.total_tokens, args.segment_len, device)

    def recompute_fn(inp: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        return recompute.forward_flat(inp, pos=xyz, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

    def cached_fn(inp: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        return cached.forward_flat(inp, pos=xyz, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

    def fused_fn(inp: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        if fused is None:
            raise RuntimeError("fused_qv benchmark is unavailable when use_key=True")
        return fused.forward_flat(inp, pos=xyz, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

    timed_recompute_fn = torch.compile(recompute_fn, mode="default") if args.compile else recompute_fn
    timed_cached_fn = torch.compile(cached_fn, mode="default") if args.compile else cached_fn
    timed_fused_fn = torch.compile(fused_fn, mode="default") if args.compile and fused is not None else fused_fn

    with torch.no_grad():
        y_recompute = timed_recompute_fn(x, pos)
        y_cached = timed_cached_fn(x, pos)
        cache_max_error = (y_recompute - y_cached).abs().max().item()
        fused_max_error = float("nan")
        if fused is not None:
            y_fused = timed_fused_fn(x, pos)
            fused_max_error = (y_cached - y_fused).abs().max().item()

    print(
        "torch",
        torch.__version__,
        "cuda",
        torch.version.cuda,
        "device",
        torch.cuda.get_device_name(0),
        "flash_available",
        flash_attn_varlen_func is not None,
        "compile",
        args.compile,
        "matmul_precision",
        args.matmul_precision,
        "allow_tf32",
        torch.backends.cuda.matmul.allow_tf32,
        "backend",
        args.attention_backend,
        "linear_backend",
        args.linear_backend,
        "use_key",
        args.use_key,
        "rope_on_values",
        args.rope_on_values,
        "total_tokens",
        args.total_tokens,
        "segment_len",
        args.segment_len,
        "cache_max_error",
        f"{cache_max_error:.3e}",
        "fused_max_error",
        f"{fused_max_error:.3e}",
    )

    if args.mode in {"forward", "both"}:
        recompute_ms = _time_forward(
            timed_recompute_fn,
            recompute,
            x,
            pos,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        cached_ms = _time_forward(
            timed_cached_fn,
            cached,
            x,
            pos,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        speedup = recompute_ms / cached_ms if cached_ms > 0 else float("inf")
        print(f"mode=forward recompute={recompute_ms:.3f}ms cached={cached_ms:.3f}ms speedup={speedup:.3f}x")
        if fused is not None:
            fused_ms = _time_forward(
                timed_fused_fn,
                fused,
                x,
                pos,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            cache_to_fused = cached_ms / fused_ms if fused_ms > 0 else float("inf")
            recompute_to_fused = recompute_ms / fused_ms if fused_ms > 0 else float("inf")
            print(
                f"mode=forward cached={cached_ms:.3f}ms fused_qv={fused_ms:.3f}ms "
                f"cache_to_fused={cache_to_fused:.3f}x recompute_to_fused={recompute_to_fused:.3f}x"
            )

    if args.mode in {"backward", "both"}:
        recompute_ms = _time_forward_backward(
            timed_recompute_fn,
            recompute,
            x,
            pos,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        cached_ms = _time_forward_backward(
            timed_cached_fn,
            cached,
            x,
            pos,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        speedup = recompute_ms / cached_ms if cached_ms > 0 else float("inf")
        print(f"mode=fwd_bwd recompute={recompute_ms:.3f}ms cached={cached_ms:.3f}ms speedup={speedup:.3f}x")
        if fused is not None:
            fused_ms = _time_forward_backward(
                timed_fused_fn,
                fused,
                x,
                pos,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            cache_to_fused = cached_ms / fused_ms if fused_ms > 0 else float("inf")
            recompute_to_fused = recompute_ms / fused_ms if fused_ms > 0 else float("inf")
            print(
                f"mode=fwd_bwd cached={cached_ms:.3f}ms fused_qv={fused_ms:.3f}ms "
                f"cache_to_fused={cache_to_fused:.3f}x recompute_to_fused={recompute_to_fused:.3f}x"
            )


if __name__ == "__main__":
    main()
