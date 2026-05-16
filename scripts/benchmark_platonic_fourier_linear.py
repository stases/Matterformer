from __future__ import annotations

import argparse
import time

import torch

from matterformer.models.platonic import PlatonicLinear


def _time_forward(layer: PlatonicLinear, x: torch.Tensor, *, warmup: int, repeats: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            layer(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            layer(x)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / float(repeats)


def _time_forward_backward(layer: PlatonicLinear, x: torch.Tensor, *, warmup: int, repeats: int) -> float:
    x = x.detach().clone().requires_grad_(True)
    for _ in range(warmup):
        layer.zero_grad(set_to_none=True)
        x.grad = None
        layer(x).square().mean().backward()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        layer.zero_grad(set_to_none=True)
        x.grad = None
        layer(x).square().mean().backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / float(repeats)


def benchmark_shape(tokens: int, cin: int, cout: int, *, warmup: int, repeats: int, mode: str) -> None:
    device = torch.device("cuda")
    spatial = PlatonicLinear(12 * cin, 12 * cout, solid="tetrahedron", linear_backend="spatial").to(device)
    fourier = PlatonicLinear(12 * cin, 12 * cout, solid="tetrahedron", linear_backend="fourier").to(device)
    direct = PlatonicLinear(12 * cin, 12 * cout, solid="tetrahedron", linear_backend="fourier_direct").to(device)
    fourier.load_state_dict(spatial.state_dict())
    direct.set_spatial_parameters_(spatial.kernel, spatial.bias)
    x = torch.randn(tokens, 12 * cin, device=device)
    with torch.no_grad():
        max_error = (spatial(x) - fourier(x)).abs().max().item()
        direct_max_error = (spatial(x) - direct(x)).abs().max().item()
    if mode in {"forward", "both"}:
        spatial_ms = _time_forward(spatial, x, warmup=warmup, repeats=repeats)
        fourier_ms = _time_forward(fourier, x, warmup=warmup, repeats=repeats)
        direct_ms = _time_forward(direct, x, warmup=warmup, repeats=repeats)
        speedup = spatial_ms / fourier_ms if fourier_ms > 0 else float("inf")
        direct_speedup = spatial_ms / direct_ms if direct_ms > 0 else float("inf")
        print(
            f"mode=forward tokens={tokens:6d} cin={cin:4d} cout={cout:4d} "
            f"spatial={spatial_ms:8.3f}ms fourier={fourier_ms:8.3f}ms "
            f"direct={direct_ms:8.3f}ms scaffold_speedup={speedup:6.2f}x "
            f"direct_speedup={direct_speedup:6.2f}x max_error={max_error:.3e} "
            f"direct_max_error={direct_max_error:.3e}"
        )
    if mode in {"backward", "both"}:
        spatial_ms = _time_forward_backward(spatial, x, warmup=warmup, repeats=repeats)
        fourier_ms = _time_forward_backward(fourier, x, warmup=warmup, repeats=repeats)
        direct_ms = _time_forward_backward(direct, x, warmup=warmup, repeats=repeats)
        speedup = spatial_ms / fourier_ms if fourier_ms > 0 else float("inf")
        direct_speedup = spatial_ms / direct_ms if direct_ms > 0 else float("inf")
        print(
            f"mode=fwd_bwd tokens={tokens:6d} cin={cin:4d} cout={cout:4d} "
            f"spatial={spatial_ms:8.3f}ms fourier={fourier_ms:8.3f}ms "
            f"direct={direct_ms:8.3f}ms scaffold_speedup={speedup:6.2f}x "
            f"direct_speedup={direct_speedup:6.2f}x max_error={max_error:.3e} "
            f"direct_max_error={direct_max_error:.3e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark spatial vs Fourier tetra PlatonicLinear.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--tokens", type=int, nargs="*", default=[512, 2048, 8192, 16384])
    parser.add_argument("--mode", choices=["forward", "backward", "both"], default="forward")
    parser.add_argument("--matmul-precision", choices=["default", "highest", "high", "medium"], default="default")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    if args.matmul_precision != "default":
        torch.set_float32_matmul_precision(args.matmul_precision)
    print(
        "torch",
        torch.__version__,
        "cuda",
        torch.version.cuda,
        "device",
        torch.cuda.get_device_name(0),
        "matmul_precision",
        args.matmul_precision,
        "allow_tf32",
        torch.backends.cuda.matmul.allow_tf32,
        "mode",
        args.mode,
    )
    torch.manual_seed(0)
    for tokens in args.tokens:
        benchmark_shape(tokens, 160, 160, warmup=args.warmup, repeats=args.repeats, mode=args.mode)
        benchmark_shape(tokens, 160, 320, warmup=args.warmup, repeats=args.repeats, mode=args.mode)
        benchmark_shape(tokens, 320, 160, warmup=args.warmup, repeats=args.repeats, mode=args.mode)


if __name__ == "__main__":
    main()
