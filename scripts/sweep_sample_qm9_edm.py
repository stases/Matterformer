#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import gc
import math
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from matterformer.data import QM9Dataset
from matterformer.metrics import build_rdkit_metrics, sample_and_evaluate_qm9
from matterformer.models import QM9EDMModel
from matterformer.tasks import EDMPreconditioner
from matterformer.utils import EMA, default_device, seed_everything


DEFAULT_VARIANTS = (
    "base",
    "half_steps",
    "double_steps",
    "zero_churn",
    "churn_1",
    "sigma_max_80",
)
SAMPLER_KEYS = (
    "num_steps",
    "sigma_min",
    "sigma_max",
    "rho",
    "s_churn",
    "s_min",
    "s_max",
    "s_noise",
)
CHECKPOINT_SAMPLER_KEYS = {
    "num_steps": "sample_num_steps",
    "sigma_min": "sample_sigma_min",
    "sigma_max": "sample_sigma_max",
    "rho": "sample_rho",
    "s_churn": "sample_s_churn",
    "s_min": "sample_s_min",
    "s_max": "sample_s_max",
    "s_noise": "sample_s_noise",
}
FALLBACK_SAMPLER = {
    "num_steps": 100,
    "sigma_min": 0.002,
    "sigma_max": 10.0,
    "rho": 7.0,
    "s_churn": 30.0,
    "s_min": 0.0,
    "s_max": float("inf"),
    "s_noise": 1.003,
}
PREFERRED_METRICS = (
    "atom_stability",
    "molecule_stability",
    "validity",
    "uniqueness",
    "composite_score",
    "rdkit_available",
)


def make_autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def checkpoint_args_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    raw_args = checkpoint.get("args", {})
    if isinstance(raw_args, dict):
        return raw_args
    return vars(raw_args)


def float_or_inf(value: Any) -> float:
    if isinstance(value, str) and value.lower() == "inf":
        return float("inf")
    return float(value)


def get_checkpoint_value(model_args: dict[str, Any], key: str, default: Any) -> Any:
    return model_args.get(key, default)


def build_base_sampler(args: argparse.Namespace, model_args: dict[str, Any]) -> dict[str, float | int]:
    base: dict[str, float | int] = {}
    for key in SAMPLER_KEYS:
        checkpoint_key = CHECKPOINT_SAMPLER_KEYS[key]
        cli_value = getattr(args, key)
        value = cli_value
        if value is None:
            value = get_checkpoint_value(model_args, checkpoint_key, FALLBACK_SAMPLER[key])
        if key == "num_steps":
            base[key] = int(value)
        else:
            base[key] = float_or_inf(value)
    return base


def parse_variants(raw_variants: str) -> list[str]:
    if raw_variants.strip().lower() == "all":
        return list(DEFAULT_VARIANTS)
    variants = [variant.strip() for variant in raw_variants.split(",") if variant.strip()]
    if not variants:
        raise ValueError("At least one variant must be specified")
    unknown = sorted(set(variants) - set(DEFAULT_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {', '.join(unknown)}")
    return variants


def variant_sampler(name: str, base: dict[str, float | int]) -> dict[str, float | int]:
    settings = dict(base)
    if name == "base":
        return settings
    if name == "half_steps":
        settings["num_steps"] = max(1, int(settings["num_steps"]) // 2)
    elif name == "double_steps":
        settings["num_steps"] = int(settings["num_steps"]) * 2
    elif name == "zero_churn":
        settings["s_churn"] = 0.0
    elif name == "churn_1":
        settings["s_churn"] = 1.0
    elif name == "sigma_max_80":
        settings["sigma_max"] = 80.0
    else:
        raise ValueError(f"Unknown variant: {name}")
    return settings


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[EDMPreconditioner, dict[str, Any]]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = checkpoint_args_dict(checkpoint)

    model = QM9EDMModel(
        d_model=int(model_args.get("d_model", 256)),
        n_heads=int(model_args.get("n_heads", 8)),
        n_layers=int(model_args.get("n_layers", 8)),
        mlp_ratio=float(model_args.get("mlp_ratio", 4.0)),
        dropout=float(model_args.get("dropout", 0.0)),
        attn_dropout=float(model_args.get("attn_dropout", 0.0)),
        attn_type=str(model_args.get("attn_type", "mha")),
        simplicial_geom_mode=str(model_args.get("simplicial_geom_mode", "factorized")),
        simplicial_angle_rank=int(model_args.get("simplicial_angle_rank", 16)),
        simplicial_impl=args.simplicial_impl or str(model_args.get("simplicial_impl", "auto")),
        simplicial_precision=args.simplicial_precision
        or str(model_args.get("simplicial_precision", "ieee_fp32")),
        mha_geom_bias_mode=str(model_args.get("mha_geom_bias_mode", "standard")),
        use_geometry_bias=not bool(model_args.get("disable_geometry_bias", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    if args.sample_mode == "ema":
        ema_state = checkpoint.get("ema_state_dict")
        if ema_state is None:
            raise ValueError(f"Checkpoint has no EMA state: {args.checkpoint}")
        ema = EMA(model, decay=float(model_args.get("ema_decay", 0.9999)))
        ema.load_state_dict(ema_state)
        ema.apply(model)
        del ema

    net = EDMPreconditioner(
        model,
        sigma_data=float(model_args.get("sigma_data", 1.0)),
    ).to(device)
    net.eval()

    del checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return net, model_args


def composite_score(
    metrics: dict[str, float],
    *,
    molecule_stability_weight: float,
    validity_weight: float,
    uniqueness_weight: float,
) -> float:
    molecule_stability = float(metrics.get("molecule_stability", math.nan))
    validity = float(metrics.get("validity", math.nan))
    uniqueness = float(metrics.get("uniqueness", math.nan))
    if not all(math.isfinite(value) for value in (molecule_stability, validity, uniqueness)):
        return math.nan
    return (
        molecule_stability_weight * molecule_stability
        + validity_weight * validity
        + uniqueness_weight * uniqueness
    )


def metric_order(metric_keys: set[str]) -> list[str]:
    ordered = [metric for metric in PREFERRED_METRICS if metric in metric_keys]
    ordered.extend(sorted(metric for metric in metric_keys if metric not in ordered))
    return ordered


def finite_stats(values: list[float]) -> tuple[float, float, int]:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return math.nan, math.nan, 0
    mean = sum(finite_values) / len(finite_values)
    if len(finite_values) == 1:
        return mean, 0.0, 1
    variance = sum((value - mean) ** 2 for value in finite_values) / (len(finite_values) - 1)
    return mean, math.sqrt(variance), len(finite_values)


def raw_fieldnames() -> list[str]:
    return [
        "checkpoint",
        "setting",
        "repeat_index",
        "seed",
        "num_molecules",
        "sample_batch_size",
        "sample_mode",
        "elapsed_sec",
        *SAMPLER_KEYS,
        "metric",
        "value",
    ]


def append_raw_metrics(
    path: Path,
    *,
    row_prefix: dict[str, object],
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fieldnames())
        if write_header:
            writer.writeheader()
        for metric, value in metrics.items():
            writer.writerow({**row_prefix, "metric": metric, "value": value})


def write_summary(
    path: Path,
    *,
    rows: list[dict[str, object]],
    metric_keys: set[str],
) -> None:
    metrics = metric_order(metric_keys)
    fieldnames = [
        "checkpoint",
        "setting",
        "repeats_requested",
        "repeats_completed",
        "num_molecules",
        "sample_batch_size",
        "sample_mode",
        *SAMPLER_KEYS,
    ]
    for metric in metrics:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_n"])

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp_path.replace(path)


def build_summary_row(
    *,
    args: argparse.Namespace,
    setting: str,
    settings: dict[str, float | int],
    repeat_metrics: list[dict[str, float]],
    metric_keys: set[str],
) -> dict[str, object]:
    row: dict[str, object] = {
        "checkpoint": str(Path(args.checkpoint)),
        "setting": setting,
        "repeats_requested": args.repeats,
        "repeats_completed": len(repeat_metrics),
        "num_molecules": args.num_molecules,
        "sample_batch_size": args.sample_batch_size,
        "sample_mode": args.sample_mode,
    }
    row.update(settings)
    for metric in metric_order(metric_keys):
        values = [metrics.get(metric, math.nan) for metrics in repeat_metrics]
        mean, std, count = finite_stats(values)
        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = std
        row[f"{metric}_n"] = count
    return row


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main(args: argparse.Namespace) -> None:
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.num_molecules <= 0:
        raise ValueError("--num-molecules must be positive")

    args.summary_csv = str(Path(args.summary_csv))
    if args.raw_csv is None:
        summary_path = Path(args.summary_csv)
        args.raw_csv = str(summary_path.with_name(f"{summary_path.stem}_raw.csv"))

    device = default_device()
    print(f"device={device}", flush=True)
    print(f"checkpoint={args.checkpoint}", flush=True)
    net, model_args = load_model(args, device)
    bf16_enabled = bool(model_args.get("bf16", False)) if args.bf16 is None else bool(args.bf16)

    if args.sample_batch_size is None:
        args.sample_batch_size = int(model_args.get("sample_batch_size", 128))

    base_sampler = build_base_sampler(args, model_args)
    variants = parse_variants(args.variants)
    variant_settings = [(variant, variant_sampler(variant, base_sampler)) for variant in variants]

    train_dataset = QM9Dataset(args.data_dir, split="train")
    num_atoms_sampler = train_dataset.make_num_atoms_sampler()
    rdkit_metrics = build_rdkit_metrics(train_dataset.smiles_list)
    print(
        "dataset: "
        f"train={len(train_dataset)} rdkit_available={rdkit_metrics is not None} "
        f"repeats={args.repeats} num_molecules={args.num_molecules} "
        f"sample_batch_size={args.sample_batch_size} bf16={bf16_enabled}",
        flush=True,
    )
    print(f"variants={','.join(variants)}", flush=True)
    print(f"summary_csv={args.summary_csv}", flush=True)
    print(f"raw_csv={args.raw_csv}", flush=True)

    molecule_weight = (
        args.composite_molecule_stability_weight
        if args.composite_molecule_stability_weight is not None
        else float(model_args.get("checkpoint_molecule_stability_weight", 0.60))
    )
    validity_weight = (
        args.composite_validity_weight
        if args.composite_validity_weight is not None
        else float(model_args.get("checkpoint_validity_weight", 0.30))
    )
    uniqueness_weight = (
        args.composite_uniqueness_weight
        if args.composite_uniqueness_weight is not None
        else float(model_args.get("checkpoint_uniqueness_weight", 0.10))
    )

    summary_rows: list[dict[str, object]] = []
    metric_keys: set[str] = set()
    raw_path = Path(args.raw_csv)
    if raw_path.exists() and not args.append:
        raw_path.unlink()

    for setting_index, (setting, sampler_kwargs) in enumerate(variant_settings):
        repeat_metrics: list[dict[str, float]] = []
        print(f"setting={setting} sampler={sampler_kwargs}", flush=True)
        for repeat_index in range(args.repeats):
            seed = args.base_seed + setting_index * 1000 + repeat_index
            seed_everything(seed)
            maybe_sync(device)
            start_time = time.perf_counter()
            with make_autocast_context(device, bf16_enabled):
                metrics = sample_and_evaluate_qm9(
                    net,
                    num_atoms_sampler,
                    device=device,
                    num_molecules=args.num_molecules,
                    sample_batch_size=args.sample_batch_size,
                    sampler_kwargs=sampler_kwargs,
                    rdkit_metrics=rdkit_metrics,
                )
            if args.composite_score:
                metrics["composite_score"] = composite_score(
                    metrics,
                    molecule_stability_weight=molecule_weight,
                    validity_weight=validity_weight,
                    uniqueness_weight=uniqueness_weight,
                )
            maybe_sync(device)
            elapsed_sec = time.perf_counter() - start_time

            metric_keys.update(metrics)
            repeat_metrics.append(metrics)
            row_prefix: dict[str, object] = {
                "checkpoint": str(Path(args.checkpoint)),
                "setting": setting,
                "repeat_index": repeat_index,
                "seed": seed,
                "num_molecules": args.num_molecules,
                "sample_batch_size": args.sample_batch_size,
                "sample_mode": args.sample_mode,
                "elapsed_sec": elapsed_sec,
                **sampler_kwargs,
            }
            append_raw_metrics(raw_path, row_prefix=row_prefix, metrics=metrics)
            metrics_str = " ".join(f"{key}={value:.6f}" for key, value in metrics.items())
            print(
                f"completed setting={setting} repeat={repeat_index} "
                f"seed={seed} elapsed_sec={elapsed_sec:.1f} {metrics_str}",
                flush=True,
            )

        summary_rows.append(
            build_summary_row(
                args=args,
                setting=setting,
                settings=sampler_kwargs,
                repeat_metrics=repeat_metrics,
                metric_keys=metric_keys,
            )
        )
        write_summary(Path(args.summary_csv), rows=summary_rows, metric_keys=metric_keys)
        print(f"updated summary_csv={args.summary_csv}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run repeated QM9 EDM sampling metrics for sampler ablations."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="./data/qm9")
    parser.add_argument("--summary-csv", type=str, required=True)
    parser.add_argument("--raw-csv", type=str, default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--num-molecules", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sample-batch-size", type=int, default=None)
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--sample-mode", type=str, default="ema", choices=["ema", "regular"])
    parser.add_argument("--simplicial-impl", type=str, default=None, choices=["auto", "torch", "triton"])
    parser.add_argument(
        "--simplicial-precision",
        type=str,
        default=None,
        choices=["bf16_tc", "tf32", "ieee_fp32"],
    )
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--sigma-min", type=float, default=None)
    parser.add_argument("--sigma-max", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--s-churn", type=float, default=None)
    parser.add_argument("--s-min", type=float, default=None)
    parser.add_argument("--s-max", type=float, default=None)
    parser.add_argument("--s-noise", type=float, default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--composite-score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--composite-molecule-stability-weight", type=float, default=None)
    parser.add_argument("--composite-validity-weight", type=float, default=None)
    parser.add_argument("--composite-uniqueness-weight", type=float, default=None)
    main(parser.parse_args())
