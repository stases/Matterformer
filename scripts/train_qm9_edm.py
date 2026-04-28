#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from torch.utils.data import DataLoader

from matterformer.data import QM9Dataset, collate_qm9
from matterformer.metrics import build_rdkit_metrics, sample_and_evaluate_qm9
from matterformer.models import QM9EDMModel
from matterformer.tasks import EDMLoss, EDMPreconditioner
from matterformer.utils import (
    CosineWarmupScheduler,
    EMA,
    default_device,
    random_rotation_matrices,
    seed_everything,
)


def make_autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def evaluate_loss(
    net: EDMPreconditioner,
    criterion: EDMLoss,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None = None,
    bf16_enabled: bool = False,
) -> tuple[float, float, float]:
    net.eval()
    total_loss = 0.0
    total_atom_loss = 0.0
    total_coord_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            with make_autocast_context(device, bf16_enabled):
                loss, diagnostics = criterion(net, batch)
            batch_size = int(batch.num_atoms.shape[0])
            total_loss += loss.item() * batch_size
            total_atom_loss += diagnostics["atom_loss"].mean().item() * batch_size
            total_coord_loss += diagnostics["coord_loss"].mean().item() * batch_size
            count += batch_size
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break
    denom = max(count, 1)
    return total_loss / denom, total_atom_loss / denom, total_coord_loss / denom


def maybe_init_wandb(args: argparse.Namespace):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is required unless --wandb-mode disabled is used") from exc

    wandb_dir = Path(args.wandb_dir)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        group=args.wandb_group,
        dir=str(wandb_dir),
        mode=args.wandb_mode,
        config=vars(args),
    )


def sampler_kwargs_from_args(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "num_steps": args.sample_num_steps,
        "sigma_min": args.sample_sigma_min,
        "sigma_max": args.sample_sigma_max,
        "rho": args.sample_rho,
        "s_churn": args.sample_s_churn,
        "s_min": args.sample_s_min,
        "s_max": args.sample_s_max,
        "s_noise": args.sample_s_noise,
    }


def maybe_configure_wandb(
    run,
    args: argparse.Namespace,
    *,
    train_dataset: QM9Dataset,
    val_dataset: QM9Dataset,
    test_dataset: QM9Dataset,
    model: QM9EDMModel,
) -> None:
    if run is None:
        return
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    num_trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    run.config.update(
        {
            "dataset": {
                "name": "QM9",
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "test_samples": len(test_dataset),
            },
            "model": {
                "attn_type": args.attn_type,
                "simplicial_geom_mode": args.simplicial_geom_mode,
                "simplicial_angle_rank": args.simplicial_angle_rank,
                "simplicial_message_mode": args.simplicial_message_mode,
                "simplicial_message_rank": args.simplicial_message_rank,
                "mha_geom_bias_mode": args.mha_geom_bias_mode,
                "simplicial_impl": args.simplicial_impl,
                "simplicial_precision": args.simplicial_precision,
                "num_parameters": num_parameters,
                "num_trainable_parameters": num_trainable_parameters,
            },
            "logging": {
                "log_every_steps": args.log_every_steps,
                "val_estimate_every_steps": args.val_estimate_every_steps,
                "val_estimate_batches": args.val_estimate_batches,
                "full_val_every_steps": args.full_val_every_steps,
                "approx_metrics_every_steps": args.approx_metrics_every_steps,
                "approx_metrics_num_molecules": args.approx_metrics_num_molecules,
                "precise_metrics_every_steps": args.precise_metrics_every_steps,
                "precise_metrics_num_molecules": args.precise_metrics_num_molecules,
                "sample_batch_size": args.sample_batch_size,
                "warmup_steps": args.warmup_steps,
                "max_steps": args.max_steps,
            },
            "ema": {
                "decay": args.ema_decay,
                "use_for_sampling": args.ema_use_for_sampling,
            },
            "sampler": sampler_kwargs_from_args(args),
            "checkpointing": {
                "save_checkpoint": args.save_checkpoint,
                "output": args.output,
                "selector": args.best_checkpoint_selector,
                "molecule_stability_weight": args.checkpoint_molecule_stability_weight,
                "validity_weight": args.checkpoint_validity_weight,
                "uniqueness_weight": args.checkpoint_uniqueness_weight,
            },
        },
        allow_val_change=True,
    )
    run.summary["dataset/train_samples"] = len(train_dataset)
    run.summary["dataset/val_samples"] = len(val_dataset)
    run.summary["dataset/test_samples"] = len(test_dataset)
    run.summary["model/num_parameters"] = num_parameters
    run.summary["model/num_trainable_parameters"] = num_trainable_parameters


def apply_rotation_augmentation(batch, enabled: bool):
    if not enabled:
        return batch
    rotations = random_rotation_matrices(
        batch.coords.shape[0],
        device=batch.coords.device,
        dtype=batch.coords.dtype,
    )
    batch.coords = torch.einsum("bij,bnj->bni", rotations, batch.coords)
    batch.coords = batch.coords.masked_fill(batch.pad_mask[..., None], 0.0)
    return batch


def initialize_model_parameters(
    net: EDMPreconditioner,
    criterion: EDMLoss,
    dataset: QM9Dataset,
    device: torch.device,
    *,
    bf16_enabled: bool = False,
) -> None:
    if len(dataset) == 0:
        raise ValueError("Cannot initialize model parameters from an empty dataset")
    init_batch = collate_qm9([dataset[0]]).to(device)
    with torch.no_grad():
        with make_autocast_context(device, bf16_enabled):
            _ = criterion(net, init_batch)


def clone_model_state_to_cpu(model: QM9EDMModel) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def maybe_clone_ema_state(ema: EMA | None) -> dict[str, torch.Tensor] | None:
    if ema is None:
        return None
    return ema.state_dict()


def compute_qm9_composite_score(
    metrics: dict[str, float],
    *,
    molecule_stability_weight: float,
    validity_weight: float,
    uniqueness_weight: float,
) -> float:
    molecule_stability = float(metrics.get("molecule_stability", math.nan))
    validity = float(metrics.get("validity", math.nan))
    uniqueness = float(metrics.get("uniqueness", math.nan))
    values = (molecule_stability, validity, uniqueness)
    if not all(math.isfinite(value) for value in values):
        return math.nan
    return (
        molecule_stability_weight * molecule_stability
        + validity_weight * validity
        + uniqueness_weight * uniqueness
    )


def build_checkpoint_payload(
    *,
    model: QM9EDMModel,
    ema: EMA | None,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    args: argparse.Namespace,
    best_val: float,
    best_precise_composite_score: float,
    best_checkpoint_score: float,
    global_step: int,
    epoch: int,
    last_full_val_step: int,
    last_full_val: tuple[float, float, float] | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "args": vars(args),
        "best_val_loss": float(best_val),
        "best_precise_composite_score": float(best_precise_composite_score),
        "best_checkpoint_score": float(best_checkpoint_score),
        "best_checkpoint_selector": args.best_checkpoint_selector,
        "global_step": int(global_step),
        "epoch": int(epoch),
        "last_full_val_step": int(last_full_val_step),
    }
    if last_full_val is not None:
        payload["last_full_val"] = tuple(float(value) for value in last_full_val)
    if ema is not None:
        payload["ema_state_dict"] = ema.state_dict()
        payload["ema_decay"] = args.ema_decay
    return payload


def log_sampling_metrics(
    run,
    model: QM9EDMModel,
    ema: EMA | None,
    net: EDMPreconditioner,
    num_atoms_sampler,
    device: torch.device,
    *,
    global_step: int,
    epoch_float: float,
    split_name: str,
    num_molecules: int,
    sample_batch_size: int,
    sampler_kwargs: dict[str, float | int],
    rdkit_metrics,
    use_ema: bool,
    composite_weights: tuple[float, float, float] | None = None,
    bf16_enabled: bool = False,
) -> dict[str, float]:
    ema_backup = None
    if use_ema and ema is not None:
        ema_backup = ema.apply(model)
    try:
        with make_autocast_context(device, bf16_enabled):
            metrics = sample_and_evaluate_qm9(
                net,
                num_atoms_sampler,
                device=device,
                num_molecules=num_molecules,
                sample_batch_size=sample_batch_size,
                sampler_kwargs=sampler_kwargs,
                rdkit_metrics=rdkit_metrics,
            )
    finally:
        if ema_backup is not None:
            ema.restore(model, ema_backup)
    if composite_weights is not None:
        metrics["composite_score"] = compute_qm9_composite_score(
            metrics,
            molecule_stability_weight=composite_weights[0],
            validity_weight=composite_weights[1],
            uniqueness_weight=composite_weights[2],
        )
    log_payload = {
        "trainer/global_step": global_step,
        "trainer/epoch": epoch_float,
    }
    for key, value in metrics.items():
        log_payload[f"{split_name}/{key}"] = value
    if run is not None:
        run.log(log_payload, step=global_step)
    else:
        metrics_str = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print(f"step={global_step:07d} {split_name} {metrics_str}")
    return metrics


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = default_device()

    train_dataset = QM9Dataset(args.data_dir, split="train")
    val_dataset = QM9Dataset(args.data_dir, split="val")
    test_dataset = QM9Dataset(args.data_dir, split="test")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_qm9,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_qm9,
        num_workers=args.num_workers,
    )
    num_atoms_sampler = train_dataset.make_num_atoms_sampler()
    rdkit_metrics = build_rdkit_metrics(train_dataset.smiles_list)
    run = maybe_init_wandb(args)

    model = QM9EDMModel(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        attn_type=args.attn_type,
        simplicial_geom_mode=args.simplicial_geom_mode,
        simplicial_impl=args.simplicial_impl,
        simplicial_precision=args.simplicial_precision,
        simplicial_angle_rank=args.simplicial_angle_rank,
        simplicial_message_mode=args.simplicial_message_mode,
        simplicial_message_rank=args.simplicial_message_rank,
        mha_geom_bias_mode=args.mha_geom_bias_mode,
        use_geometry_bias=not args.disable_geometry_bias,
    ).to(device)
    net = EDMPreconditioner(model, sigma_data=args.sigma_data).to(device)
    criterion = EDMLoss(
        sigma_data=args.sigma_data,
        atom_feature_scale=args.atom_feature_scale,
        p_mean=args.p_mean,
        p_std=args.p_std,
    )
    initialize_model_parameters(net, criterion, train_dataset, device, bf16_enabled=args.bf16)
    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0.0 else None

    maybe_configure_wandb(
        run,
        args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        model=model,
    )
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    num_trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        "dataset: "
        f"train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}"
    )
    print(
        "model: "
        f"num_parameters={num_parameters} num_trainable_parameters={num_trainable_parameters}"
    )
    print(
        "schedule: "
        f"warmup_steps={args.warmup_steps} max_steps={args.max_steps}"
    )
    print(
        "sampler: "
        f"num_steps={args.sample_num_steps} sigma_min={args.sample_sigma_min} "
        f"sigma_max={args.sample_sigma_max} rho={args.sample_rho} "
        f"s_churn={args.sample_s_churn} s_min={args.sample_s_min} "
        f"s_max={args.sample_s_max} s_noise={args.sample_s_noise}"
    )
    print(
        "metrics: "
        f"approx_every={args.approx_metrics_every_steps} approx_num_molecules={args.approx_metrics_num_molecules} "
        f"precise_every={args.precise_metrics_every_steps} precise_num_molecules={args.precise_metrics_num_molecules} "
        f"sample_batch_size={args.sample_batch_size} rdkit_available={rdkit_metrics is not None}"
    )
    print(
        "ema: "
        f"decay={args.ema_decay} use_for_sampling={args.ema_use_for_sampling}"
    )
    print(f"precision: bf16={args.bf16}")
    print(
        "checkpointing: "
        f"save_checkpoint={args.save_checkpoint} output={args.output} "
        f"selector={args.best_checkpoint_selector}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup=args.warmup_steps,
        max_iters=args.max_steps,
    )

    checkpoint_path = Path(args.output)
    if args.save_checkpoint:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_precise_composite_score = float("-inf")
    best_state_in_memory: dict[str, torch.Tensor] | None = None
    best_ema_state_in_memory: dict[str, torch.Tensor] | None = None
    global_step = 0
    last_full_val_step = -1
    last_full_val: tuple[float, float, float] | None = None
    sampler_kwargs = sampler_kwargs_from_args(args)
    composite_weights = (
        args.checkpoint_molecule_stability_weight,
        args.checkpoint_validity_weight,
        args.checkpoint_uniqueness_weight,
    )
    batches_per_epoch = len(train_loader)
    epoch = 0
    best_checkpoint_score = (
        float("inf") if args.best_checkpoint_selector == "val_loss" else float("-inf")
    )

    if args.resume_checkpoint is not None and args.warm_start_checkpoint is not None:
        raise ValueError("Only one of --resume-checkpoint or --warm-start-checkpoint may be set")

    if args.resume_checkpoint is not None:
        resume_path = Path(args.resume_checkpoint)
        checkpoint = torch.load(resume_path, map_location=device)
        required_keys = ("model_state", "optimizer_state", "scheduler_state", "global_step")
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise KeyError(
                f"Checkpoint {resume_path} is missing resume fields: {', '.join(missing_keys)}"
            )
        model.load_state_dict(checkpoint["model_state"])
        if ema is not None and checkpoint.get("ema_state_dict") is not None:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_val = float(checkpoint.get("best_val_loss", float("inf")))
        best_precise_composite_score = float(
            checkpoint.get("best_precise_composite_score", float("-inf"))
        )
        best_checkpoint_score = float(
            checkpoint.get(
                "best_checkpoint_score",
                best_val if args.best_checkpoint_selector == "val_loss" else best_precise_composite_score,
            )
        )
        global_step = int(checkpoint["global_step"])
        epoch = int(checkpoint.get("epoch", 0))
        last_full_val_step = int(checkpoint.get("last_full_val_step", -1))
        checkpoint_last_full_val = checkpoint.get("last_full_val")
        if checkpoint_last_full_val is not None:
            last_full_val = tuple(float(value) for value in checkpoint_last_full_val)
        best_state_in_memory = clone_model_state_to_cpu(model)
        best_ema_state_in_memory = maybe_clone_ema_state(ema)
        print(
            "resume_checkpoint: "
            f"path={resume_path} global_step={global_step} epoch={epoch} "
            f"best_val_loss={best_val:.6f} "
            f"best_precise_composite_score={best_precise_composite_score:.6f}"
        )
        if (
            args.save_checkpoint
            and checkpoint_path.expanduser().resolve() != resume_path.expanduser().resolve()
        ):
            torch.save(checkpoint, checkpoint_path)
    elif args.warm_start_checkpoint is not None:
        warm_start_path = Path(args.warm_start_checkpoint)
        checkpoint = torch.load(warm_start_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if ema is not None and checkpoint.get("ema_state_dict") is not None:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        print(
            "warm_start_checkpoint: "
            f"path={warm_start_path} source_best_val_loss={float(checkpoint.get('best_val_loss', float('nan'))):.6f}"
        )

    while global_step < args.max_steps:
        epoch += 1
        net.train()
        total_loss = 0.0
        total_atom_loss = 0.0
        total_coord_loss = 0.0
        total_samples = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            if global_step >= args.max_steps:
                break
            batch = batch.to(device)
            batch = apply_rotation_augmentation(batch, enabled=args.train_augm)
            with make_autocast_context(device, args.bf16):
                loss, diagnostics = criterion(net, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            global_step += 1

            batch_size = int(batch.num_atoms.shape[0])
            total_loss += loss.item() * batch_size
            total_atom_loss += diagnostics["atom_loss"].mean().item() * batch_size
            total_coord_loss += diagnostics["coord_loss"].mean().item() * batch_size
            total_samples += batch_size

            epoch_float = (epoch - 1) + batch_idx / max(batches_per_epoch, 1)
            lr = optimizer.param_groups[0]["lr"]

            if run is not None and args.log_every_steps > 0 and global_step % args.log_every_steps == 0:
                run.log(
                    {
                        "trainer/global_step": global_step,
                        "trainer/epoch": epoch_float,
                        "train/loss": loss.item(),
                        "train/atom_loss": diagnostics["atom_loss"].mean().item(),
                        "train/coord_loss": diagnostics["coord_loss"].mean().item(),
                        "noise/sigma_mean": diagnostics["sigma"].mean().item(),
                        "noise/sigma_std": diagnostics["sigma"].std(unbiased=False).item(),
                        "noise/log_sigma_over_4_mean": diagnostics["log_sigma_over_4"].mean().item(),
                        "noise/log_sigma_over_4_std": diagnostics["log_sigma_over_4"].std(unbiased=False).item(),
                        "optim/grad_norm": float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm),
                        "optim/lr": lr,
                    },
                    step=global_step,
                )

            if (
                run is not None
                and args.val_estimate_every_steps > 0
                and global_step % args.val_estimate_every_steps == 0
            ):
                val_est_loss, val_est_atom_loss, val_est_coord_loss = evaluate_loss(
                    net,
                    criterion,
                    val_loader,
                    device,
                    max_batches=args.val_estimate_batches,
                    bf16_enabled=args.bf16,
                )
                run.log(
                    {
                        "trainer/global_step": global_step,
                        "trainer/epoch": epoch_float,
                        "val_estimate/loss": val_est_loss,
                        "val_estimate/atom_loss": val_est_atom_loss,
                        "val_estimate/coord_loss": val_est_coord_loss,
                    },
                    step=global_step,
                )

            if args.full_val_every_steps > 0 and global_step % args.full_val_every_steps == 0:
                val_loss, val_atom_loss, val_coord_loss = evaluate_loss(
                    net,
                    criterion,
                    val_loader,
                    device,
                    bf16_enabled=args.bf16,
                )
                last_full_val = (val_loss, val_atom_loss, val_coord_loss)
                last_full_val_step = global_step
                if run is not None:
                    run.log(
                        {
                            "trainer/global_step": global_step,
                            "trainer/epoch": epoch_float,
                            "val/loss": val_loss,
                            "val/atom_loss": val_atom_loss,
                            "val/coord_loss": val_coord_loss,
                            "val/best_loss": min(best_val, val_loss),
                            "checkpoint/best_score": (
                                min(best_checkpoint_score, val_loss)
                                if args.best_checkpoint_selector == "val_loss"
                                else best_checkpoint_score
                            ),
                        },
                        step=global_step,
                    )
                if val_loss < best_val:
                    best_val = val_loss
                if args.best_checkpoint_selector == "val_loss" and val_loss < best_checkpoint_score:
                    best_checkpoint_score = val_loss
                    best_payload = build_checkpoint_payload(
                        model=model,
                        ema=ema,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args,
                        best_val=best_val,
                        best_precise_composite_score=best_precise_composite_score,
                        best_checkpoint_score=best_checkpoint_score,
                        global_step=global_step,
                        epoch=epoch,
                        last_full_val_step=last_full_val_step,
                        last_full_val=last_full_val,
                    )
                    best_state_in_memory = clone_model_state_to_cpu(model)
                    best_ema_state_in_memory = maybe_clone_ema_state(ema)
                    if args.save_checkpoint:
                        torch.save(best_payload, checkpoint_path)

            if (
                args.approx_metrics_every_steps > 0
                and global_step % args.approx_metrics_every_steps == 0
            ):
                log_sampling_metrics(
                    run,
                    model,
                    ema,
                    net,
                    num_atoms_sampler,
                    device,
                    global_step=global_step,
                    epoch_float=epoch_float,
                    split_name="approx_metrics",
                    num_molecules=args.approx_metrics_num_molecules,
                    sample_batch_size=args.sample_batch_size,
                    sampler_kwargs=sampler_kwargs,
                    rdkit_metrics=rdkit_metrics,
                    use_ema=args.ema_use_for_sampling,
                    composite_weights=composite_weights,
                    bf16_enabled=args.bf16,
                )

            if (
                args.precise_metrics_every_steps > 0
                and global_step % args.precise_metrics_every_steps == 0
            ):
                precise_metrics = log_sampling_metrics(
                    run,
                    model,
                    ema,
                    net,
                    num_atoms_sampler,
                    device,
                    global_step=global_step,
                    epoch_float=epoch_float,
                    split_name="precise_metrics",
                    num_molecules=args.precise_metrics_num_molecules,
                    sample_batch_size=args.sample_batch_size,
                    sampler_kwargs=sampler_kwargs,
                    rdkit_metrics=rdkit_metrics,
                    use_ema=args.ema_use_for_sampling,
                    composite_weights=composite_weights,
                    bf16_enabled=args.bf16,
                )
                precise_composite_score = float(precise_metrics.get("composite_score", math.nan))
                if math.isfinite(precise_composite_score):
                    best_precise_composite_score = max(
                        best_precise_composite_score,
                        precise_composite_score,
                    )
                    if run is not None:
                        run.log(
                            {
                                "trainer/global_step": global_step,
                                "trainer/epoch": epoch_float,
                                "precise_metrics/best_composite_score": best_precise_composite_score,
                                "checkpoint/best_score": (
                                    best_precise_composite_score
                                    if args.best_checkpoint_selector == "precise_composite"
                                    else best_checkpoint_score
                                ),
                            },
                            step=global_step,
                        )
                    if (
                        args.best_checkpoint_selector == "precise_composite"
                        and precise_composite_score > best_checkpoint_score
                    ):
                        best_checkpoint_score = precise_composite_score
                        best_payload = build_checkpoint_payload(
                            model=model,
                            ema=ema,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            args=args,
                            best_val=best_val,
                            best_precise_composite_score=best_precise_composite_score,
                            best_checkpoint_score=best_checkpoint_score,
                            global_step=global_step,
                            epoch=epoch,
                            last_full_val_step=last_full_val_step,
                            last_full_val=last_full_val,
                        )
                        best_state_in_memory = clone_model_state_to_cpu(model)
                        best_ema_state_in_memory = maybe_clone_ema_state(ema)
                        if args.save_checkpoint:
                            torch.save(best_payload, checkpoint_path)

        train_loss = total_loss / max(total_samples, 1)
        train_atom_loss = total_atom_loss / max(total_samples, 1)
        train_coord_loss = total_coord_loss / max(total_samples, 1)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch:03d} "
            f"step={global_step:07d} "
            f"train_epoch_loss={train_loss:.6f} "
            f"train_epoch_atom_loss={train_atom_loss:.6f} "
            f"train_epoch_coord_loss={train_coord_loss:.6f} "
            f"lr={lr:.6e}"
        )
        if run is not None:
            run.log(
                {
                    "trainer/global_step": global_step,
                    "trainer/epoch": float(epoch),
                    "train_epoch/loss": train_loss,
                    "train_epoch/atom_loss": train_atom_loss,
                    "train_epoch/coord_loss": train_coord_loss,
                    "optim/lr": lr,
                },
                step=global_step,
            )

    if last_full_val_step != global_step:
        val_loss, val_atom_loss, val_coord_loss = evaluate_loss(
            net,
            criterion,
            val_loader,
            device,
            bf16_enabled=args.bf16,
        )
        last_full_val = (val_loss, val_atom_loss, val_coord_loss)
        last_full_val_step = global_step
        if run is not None:
            run.log(
                {
                    "trainer/global_step": global_step,
                    "trainer/epoch": float(epoch),
                    "val/loss": val_loss,
                    "val/atom_loss": val_atom_loss,
                    "val/coord_loss": val_coord_loss,
                    "val/best_loss": min(best_val, val_loss),
                    "checkpoint/best_score": (
                        min(best_checkpoint_score, val_loss)
                        if args.best_checkpoint_selector == "val_loss"
                        else best_checkpoint_score
                    ),
                },
                step=global_step,
            )
        if val_loss < best_val:
            best_val = val_loss
        if args.best_checkpoint_selector == "val_loss" and val_loss < best_checkpoint_score:
            best_checkpoint_score = val_loss
            best_payload = build_checkpoint_payload(
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                best_val=best_val,
                best_precise_composite_score=best_precise_composite_score,
                best_checkpoint_score=best_checkpoint_score,
                global_step=global_step,
                epoch=epoch,
                last_full_val_step=last_full_val_step,
                last_full_val=last_full_val,
            )
            best_state_in_memory = clone_model_state_to_cpu(model)
            best_ema_state_in_memory = maybe_clone_ema_state(ema)
            if args.save_checkpoint:
                torch.save(best_payload, checkpoint_path)
    else:
        val_loss, val_atom_loss, val_coord_loss = last_full_val

    print(
        f"final_val step={global_step:07d} "
        f"val_loss={val_loss:.6f} "
        f"val_atom_loss={val_atom_loss:.6f} "
        f"val_coord_loss={val_coord_loss:.6f} "
        f"best_val_loss={best_val:.6f} "
        f"best_precise_composite_score={best_precise_composite_score:.6f} "
        f"best_checkpoint_score={best_checkpoint_score:.6f} "
        f"selector={args.best_checkpoint_selector}"
    )

    if best_state_in_memory is not None:
        model.load_state_dict(best_state_in_memory)
        if ema is not None and best_ema_state_in_memory is not None:
            ema.load_state_dict(best_ema_state_in_memory)
    elif args.save_checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if ema is not None and checkpoint.get("ema_state_dict") is not None:
            ema.load_state_dict(checkpoint["ema_state_dict"])
    if run is not None:
        run.summary["val/best_loss"] = best_val
        run.summary["precise_metrics/best_composite_score"] = best_precise_composite_score
        run.summary["checkpoint/best_selector"] = args.best_checkpoint_selector
        run.summary["checkpoint/best_score"] = best_checkpoint_score
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Matterformer QM9 EDM")
    parser.add_argument("--data-dir", type=str, default="./data/qm9")
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/Matterformer_QM9_edm/checkpoints/qm9_edm.pt",
    )
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--warm-start-checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--attn-type", type=str, default="simplicial", choices=["mha", "simplicial"])
    parser.add_argument(
        "--simplicial-geom-mode",
        type=str,
        default="factorized",
        choices=["none", "factorized", "angle_residual", "angle_low_rank"],
    )
    parser.add_argument(
        "--simplicial-impl",
        type=str,
        default="auto",
        choices=["auto", "torch", "triton"],
    )
    parser.add_argument("--simplicial-angle-rank", type=int, default=16)
    parser.add_argument("--simplicial-message-mode", type=str, default="none", choices=["none", "low_rank"])
    parser.add_argument("--simplicial-message-rank", type=int, default=16)
    parser.add_argument(
        "--simplicial-precision",
        type=str,
        default="bf16_tc",
        choices=["bf16_tc", "tf32", "ieee_fp32"],
    )
    parser.add_argument(
        "--mha-geom-bias-mode",
        type=str,
        default="standard",
        choices=["standard", "factorized_marginal"],
    )
    parser.add_argument("--train-augm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sigma-data", type=float, default=1.0)
    parser.add_argument("--p-mean", type=float, default=-1.2)
    parser.add_argument("--p-std", type=float, default=1.2)
    parser.add_argument("--atom-feature-scale", type=float, default=4.0)
    parser.add_argument("--disable-geometry-bias", action="store_true")
    parser.add_argument("--sample-num-steps", type=int, default=100)
    parser.add_argument("--sample-sigma-min", type=float, default=0.002)
    parser.add_argument("--sample-sigma-max", type=float, default=10.0)
    parser.add_argument("--sample-rho", type=float, default=7.0)
    parser.add_argument("--sample-s-churn", type=float, default=30.0)
    parser.add_argument("--sample-s-min", type=float, default=0.0)
    parser.add_argument("--sample-s-max", type=float, default=float("inf"))
    parser.add_argument("--sample-s-noise", type=float, default=1.003)
    parser.add_argument("--sample-batch-size", type=int, default=128)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--ema-use-for-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--best-checkpoint-selector",
        type=str,
        default="val_loss",
        choices=["val_loss", "precise_composite"],
    )
    parser.add_argument("--checkpoint-molecule-stability-weight", type=float, default=0.60)
    parser.add_argument("--checkpoint-validity-weight", type=float, default=0.30)
    parser.add_argument("--checkpoint-uniqueness-weight", type=float, default=0.10)
    parser.add_argument("--wandb-project", type=str, default="Matterformer_QM9_edm")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--wandb-dir",
        type=str,
        default="./outputs/Matterformer_QM9_edm/wandb",
    )
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--val-estimate-every-steps", type=int, default=500)
    parser.add_argument("--val-estimate-batches", type=int, default=16)
    parser.add_argument("--full-val-every-steps", type=int, default=5000)
    parser.add_argument("--approx-metrics-every-steps", type=int, default=2_000)
    parser.add_argument("--approx-metrics-num-molecules", type=int, default=128)
    parser.add_argument("--precise-metrics-every-steps", type=int, default=10_000)
    parser.add_argument("--precise-metrics-num-molecules", type=int, default=10_000)
    main(parser.parse_args())
