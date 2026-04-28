#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from matterformer.data import QM9_TARGETS, QM9Dataset, collate_qm9, compute_target_stats
from matterformer.models import QM9RegressionModel
from matterformer.utils import (
    CosineWarmupScheduler,
    default_device,
    random_rotation_matrices,
    seed_everything,
)


def make_autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def evaluate(
    model: QM9RegressionModel,
    loader: DataLoader,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
    max_batches: int | None = None,
    bf16_enabled: bool = False,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    mae = 0.0
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            targets_norm = (batch.targets[:, 0] - target_mean[0]) / target_std[0]
            with make_autocast_context(device, bf16_enabled):
                predictions = model(
                    batch.atom_types,
                    batch.coords,
                    batch.pad_mask,
                    lattice=batch.lattice,
                )
            total_loss += F.l1_loss(predictions, targets_norm, reduction="sum").item()
            predictions = predictions * target_std[0] + target_mean[0]
            mae += torch.abs(predictions - batch.targets[:, 0]).sum().item()
            count += batch.targets.shape[0]
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break
    return total_loss / max(count, 1), mae / max(count, 1)


def maybe_init_wandb(args: argparse.Namespace):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is required unless --wandb-mode disabled is used") from exc

    wandb_dir = Path(args.wandb_dir)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        group=args.wandb_group,
        dir=str(wandb_dir),
        mode=args.wandb_mode,
        config=vars(args),
    )
    return run


def maybe_configure_wandb(
    run,
    args: argparse.Namespace,
    *,
    train_dataset: QM9Dataset,
    val_dataset: QM9Dataset,
    test_dataset: QM9Dataset,
    model: QM9RegressionModel,
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
                "target": args.target,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "test_samples": len(test_dataset),
            },
            "model": {
                "attn_type": args.attn_type,
                "simplicial_geom_mode": args.simplicial_geom_mode,
                "simplicial_angle_rank": args.simplicial_angle_rank,
                "simplicial_impl": args.simplicial_impl,
                "simplicial_precision": args.simplicial_precision,
                "readout_mode": args.readout_mode,
                "num_parameters": num_parameters,
                "num_trainable_parameters": num_trainable_parameters,
            },
            "logging": {
                "log_every_steps": args.log_every_steps,
                "val_estimate_every_steps": args.val_estimate_every_steps,
                "val_estimate_batches": args.val_estimate_batches,
                "full_val_every_steps": args.full_val_every_steps,
                "warmup_steps": args.warmup_steps,
                "max_steps": args.max_steps,
            },
            "checkpointing": {
                "save_checkpoint": args.save_checkpoint,
                "output": args.output,
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
    model: QM9RegressionModel,
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
            _ = model(
                init_batch.atom_types,
                init_batch.coords,
                init_batch.pad_mask,
                lattice=init_batch.lattice,
            )


def clone_model_state_to_cpu(model: QM9RegressionModel) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def build_checkpoint_payload(
    *,
    model: QM9RegressionModel,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    args: argparse.Namespace,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    best_val: float,
    global_step: int,
    epoch: int,
    last_full_val_step: int,
    last_full_val: tuple[float, float] | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "args": vars(args),
        "target_mean": target_mean.cpu(),
        "target_std": target_std.cpu(),
        "target": args.target,
        "best_val_mae": float(best_val),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "last_full_val_step": int(last_full_val_step),
    }
    if last_full_val is not None:
        payload["last_full_val"] = tuple(float(value) for value in last_full_val)
    return payload


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = default_device()

    train_dataset = QM9Dataset(args.data_dir, split="train", target=args.target)
    val_dataset = QM9Dataset(args.data_dir, split="val", target=args.target)
    test_dataset = QM9Dataset(args.data_dir, split="test", target=args.target)
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_qm9,
        num_workers=args.num_workers,
    )

    target_mean, target_std = compute_target_stats(train_dataset)
    target_mean = target_mean.to(device)
    target_std = target_std.to(device)
    run = maybe_init_wandb(args)

    model = QM9RegressionModel(
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
        readout_mode=args.readout_mode,
        use_geometry_bias=not args.disable_geometry_bias,
    ).to(device)
    initialize_model_parameters(model, train_dataset, device, bf16_enabled=args.bf16)
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
        f"readout_mode={args.readout_mode} "
        f"num_parameters={num_parameters} num_trainable_parameters={num_trainable_parameters}"
    )
    print(f"precision: bf16={args.bf16}")
    print(
        "schedule: "
        f"warmup_steps={args.warmup_steps} max_steps={args.max_steps}"
    )
    print(
        "checkpointing: "
        f"save_checkpoint={args.save_checkpoint} output={args.output}"
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
    best_state_in_memory: dict[str, torch.Tensor] | None = None
    global_step = 0
    last_full_val_step = -1
    last_full_val: tuple[float, float] | None = None
    batches_per_epoch = len(train_loader)
    epoch = 0

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
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_val = float(checkpoint.get("best_val_mae", float("inf")))
        global_step = int(checkpoint["global_step"])
        epoch = int(checkpoint.get("epoch", 0))
        last_full_val_step = int(checkpoint.get("last_full_val_step", -1))
        checkpoint_last_full_val = checkpoint.get("last_full_val")
        if checkpoint_last_full_val is not None:
            last_full_val = tuple(float(value) for value in checkpoint_last_full_val)
        best_state_in_memory = clone_model_state_to_cpu(model)
        print(
            "resume_checkpoint: "
            f"path={resume_path} global_step={global_step} epoch={epoch} best_val_mae={best_val:.6f}"
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
        print(
            "warm_start_checkpoint: "
            f"path={warm_start_path} source_best_val_mae={float(checkpoint.get('best_val_mae', float('nan'))):.6f}"
        )

    while global_step < args.max_steps:
        epoch += 1
        model.train()
        total_loss = 0.0
        total_mae = 0.0
        total_samples = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            if global_step >= args.max_steps:
                break
            batch = batch.to(device)
            batch = apply_rotation_augmentation(batch, enabled=args.train_augm)
            targets = (batch.targets[:, 0] - target_mean[0]) / target_std[0]
            with make_autocast_context(device, args.bf16):
                predictions = model(
                    batch.atom_types,
                    batch.coords,
                    batch.pad_mask,
                    lattice=batch.lattice,
                )
                loss = F.l1_loss(predictions, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            total_loss += loss.item() * targets.shape[0]
            predictions_denorm = predictions.detach() * target_std[0] + target_mean[0]
            batch_mae = torch.abs(predictions_denorm - batch.targets[:, 0]).mean().item()
            total_mae += torch.abs(predictions_denorm - batch.targets[:, 0]).sum().item()
            total_samples += targets.shape[0]
            epoch_float = (epoch - 1) + batch_idx / max(batches_per_epoch, 1)
            lr = optimizer.param_groups[0]["lr"]

            if run is not None and args.log_every_steps > 0 and global_step % args.log_every_steps == 0:
                run.log(
                    {
                        "trainer/global_step": global_step,
                        "trainer/epoch": epoch_float,
                        "train/loss": loss.item(),
                        "train/mae": batch_mae,
                        "optim/lr": lr,
                    },
                    step=global_step,
                )

            if (
                run is not None
                and args.val_estimate_every_steps > 0
                and global_step % args.val_estimate_every_steps == 0
            ):
                val_est_loss, val_est_mae = evaluate(
                    model,
                    val_loader,
                    target_mean,
                    target_std,
                    device,
                    max_batches=args.val_estimate_batches,
                    bf16_enabled=args.bf16,
                )
                run.log(
                    {
                        "trainer/global_step": global_step,
                        "trainer/epoch": epoch_float,
                        "val_estimate/loss": val_est_loss,
                        "val_estimate/mae": val_est_mae,
                    },
                    step=global_step,
                )

            if args.full_val_every_steps > 0 and global_step % args.full_val_every_steps == 0:
                val_loss, val_mae = evaluate(
                    model,
                    val_loader,
                    target_mean,
                    target_std,
                    device,
                    bf16_enabled=args.bf16,
                )
                last_full_val = (val_loss, val_mae)
                last_full_val_step = global_step
                if run is not None:
                    run.log(
                        {
                            "trainer/global_step": global_step,
                            "trainer/epoch": epoch_float,
                            "val/loss": val_loss,
                            "val/mae": val_mae,
                            "val/best_mae": min(best_val, val_mae),
                        },
                        step=global_step,
                    )
                if val_mae < best_val:
                    best_val = val_mae
                    best_payload = build_checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        args=args,
                        target_mean=target_mean,
                        target_std=target_std,
                        best_val=best_val,
                        global_step=global_step,
                        epoch=epoch,
                        last_full_val_step=last_full_val_step,
                        last_full_val=last_full_val,
                    )
                    best_state_in_memory = clone_model_state_to_cpu(model)
                    if args.save_checkpoint:
                        torch.save(best_payload, checkpoint_path)

        train_loss = total_loss / max(total_samples, 1)
        train_mae = total_mae / max(total_samples, 1)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch:03d} "
            f"step={global_step:07d} "
            f"train_epoch_loss={train_loss:.6f} train_epoch_mae={train_mae:.6f} "
            f"lr={lr:.6e}"
        )
        if run is not None:
            run.log(
                {
                    "trainer/global_step": global_step,
                    "trainer/epoch": float(epoch),
                    "train_epoch/loss": train_loss,
                    "train_epoch/mae": train_mae,
                    "optim/lr": lr,
                },
                step=global_step,
            )

    if last_full_val_step != global_step:
        val_loss, val_mae = evaluate(
            model,
            val_loader,
            target_mean,
            target_std,
            device,
            bf16_enabled=args.bf16,
        )
        last_full_val = (val_loss, val_mae)
        last_full_val_step = global_step
        if run is not None:
            run.log(
                {
                    "trainer/global_step": global_step,
                    "trainer/epoch": float(epoch),
                    "val/loss": val_loss,
                    "val/mae": val_mae,
                    "val/best_mae": min(best_val, val_mae),
                },
                step=global_step,
            )
        if val_mae < best_val:
            best_val = val_mae
            best_payload = build_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                target_mean=target_mean,
                target_std=target_std,
                best_val=best_val,
                global_step=global_step,
                epoch=epoch,
                last_full_val_step=last_full_val_step,
                last_full_val=last_full_val,
            )
            best_state_in_memory = clone_model_state_to_cpu(model)
            if args.save_checkpoint:
                torch.save(best_payload, checkpoint_path)
    else:
        val_loss, val_mae = last_full_val

    print(
        f"final_val step={global_step:07d} "
        f"val_loss={val_loss:.6f} val_mae={val_mae:.6f} "
        f"best_val_mae={best_val:.6f}"
    )

    if best_state_in_memory is not None:
        model.load_state_dict(best_state_in_memory)
    elif args.save_checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    test_loss, test_mae = evaluate(
        model,
        test_loader,
        target_mean,
        target_std,
        device,
        bf16_enabled=args.bf16,
    )
    print(f"best_val_mae={best_val:.6f} test_loss={test_loss:.6f} test_mae={test_mae:.6f}")
    if run is not None:
        run.log(
            {
                "trainer/global_step": global_step,
                "trainer/epoch": float(epoch),
                "test/loss": test_loss,
                "test/mae": test_mae,
                "val/best_mae": best_val,
            },
            step=global_step,
        )
        run.summary["test/loss"] = test_loss
        run.summary["test/mae"] = test_mae
        run.summary["val/best_mae"] = best_val
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Matterformer QM9 regression")
    parser.add_argument("--data-dir", type=str, default="./data/qm9")
    parser.add_argument("--target", type=str, default="homo", choices=QM9_TARGETS)
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/Matterformer_QM9_regr/checkpoints/qm9_regression.pt",
    )
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--warm-start-checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-8)
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
    parser.add_argument(
        "--simplicial-precision",
        type=str,
        default="bf16_tc",
        choices=["bf16_tc", "tf32", "ieee_fp32"],
    )
    parser.add_argument(
        "--readout-mode",
        type=str,
        default="cls",
        choices=["cls", "sum", "mean"],
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-augm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-geometry-bias", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="Matterformer_QM9_regr")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--wandb-dir",
        type=str,
        default="./outputs/Matterformer_QM9_regr/wandb",
    )
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--val-estimate-every-steps", type=int, default=500)
    parser.add_argument("--val-estimate-batches", type=int, default=16)
    parser.add_argument("--full-val-every-steps", type=int, default=5000)
    main(parser.parse_args())
