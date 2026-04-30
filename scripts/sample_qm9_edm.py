#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from matterformer.data import QM9Dataset, QM9_NUM_ATOM_TYPES
from matterformer.metrics import build_rdkit_metrics, sample_and_evaluate_qm9
from matterformer.models import QM9EDMModel
from matterformer.tasks import EDMPreconditioner
from matterformer.utils import EMA, default_device


def main(args: argparse.Namespace) -> None:
    device = default_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint.get("args", {})

    train_dataset = QM9Dataset(args.data_dir, split="train")
    num_atoms_sampler = train_dataset.make_num_atoms_sampler()
    rdkit_metrics = build_rdkit_metrics(train_dataset.smiles_list)

    model = QM9EDMModel(
        atom_channels=QM9_NUM_ATOM_TYPES + (1 if bool(model_args.get("use_charges", False)) else 0),
        d_model=int(model_args.get("d_model", 256)),
        n_heads=int(model_args.get("n_heads", 8)),
        n_layers=int(model_args.get("n_layers", 8)),
        mlp_ratio=float(model_args.get("mlp_ratio", 4.0)),
        dropout=float(model_args.get("dropout", 0.0)),
        attn_dropout=float(model_args.get("attn_dropout", 0.0)),
        attn_type=str(model_args.get("attn_type", "mha")),
        simplicial_geom_mode=str(model_args.get("simplicial_geom_mode", "factorized")),
        simplicial_angle_rank=int(model_args.get("simplicial_angle_rank", 16)),
        simplicial_message_mode=str(model_args.get("simplicial_message_mode", "none")),
        simplicial_message_rank=int(model_args.get("simplicial_message_rank", 16)),
        simplicial_impl=args.simplicial_impl or str(model_args.get("simplicial_impl", "auto")),
        simplicial_precision=args.simplicial_precision
        or str(model_args.get("simplicial_precision", "ieee_fp32")),
        mha_geom_bias_mode=str(model_args.get("mha_geom_bias_mode", "standard")),
        mha_position_mode=str(model_args.get("mha_position_mode", "none")),
        mha_rope_freq_sigma=float(model_args.get("mha_rope_freq_sigma", 1.0)),
        mha_rope_learned_freqs=bool(model_args.get("mha_rope_learned_freqs", False)),
        mha_rope_use_key=bool(model_args.get("mha_rope_use_key", True)),
        mha_rope_on_values=bool(model_args.get("mha_rope_on_values", False)),
        use_geometry_bias=not bool(model_args.get("disable_geometry_bias", False)),
        coord_embed_mode=str(model_args.get("coord_embed_mode", "none")),
        coord_n_freqs=int(model_args.get("coord_n_freqs", 32)),
        coord_rff_dim=(
            None
            if model_args.get("coord_rff_dim", None) is None
            else int(model_args.get("coord_rff_dim"))
        ),
        coord_rff_sigma=float(model_args.get("coord_rff_sigma", 1.0)),
        coord_embed_normalize=bool(model_args.get("coord_embed_normalize", False)),
        coord_head_mode=str(model_args.get("coord_head_mode", "equivariant")),
        noise_conditioning=model_args.get("noise_conditioning", None),
        concat_sigma_condition=bool(model_args.get("concat_sigma_condition", False)),
        charge_feature_scale=float(model_args.get("charge_feature_scale", 8.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    net = EDMPreconditioner(
        model,
        sigma_data=float(model_args.get("sigma_data", 1.0)),
    ).to(device)
    if args.sample_mode == "ema" and checkpoint.get("ema_state_dict") is not None:
        ema = EMA(model, decay=float(model_args.get("ema_decay", 0.9999)))
        ema.load_state_dict(checkpoint["ema_state_dict"])
        ema.apply(model)
    net.eval()

    sampler_kwargs = {
        "num_steps": args.num_steps,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "rho": args.rho,
        "s_churn": args.s_churn,
        "s_min": args.s_min,
        "s_max": args.s_max,
        "s_noise": args.s_noise,
    }
    print(
        "sampler: "
        f"num_steps={args.num_steps} sigma_min={args.sigma_min} sigma_max={args.sigma_max} "
        f"rho={args.rho} s_churn={args.s_churn} s_min={args.s_min} s_max={args.s_max} "
        f"s_noise={args.s_noise} sample_batch_size={args.sample_batch_size}"
    )
    metrics = sample_and_evaluate_qm9(
        net,
        num_atoms_sampler,
        device=device,
        num_molecules=args.num_molecules,
        sample_batch_size=args.sample_batch_size,
        sampler_kwargs=sampler_kwargs,
        rdkit_metrics=rdkit_metrics,
    )
    for key, value in metrics.items():
        print(f"{key}={value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample QM9 molecules from a Matterformer EDM checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="./data/qm9")
    parser.add_argument("--num-molecules", type=int, default=128)
    parser.add_argument("--sample-batch-size", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--s-churn", type=float, default=30.0)
    parser.add_argument("--s-min", type=float, default=0.0)
    parser.add_argument("--s-max", type=float, default=float("inf"))
    parser.add_argument("--s-noise", type=float, default=1.003)
    parser.add_argument("--sample-mode", type=str, default="ema", choices=["ema", "regular"])
    parser.add_argument("--simplicial-impl", type=str, default=None, choices=["auto", "torch", "triton"])
    parser.add_argument(
        "--simplicial-precision",
        type=str,
        default=None,
        choices=["bf16_tc", "tf32", "ieee_fp32"],
    )
    main(parser.parse_args())
