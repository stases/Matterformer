#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from tqdm import trange

from matterformer.data import (
    SynthMolForceConfig,
    SynthMolForceDataset,
    pack_synthmolforce_samples,
)


def parse_num_atoms(value: str) -> int | tuple[int, int]:
    value = value.strip()
    separator = ":" if ":" in value else "-"
    if separator in value:
        left, right = value.split(separator, maxsplit=1)
        return int(left), int(right)
    return int(value)


def build_config(args: argparse.Namespace, *, length: int) -> SynthMolForceConfig:
    return SynthMolForceConfig(
        level=args.level,
        pair_mode=args.pair_mode,
        num_atoms=parse_num_atoms(args.num_atoms),
        length=length,
        seed=args.seed,
        density=args.density,
        d_min=args.d_min,
        max_resample_attempts=args.max_resample_attempts,
        cutoff=args.cutoff,
        radial_width=args.radial_width,
        pair_sigma=args.pair_sigma,
        pair_tau=args.pair_tau,
        pair_scale=args.pair_scale,
        coord_scale=args.coord_scale,
        angle_scale=args.angle_scale,
        chiral_scale=args.chiral_scale,
    )


def write_split(
    *,
    root: Path,
    base_config: SynthMolForceConfig,
    config_name: str,
    split: str,
    size: int,
    force: bool,
) -> Path:
    output_path = root / config_name / f"{split}.pt"
    if output_path.exists() and not force:
        print(f"exists, skipping: {output_path}")
        return output_path

    split_config = SynthMolForceConfig.from_dict({**asdict(base_config), "length": int(size)})
    dataset = SynthMolForceDataset(
        root,
        split=split,
        config=split_config,
        mode="online",
        config_name=config_name,
    )
    samples = [
        dataset[index]
        for index in trange(size, desc=f"SynthMolForce {split}", unit="sample")
    ]
    packed = pack_synthmolforce_samples(samples, config=split_config, split=split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed, output_path)
    print(f"wrote {size} samples: {output_path}")
    return output_path


def main(args: argparse.Namespace) -> None:
    root = Path(args.root)
    base_config = build_config(args, length=args.train_size)
    config_name = args.config_name or base_config.cache_name()
    write_split(
        root=root,
        base_config=base_config,
        config_name=config_name,
        split="train",
        size=args.train_size,
        force=args.force,
    )
    write_split(
        root=root,
        base_config=base_config,
        config_name=config_name,
        split="val",
        size=args.val_size,
        force=args.force,
    )
    write_split(
        root=root,
        base_config=base_config,
        config_name=config_name,
        split="test",
        size=args.test_size,
        force=args.force,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Materialize fixed SynthMolForce splits.")
    parser.add_argument("--root", type=Path, default=Path("data/synthmolforce"))
    parser.add_argument("--config-name", type=str, default=None)
    parser.add_argument("--level", choices=("v0", "v1", "v2", "v3"), default="v2")
    parser.add_argument("--pair-mode", choices=("complete", "cutoff"), default="cutoff")
    parser.add_argument("--num-atoms", type=str, default="16", help="Fixed count like 16 or range like 8:20.")
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--val-size", type=int, default=2_000)
    parser.add_argument("--test-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--density", type=float, default=0.12)
    parser.add_argument("--d-min", type=float, default=0.25)
    parser.add_argument("--max-resample-attempts", type=int, default=64)
    parser.add_argument("--cutoff", type=float, default=3.0)
    parser.add_argument("--radial-width", type=float, default=0.55)
    parser.add_argument("--pair-sigma", type=float, default=1.1)
    parser.add_argument("--pair-tau", type=float, default=0.45)
    parser.add_argument("--pair-scale", type=float, default=1.0)
    parser.add_argument("--coord-scale", type=float, default=0.08)
    parser.add_argument("--angle-scale", type=float, default=0.025)
    parser.add_argument("--chiral-scale", type=float, default=0.02)
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args())

