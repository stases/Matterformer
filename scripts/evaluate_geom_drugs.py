#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from matterformer.evaluators.geom_drugs import (
    evaluate_generated_geom_drugs,
    load_generated_samples,
    load_or_build_train_reference_smiles,
)


def main(args: argparse.Namespace) -> None:
    generated = load_generated_samples(args.input)
    train_reference_smiles = None
    if args.train_reference_cache is not None:
        with Path(args.train_reference_cache).open("rb") as handle:
            train_reference_smiles = set(pickle.load(handle))
    else:
        train_reference_smiles = load_or_build_train_reference_smiles(data_root=args.data_dir)

    report = evaluate_generated_geom_drugs(
        generated,
        train_reference_smiles=train_reference_smiles,
        data_root=args.data_dir,
    )
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GEOM-Drugs generated samples in Matterformer")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="./data/geom_drugs")
    parser.add_argument("--train-reference-cache", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    main(parser.parse_args())
