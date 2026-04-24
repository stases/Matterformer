#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from matterformer.data.geom_drugs_processing import prepare_geom_drugs_dataset


def main(args: argparse.Namespace) -> None:
    report = prepare_geom_drugs_dataset(
        data_root=args.data_dir,
        force_download=args.force_download,
        force_clean=args.force_clean,
        force_processed=args.force_processed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download, clean, and process GEOM-Drugs for Matterformer")
    parser.add_argument("--data-dir", type=str, default="./data/geom_drugs")
    parser.add_argument("--force-download", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force-clean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force-processed", action=argparse.BooleanOptionalAction, default=False)
    main(parser.parse_args())
