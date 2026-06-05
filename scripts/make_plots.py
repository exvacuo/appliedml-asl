"""Make evaluation plots from saved model training histories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asl_detector.evaluation.evaluate_models import make_graphs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path("models/evaluation"),
        help="Directory containing *_accuracy.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/evaluation/plots"),
        help="Directory where plot PNGs should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_paths = make_graphs(args.evaluation_dir, args.output_dir)
    for path in plot_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
