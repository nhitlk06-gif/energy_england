#!/usr/bin/env python3
"""Run the full pipeline: clean raw data -> build features -> train models.

Usage:
    python scripts/run_pipeline.py            # full hyperparameters
    python scripts/run_pipeline.py --fast      # lighter/faster hyperparameters
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from energyforecast.pipeline import run_full_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="Use lighter hyperparameters for a quick run.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_full_pipeline(fast_mode=args.fast)


if __name__ == "__main__":
    main()
