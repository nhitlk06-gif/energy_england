#!/usr/bin/env python3
"""Run only the training stage: uk_electricity_features.csv -> models/*.pkl"""
import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from energyforecast.train import run_training_pipeline  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="Use lighter hyperparameters for a quick run.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run_training_pipeline(fast_mode=args.fast)
    print(summary.to_string())
