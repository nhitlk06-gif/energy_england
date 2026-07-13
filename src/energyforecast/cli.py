"""Command-line entry points (installed as console scripts, see pyproject.toml)."""
from __future__ import annotations

import argparse
import logging

from .cleaning import run_cleaning_pipeline
from .features import run_feature_pipeline
from .pipeline import run_full_pipeline
from .train import run_training_pipeline


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def clean_main() -> None:
    _configure_logging()
    run_cleaning_pipeline()


def features_main() -> None:
    _configure_logging()
    run_feature_pipeline()


def train_main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Train all energyforecast models.")
    parser.add_argument("--fast", action="store_true", help="Use lighter hyperparameters for a quick run.")
    args = parser.parse_args()
    run_training_pipeline(fast_mode=args.fast)


def pipeline_main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Run the full energyforecast pipeline.")
    parser.add_argument("--fast", action="store_true", help="Use lighter hyperparameters for a quick run.")
    args = parser.parse_args()
    run_full_pipeline(fast_mode=args.fast)


if __name__ == "__main__":
    pipeline_main()
