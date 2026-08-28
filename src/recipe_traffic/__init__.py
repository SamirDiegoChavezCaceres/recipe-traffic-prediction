"""Predict high-traffic recipes from nutrition and category (synthetic data)."""

from .data import CATEGORICAL, FEATURES, NUMERIC, TARGET, make_dataset
from .model import build_pipeline, train

__all__ = [
    "make_dataset",
    "FEATURES",
    "NUMERIC",
    "CATEGORICAL",
    "TARGET",
    "build_pipeline",
    "train",
]
