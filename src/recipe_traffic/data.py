"""A synthetic 'recipe site traffic' dataset.

Inspired by a common business problem - predict which recipes will drive high
traffic so the team knows what to feature - but the data here is generated, not
taken from any course or platform dataset. Some nutrition values are left
missing on purpose, because handling missing data is part of the job.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC = ["calories", "carbohydrate", "sugar", "protein"]
CATEGORICAL = ["category", "servings"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "high_traffic"

_CATEGORIES = [
    "Beverages", "Breakfast", "Chicken", "Dessert", "Lunch/Snacks",
    "Meat", "One Dish Meal", "Pork", "Potato", "Vegetable",
]
# Categories that tend to drive traffic (the signal the model should find).
_HIGH_TRAFFIC_CATS = {"Vegetable", "Potato", "Pork", "One Dish Meal"}


def make_dataset(n: int = 2000, seed: int = 0, missing_rate: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    category = rng.choice(_CATEGORIES, n)
    servings = rng.choice([1, 2, 4, 6], n)
    calories = rng.gamma(4.0, 120, n).clip(0, 3000)
    carbohydrate = rng.gamma(2.0, 20, n).clip(0, 400)
    sugar = rng.gamma(1.5, 6, n).clip(0, 150)
    protein = rng.gamma(2.5, 12, n).clip(0, 400)

    logit = 1.4 * (
        -1.5
        + 2.6 * np.isin(category, list(_HIGH_TRAFFIC_CATS))
        + 0.03 * (protein - 30)
        - 0.0015 * (calories - 700)
        + 0.5 * (servings >= 4)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    high_traffic = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "calories": calories.round(2),
            "carbohydrate": carbohydrate.round(2),
            "sugar": sugar.round(2),
            "protein": protein.round(2),
            "category": category,
            "servings": servings,
            TARGET: high_traffic,
        }
    )
    # Punch holes in the nutrition columns to mimic real, messy data.
    for col in NUMERIC:
        mask = rng.random(n) < missing_rate
        df.loc[mask, col] = np.nan
    return df
