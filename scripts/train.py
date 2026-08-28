"""Train on synthetic data and print metrics.

    python scripts/train.py
"""

from __future__ import annotations

from recipe_traffic import make_dataset, train


def main() -> None:
    df = make_dataset(n=4000, seed=0)
    print(f"dataset: {len(df)} rows, "
          f"{int(df[['calories','carbohydrate','sugar','protein']].isna().sum().sum())} missing values")
    _, metrics = train(df)
    print("metrics:", metrics)


if __name__ == "__main__":
    main()
