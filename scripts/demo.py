"""Walkthrough: train (with missing-value imputation), then sample predictions.

    python scripts/demo.py
"""

from __future__ import annotations

from recipe_traffic import FEATURES, make_dataset, train


def rule(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    df = make_dataset(n=4000, seed=0)
    missing = int(df[["calories", "carbohydrate", "sugar", "protein"]].isna().sum().sum())

    rule("1. Train")
    print(f"  dataset: {len(df)} rows, {missing} missing nutrition values (imputed in-pipeline)")
    pipe, metrics = train(df)
    print(f"  metrics: {metrics}")

    rule("2. Sample predictions (1 = predicted high traffic)")
    sample = make_dataset(n=8, seed=123)
    X = sample[FEATURES]
    proba = pipe.predict_proba(X)[:, 1]
    pred = pipe.predict(X)
    for i in range(len(X)):
        row = X.iloc[i]
        print(f"  {row.category:<13} protein={row.protein:<6} servings={row.servings}  "
              f"->  p(high)={proba[i]:.2f}  pred={pred[i]}")


if __name__ == "__main__":
    main()
