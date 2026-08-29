from recipe_traffic import make_dataset, train


def test_dataset_has_missing_values():
    df = make_dataset(n=1000, seed=1, missing_rate=0.1)
    assert df[["calories", "carbohydrate", "sugar", "protein"]].isna().sum().sum() > 0


def test_model_trains_through_missing_values():
    pipe, metrics = train(make_dataset(n=3000, seed=1))
    # The pipeline imputes, so training never sees a NaN and the model learns.
    assert metrics["accuracy"] > 0.70
    assert metrics["n_missing_handled"] > 0


def test_precision_is_reported():
    _, metrics = train(make_dataset(n=2000, seed=2))
    assert 0.0 <= metrics["precision_high_traffic"] <= 1.0
