# recipe-traffic-prediction

Predict which recipes will drive high site traffic, so a team knows what to
feature on the homepage. A compact, honest tabular ML pipeline.

The data is **synthetic** (generated here, not taken from any course or platform
dataset), and deliberately includes missing values, because cleaning messy data
is part of the task.

## What it shows

- **Missing-value handling inside the pipeline.** Nutrition columns have gaps;
  median imputation is a pipeline step, so the fill values learned at training
  time are reused at serving time (no leakage, no train/serve skew).
- **Mixed features.** Numeric nutrition (imputed + scaled) and categorical
  fields (one-hot encoded) in one `ColumnTransformer`.
- **The metric that matches the goal.** The business cost is featuring a recipe
  that flops, so **precision on the high-traffic class** is reported next to
  accuracy.

## Run it

```bash
pip install -e .
python scripts/train.py
# dataset: 4000 rows, ~800 missing values
# metrics: {'accuracy': 0.8x, 'precision_high_traffic': 0.8x, 'n_missing_handled': ...}
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Covers that the data really contains gaps, that the pipeline trains straight
through them via imputation, and that precision is reported.

## License

MIT.
