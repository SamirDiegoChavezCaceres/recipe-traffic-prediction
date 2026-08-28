"""Training pipeline with missing-value imputation.

The business cares about **precision on the high-traffic class**: if the site
features a recipe the model flagged, it should usually be right. So precision is
reported alongside accuracy. Median imputation lives inside the pipeline, so the
same fill values learned at training time are applied at serving time.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, FEATURES, NUMERIC, TARGET


def build_pipeline() -> Pipeline:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    pre = ColumnTransformer(
        [
            ("num", numeric, NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", clf)])


def train(df: pd.DataFrame) -> Tuple[Pipeline, dict]:
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    metrics = {
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "precision_high_traffic": round(float(precision_score(y_te, pred)), 4),
        "n_missing_handled": int(X[NUMERIC].isna().sum().sum()),
    }
    return pipe, metrics
