"""Standalone helpers for the Workflow-CI repository."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "target"


def load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train = pd.read_csv(data_dir / "bank_train.csv")
    test = pd.read_csv(data_dir / "bank_test.csv")
    if TARGET not in train or TARGET not in test:
        raise ValueError("Dataset wajib memiliki target.")
    if "duration" in train or "duration" in test:
        raise ValueError("Leakage column 'duration' ditemukan.")
    x_train = train.drop(columns=[TARGET])
    x_test = test.drop(columns=[TARGET])
    numeric = x_train.select_dtypes(include=["number"]).columns
    x_train[numeric] = x_train[numeric].astype("float64")
    x_test[numeric] = x_test[numeric].astype("float64")
    return x_train, x_test, train[TARGET].astype(int), test[TARGET].astype(int)


def build_pipeline(model: Any, features: pd.DataFrame) -> Pipeline:
    numeric = features.select_dtypes(include=["number"]).columns.tolist()
    categorical = features.select_dtypes(exclude=["number"]).columns.tolist()
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocessor", transformer), ("model", model)])


def metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }
