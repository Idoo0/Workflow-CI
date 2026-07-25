from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


PROJECT_DIR = Path(__file__).resolve().parents[1] / "MLProject"
sys.path.insert(0, str(PROJECT_DIR))

from common import build_pipeline, metrics  # noqa: E402


def test_serving_pipeline_accepts_raw_features() -> None:
    features = pd.DataFrame(
        {
            "age": [21, 31, 41, 51, 61, 71],
            "balance": [0, 10, 20, 30, 40, 50],
            "job": ["student", "admin", "admin", "services", "retired", None],
        }
    )
    target = pd.Series([0, 0, 0, 1, 1, 1])
    pipeline = build_pipeline(
        RandomForestClassifier(n_estimators=10, random_state=42), features
    )
    pipeline.fit(features, target)
    assert pipeline.predict(features.head(1)).shape == (1,)


def test_metrics_are_finite() -> None:
    result = metrics(
        pd.Series([0, 0, 1, 1]),
        np.array([0, 0, 0, 1]),
        np.array([0.1, 0.2, 0.4, 0.8]),
    )
    assert all(np.isfinite(value) for value in result.values())

