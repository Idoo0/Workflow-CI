"""Train and log a serving-ready model from an MLflow Project."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

from common import build_pipeline, load_splits, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("bank_preprocessing"))
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--max-depth", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.sklearn.autolog(
        log_models=False,
        log_input_examples=False,
        log_model_signatures=False,
        silent=False,
    )

    x_train, x_test, y_train, y_test = load_splits(args.data_dir.resolve())
    classifier = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model = build_pipeline(classifier, x_train)

    with mlflow.start_run(run_name="ci-retraining") as run:
        mlflow.set_tags(
            {
                "criteria": "3-mlproject-ci",
                "git_sha": os.getenv("GITHUB_SHA", "local"),
                "dataset": "UCI Bank Marketing",
            }
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        evaluation = metrics(y_test, predictions, probabilities)
        mlflow.log_metrics({f"test_{key}": value for key, value in evaluation.items()})

        signature = mlflow.models.infer_signature(x_train, model.predict(x_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=x_train.head(5),
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "tracking_uri": args.tracking_uri,
            "metrics": evaluation,
        }
        (args.output_dir / "run_id.txt").write_text(run.info.run_id, encoding="utf-8")
        (args.output_dir / "metrics.json").write_text(
            json.dumps(evaluation, indent=2), encoding="utf-8"
        )
        (args.output_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        mlflow.log_artifacts(str(args.output_dir), artifact_path="ci-output")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
