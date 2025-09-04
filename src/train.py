from __future__ import annotations
import argparse
import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from utils import load_config, ensure_dir, write_text
from data_ingest import load_iris_df, train_test_split_df


def build_model(model_type: str, params: dict):
    if model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=params.get("max_iter", 200), C=params.get("C", 1.0))
        # Standardize for LR
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])
    elif model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=params.get("n_estimators", 200), random_state=42)
        return clf
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def main(config_path: str):
    cfg = load_config(config_path)

    # Prepare dirs
    artifact_dir = cfg.paths["artifact_dir"]
    ensure_dir(artifact_dir)

    # Data
    X, y = load_iris_df()
    X_train, X_test, y_train, y_test = train_test_split_df(
        X, y, test_size=cfg.training["test_size"], random_state=cfg.training["random_state"]
    )

    # Persist test set for evaluation/serving sanity checks
    test_csv = cfg.paths["test_csv"]
    pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True).rename("target")], axis=1).to_csv(test_csv, index=False)

    # Model
    model = build_model(cfg.training["model_type"], cfg.training.get("model_params", {}))

    if cfg.mlflow.get("enabled", True):
        mlflow.set_tracking_uri(cfg.mlflow.get("tracking_uri", "./mlruns"))
        mlflow.set_experiment(cfg.experiment_name)
        with mlflow.start_run() as run:
            mlflow.log_params({
                "model_type": cfg.training["model_type"],
                **cfg.training.get("model_params", {})
            })
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = float(accuracy_score(y_test, preds))
            f1 = float(f1_score(y_test, preds, average="macro"))
            mlflow.log_metrics({"accuracy": acc, "f1_macro": f1})
            # Log model
            mlflow.sklearn.log_model(model, artifact_path="model")
            # Also persist a local copy for serving
            model_path = cfg.paths["model_path"]
            joblib.dump(model, model_path)
            write_text(cfg.paths["latest_ptr"], os.path.abspath(model_path))
            print({"run_id": run.info.run_id, "accuracy": acc, "f1_macro": f1, "model_path": model_path})
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, average="macro"))
        model_path = cfg.paths["model_path"]
        joblib.dump(model, model_path)
        write_text(cfg.paths["latest_ptr"], os.path.abspath(model_path))
        print({"accuracy": acc, "f1_macro": f1, "model_path": model_path})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)