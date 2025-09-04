from __future__ import annotations
import argparse
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils import load_config, read_text


def main(config_path: str):
    cfg = load_config(config_path)
    model_path = read_text(cfg.paths["latest_ptr"])  # absolute path to latest model
    model = joblib.load(model_path)

    df = pd.read_csv(cfg.paths["test_csv"])  # written by train.py
    X_test = df.drop(columns=["target"])
    y_test = df["target"]

    preds = model.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds, average="macro"))
    print({"accuracy": acc, "f1_macro": f1})
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)