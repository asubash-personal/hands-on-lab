from __future__ import annotations
import os
import yaml
from dataclasses import dataclass

@dataclass
class Config:
    experiment_name: str
    paths: dict
    training: dict
    mlflow: dict


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(text)


def read_text(path: str) -> str:
    with open(path, "r") as f:
        return f.read().strip()