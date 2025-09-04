from __future__ import annotations
from typing import Tuple
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_iris_df() -> Tuple[pd.DataFrame, pd.Series]:
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    return X, y


def train_test_split_df(X, y, test_size: float, random_state: int):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test