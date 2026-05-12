from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def train_models(X_train: np.ndarray, y_train: np.ndarray) -> dict[str, Any]:
    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=400,
            max_depth=6,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=42,
            class_weight="balanced",
        ),
    }
    for m in models.values():
        m.fit(X_train, y_train)
    return models


def evaluate_models(models: dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        row = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
        }
        if proba is not None:
            row["roc_auc"] = float(roc_auc_score(y_test, proba))
        else:
            row["roc_auc"] = float("nan")
        out[name] = row
    return out


def append_cluster_onehot(X: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    oh = np.zeros((X.shape[0], n_clusters), dtype=np.float32)
    oh[np.arange(X.shape[0]), labels] = 1.0
    return np.hstack([X, oh])
