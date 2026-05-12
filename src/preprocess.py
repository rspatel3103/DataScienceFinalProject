from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_raw(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return features X (no churn) and binary label y. No leakage: churn not in X."""
    work = df.copy()
    if "customerID" in work.columns:
        work = work.drop(columns=["customerID"])
    work["TotalCharges"] = pd.to_numeric(work["TotalCharges"], errors="coerce")
    work["TotalCharges"] = work["TotalCharges"].fillna(work["MonthlyCharges"])
    y = (work["Churn"] == "Yes").astype(np.int8)
    X = work.drop(columns=["Churn"])
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def fit_scaler(X_train: pd.DataFrame) -> tuple[StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    return scaler, X_train_s


def transform(scaler: StandardScaler, X: pd.DataFrame) -> np.ndarray:
    return scaler.transform(X)
