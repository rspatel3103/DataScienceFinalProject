from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def kmeans_range_inertia(X: np.ndarray, k_min: int = 2, k_max: int = 10) -> tuple[list[int], list[float]]:
    ks = list(range(k_min, k_max + 1))
    inertias: list[float] = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        km.fit(X)
        inertias.append(float(km.inertia_))
    return ks, inertias


def kmeans_range_silhouette(X: np.ndarray, k_min: int = 2, k_max: int = 10) -> tuple[list[int], list[float]]:
    ks = list(range(k_min, k_max + 1))
    scores: list[float] = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
        scores.append(float(silhouette_score(X, labels)))
    return ks, scores


def fit_kmeans(X_train: np.ndarray, n_clusters: int = 4, random_state: int = 42) -> KMeans:
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    km.fit(X_train)
    return km


def pca_2d(X: np.ndarray, random_state: int = 42) -> np.ndarray:
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(X)
