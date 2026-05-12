#!/usr/bin/env python3
"""End-to-end experiment script: figures + metrics/metrics.json for the report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, confusion_matrix
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clustering import (  # noqa: E402
    fit_kmeans,
    kmeans_range_inertia,
    kmeans_range_silhouette,
    pca_2d,
)
from src.preprocess import clean_dataframe, fit_scaler, load_raw, stratified_split, transform  # noqa: E402
from src.train_eval import append_cluster_onehot, evaluate_models, train_models  # noqa: E402


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_path = ROOT / "data" / "Telco-Customer-Churn.csv"
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}. Run: python3 scripts/download_data.py")

    df = load_raw(data_path)
    X, y = clean_dataframe(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    scaler, X_train_s = fit_scaler(X_train)
    X_test_s = transform(scaler, X_test)

    ks, inertias = kmeans_range_inertia(X_train_s, 2, 10)
    _, sils = kmeans_range_silhouette(X_train_s, 2, 10)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ks, inertias, marker="o")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia (SSE)")
    axes[0].set_title("Elbow method")
    axes[1].plot(ks, sils, marker="o", color="C1")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette vs. k")
    fig.tight_layout()
    fig.savefig(fig_dir / "cluster_selection.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "cluster_selection.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    n_clusters = 4
    km = fit_kmeans(X_train_s, n_clusters=n_clusters)
    c_train = km.predict(X_train_s)
    c_test = km.predict(X_test_s)

    Z = pca_2d(np.vstack([X_train_s, X_test_s]))
    labels_vis = np.concatenate([c_train, c_test])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    scatter = ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=labels_vis,
        cmap="tab10",
        alpha=0.65,
        s=12,
        edgecolors="none",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA projection (2D) colored by K-Means cluster (k=4)")
    fig.colorbar(scatter, ax=ax, label="cluster id")
    fig.tight_layout()
    fig.savefig(fig_dir / "pca_clusters.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "pca_clusters.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    X_train_h = append_cluster_onehot(X_train_s, c_train, n_clusters)
    X_test_h = append_cluster_onehot(X_test_s, c_test, n_clusters)

    models_plain = train_models(X_train_s, y_train.to_numpy())
    models_hybrid = train_models(X_train_h, y_train.to_numpy())
    metrics_plain = evaluate_models(models_plain, X_test_s, y_test.to_numpy())
    metrics_hybrid = evaluate_models(models_hybrid, X_test_h, y_test.to_numpy())

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, style in [
        ("logistic_regression", ":"),
        ("random_forest", "-"),
        ("hist_gradient_boosting", "--"),
    ]:
        m = models_plain[name]
        if hasattr(m, "predict_proba"):
            RocCurveDisplay.from_predictions(
                y_test,
                m.predict_proba(X_test_s)[:, 1],
                ax=ax,
                name=name.replace("_", " ").title(),
                linestyle=style,
            )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="chance")
    ax.set_title("ROC curves (held-out test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_curves.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "roc_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    best_name = "hist_gradient_boosting"
    best = models_plain[best_name]
    pred = best.predict(X_test_s)
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {best_name} (plain features)")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_hgb.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "confusion_hgb.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Churn rate by cluster (train) for interpretability
    churn_by_cluster = []
    for k in range(n_clusters):
        mask = c_train == k
        rate = float(y_train.to_numpy()[mask].mean()) if mask.any() else 0.0
        churn_by_cluster.append({"cluster": k, "churn_rate": rate, "n": int(mask.sum())})
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(
        [str(c["cluster"]) for c in churn_by_cluster],
        [c["churn_rate"] for c in churn_by_cluster],
        color="steelblue",
    )
    ax.set_xlabel("Cluster (train assignments)")
    ax.set_ylabel("Churn rate")
    ax.set_title("Churn rate by K-Means segment (training data)")
    fig.tight_layout()
    fig.savefig(fig_dir / "churn_by_cluster.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "churn_by_cluster.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    rf = models_plain["random_forest"]
    pi = permutation_importance(
        rf,
        X_test_s,
        y_test.to_numpy(),
        n_repeats=15,
        random_state=42,
        n_jobs=-1,
    )
    order = np.argsort(pi.importances_mean)[::-1][:12]
    names = np.array(X_train.columns.tolist())[order]
    means = pi.importances_mean[order]
    stds = pi.importances_std[order]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names[::-1], means[::-1], xerr=stds[::-1], color="teal", alpha=0.85)
    ax.set_xlabel("Mean decrease in accuracy (permutation)")
    ax.set_title("Permutation importance — Random Forest (test set)")
    fig.tight_layout()
    fig.savefig(fig_dir / "permutation_importance.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "permutation_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "n_clusters": n_clusters,
        "silhouette_at_k4": float(sils[ks.index(4)]) if 4 in ks else None,
        "metrics_plain": metrics_plain,
        "metrics_hybrid": metrics_hybrid,
        "churn_by_cluster_train": churn_by_cluster,
        "class_balance": {
            "train_churn_rate": float(y_train.mean()),
            "test_churn_rate": float(y_test.mean()),
        },
    }
    (ROOT / "figures" / "metrics.json").write_text(json.dumps(payload, indent=2))
    print("Wrote figures and figures/metrics.json")


if __name__ == "__main__":
    main()
