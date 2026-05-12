# CS439 Final Project — Telco Churn (Segmentation + Supervised Learning)

This repository implements the course-style **hybrid pipeline** from the provided examples: **K-Means customer segmentation** plus **supervised churn classifiers**, with preprocessing, baselines, multiple metrics, and publication-style figures.

## Reproduce

```bash
python3 -m pip install -r requirements.txt
python3 scripts/download_data.py
python3 scripts/reproduce.py
```

Figures and `figures/metrics.json` are written for the written report. Interactive work lives in `notebooks/final_project.ipynb`.

## Report

The NeurIPS-inspired LaTeX source is in `report/main.tex` (compile from `report/` with `pdflatex main.tex`, or upload the `report` folder to [Overleaf](https://www.overleaf.com/)). **Replace** the placeholder GitHub URL on the title page before submission.

## Layout

| Path                   | Role                                                                |
| ---------------------- | ------------------------------------------------------------------- |
| `src/preprocess.py`    | Load/clean, dummies, stratified split, scaler (train-only fit)      |
| `src/clustering.py`    | Elbow/silhouette scans, K-Means, PCA projection                     |
| `src/train_eval.py`    | Baseline + LR + RF + HistGradientBoosting; cluster ablation helpers |
| `scripts/reproduce.py` | End-to-end experiment + all figures                                 |

## Note on XGBoost

The reference examples use **XGBoost**; this repo defaults to **`HistGradientBoostingClassifier`** from scikit-learn so the project runs on macOS/Python without a separate OpenMP (`libomp`) install. You may swap in XGBoost locally if your environment supports it.
