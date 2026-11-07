"""
Handover Prediction — Model Training & Comparison
==================================================
Trains XGBoost, Random Forest, LightGBM, CatBoost and Logistic Regression
on each of 3 datasets, tunes hyperparameters with RandomizedSearchCV,
and reports Accuracy / Precision / Recall / F1 / ROC-AUC.

Usage:
    python train_handover_models.py
    python train_handover_models.py --data-dir data --output-dir results --n-iter 20

Expects the 3 CSVs in --data-dir (defaults below match the filenames supplied):
    4G5G_5G4G_DownSampling.csv
    5G_5G_DownSampling.csv
    5G5G_4G5G_5G4G_DowbSamoling.csv
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
FEATURES = [
    'Temperature_C', 'Humidity_pct', 'Level', 'SNR', 'Altitude', 'NRxLev1',
    'Serving_RSRP', 'Best_Neighbor_RSRP', 'RSRP_Difference',
    'Prev_Primary_SNR', 'Prev_Primary_RSRP',
]
TARGET = 'IsHandover'

DATASETS = {
    "4G5G_5G4G": "4G5G_5G4G_DownSampling.csv",
    "5G_5G": "5G_5G_DownSampling.csv",
    "5G5G_4G5G_5G4G": "5G5G_4G5G_5G4G_DowbSamoling.csv",
}

RANDOM_STATE = 42
CV_FOLDS = 5

PARAM_GRIDS = {
    "LogisticRegression": {
        "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    },
    "RandomForest": {
        "n_estimators": [150, 250, 400, 600],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    },
    "XGBoost": {
        "n_estimators": [150, 250, 400, 600],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
    },
    "LightGBM": {
        "n_estimators": [150, 250, 400, 600],
        "num_leaves": [15, 31, 63, 127],
        "max_depth": [-1, 5, 10, 15],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    },
    "CatBoost": {
        "iterations": [200, 400, 600],
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5, 7],
    },
}


def build_estimators():
    """Returns dict of name -> (estimator, is_pipeline_scaled)."""
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced",
                                        max_iter=2000,
                                        random_state=RANDOM_STATE)),
        ]),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE,
            n_jobs=-1, use_label_encoder=False,
        ),
        "LightGBM": LGBMClassifier(
            class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=-1, verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            random_state=RANDOM_STATE, verbose=0, auto_class_weights="Balanced",
        ),
    }


def tune_model(name, estimator, X_train, y_train, n_iter, cv):
    grid = PARAM_GRIDS[name]
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=grid,
        n_iter=min(n_iter, np.prod([len(v) for v in grid.values()])),
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=4,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["No-Handover", "Handover"],
                                    zero_division=0)
    return metrics, cm, report


def plot_confusion_matrices(dataset_name, results, output_dir):
    """One figure per dataset, one subplot per model. Counts + row-% shown."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(4.6 * n_models, 4.2))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, r) in zip(axes, results.items()):
        cm = r["cm"]
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        annot = np.array([
            [f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ])
        sns.heatmap(
            cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["No-Handover", "Handover"],
            yticklabels=["No-Handover", "Handover"],
            linewidths=0.5, linecolor="grey",
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{model_name}\nAcc={r['metrics']['Accuracy']:.3f}  "
                      f"F1={r['metrics']['F1']:.3f}")

    fig.suptitle(f"Confusion Matrices — {dataset_name}", fontsize=14, y=1.03)
    fig.tight_layout()
    out_path = output_dir / f"confusion_matrices_{dataset_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_dataset(dataset_name, csv_path, output_dir, n_iter, test_size):
    print("\n" + "=" * 90)
    print(f"DATASET: {dataset_name}  ({csv_path.name})")
    print("=" * 90)

    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing}")

    df = df[FEATURES + [TARGET]].dropna()
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    print(f"Rows: {len(df)} | Class balance -> {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    estimators = build_estimators()

    results = {}
    summary_rows = []

    for name, est in estimators.items():
        t0 = time.time()
        best_model, best_params, best_cv_f1 = tune_model(name, est, X_train, y_train, n_iter, cv)
        metrics, cm, report = evaluate(name, best_model, X_test, y_test)
        elapsed = time.time() - t0

        results[name] = {"model": best_model, "metrics": metrics, "cm": cm,
                          "report": report, "best_params": best_params}

        print(f"\n--- {name} ({elapsed:.1f}s, best CV F1={best_cv_f1:.4f}) ---")
        print(f"Best params: {best_params}")
        print(f"Accuracy={metrics['Accuracy']:.4f}  Precision={metrics['Precision']:.4f}  "
              f"Recall={metrics['Recall']:.4f}  F1={metrics['F1']:.4f}  "
              f"ROC_AUC={metrics['ROC_AUC']:.4f}")
        print("Confusion matrix [[TN FP] [FN TP]]:")
        print(cm)
        print(report)

        summary_rows.append({"Dataset": dataset_name, "Model": name, **metrics})

    img_path = plot_confusion_matrices(dataset_name, results, output_dir)
    print(f"\nSaved confusion-matrix image -> {img_path}")

    summary_df = pd.DataFrame(summary_rows).sort_values("F1", ascending=False)
    print(f"\n{dataset_name} — Model comparison (sorted by F1):")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    summary_df.to_csv(output_dir / f"summary_{dataset_name}.csv", index=False)

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Train & compare handover-prediction models.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--n-iter", type=int, default=25,
                         help="RandomizedSearchCV iterations per model.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--datasets", type=str, default=None,
                         help=f"Comma-separated subset of {list(DATASETS.keys())}. Default: all.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_run = DATASETS
    if args.datasets:
        keep = {k.strip() for k in args.datasets.split(",")}
        datasets_to_run = {k: v for k, v in DATASETS.items() if k in keep}

    all_summaries = []
    for dataset_name, filename in datasets_to_run.items():
        csv_path = data_dir / filename
        if not csv_path.exists():
            print(f"[SKIP] {csv_path} not found.")
            continue
        summary_df = run_dataset(dataset_name, csv_path, output_dir, args.n_iter, args.test_size)
        all_summaries.append(summary_df)

    if all_summaries:
        full_summary = pd.concat(all_summaries, ignore_index=True)
        summary_csv = output_dir / "model_comparison_summary.csv"
        full_summary.to_csv(summary_csv, index=False)

        print("\n" + "=" * 90)
        print("OVERALL SUMMARY (all datasets, all models)")
        print("=" * 90)
        print(full_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\nSaved summary table -> {summary_csv}")

        best_overall = full_summary.sort_values("F1", ascending=False).iloc[0]
        print(f"\nBest overall (by F1): {best_overall['Model']} on {best_overall['Dataset']} "
              f"(F1={best_overall['F1']:.4f}, Accuracy={best_overall['Accuracy']:.4f})")


if __name__ == "__main__":
    main()
