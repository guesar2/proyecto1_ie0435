"""
Retrain binary classifiers on all available dataset CSV files.
Combines dataset.csv and dataset_*.csv files, then evaluates Decision Tree, Naive Bayes, KNN, and SVM.
Also compares baseline, dataset-level normalization, and PCA.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Utility functions


def load_datasets(pattern="dataset*.csv"):
    files = sorted(Path(".").glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching {pattern}")
    raw_arrays = []
    print("Found dataset files:")
    for path in files:
        print(f" - {path}")
        data = np.loadtxt(path, delimiter=",")
        raw_arrays.append(data)
        counts = np.bincount(data[:, -1].astype(int))
        print(f"  Loaded {path}: shape={data.shape}, label counts={counts}")
    return raw_arrays


def combine_arrays(raw_arrays):
    combined = np.vstack(raw_arrays)
    print(f"\nCombined dataset shape: {combined.shape}")
    print(f"Combined label counts: {np.bincount(combined[:, -1].astype(int))}")
    X = combined[:, :-1]
    y = combined[:, -1].astype(int)
    return X, y


def normalize_per_dataset(raw_arrays):
    normalized = []
    print("\nApplying dataset-level normalization per source file...")
    for idx, arr in enumerate(raw_arrays, start=1):
        X_i = arr[:, :-1]
        y_i = arr[:, -1].astype(int)
        scaler = StandardScaler()
        X_i_scaled = scaler.fit_transform(X_i)
        normalized.append(np.hstack([X_i_scaled, y_i.reshape(-1, 1)]))
        print(
            f"  Normalized dataset {idx}: shape={X_i_scaled.shape}, mean={X_i_scaled.mean():.4f}, std={X_i_scaled.std():.4f}"
        )
    return normalized


def build_pipeline(model, use_scaling=False, use_pca=False, pca_components=None):
    steps = []
    if use_scaling:
        steps.append(("scaler", StandardScaler()))
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_components, random_state=42)))
    steps.append(("model", model))
    return Pipeline(steps)


def evaluate_models(
    X, y, setting_name, use_scaling=True, use_pca=False, pca_components=None
):
    print(f"\n{'#'*80}")
    print(f"EVALUATING SETTING: {setting_name}")
    print(f"{'#'*80}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Mean feature value: {X.mean():.4f}, std: {X.std():.4f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
    }

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Naive Bayes": GaussianNB(),
        "K-Nearest Neighbors (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine (SVM)": SVC(
            kernel="linear", C=0.1, gamma="auto", random_state=42
        ),
    }

    results = {}
    fold_predictions = {}

    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")

        model_scaling = use_scaling and (
            "KNeighbors" in model_name or "Support Vector" in model_name
        )
        pipeline = build_pipeline(
            model,
            use_scaling=model_scaling,
            use_pca=use_pca,
            pca_components=pca_components,
        )

        cv_results = cross_validate(
            pipeline, X, y, cv=skf, scoring=scoring, return_train_score=True
        )
        y_pred_cv = cross_val_predict(pipeline, X, y, cv=skf)

        results[model_name] = {
            "accuracy": cv_results["test_accuracy"],
            "precision": cv_results["test_precision"],
            "recall": cv_results["test_recall"],
            "f1": cv_results["test_f1"],
            "accuracy_train": cv_results["train_accuracy"],
            "precision_train": cv_results["train_precision"],
            "recall_train": cv_results["train_recall"],
            "f1_train": cv_results["train_f1"],
        }
        fold_predictions[model_name] = y_pred_cv

        for fold in range(5):
            print(
                f" Fold {fold+1}: Accuracy={cv_results['test_accuracy'][fold]:.4f}, "
                f"Precision={cv_results['test_precision'][fold]:.4f}, "
                f"Recall={cv_results['test_recall'][fold]:.4f}, "
                f"F1={cv_results['test_f1'][fold]:.4f}"
            )

    summary_rows = []
    for model_name, r in results.items():
        summary_rows.append(
            {
                "Model": model_name,
                "Accuracy": r["accuracy"].mean(),
                "Precision": r["precision"].mean(),
                "Recall": r["recall"].mean(),
                "F1": r["f1"].mean(),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("F1", ascending=False)
    print(f"\n{setting_name} SUMMARY")
    print(summary_df.to_string(index=False, float_format="%.4f"))

    for model_name, r in results.items():
        print(f"\n{model_name} ({setting_name})")
        print(f"  Accuracy:  {r['accuracy'].mean():.4f} ± {r['accuracy'].std():.4f}")
        print(f"  Precision: {r['precision'].mean():.4f} ± {r['precision'].std():.4f}")
        print(f"  Recall:    {r['recall'].mean():.4f} ± {r['recall'].std():.4f}")
        print(f"  F1:        {r['f1'].mean():.4f} ± {r['f1'].std():.4f}")

    plot_confusion_matrices(fold_predictions, y, setting_name)
    plot_comparison(summary_df, setting_name)

    return summary_df, results


def tune_svm_dataset_norm(X, y, filename="svm_dataset_norm.joblib"):
    print(f"\n{'#'*80}")
    print("TUNING SVM ON DATASET-LEVEL NORMALIZED DATA")
    print(f"{'#'*80}")

    param_grid = [
        {"svc__kernel": ["linear"], "svc__C": [0.1, 1, 10, 100]},
        {
            "svc__kernel": ["rbf"],
            "svc__C": [0.1, 1, 10, 100],
            "svc__gamma": ["scale", "auto"],
        },
        {
            "svc__kernel": ["poly"],
            "svc__C": [0.1, 1, 10, 100],
            "svc__gamma": ["scale", "auto"],
            "svc__degree": [2, 3, 4],
        },
    ]

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(random_state=42)),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
        refit=True,
    )
    grid_search.fit(X, y)

    print("\nBest SVM parameters:", grid_search.best_params_)
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    dump(grid_search.best_estimator_, filename)
    print(f"Saved tuned SVM model pipeline to {filename}")
    return grid_search


def plot_confusion_matrices(fold_predictions, y, setting_name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    for idx, (model_name, y_pred) in enumerate(fold_predictions.items()):
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx], cbar=False)
        axes[idx].set_title(f"{model_name} - {setting_name}")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")
    plt.tight_layout()
    filename = f"confusion_matrices_{setting_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved confusion matrix plot as {filename}")
    plt.close(fig)


def plot_comparison(summary_df, setting_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    width = 0.18
    x = np.arange(len(summary_df))
    for i, metric in enumerate(metrics):
        values = summary_df[metric].values
        ax.bar(x + i * width, values, width, label=metric)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(summary_df["Model"].values, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(f"Model Performance - {setting_name}")
    plt.tight_layout()
    filename = f"model_comparison_{setting_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot as {filename}")
    plt.close(fig)


# Load and combine datasets
raw_arrays = load_datasets()
X_base, y = combine_arrays(raw_arrays)

# Baseline evaluation
baseline_df, baseline_results = evaluate_models(
    X_base,
    y,
    setting_name="baseline",
    use_scaling=True,
    use_pca=False,
    pca_components=None,
)

# Dataset-level normalization and evaluation
normalized_arrays = normalize_per_dataset(raw_arrays)
X_norm, y_norm = combine_arrays(normalized_arrays)
normalized_df, normalized_results = evaluate_models(
    X_norm,
    y_norm,
    setting_name="dataset_level_normalization",
    use_scaling=True,
    use_pca=False,
    pca_components=None,
)


# Tune SVM on the dataset-level normalized data and export the final model
svm_grid = tune_svm_dataset_norm(X_norm, y_norm)
