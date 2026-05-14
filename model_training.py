"""
Binary Classification Models Training and Evaluation
Dataset: Preprocessed rice contamination images (30 samples, 16384 features, binary labels)
Models: Decision Tree, Naive Bayes, KNN, SVM
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ==================== TASK 1: Load and Inspect Dataset ====================
print("=" * 80)
print("TASK 1: DATASET LOADING AND INSPECTION")
print("=" * 80)

# Load dataset
data = np.loadtxt("dataset.csv", delimiter=",")
print(f"\nDataset shape: {data.shape}")
print(f"Number of samples: {data.shape[0]}")
print(f"Number of features (including label): {data.shape[1]}")

# Separate features and labels
X = data[:, :-1]  # All columns except last
y = data[:, -1].astype(int)  # Last column is label

print(f"\nFeatures shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Label distribution: {np.bincount(y)}")
print(f"  - Class 0 (no contamination): {np.sum(y == 0)} samples")
print(f"  - Class 1 (contamination present): {np.sum(y == 1)} samples")
print(
    f"  - Class balance: {np.sum(y == 0)/len(y)*100:.1f}% vs {np.sum(y == 1)/len(y)*100:.1f}%"
)

# Dataset statistics
print(f"\nFeature statistics:")
print(f"  - Feature data type: {X.dtype}")
print(f"  - Min value: {X.min()}")
print(f"  - Max value: {X.max()}")
print(f"  - Mean: {X.mean():.4f}")
print(f"  - Std Dev: {X.std():.4f}")

# ==================== TASK 2: Separate Features and Labels ====================
print("\n" + "=" * 80)
print("TASK 2: FEATURES AND LABELS SEPARATION (Already done above)")
print("=" * 80)
print(f"X: {X.shape} - Feature matrix")
print(f"y: {y.shape} - Label vector")

# ==================== TASK 3: Preprocessing ====================
print("\n" + "=" * 80)
print("TASK 3: PREPROCESSING")
print("=" * 80)

# Create scalers for KNN and SVM
scaler_knn = StandardScaler()
scaler_svm = StandardScaler()

# Fit scaler on entire data (will be applied within CV)
X_scaled_demo = StandardScaler().fit_transform(X)
print(f"\nScaling applied (StandardScaler):")
print(
    f"  - Original X: min={X.min()}, max={X.max()}, mean={X.mean():.4f}, std={X.std():.4f}"
)
print(
    f"  - Scaled X: min={X_scaled_demo.min():.4f}, max={X_scaled_demo.max():.4f}, mean={X_scaled_demo.mean():.4f}, std={X_scaled_demo.std():.4f}"
)
print(
    f"\nNote: During cross-validation, scaler is fit on training fold only to prevent data leakage"
)

# ==================== TASK 4: Model Training with Stratified K-Fold CV ====================
print("\n" + "=" * 80)
print("TASK 4: MODEL TRAINING WITH STRATIFIED K-FOLD CROSS-VALIDATION")
print("=" * 80)

# Set up stratified k-fold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

print(f"\nUsing Stratified {n_splits}-Fold Cross-Validation")
print(f"This ensures each fold has approximately the same label distribution")

# Initialize models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors (k=3)": KNeighborsClassifier(n_neighbors=3),
    "Support Vector Machine (SVM)": SVC(kernel="rbf", random_state=42),
}

# Store results
results = {}
fold_predictions = {}

# Training and evaluation
for model_name, model in models.items():
    print(f"\n{'-'*80}")
    print(f"Training: {model_name}")
    print(f"{'-'*80}")

    # Define scoring metrics
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
    }

    # Apply scaling if needed
    if "KNeighbors" in model_name or "Support Vector" in model_name:
        # For KNN and SVM, we need to scale data within CV loop
        from sklearn.pipeline import Pipeline

        if "KNeighbors" in model_name:
            pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
        else:  # SVM
            pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
        cv_model = pipeline
    else:
        cv_model = model

    # Perform cross-validation
    cv_results = cross_validate(
        cv_model, X, y, cv=skf, scoring=scoring, return_train_score=True
    )

    # Get predictions for confusion matrix (using all data with CV)
    y_pred_cv = cross_val_predict(cv_model, X, y, cv=skf)

    # Store results
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

    # Print fold-by-fold results
    print(f"\nFold-by-fold results:")
    for fold in range(n_splits):
        print(
            f"  Fold {fold+1}: Acc={cv_results['test_accuracy'][fold]:.4f}, "
            f"Prec={cv_results['test_precision'][fold]:.4f}, "
            f"Rec={cv_results['test_recall'][fold]:.4f}, "
            f"F1={cv_results['test_f1'][fold]:.4f}"
        )

# ==================== TASK 5: Report Metrics ====================
print("\n" + "=" * 80)
print("TASK 5: COMPREHENSIVE METRICS REPORT")
print("=" * 80)

metrics_df = pd.DataFrame()

for model_name in models.keys():
    model_results = results[model_name]

    # Calculate means and stds
    metrics_dict = {
        "Model": model_name,
        "Accuracy": f"{model_results['accuracy'].mean():.4f} ± {model_results['accuracy'].std():.4f}",
        "Precision": f"{model_results['precision'].mean():.4f} ± {model_results['precision'].std():.4f}",
        "Recall": f"{model_results['recall'].mean():.4f} ± {model_results['recall'].std():.4f}",
        "F1-Score": f"{model_results['f1'].mean():.4f} ± {model_results['f1'].std():.4f}",
    }

    print(f"\n{model_name}")
    print(f"  Accuracy:  {metrics_dict['Accuracy']}")
    print(f"  Precision: {metrics_dict['Precision']}")
    print(f"  Recall:    {metrics_dict['Recall']}")
    print(f"  F1-Score:  {metrics_dict['F1-Score']}")

# ==================== CONFUSION MATRICES ====================
print("\n" + "=" * 80)
print("CONFUSION MATRICES")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, (model_name, model) in enumerate(models.items()):
    y_pred = fold_predictions[model_name]
    cm = confusion_matrix(y, y_pred)

    print(f"\n{model_name}:")
    print(f"  True Negatives:  {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives:  {cm[1, 1]}")

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx], cbar=False)
    axes[idx].set_title(f"{model_name}\nConfusion Matrix")
    axes[idx].set_ylabel("True Label")
    axes[idx].set_xlabel("Predicted Label")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
print("\nConfusion matrices saved to 'confusion_matrices.png'")

# ==================== TASK 6: Overfitting Analysis ====================
print("\n" + "=" * 80)
print("TASK 6: OVERFITTING ANALYSIS & DISCUSSION")
print("=" * 80)

print("\nOverfitting Analysis (Train vs Test Performance):")
print("-" * 80)

overfitting_data = []
for model_name in models.keys():
    model_results = results[model_name]

    train_acc = model_results["accuracy_train"].mean()
    test_acc = model_results["accuracy"].mean()
    diff = train_acc - test_acc

    overfitting_data.append(
        {
            "Model": model_name,
            "Train Accuracy": f"{train_acc:.4f}",
            "Test Accuracy": f"{test_acc:.4f}",
            "Overfitting Gap": f"{diff:.4f}",
        }
    )

    print(f"\n{model_name}:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Overfitting Gap: {diff:.4f}")

    if diff > 0.1:
        print(f"  ⚠ WARNING: Possible overfitting detected (gap > 0.1)")
    elif diff > 0.05:
        print(f"  ⚠ CAUTION: Moderate overfitting (gap > 0.05)")
    else:
        print(f"  ✓ Acceptable generalization")

print("\n" + "-" * 80)
print("GENERAL DISCUSSION ON OVERFITTING WITH SMALL DATASETS (30 samples):")
print("-" * 80)
print("""
Challenges with small datasets (30 samples):
1. High variance: Limited data leads to high variability in model estimates
2. Limited generalization: Models may learn dataset-specific patterns rather than general rules
3. Increased overfitting risk: More likely to fit noise rather than true signal
4. Poor statistical reliability: Standard errors are larger with small samples

Mitigation strategies employed:
1. Stratified K-Fold CV: Maintains class distribution in each fold
2. Cross-validation: Multiple train-test splits reduce variance
3. Feature scaling: For KNN/SVM to ensure fair comparisons
4. Constrained models: Decision Tree with max_depth=5 to reduce complexity
5. Simple models preferred: Naive Bayes works well on small datasets

Expected behavior:
- Higher variance in fold-by-fold scores
- Potential overfitting despite regularization
- Need for ensemble methods or more data for production use
""")

# ==================== TASK 7: Model Comparison & Best Model ====================
print("\n" + "=" * 80)
print("TASK 7: MODEL COMPARISON & BEST MODEL SELECTION")
print("=" * 80)

# Create summary table
summary_data = []
for model_name in models.keys():
    model_results = results[model_name]
    summary_data.append(
        {
            "Model": model_name,
            "Accuracy": model_results["accuracy"].mean(),
            "Precision": model_results["precision"].mean(),
            "Recall": model_results["recall"].mean(),
            "F1-Score": model_results["f1"].mean(),
        }
    )

summary_df = pd.DataFrame(summary_data).sort_values("F1-Score", ascending=False)

print("\nModel Performance Ranking (by F1-Score):")
print(summary_df.to_string(index=False))

best_model_idx = summary_df["F1-Score"].idxmax()
best_model_name = summary_df.loc[best_model_idx, "Model"]

print(f"\n{'='*80}")
print(f"BEST MODEL: {best_model_name}")
print(f"{'='*80}")

best_results = results[best_model_name]
print(f"\nBest Model Performance Metrics:")
print(
    f"  Accuracy:  {best_results['accuracy'].mean():.4f} ± {best_results['accuracy'].std():.4f}"
)
print(
    f"  Precision: {best_results['precision'].mean():.4f} ± {best_results['precision'].std():.4f}"
)
print(
    f"  Recall:    {best_results['recall'].mean():.4f} ± {best_results['recall'].std():.4f}"
)
print(f"  F1-Score:  {best_results['f1'].mean():.4f} ± {best_results['f1'].std():.4f}")

# ==================== WHY THIS MODEL IS BEST ====================
print(f"\nWhy {best_model_name} is the best choice:")
print("-" * 80)

best_acc = summary_df.loc[best_model_idx, "Accuracy"]
best_f1 = summary_df.loc[best_model_idx, "F1-Score"]

# Get ranking for each metric
acc_rank = (summary_df["Accuracy"] >= best_acc).sum()
f1_rank = (summary_df["F1-Score"] >= best_f1).sum()

print(f"1. Best F1-Score ranking: #{f1_rank} out of {len(models)}")
print(f"2. Accuracy ranking: #{acc_rank} out of {len(models)}")

if best_model_name == "Naive Bayes":
    print("""
3. Advantages:
   - Simple probabilistic model suitable for small datasets
   - Fast training and prediction
   - Provides probability estimates
   - Less prone to overfitting with limited data
   - Effective baseline model
4. Why it works well here:
   - Binary pixel features are well-suited for Naive Bayes
   - Independent feature assumption is reasonable for random noise
   - Doesn't require much hyperparameter tuning
""")
elif "Decision Tree" in best_model_name:
    print("""
3. Advantages:
   - Interpretable results (can visualize decision boundaries)
   - Handles non-linear relationships
   - Max depth constraint prevents overfitting
   - Works well with categorical/binary features
4. Why it works well here:
   - Binary features naturally suited to tree splits
   - Regularization (max_depth=5) prevents overfitting
""")
elif "KNeighbors" in best_model_name:
    print("""
3. Advantages:
   - Non-parametric approach good for small datasets
   - Simple to understand and implement
   - k=3 provides reasonable generalization
4. Why it works well here:
   - Similar images should have similar contamination status
   - Local neighborhood assumption is reasonable
""")
elif "Support Vector" in best_model_name:
    print("""
3. Advantages:
   - Effective in high-dimensional spaces
   - RBF kernel captures non-linear relationships
   - Regularization built-in (C parameter)
4. Why it works well here:
   - Works in 16384-dimensional space effectively
   - Kernel trick handles complexity well
""")

# ==================== VISUALIZATION ====================
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# Plot 1: Model Performance Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
x_pos = np.arange(len(models))
width = 0.2

ax = axes[0]
metric_keys = ["accuracy", "precision", "recall", "f1"]
for i, metric in enumerate(metrics):
    values = [
        results[model_name][metric_keys[i]].mean() for model_name in models.keys()
    ]
    ax.bar(x_pos + i * width, values, width, label=metric)

ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison (Mean CV Scores)")
ax.set_xticks(x_pos + width * 1.5)
ax.set_xticklabels(models.keys(), rotation=45, ha="right")
ax.legend()
ax.grid(axis="y", alpha=0.3)

# F1-Score with error bars
ax = axes[1]
f1_means = [results[model_name]["f1"].mean() for model_name in models.keys()]
f1_stds = [results[model_name]["f1"].std() for model_name in models.keys()]
ax.bar(x_pos, f1_means, yerr=f1_stds, capsize=5, alpha=0.7)
ax.set_xlabel("Model")
ax.set_ylabel("F1-Score")
ax.set_title("F1-Score with Standard Deviation (5-Fold CV)")
ax.set_xticks(x_pos)
ax.set_xticklabels(models.keys(), rotation=45, ha="right")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
print("Model comparison plot saved to 'model_comparison.png'")

# ==================== FINAL SUMMARY ====================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"""
Dataset: 30 samples with 16384 binary pixel features
Task: Binary classification (contamination detection in rice)

Models Trained:
1. Decision Tree (max_depth=5)
2. Naive Bayes (Gaussian)
3. K-Nearest Neighbors (k=3)
4. Support Vector Machine (RBF kernel)

Best Performing Model: {best_model_name}
- F1-Score: {best_f1:.4f}
- Accuracy: {best_acc:.4f}

Key Insights:
- Small dataset (30 samples) limits model complexity
- Stratified K-Fold CV ensures reliable evaluation
- Scaling important for distance-based methods (KNN, SVM)
- Need more data for production deployment
- Ensemble methods recommended for improvement

Output files generated:
- confusion_matrices.png
- model_comparison.png
""")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
