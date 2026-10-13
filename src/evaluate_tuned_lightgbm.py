
import json
from pathlib import Path

from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from preprocess import (
    X_train,
    X_val,
    y_train,
    y_val,
    preprocessor
)


# ============================================
# STEP 1: LOAD OPTUNA PARAMETERS
# ============================================

project_root = Path(__file__).resolve().parent.parent

params_path = (
    project_root / "results" / "lightgbm_best_params.json"
)

with open(params_path, "r", encoding="utf-8") as file:
    optimization_results = json.load(file)

best_params = optimization_results["best_params"].copy()

# Convert Optuna's class-weight choice into
# the parameter expected by LightGBM.
use_balanced_weights = best_params.pop(
    "use_balanced_weights"
)

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

best_params["scale_pos_weight"] = (
    negative_count / positive_count
    if use_balanced_weights
    else 1.0
)

print("\nOptuna parameters loaded successfully!")
print("Best CV PR-AUC:",
      optimization_results["best_cv_pr_auc"])


# ============================================
# STEP 2: CREATE TUNED LIGHTGBM MODEL
# ============================================

model = Pipeline(steps=[
    ("preprocessor", clone(preprocessor)),
    ("classifier", LGBMClassifier(
        **best_params,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    ))
])


# ============================================
# STEP 3: TRAIN MODEL
# ============================================

print("\nTraining tuned LightGBM...")

model.fit(X_train, y_train)

print("Training completed!")


# ============================================
# STEP 4: VALIDATION PREDICTIONS
# ============================================

y_prob = model.predict_proba(X_val)[:, 1]

# Initial classification threshold
threshold = 0.50

y_pred = (y_prob >= threshold).astype(int)


# ============================================
# STEP 5: EVALUATE MODEL
# ============================================

print("\n===================================")
print("TUNED LIGHTGBM RESULTS")
print("===================================")

print(f"Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_val, y_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_val, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_val, y_prob):.4f}")
print(f"PR-AUC:    {average_precision_score(y_val, y_prob):.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_val, y_pred))


# ============================================
# STEP 6: COMPARE AGAINST ORIGINAL LIGHTGBM
# ============================================

print("\n===================================")
print("PREVIOUS LIGHTGBM BENCHMARK")
print("===================================")

print("Original engineered ROC-AUC: 0.6774")
print("Original engineered PR-AUC:  0.2389")

print("\nTuned model evaluation completed!")
