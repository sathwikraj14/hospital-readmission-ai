
import json
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
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
results_dir = project_root / "results"

params_path = results_dir / "catboost_best_params.json"

with open(params_path, "r", encoding="utf-8") as file:
    optimization_results = json.load(file)

best_params = optimization_results["best_params"].copy()

use_balanced_weights = best_params.pop(
    "use_balanced_weights"
)

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

best_params["scale_pos_weight"] = (
    negative_count / positive_count
    if use_balanced_weights else 1.0
)


# ============================================
# STEP 2: TRAIN TUNED CATBOOST
# ============================================

print("\nTraining tuned CatBoost...")

model = Pipeline(steps=[
    ("preprocessor", clone(preprocessor)),
    ("classifier", CatBoostClassifier(
        **best_params,
        loss_function="Logloss",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1
    ))
])

model.fit(X_train, y_train)

print("CatBoost training completed!")


# ============================================
# STEP 3: VALIDATION PROBABILITIES
# ============================================

y_prob = model.predict_proba(X_val)[:, 1]

print("\nValidation ROC-AUC:",
      round(roc_auc_score(y_val, y_prob), 4))

print("Validation PR-AUC:",
      round(average_precision_score(y_val, y_prob), 4))


# ============================================
# STEP 4: TEST DIFFERENT THRESHOLDS
# ============================================

thresholds = np.round(
    np.arange(0.05, 0.501, 0.01), 2
)

results = []

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(
            y_val, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_val, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_val, y_pred, zero_division=0
        ),
        "flagged_encounters": int(y_pred.sum()),
        "readmissions_detected": int(
            ((y_pred == 1) & (y_val.to_numpy() == 1)).sum()
        )
    })

results_df = pd.DataFrame(results)


# ============================================
# STEP 5: FIND BEST F1 THRESHOLD
# ============================================

best_row = results_df.loc[
    results_df["f1_score"].idxmax()
]

best_threshold = float(best_row["threshold"])

print("\n===================================")
print("BEST CATBOOST F1 THRESHOLD")
print("===================================")

print(best_row.to_string())


# ============================================
# STEP 6: DISPLAY THRESHOLD COMPARISON
# ============================================

selected_thresholds = [
    0.10, 0.15, 0.20, 0.25, 0.30, 0.50
]

print("\nThreshold comparison:")

print(
    results_df[
        results_df["threshold"].isin(selected_thresholds)
    ].round(4).to_string(index=False)
)


# ============================================
# STEP 7: CONFUSION MATRIX
# ============================================

best_predictions = (
    y_prob >= best_threshold
).astype(int)

print("\nConfusion matrix at best F1 threshold:")
print(confusion_matrix(y_val, best_predictions))


# ============================================
# STEP 8: SAVE RESULTS
# ============================================

results_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    results_dir / "catboost_threshold_results.csv",
    index=False
)

summary = {
    "best_f1_threshold": best_threshold,
    "validation_f1": float(best_row["f1_score"]),
    "validation_precision": float(best_row["precision"]),
    "validation_recall": float(best_row["recall"]),
    "validation_roc_auc": float(
        roc_auc_score(y_val, y_prob)
    ),
    "validation_pr_auc": float(
        average_precision_score(y_val, y_prob)
    )
}

with open(
    results_dir / "catboost_threshold_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)

print("\nThreshold optimization completed!")
print("Results saved in the results folder.")
