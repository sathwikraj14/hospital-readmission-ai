
import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score
)
from lightgbm import LGBMClassifier

from preprocess import (
    X_train,
    X_val,
    y_train,
    y_val,
    preprocessor
)


# ============================================
# STEP 1: LOAD TUNED HYPERPARAMETERS
# ============================================

project_root = Path(__file__).resolve().parent.parent

results_path = project_root / "results"
params_path = results_path / "lightgbm_best_params.json"

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
    if use_balanced_weights
    else 1.0
)


# ============================================
# STEP 2: TRAIN TUNED LIGHTGBM
# ============================================

print("\nTraining tuned LightGBM...")

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

model.fit(X_train, y_train)

print("Model training completed!")


# ============================================
# STEP 3: GET VALIDATION PROBABILITIES
# ============================================

y_prob = model.predict_proba(X_val)[:, 1]

print("\nValidation ROC-AUC:",
      round(roc_auc_score(y_val, y_prob), 4))

print("Validation PR-AUC:",
      round(average_precision_score(y_val, y_prob), 4))


# ============================================
# STEP 4: EVALUATE MULTIPLE THRESHOLDS
# ============================================

thresholds = np.arange(0.05, 0.51, 0.01)

results = []

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    results.append({
        "threshold": round(float(threshold), 2),
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
        "flagged_patients": int(y_pred.sum()),
        "readmissions_detected": int(
            ((y_pred == 1) & (y_val.to_numpy() == 1)).sum()
        )
    })

results_df = pd.DataFrame(results)


# ============================================
# STEP 5: DISPLAY BEST F1 THRESHOLD
# ============================================

best_row = results_df.loc[
    results_df["f1_score"].idxmax()
]

print("\n===================================")
print("BEST F1 THRESHOLD")
print("===================================")

print(best_row.to_string())


# ============================================
# STEP 6: DISPLAY THRESHOLD COMPARISON
# ============================================

print("\nThreshold comparison:")

selected_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.50]

print(
    results_df[
        results_df["threshold"].isin(selected_thresholds)
    ].round(4).to_string(index=False)
)


# ============================================
# STEP 7: SAVE RESULTS
# ============================================

output_path = results_path / "lightgbm_threshold_results.csv"

results_df.to_csv(output_path, index=False)

print("\nThreshold results saved to:")
print(output_path)

print("\nThreshold optimization completed!")
