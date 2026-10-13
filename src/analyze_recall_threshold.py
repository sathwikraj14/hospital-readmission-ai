
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from preprocess import X_val, y_val


# ============================================
# STEP 1: PROJECT PATHS
# ============================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "catboost_candidate.joblib"

METADATA_PATH = (
    ROOT / "models" / "catboost_candidate_metadata.json"
)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================
# STEP 2: LOAD SAVED MODEL
# ============================================

print("\nLoading saved CatBoost model...")

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, encoding="utf-8") as file:
    metadata = json.load(file)

current_threshold = float(
    metadata["classification_threshold"]
)

print("Model loaded successfully!")
print("Current threshold:", current_threshold)


# ============================================
# STEP 3: PREPARE VALIDATION DATA
# ============================================

print("\nPreparing validation data...")

X_validation = X_val.copy()

required_features = metadata["feature_names"]

X_validation = X_validation[required_features]

preprocessor = model.named_steps["preprocessor"]

categorical_columns = preprocessor.transformers_[1][2]

for column in categorical_columns:
    X_validation[column] = (
        X_validation[column]
        .astype("string")
        .fillna("Missing")
        .replace("?", "Missing")
        .astype(str)
    )

actual = np.asarray(y_val).astype(int)

print("Validation encounters:", len(actual))
print("Actual readmissions:", int(actual.sum()))


# ============================================
# STEP 4: GENERATE MODEL SCORES
# ============================================

print("\nGenerating validation predictions...")

probabilities = model.predict_proba(
    X_validation
)[:, 1]

print("Predictions generated successfully!")


# ============================================
# STEP 5: CREATE THRESHOLD GRID
# ============================================

# Evaluate 0.000 to 0.500 in steps of 0.001.
# Include current threshold exactly.

thresholds = np.unique(
    np.concatenate([
        np.round(np.arange(0.0, 0.501, 0.001), 3),
        [current_threshold]
    ])
)

print("\nNumber of thresholds:", len(thresholds))


# ============================================
# STEP 6: CALCULATE METRICS
# ============================================

print("\nEvaluating thresholds...")

records = []

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predictions,
        labels=[0, 1]
    ).ravel()

    total_flagged = int(tp + fp)

    records.append({
        "threshold": float(threshold),

        "accuracy": accuracy_score(
            actual,
            predictions
        ),

        "precision": precision_score(
            actual,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            actual,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            actual,
            predictions,
            zero_division=0
        ),

        "true_positives": int(tp),

        "false_positives": int(fp),

        "false_negatives": int(fn),

        "true_negatives": int(tn),

        "flagged_encounters": total_flagged,

        "alert_burden_pct": (
            total_flagged / len(actual)
        ) * 100
    })

results = pd.DataFrame(records)

print("Threshold evaluation completed!")


# ============================================
# STEP 7: FIND 80% RECALL THRESHOLD
# ============================================

target_recall = 0.80

eligible = results[
    results["recall"] >= target_recall
]

if eligible.empty:
    raise RuntimeError(
        "No threshold achieved the 80% recall target."
    )

# Select the highest evaluated threshold
# that achieves at least 80% recall.
# This minimizes alerts on the evaluated grid.

best = eligible.loc[
    eligible["threshold"].idxmax()
]

# Find metrics for current 0.14 threshold.

current = results.iloc[
    (results["threshold"] - current_threshold)
    .abs()
    .argmin()
]


# ============================================
# STEP 8: COMPARE CURRENT VS TARGET
# ============================================

comparison = pd.DataFrame([
    {
        "Setting": "Current threshold",
        **current.to_dict()
    },
    {
        "Setting": "80% recall target",
        **best.to_dict()
    }
])

display_columns = [
    "Setting",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "true_positives",
    "false_positives",
    "false_negatives",
    "flagged_encounters",
    "alert_burden_pct"
]

print("\n" + "=" * 60)
print("VALIDATION THRESHOLD COMPARISON")
print("=" * 60)

print(
    comparison[display_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

print("\nInterpretation:")

print(
    f"Current recall: {current['recall'] * 100:.2f}%"
)

print(
    f"Target recall: {best['recall'] * 100:.2f}%"
)

print(
    f"Readmissions detected at target: "
    f"{int(best['true_positives'])}"
)

print(
    f"Readmissions missed at target: "
    f"{int(best['false_negatives'])}"
)

print(
    f"False-positive alerts at target: "
    f"{int(best['false_positives'])}"
)

print(
    f"Total encounters flagged at target: "
    f"{int(best['flagged_encounters'])}"
)

print(
    f"Alert burden at target: "
    f"{best['alert_burden_pct']:.2f}%"
)


# ============================================
# STEP 9: SAVE RESULTS
# ============================================

results.to_csv(
    RESULTS_DIR / "recall_threshold_sweep.csv",
    index=False
)

comparison.to_csv(
    RESULTS_DIR / "recall_threshold_comparison.csv",
    index=False
)

print("\nResults saved successfully!")

print("results/recall_threshold_sweep.csv")

print("results/recall_threshold_comparison.csv")

print("\nIMPORTANT:")
print("These metrics are from the validation dataset.")
print("The saved CatBoost model has not been modified.")
print("The dashboard threshold remains unchanged.")

print("\nThreshold analysis: PASSED!")
