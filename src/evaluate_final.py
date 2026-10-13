
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
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    log_loss
)

from preprocess import (
    X_test,
    y_test
)


# ============================================
# STEP 1: LOAD SAVED MODEL
# ============================================

project_root = Path(__file__).resolve().parent.parent

model_path = (
    project_root / "models" / "catboost_candidate.joblib"
)

metadata_path = (
    project_root / "models" /
    "catboost_candidate_metadata.json"
)

model = joblib.load(model_path)

with open(metadata_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)

threshold = float(metadata["classification_threshold"])
expected_features = metadata["feature_names"]

print("\nSaved CatBoost model loaded!")
print("Classification threshold:", threshold)


# ============================================
# STEP 2: VALIDATE TEST DATA SCHEMA
# ============================================

assert list(X_test.columns) == expected_features, (
    "Test features do not match the saved model schema."
)

assert len(X_test) == len(y_test)

print("\nTest dataset shape:", X_test.shape)
print("Feature schema validation: PASSED!")


# ============================================
# STEP 3: GENERATE TEST PREDICTIONS
# ============================================

y_prob = model.predict_proba(X_test)[:, 1]

y_pred = (y_prob >= threshold).astype(int)

assert np.isfinite(y_prob).all()
assert ((y_prob >= 0) & (y_prob <= 1)).all()

print("Test predictions generated successfully!")


# ============================================
# STEP 4: COMPUTE EVALUATION METRICS
# ============================================

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(
        precision_score(y_test, y_pred, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, y_pred, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_test, y_pred, zero_division=0)
    ),
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
    "pr_auc": float(
        average_precision_score(y_test, y_prob)
    ),
    "brier_score": float(
        brier_score_loss(y_test, y_prob)
    ),
    "log_loss": float(log_loss(y_test, y_prob))
}


# ============================================
# STEP 5: DISPLAY RESULTS
# ============================================

print("\n====================================")
print("FINAL CATBOOST TEST RESULTS")
print("====================================")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\nConfusion matrix:")
print(cm)

print("\nActual readmissions:", int(y_test.sum()))
print("Readmissions detected:", int(tp))
print("Readmissions missed:", int(fn))
print("False positive flags:", int(fp))
print("Total encounters flagged:", int(tp + fp))


# ============================================
# STEP 6: SAVE TEST RESULTS
# ============================================

results_dir = project_root / "results"
results_dir.mkdir(parents=True, exist_ok=True)

output = {
    "model": "Tuned CatBoost",
    "dataset": "held_out_patient_group_test",
    "test_encounters": len(X_test),
    "threshold": threshold,
    "metrics": metrics,
    "confusion_matrix": cm.tolist()
}

output_path = results_dir / "final_test_results.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(output, file, indent=4)

print("\nTest results saved to:")
print(output_path)

print("\nFINAL TEST EVALUATION COMPLETED!")
