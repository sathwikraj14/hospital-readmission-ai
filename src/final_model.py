
import json
from pathlib import Path

import joblib
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline

from preprocess import (
    X_train,
    X_val,
    y_train,
    y_val,
    preprocessor
)
from feature_engineering import create_features


# ============================================
# STEP 1: PROJECT PATHS
# ============================================

project_root = Path(__file__).resolve().parent.parent

results_dir = project_root / "results"
models_dir = project_root / "models"

models_dir.mkdir(parents=True, exist_ok=True)


# ============================================
# STEP 2: LOAD OPTUNA PARAMETERS
# ============================================

params_path = results_dir / "catboost_best_params.json"

with open(params_path, "r", encoding="utf-8") as file:
    optimization_results = json.load(file)

best_params = optimization_results["best_params"].copy()

use_balanced_weights = best_params.pop(
    "use_balanced_weights"
)

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

best_params["scale_pos_weight"] = (
    negative_count / positive_count
    if use_balanced_weights
    else 1.0
)


# ============================================
# STEP 3: CREATE MODEL PIPELINE
# ============================================

print("\nCreating final CatBoost candidate...")

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


# ============================================
# STEP 4: TRAIN MODEL
# ============================================

print("\nTraining CatBoost candidate...")

model.fit(X_train, y_train)

print("Model training completed!")


# ============================================
# STEP 5: SAVE MODEL AND METADATA
# ============================================

model_path = models_dir / "catboost_candidate.joblib"

joblib.dump(model, model_path)

metadata = {
    "model_name": "Tuned CatBoost",
    "model_status": "candidate",
    "classification_threshold": 0.14,
    "expected_feature_count": X_train.shape[1],
    "feature_names": X_train.columns.tolist(),
    "feature_engineering": [
        "total_previous_visits",
        "frequent_inpatient",
        "frequent_emergency"
    ],
    "hyperparameters": best_params
}

metadata_path = models_dir / "catboost_candidate_metadata.json"

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)


# ============================================
# STEP 6: VERIFY SAVED MODEL
# ============================================

print("\nVerifying saved model...")

loaded_model = joblib.load(model_path)

original_probabilities = model.predict_proba(
    X_val.head(5)
)[:, 1]

loaded_probabilities = loaded_model.predict_proba(
    X_val.head(5)
)[:, 1]

import numpy as np

assert np.allclose(
    original_probabilities,
    loaded_probabilities
)

print("Saved model verification: PASSED!")

print("\n===================================")
print("CATBOOST MODEL CANDIDATE SAVED")
print("===================================")

print("Model:", model_path)
print("Metadata:", metadata_path)
print("Classification threshold:", 0.14)
print("Expected features:", X_train.shape[1])
