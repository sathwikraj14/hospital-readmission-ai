
import json
from pathlib import Path

import joblib
import pandas as pd

from feature_engineering import create_features


# ============================================
# STEP 1: PROJECT PATHS
# ============================================

project_root = Path(__file__).resolve().parent.parent

model_path = project_root / "models" / "catboost_candidate.joblib"
metadata_path = project_root / "models" / "catboost_candidate_metadata.json"
data_path = project_root / "data" / "cleaned_data.csv"


# ============================================
# STEP 2: LOAD SAVED MODEL
# ============================================

print("\nLoading saved CatBoost model...")

model = joblib.load(model_path)

with open(metadata_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)

threshold = metadata["classification_threshold"]
expected_features = metadata["feature_names"]

print("Saved model loaded successfully!")


# ============================================
# STEP 3: LOAD SAMPLE PATIENT ENCOUNTERS
# ============================================

df = pd.read_csv(data_path)

sample = df.head(5).copy()

encounter_ids = sample["encounter_id"].copy()

# Remove identifiers and outcome columns
sample_features = sample.drop(
    columns=[
        "encounter_id",
        "patient_nbr",
        "readmitted",
        "readmitted_30_days"
    ]
)


# ============================================
# STEP 4: APPLY FEATURE ENGINEERING
# ============================================

sample_features = create_features(sample_features)

# Apply the same categorical missing-value
# handling used during model training.
categorical_columns = model.named_steps[
    "preprocessor"
].transformers_[1][2]

for column in categorical_columns:
    sample_features[column] = (
        sample_features[column]
        .astype("string")
        .fillna("Missing")
        .replace("?", "Missing")
        .astype(str)
    )

# Validate input schema before prediction
missing_columns = (
    set(expected_features) - set(sample_features.columns)
)

extra_columns = (
    set(sample_features.columns) - set(expected_features)
)

assert not missing_columns, f"Missing columns: {missing_columns}"
assert not extra_columns, f"Unexpected columns: {extra_columns}"

sample_features = sample_features[expected_features]

print("\nFeature engineering and schema check: PASSED!")
print("Input shape:", sample_features.shape)


# ============================================
# STEP 5: MAKE PREDICTIONS
# ============================================

probabilities = model.predict_proba(
    sample_features
)[:, 1]

predictions = (
    probabilities >= threshold
).astype(int)


# ============================================
# STEP 6: DISPLAY RESULTS
# ============================================

results = pd.DataFrame({
    "encounter_id": encounter_ids.to_numpy(),
    "risk_score": probabilities.round(4),
    "predicted_class": predictions
})

results["risk_flag"] = results["predicted_class"].map({
    0: "Not flagged",
    1: "Flagged"
})

print("\n===================================")
print("HOSPITAL READMISSION PREDICTIONS")
print("===================================")

print(results.to_string(index=False))

print("\nClassification threshold:", threshold)
print("\nPrediction pipeline validation: PASSED!")
