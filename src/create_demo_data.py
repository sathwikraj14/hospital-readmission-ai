
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# ============================================
# STEP 1: PROJECT PATHS
# ============================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "models" / "catboost_candidate.joblib"
)

OUTPUT_PATH = (
    ROOT / "data" / "demo_encounters.csv"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================
# STEP 2: LOAD TRAINED MODEL
# ============================================

print("\nLoading trained CatBoost model...")

model = joblib.load(MODEL_PATH)

preprocessor = model.named_steps[
    "preprocessor"
]

print("Model loaded successfully!")


# ============================================
# STEP 3: IDENTIFY MODEL FEATURES
# ============================================

numeric_columns = list(
    preprocessor.transformers_[0][2]
)

categorical_columns = list(
    preprocessor.transformers_[1][2]
)

engineered_features = {
    "total_previous_visits",
    "frequent_inpatient",
    "frequent_emergency"
}

print("\nFeature information:")

print(
    "Numerical features:",
    len(numeric_columns)
)

print(
    "Categorical features:",
    len(categorical_columns)
)


# ============================================
# STEP 4: FIND ONE-HOT ENCODER
# ============================================

print("\nFinding categorical encoder...")

categorical_transformer = (
    preprocessor.named_transformers_["cat"]
)

if isinstance(
    categorical_transformer,
    Pipeline
):

    encoder = next(
        (
            step
            for step in (
                categorical_transformer
                .named_steps
                .values()
            )
            if isinstance(
                step,
                OneHotEncoder
            )
        ),
        None
    )

else:

    encoder = categorical_transformer


if not isinstance(
    encoder,
    OneHotEncoder
):

    raise ValueError(
        "OneHotEncoder not found in "
        "categorical preprocessing pipeline."
    )


print("OneHotEncoder found successfully!")


# ============================================
# STEP 5: GENERATE SYNTHETIC RECORDS
# ============================================

print("\nGenerating synthetic encounters...")

rng = np.random.default_rng(42)

number_of_encounters = 20

data = {}

numeric_ranges = {
    "time_in_hospital": (1, 14),
    "num_lab_procedures": (1, 100),
    "num_procedures": (0, 6),
    "num_medications": (1, 50),
    "number_outpatient": (0, 5),
    "number_emergency": (0, 5),
    "number_inpatient": (0, 6),
    "number_diagnoses": (1, 9)
}


# ============================================
# STEP 6: GENERATE NUMERICAL FEATURES
# ============================================

for column in numeric_columns:

    if column in engineered_features:
        continue

    low, high = numeric_ranges.get(
        column,
        (0, 10)
    )

    data[column] = rng.integers(
        low,
        high + 1,
        size=number_of_encounters
    )


# ============================================
# STEP 7: GENERATE CATEGORICAL FEATURES
# ============================================

for column, categories in zip(
    categorical_columns,
    encoder.categories_
):

    valid_categories = [
        str(value)
        for value in categories
        if str(value) not in [
            "Missing",
            "?",
            "nan",
            "<NA>"
        ]
    ]

    if not valid_categories:
        valid_categories = ["Missing"]

    data[column] = rng.choice(
        valid_categories,
        size=number_of_encounters
    )


# ============================================
# STEP 8: CREATE DATAFRAME
# ============================================

demo_df = pd.DataFrame(data)

required_columns = [
    column
    for column in (
        numeric_columns +
        categorical_columns
    )
    if column not in engineered_features
]

demo_df = demo_df[
    required_columns
].copy()


# ============================================
# STEP 9: VALIDATE DEMO DATA
# ============================================

expected_column_count = (
    len(numeric_columns) +
    len(categorical_columns) -
    len(engineered_features)
)

assert len(demo_df) == number_of_encounters

assert (
    demo_df.shape[1] ==
    expected_column_count
)

assert not demo_df.empty

print("\nValidation checks passed!")

print(
    "Total encounters:",
    len(demo_df)
)

print(
    "Input columns:",
    len(demo_df.columns)
)


# ============================================
# STEP 10: SAVE DEMO CSV
# ============================================

demo_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSynthetic demo dataset created!")

print("Saved to:", OUTPUT_PATH)

print("\nFirst 5 records:")

print(
    demo_df.head().to_string(
        index=False,
        max_cols=8
    )
)

print("\nDemo data generation: PASSED!")

print(
    "\nNOTE: These records are randomly "
    "generated for software testing only. "
    "They are not clinically representative."
)
