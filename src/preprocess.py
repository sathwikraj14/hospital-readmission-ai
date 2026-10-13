
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from feature_engineering import create_features


# ==================================================
# STEP 1: LOAD CLEANED DATASET
# ==================================================

project_root = Path(__file__).resolve().parent.parent

data_path = project_root / "data" / "cleaned_data.csv"

df = pd.read_csv(data_path)

print("Cleaned dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==================================================
# STEP 2: SEPARATE FEATURES AND TARGET
# ==================================================

y = df["readmitted_30_days"].copy()

columns_to_exclude = [
    "encounter_id",
    "patient_nbr",
    "readmitted",
    "readmitted_30_days"
]

# Apply our three engineered features
X = create_features(
    df.drop(columns=columns_to_exclude)
)

# Keep patient IDs for group-based splitting
patient_groups = df["patient_nbr"]

print("\nFeature engineering applied successfully!")
print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)
print("Unique patients:", patient_groups.nunique())

assert X.shape[1] == 48


# ==================================================
# STEP 3: TRAIN / VALIDATION / TEST SPLIT
# ==================================================

splitter1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_idx, temp_idx = next(
    splitter1.split(X, y, groups=patient_groups)
)

splitter2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_relative_idx, test_relative_idx = next(
    splitter2.split(
        X.iloc[temp_idx],
        y.iloc[temp_idx],
        groups=patient_groups.iloc[temp_idx]
    )
)

val_idx = temp_idx[val_relative_idx]
test_idx = temp_idx[test_relative_idx]


# ==================================================
# STEP 4: CREATE INDEPENDENT DATASETS
# ==================================================

X_train = X.iloc[train_idx].copy()
X_val = X.iloc[val_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_val = y.iloc[val_idx].copy()
y_test = y.iloc[test_idx].copy()

print("\nDataset split completed!")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# ==================================================
# STEP 5: VERIFY PATIENT-LEVEL SEPARATION
# ==================================================

train_patients = set(patient_groups.iloc[train_idx])
val_patients = set(patient_groups.iloc[val_idx])
test_patients = set(patient_groups.iloc[test_idx])

assert train_patients.isdisjoint(val_patients)
assert train_patients.isdisjoint(test_patients)
assert val_patients.isdisjoint(test_patients)

print("\nPatient overlap check: PASSED")


# ==================================================
# STEP 6: CHECK READMISSION DISTRIBUTIONS
# ==================================================

print("\nReadmission percentages:")
print(f"Training: {y_train.mean() * 100:.2f}%")
print(f"Validation: {y_val.mean() * 100:.2f}%")
print(f"Testing: {y_test.mean() * 100:.2f}%")


# ==================================================
# STEP 7: IDENTIFY FEATURE TYPES
# ==================================================

categorical_features = X_train.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

coded_categories = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id"
]

categorical_features = sorted(
    set(categorical_features + coded_categories)
)

numerical_features = [
    col for col in X_train.columns
    if col not in categorical_features
]

print("\nFeature type analysis:")
print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))

print("\nNumerical feature names:")
print(numerical_features)

assert (
    len(numerical_features) + len(categorical_features)
    == X_train.shape[1]
)

print("\nFeature classification check: PASSED")


# ==================================================
# STEP 8: HANDLE MISSING CATEGORICAL VALUES
# ==================================================

print("\nHandling missing categorical values...")

for column in categorical_features:

    X_train[column] = (
        X_train[column]
        .astype("string")
        .fillna("Missing")
        .replace("?", "Missing")
        .astype(str)
    )

    X_val[column] = (
        X_val[column]
        .astype("string")
        .fillna("Missing")
        .replace("?", "Missing")
        .astype(str)
    )

    X_test[column] = (
        X_test[column]
        .astype("string")
        .fillna("Missing")
        .replace("?", "Missing")
        .astype(str)
    )

print("Missing categorical values handled successfully!")

print("\nRemaining '?' values:")
print("Training:", (X_train == "?").sum().sum())
print("Validation:", (X_val == "?").sum().sum())
print("Testing:", (X_test == "?").sum().sum())


# ==================================================
# STEP 9: BUILD ML PREPROCESSING PIPELINE
# ==================================================

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )
    )
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

print("\nML preprocessing pipeline created!")


# ==================================================
# STEP 10: FIT AND VALIDATE PREPROCESSING
# ==================================================

print("\nFitting preprocessing on training data...")

X_train_processed = preprocessor.fit_transform(X_train)

print("Transforming validation data...")

X_val_processed = preprocessor.transform(X_val)

print("\nProcessed dataset shapes:")
print("Training:", X_train_processed.shape)
print("Validation:", X_val_processed.shape)


# ==================================================
# STEP 11: CHECK INVALID VALUES
# ==================================================

def check_invalid_values(data, name):

    values = data.data if hasattr(data, "toarray") else data

    missing_count = np.isnan(values).sum()
    infinite_count = np.isinf(values).sum()

    print(f"{name} missing values: {missing_count}")
    print(f"{name} infinite values: {infinite_count}")

    assert missing_count == 0
    assert infinite_count == 0


check_invalid_values(X_train_processed, "Training")
check_invalid_values(X_val_processed, "Validation")


# ==================================================
# STEP 12: FINAL VERIFICATION
# ==================================================

print("\n====================================")
print("FEATURE ENGINEERING + PREPROCESSING")
print("====================================")

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Testing shape:", X_test.shape)

print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))

print("\nEngineered features:")
print("1. total_previous_visits")
print("2. frequent_inpatient")
print("3. frequent_emergency")

print("\nPreprocessing validation: PASSED!")
