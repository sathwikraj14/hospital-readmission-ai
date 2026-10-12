
import pandas as pd
from pathlib import Path

# Locate the dataset
project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "data" / "diabetic_data.csv"

# Load the hospital dataset
df = pd.read_csv(file_path)

# Display basic information
print("Dataset loaded successfully!")

print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nFirst 5 records:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nReadmission category counts:")
print(df["readmitted"].value_counts())

print("\nReadmission category percentages:")
print(df["readmitted"].value_counts(normalize=True) * 100)

# Create binary target for 30-day readmission
df["readmitted_30_days"] = (
    df["readmitted"] == "<30"
).astype(int)

print("\nBinary target distribution:")
print(df["readmitted_30_days"].value_counts())

print("\nBinary target percentages:")
print(
    df["readmitted_30_days"]
    .value_counts(normalize=True) * 100
)

# Check missing values
print("\nMissing values in each column:")

missing_values = df.replace("?", pd.NA).isna().sum()

print(missing_values[missing_values > 0].sort_values(ascending=False))

# Check duplicate records
print("\nDuplicate analysis:")

print("Exact duplicate rows:", df.duplicated().sum())

print(
    "Duplicate encounter IDs:",
    df["encounter_id"].duplicated().sum()
)

print(
    "Unique patients:",
    df["patient_nbr"].nunique()
)

# Check duplicate records
print("\nDuplicate analysis:")

print("Exact duplicate rows:", df.duplicated().sum())

print("Duplicate encounter IDs:", df["encounter_id"].duplicated().sum())

print("Unique patients:", df["patient_nbr"].nunique())

# Check column data types
print("\nDataset column data types:")
print(df.dtypes)

print("\nData type summary:")
print(df.dtypes.value_counts())

# Identify columns with only one unique value
print("\nConstant columns:")

unique_counts = df.nunique(dropna=False)

constant_columns = unique_counts[unique_counts <= 1]

print(constant_columns)

print("\nTotal constant columns:", len(constant_columns))

# Examine important categorical features
print("\nGender distribution:")
print(df["gender"].value_counts(dropna=False))

print("\nAge distribution:")
print(df["age"].value_counts(dropna=False).sort_index())

print("\nDischarge disposition distribution:")
print(df["discharge_disposition_id"].value_counts().sort_index())

# Calculate 30-day readmission rate by age group
print("\n30-day readmission rate by age group:")

age_readmission = (
    df.groupby("age")["readmitted_30_days"]
    .agg(["count", "sum", "mean"])
)

age_readmission["readmission_rate_%"] = (
    age_readmission["mean"] * 100
)

print(
    age_readmission[
        ["count", "sum", "readmission_rate_%"]
    ].round(2)
)

# Analyze previous inpatient visits and readmission risk
print("\n30-day readmission by previous inpatient visits:")

df["previous_admission_group"] = pd.cut(
    df["number_inpatient"],
    bins=[-1, 0, 1, 2, 5, float("inf")],
    labels=["0", "1", "2", "3-5", "6+"]
)

inpatient_analysis = (
    df.groupby(
        "previous_admission_group",
        observed=True
    )["readmitted_30_days"]
    .agg(["count", "sum", "mean"])
)

inpatient_analysis["readmission_rate_%"] = (
    inpatient_analysis["mean"] * 100
)

print(
    inpatient_analysis[
        ["count", "sum", "readmission_rate_%"]
    ].round(2)
)

# Analyze emergency visits and readmission risk
print("\n30-day readmission by previous emergency visits:")

df["emergency_visit_group"] = pd.cut(
    df["number_emergency"],
    bins=[-1, 0, 1, 2, 5, float("inf")],
    labels=["0", "1", "2", "3-5", "6+"]
)

emergency_analysis = (
    df.groupby(
        "emergency_visit_group",
        observed=True
    )["readmitted_30_days"]
    .agg(["count", "sum", "mean"])
)

emergency_analysis["readmission_rate_%"] = (
    emergency_analysis["mean"] * 100
)

print(
    emergency_analysis[
        ["count", "sum", "readmission_rate_%"]
    ].round(2)
)

# Analyze hospital stay duration and readmission
print("\n30-day readmission by hospital stay duration:")

df["stay_duration_group"] = pd.cut(
    df["time_in_hospital"],
    bins=[0, 2, 4, 7, float("inf")],
    labels=["1-2 days", "3-4 days", "5-7 days", "8+ days"]
)

stay_analysis = (
    df.groupby(
        "stay_duration_group",
        observed=True
    )["readmitted_30_days"]
    .agg(["count", "sum", "mean"])
)

stay_analysis["readmission_rate_%"] = (
    stay_analysis["mean"] * 100
)

print(
    stay_analysis[
        ["count", "sum", "readmission_rate_%"]
    ].round(2)
)

# Identify records requiring exclusion review
print("\nDischarge exclusion analysis:")

exclusion_codes = [11, 13, 14, 19, 20, 21]

excluded = df[
    df["discharge_disposition_id"].isin(exclusion_codes)
]

print("Total records:", len(df))
print("Records flagged for exclusion:", len(excluded))
print("Remaining records:", len(df) - len(excluded))

print("\nRecords by exclusion code:")
print(
    excluded["discharge_disposition_id"]
    .value_counts()
    .sort_index()
)

# Create an initial modeling cohort
print("\nCreating initial modeling cohort...")

cleaned_df = df[
    ~df["discharge_disposition_id"].isin(exclusion_codes)
].copy()

# Remove temporary columns created during exploration
temporary_columns = [
    "previous_admission_group",
    "emergency_visit_group",
    "stay_duration_group"
]

cleaned_df = cleaned_df.drop(
    columns=temporary_columns,
    errors="ignore"
)

# Remove constant columns
cleaned_df = cleaned_df.drop(
    columns=["examide", "citoglipton"]
)

# Save cleaned data
output_path = project_root / "data" / "cleaned_data.csv"
cleaned_df.to_csv(output_path, index=False)

print("Cleaned dataset saved successfully!")
print("Cleaned dataset shape:", cleaned_df.shape)
print("Saved to:", output_path)

# Check target distribution after cohort cleaning
print("\nCleaned dataset target distribution:")

print(
    cleaned_df["readmitted_30_days"].value_counts()
)

print("\nCleaned dataset target percentages:")

print(
    (
        cleaned_df["readmitted_30_days"]
        .value_counts(normalize=True) * 100
    ).round(2)
)

print("\nUnique patients after cleaning:")
print(cleaned_df["patient_nbr"].nunique())
