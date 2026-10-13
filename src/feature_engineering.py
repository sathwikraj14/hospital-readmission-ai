
import pandas as pd
from pathlib import Path


# ============================================
# FEATURE ENGINEERING FUNCTION
# ============================================

def create_features(df):
    """
    Create additional healthcare utilization features.

    These transformations use only information
    available within each hospital encounter.
    """

    df = df.copy()

    # Total previous healthcare visits
    df["total_previous_visits"] = (
        df["number_inpatient"]
        + df["number_emergency"]
        + df["number_outpatient"]
    )

    # Frequent previous inpatient visits
    df["frequent_inpatient"] = (
        df["number_inpatient"] >= 3
    ).astype(int)

    # Frequent previous emergency visits
    df["frequent_emergency"] = (
        df["number_emergency"] >= 3
    ).astype(int)

    return df


# ============================================
# TEST FEATURE ENGINEERING
# ============================================

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent

    data_path = project_root / "data" / "cleaned_data.csv"

    df = pd.read_csv(data_path)

    print("Original dataset shape:", df.shape)

    df_engineered = create_features(df)

    print("Engineered dataset shape:", df_engineered.shape)

    print("\nNew feature names:")
    print([
        "total_previous_visits",
        "frequent_inpatient",
        "frequent_emergency"
    ])

    print("\nFirst 5 records:")
    print(
        df_engineered[
            [
                "total_previous_visits",
                "frequent_inpatient",
                "frequent_emergency"
            ]
        ].head()
    )

    assert df_engineered.shape[1] == df.shape[1] + 3

    print("\nFeature engineering validation: PASSED!")
