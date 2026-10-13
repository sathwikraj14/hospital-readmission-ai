
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

from pathlib import Path
from scipy import sparse

from preprocess import X_val

# ============================================
# STEP 1: LOAD SAVED MODEL
# ============================================

project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "models" / "catboost_candidate.joblib"
results_dir = project_root / "results"

results_dir.mkdir(exist_ok=True)

print("\nLoading saved CatBoost model...")

model = joblib.load(model_path)

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

print("Model loaded successfully!")


# ============================================
# STEP 2: SELECT VALIDATION SAMPLE
# ============================================

sample = X_val.sample(
    n=200,
    random_state=42
).copy()

# Apply the categorical normalization
# used during model training.
categorical_columns = preprocessor.transformers_[1][2]

for column in categorical_columns:
    sample[column] = (
        sample[column]
        .astype("string")
        .fillna("Missing")
        .replace("?", "Missing")
        .astype(str)
    )

print("Sample size:", len(sample))


# ============================================
# STEP 3: PREPROCESS SAMPLE
# ============================================

X_processed = preprocessor.transform(sample)

if sparse.issparse(X_processed):
    X_processed = X_processed.toarray()

feature_names = preprocessor.get_feature_names_out()

print("Processed feature count:", len(feature_names))


# ============================================
# STEP 4: CALCULATE SHAP VALUES
# ============================================

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(classifier)

shap_values = explainer.shap_values(X_processed)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

shap_values = np.asarray(shap_values)

print("SHAP calculation completed!")


# ============================================
# STEP 5: SAVE FEATURE IMPORTANCE
# ============================================

importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
})

importance = importance.sort_values(
    "mean_abs_shap",
    ascending=False
)

importance.to_csv(
    results_dir / "shap_feature_importance.csv",
    index=False
)

print("\nTop 10 SHAP features:")
print(importance.head(10).to_string(index=False))


# ============================================
# STEP 6: CREATE SHAP CHART
# ============================================

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_values,
    X_processed,
    feature_names=feature_names,
    plot_type="bar",
    max_display=20,
    show=False
)

plt.tight_layout()

output_path = results_dir / "shap_feature_importance.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSHAP chart saved to:")
print(output_path)

print("\nSHAP explainability analysis: PASSED!")
