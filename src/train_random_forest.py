
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from preprocess import (
    X_train,
    X_val,
    y_train,
    y_val,
    preprocessor
)


# ============================================
# STEP 1: CREATE RANDOM FOREST PIPELINE
# ============================================

print("\nCreating Random Forest model...")

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])


# ============================================
# STEP 2: TRAIN MODEL
# ============================================

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Random Forest training completed!")


# ============================================
# STEP 3: MAKE VALIDATION PREDICTIONS
# ============================================

y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]


# ============================================
# STEP 4: EVALUATE MODEL
# ============================================

print("\n===================================")
print("RANDOM FOREST RESULTS")
print("===================================")

print(f"Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_val, y_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_val, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_val, y_prob):.4f}")
print(f"PR-AUC:    {average_precision_score(y_val, y_prob):.4f}")

print("\nRandom Forest evaluation completed!")
