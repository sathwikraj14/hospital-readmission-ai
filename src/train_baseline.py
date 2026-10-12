
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.pipeline import Pipeline

from preprocess import (
    X_train,
    X_val,
    y_train,
    y_val,
    preprocessor
)


# ============================================
# STEP 1: CREATE LOGISTIC REGRESSION PIPELINE
# ============================================

print("\nCreating Logistic Regression model...")

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42
    ))
])


# ============================================
# STEP 2: TRAIN MODEL
# ============================================

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

print("Model training completed!")


# ============================================
# STEP 3: MAKE VALIDATION PREDICTIONS
# ============================================

y_pred = model.predict(X_val)

y_prob = model.predict_proba(X_val)[:, 1]


# ============================================
# STEP 4: EVALUATE MODEL
# ============================================

print("\n===================================")
print("LOGISTIC REGRESSION RESULTS")
print("===================================")

print(f"Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_val, y_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_val, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_val, y_prob):.4f}")
print(f"PR-AUC:    {average_precision_score(y_val, y_prob):.4f}")

print("\nBaseline model evaluation completed!")
