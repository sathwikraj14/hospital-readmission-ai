
from xgboost import XGBClassifier
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
# STEP 1: CALCULATE CLASS IMBALANCE
# ============================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nClass imbalance analysis:")
print("Negative samples:", negative_count)
print("Positive samples:", positive_count)
print("Scale positive weight:", round(scale_pos_weight, 2))


# ============================================
# STEP 2: CREATE XGBOOST PIPELINE
# ============================================

print("\nCreating XGBoost model...")

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    ))
])


# ============================================
# STEP 3: TRAIN MODEL
# ============================================

print("\nTraining XGBoost...")

model.fit(X_train, y_train)

print("XGBoost training completed!")


# ============================================
# STEP 4: MAKE VALIDATION PREDICTIONS
# ============================================

y_pred = model.predict(X_val)

y_prob = model.predict_proba(X_val)[:, 1]


# ============================================
# STEP 5: EVALUATE MODEL
# ============================================

print("\n===================================")
print("XGBOOST RESULTS")
print("===================================")

print(f"Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_val, y_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_val, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_val, y_prob):.4f}")
print(f"PR-AUC:    {average_precision_score(y_val, y_prob):.4f}")

print("\nXGBoost evaluation completed!")
