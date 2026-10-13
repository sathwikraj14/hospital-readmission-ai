
import json
from pathlib import Path

import numpy as np
import optuna

from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from preprocess import (
    X_train,
    y_train,
    patient_groups,
    train_idx,
    preprocessor
)


# ============================================
# STEP 1: EXPERIMENT CONFIGURATION
# ============================================

RANDOM_STATE = 42
N_TRIALS = 10
N_FOLDS = 3

project_root = Path(__file__).resolve().parent.parent
results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)

# Patient IDs corresponding to the training data
train_groups = patient_groups.iloc[train_idx]

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())
imbalance_ratio = negative_count / positive_count

print("\nLIGHTGBM HYPERPARAMETER OPTIMIZATION")
print("Training encounters:", len(X_train))
print("Unique training patients:", train_groups.nunique())
print("Class imbalance ratio:", round(imbalance_ratio, 2))


# ============================================
# STEP 2: OPTUNA OBJECTIVE FUNCTION
# ============================================

def objective(trial):

    parameters = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 150, 500, step=50
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.15, log=True
        ),
        "num_leaves": trial.suggest_int(
            "num_leaves", 15, 63
        ),
        "max_depth": trial.suggest_int(
            "max_depth", 3, 10
        ),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", 20, 100
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-4, 10.0, log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-4, 10.0, log=True
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.7, 1.0
        ),
    }

    use_balanced_weights = trial.suggest_categorical(
        "use_balanced_weights", [False, True]
    )

    parameters["scale_pos_weight"] = (
        imbalance_ratio if use_balanced_weights else 1.0
    )

    parameters.update({
        "objective": "binary",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    })

    group_cv = GroupKFold(n_splits=N_FOLDS)
    fold_scores = []

    for fold, (fit_idx, eval_idx) in enumerate(
        group_cv.split(
            X_train, y_train, groups=train_groups
        ),
        start=1
    ):

        # Every fold gets its own preprocessing fit.
        # This prevents validation-fold leakage.
        model = Pipeline(steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", LGBMClassifier(**parameters))
        ])

        model.fit(
            X_train.iloc[fit_idx],
            y_train.iloc[fit_idx]
        )

        probabilities = model.predict_proba(
            X_train.iloc[eval_idx]
        )[:, 1]

        score = average_precision_score(
            y_train.iloc[eval_idx],
            probabilities
        )

        fold_scores.append(score)

    return float(np.mean(fold_scores))


# ============================================
# STEP 3: RUN OPTUNA
# ============================================

optuna.logging.set_verbosity(
    optuna.logging.WARNING
)

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(
        seed=RANDOM_STATE
    )
)

print("\nStarting Optuna trials...")

study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True
)


# ============================================
# STEP 4: DISPLAY RESULTS
# ============================================

print("\n===================================")
print("OPTUNA RESULTS")
print("===================================")

print("Completed trials:", len(study.trials))
print(f"Best cross-validation PR-AUC: {study.best_value:.4f}")

print("\nBest hyperparameters:")
for name, value in study.best_params.items():
    print(f"{name}: {value}")


# ============================================
# STEP 5: SAVE BEST PARAMETERS
# ============================================

output_path = results_dir / "lightgbm_best_params.json"

results = {
    "best_cv_pr_auc": study.best_value,
    "n_trials": N_TRIALS,
    "n_folds": N_FOLDS,
    "best_params": study.best_params
}

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nOptimization results saved to:")
print(output_path)
