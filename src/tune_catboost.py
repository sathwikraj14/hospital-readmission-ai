
import json
from pathlib import Path

import numpy as np
import optuna

from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.metrics import average_precision_score
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
# STEP 1: EXPERIMENT SETTINGS
# ============================================

RANDOM_STATE = 42
N_TRIALS = 10
N_FOLDS = 3

project_root = Path(__file__).resolve().parent.parent
results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)

train_groups = patient_groups.iloc[train_idx]

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

imbalance_ratio = negative_count / positive_count

print("\nCATBOOST HYPERPARAMETER OPTIMIZATION")
print("Training encounters:", len(X_train))
print("Unique training patients:", train_groups.nunique())
print("Class imbalance ratio:", round(imbalance_ratio, 2))


# ============================================
# STEP 2: OPTUNA OBJECTIVE
# ============================================

def objective(trial):

    parameters = {
        "iterations": trial.suggest_int(
            "iterations", 200, 500, step=50
        ),
        "depth": trial.suggest_int(
            "depth", 4, 8
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.15, log=True
        ),
        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg", 1.0, 10.0
        ),
        "random_strength": trial.suggest_float(
            "random_strength", 0.0, 2.0
        ),
    }

    use_balanced_weights = trial.suggest_categorical(
        "use_balanced_weights", [False, True]
    )

    parameters["scale_pos_weight"] = (
        imbalance_ratio if use_balanced_weights else 1.0
    )

    parameters.update({
        "loss_function": "Logloss",
        "random_seed": RANDOM_STATE,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1
    })

    group_cv = GroupKFold(n_splits=N_FOLDS)
    fold_scores = []

    for fit_idx, eval_idx in group_cv.split(
        X_train,
        y_train,
        groups=train_groups
    ):

        model = Pipeline(steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", CatBoostClassifier(**parameters))
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

print("\nStarting CatBoost optimization...")

study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True
)


# ============================================
# STEP 4: DISPLAY BEST RESULTS
# ============================================

print("\n===================================")
print("CATBOOST OPTUNA RESULTS")
print("===================================")

print("Completed trials:", len(study.trials))

print(
    f"Best cross-validation PR-AUC: "
    f"{study.best_value:.4f}"
)

print("\nBest hyperparameters:")

for name, value in study.best_params.items():
    print(f"{name}: {value}")


# ============================================
# STEP 5: SAVE RESULTS
# ============================================

output_path = (
    results_dir / "catboost_best_params.json"
)

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

print("\nCatBoost optimization completed!")
