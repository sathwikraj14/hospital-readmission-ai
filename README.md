# Hospital Readmission Intelligence
### End-to-End Machine Learning | Healthcare Analytics | Explainable AI

An end-to-end machine learning project that investigates **30-day hospital readmission prediction** using historical hospital encounter data from patients with diabetes.

The project combines data preprocessing, feature engineering, machine learning model comparison, hyperparameter optimization, explainable AI, and an interactive Streamlit dashboard.

**Status:** Functional research prototype | Not clinically validated

---

## 1. Project Overview

Hospital readmissions are an important challenge in healthcare analytics. Identifying patterns associated with readmission may help researchers explore opportunities for improving care planning and evaluating predictive decision-support systems.

This project investigates whether demographic information, diagnoses, medications, previous healthcare utilization, and hospitalization characteristics can help predict readmission within 30 days.

The final application presents model performance, experimental classification-threshold scenarios, synthetic encounter scoring, and SHAP explanations.

## 2. Dataset

**Source:** UCI Machine Learning Repository — Diabetes 130-US Hospitals for Years 1999–2008.

Dataset information: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

| Dataset characteristic | Value |
|---|---:|
| Original hospital encounters | 101,766 |
| Cleaned encounters | 99,343 |
| Unique patients after cleaning | 69,990 |
| Model input features | 48 |
| Train encounters | 69,519 |
| Validation encounters | 14,911 |
| Test encounters | 14,913 |
| 30-day readmission rate | Approximately 11.4% |

The target variable is binary:

- **1:** Readmitted within 30 days
- **0:** Not readmitted within 30 days

Patient-group-separated train, validation, and test partitions were used to reduce leakage across encounters belonging to the same patient.

## 3. Machine Learning Workflow

The project includes:

1. Exploratory data analysis and data cleaning.
2. Removal of excluded discharge dispositions and noninformative variables.
3. Missing-value handling and categorical encoding.
4. Patient-group-aware dataset splitting.
5. Feature engineering using previous inpatient, emergency, and outpatient utilization.
6. Benchmarking five machine learning algorithms.
7. Optuna hyperparameter optimization.
8. Classification-threshold analysis using validation data.
9. Final evaluation on a held-out patient-group test set.
10. SHAP-based global model explainability.
11. Streamlit dashboard development and synthetic-data inference demonstration.

### Models evaluated

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

**Selected candidate:** Tuned CatBoost classifier.

## 4. Final Model Performance

The selected CatBoost pipeline was evaluated on **14,913 held-out hospital encounters** at a classification threshold of **0.14**.

| Metric | Final Test Result |
|---|---:|
| ROC-AUC | 0.6761 |
| PR-AUC | 0.2354 |
| Accuracy | 75.40% |
| Precision | 20.64% |
| Recall | 42.64% |
| F1-score | 0.2782 |

### Confusion Matrix

| Actual / Predicted | No Readmission | Readmission |
|---|---:|---:|
| No Readmission | 10,537 | 2,718 |
| Readmission | 951 | 707 |

At the saved classification threshold, the model identified **707 of 1,658 actual readmissions**, while missing **951**.

These results demonstrate moderate discriminatory performance and highlight the importance of evaluating false negatives, false positives, and alert burden rather than accuracy alone.

## 5. Classification Threshold Analysis

Additional experiments investigated whether adjusting the prediction threshold could increase readmission detection.

**Validation-set results:**

| Setting | Threshold | Recall | Precision | Alert Burden |
|---|---:|---:|---:|---:|
| Current setting | 0.140 | 44.71% | 22.09% | 23.62% |
| 70% recall scenario | 0.096 | 70.69% | 16.88% | 48.86% |
| 80% recall scenario | 0.081 | 80.00% | 15.05% | 62.02% |

Higher recall was achievable on the validation set by lowering the classification threshold, but this substantially increased the proportion of encounters flagged for review.

These scenarios are **retrospective validation experiments**, not evaluated clinical operating recommendations or replacements for the final test results.

Changing the classification threshold does not improve the model's ROC-AUC.

## 6. Explainable AI with SHAP

SHAP analysis was used to examine global feature importance using a sample of 200 validation encounters.

Influential encoded features included:

- Previous inpatient visits
- Discharge disposition
- Total previous healthcare visits
- Number of diagnoses
- Hospital length of stay
- Missing payer information

SHAP values describe how input features contribute to model predictions. They do not establish causal relationships or clinical treatment effects.

## 7. Interactive Streamlit Dashboard

The **Hospital Readmission Intelligence** dashboard contains four sections.

**Executive Overview:** Displays the final test results, readmission detection breakdown, executive interpretation, and interactive validation-threshold scenarios.

**Risk Assessment:** Provides a synthetic sample CSV, encounter file upload, CatBoost scoring, classification flags, and downloadable results.

**Model Performance:** Compares five machine learning algorithms and explains the selected candidate's final held-out test performance.

**Explainable AI:** Displays readable SHAP feature-importance charts, detailed encoded feature contributions, and model transparency guidance.

The dashboard is designed as a research and educational demonstration. It is not a clinically validated patient-risk assessment product.

## 8. Technology Stack

**Programming and data:** Python, pandas, NumPy

**Machine learning:** scikit-learn, Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost

**Optimization:** Optuna

**Explainability:** SHAP

**Visualization and UI:** Streamlit, Altair, Matplotlib

**Development and version control:** VS Code, Git, GitHub

## 9. Local Setup

Clone the repository:

```bash
git clone https://github.com/sathwikraj14/hospital-readmission-ai.git
cd hospital-readmission-ai
```

Create and activate a Python virtual environment.

For Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Data and model artifacts

The original dataset, cleaned dataset, and serialized model are intentionally excluded from Git tracking.

To run the full pipeline locally, obtain the source dataset from UCI and reproduce the preprocessing, training, and final model artifact generation using the scripts in `src/`.

The dashboard's scoring functionality requires:

- `models/catboost_candidate.joblib`
- `models/catboost_candidate_metadata.json`

These files are generated by the model-training workflow and are not currently distributed with the repository.

To launch the dashboard after generating the required artifacts:

```powershell
python -m streamlit run app/app.py
```

The application's Executive Overview, Model Performance, and Explainable AI pages also depend on the corresponding saved results files in `results/`.

**Note:** A fully automated clean-environment reproduction and public deployment workflow has not yet been completed.

## 10. Current Status and Future Improvements

Completed functionality includes the machine learning pipeline, five-model benchmarks, CatBoost optimization, final test evaluation, SHAP global explanations, threshold analysis, synthetic encounter scoring, and an interactive Streamlit application.

Potential future improvements include expanded diagnosis feature engineering, native categorical modeling, probability calibration, additional validation, and improved model-artifact packaging.

## 11. Limitations and Responsible Use

- The dataset represents hospital encounters recorded between 1999 and 2008 and may not reflect modern healthcare populations.
- The final model has moderate discrimination and misses a substantial proportion of readmissions at its selected threshold.
- Accuracy alone is misleading because the dataset is imbalanced.
- Predicted model scores are not established clinically calibrated risk probabilities.
- Synthetic demo encounters are intended for software testing and do not represent realistic patient outcome distributions.
- The system has not undergone external validation, prospective evaluation, clinical safety assessment, or deployment governance review.

**Disclaimer:** This project is intended solely for research, education, and machine learning portfolio demonstration. It must not be used for medical diagnosis, patient triage, treatment decisions, or clinical decision-making.