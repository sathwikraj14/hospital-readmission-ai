
import json
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from threshold_panel import show_threshold_panel


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
DATA = ROOT / "data"

sys.path.insert(0, str(ROOT / "src"))

from feature_engineering import create_features


st.set_page_config(
    page_title="Hospital Readmission Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# 2. THEME
# =====================================================

st.markdown(
    """
<style>
.stApp {
    background-color: #F5F8FC;
    color: #15283B;
}

[data-testid="stSidebar"] {
    background-color: #172B45;
}

[data-testid="stSidebar"] * {
    color: white;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #15283B;
}

[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #E1E9F1;
    box-shadow: 0 3px 10px rgba(20,40,60,.03);
}

[data-testid="stMetricLabel"] {
    color: #51677B;
}

[data-testid="stMetricValue"] {
    color: #123B57;
}

.hero {
    padding: 34px;
    border-radius: 18px;
    background: linear-gradient(115deg,#102840,#166A80);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 34px;
}

.hero p {
    color: #E2F1F5;
    font-size: 16px;
}

.section-label {
    color: #167C91;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.info-card {
    background: white;
    padding: 22px;
    border: 1px solid #E1E9F1;
    border-radius: 14px;
    color: #243B53;
    min-height: 145px;
}

.info-card h3 {
    color: #123B57;
    margin-top: 0;
}

.insight {
    background: #EAF4F6;
    border-left: 5px solid #167C91;
    padding: 24px;
    border-radius: 12px;
    color: #173F50;
}

.insight h3 {
    color: #12536A;
    margin-top: 0;
}

.detection-card {
    background: white;
    padding: 24px;
    border: 1px solid #E1E9F1;
    border-radius: 15px;
    margin: 16px 0 24px;
}

.detection-track {
    display: flex;
    width: 100%;
    height: 30px;
    border-radius: 9px;
    overflow: hidden;
}

.legend {
    display: flex;
    flex-wrap: wrap;
    gap: 24px;
    margin-top: 15px;
}

.dot {
    display: inline-block;
    width: 11px;
    height: 11px;
    border-radius: 3px;
    margin-right: 7px;
}

.small-note {
    color: #64748B;
    font-size: 13px;
}
</style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# 3. LOAD PROJECT ARTIFACTS
# =====================================================

@st.cache_data
def load_json(path_string):
    path = Path(path_string)

    if not path.exists():
        return None

    with path.open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_csv(path_string):
    path = Path(path_string)

    if not path.exists():
        return None

    return pd.read_csv(path)


@st.cache_resource
def load_prediction_model():
    model_path = MODELS / "catboost_candidate.joblib"
    metadata_path = MODELS / "catboost_candidate_metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        return None, None

    model = joblib.load(model_path)

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    return model, metadata


test_results = load_json(
    str(RESULTS / "final_test_results.json")
)

comparison = load_csv(
    str(RESULTS / "model_comparison.csv")
)


# =====================================================
# 4. HELPER FUNCTIONS
# =====================================================

def section_label(text):
    st.markdown(
        f'<div class="section-label">{text}</div>',
        unsafe_allow_html=True,
    )


def detection_bar(detected, missed):
    total = detected + missed

    if total == 0:
        st.info("No readmissions available.")
        return

    detected_pct = 100 * detected / total
    missed_pct = 100 - detected_pct

    html = f"""
<div class="detection-card">
  <h3>Readmission Detection Breakdown</h3>
  <p>Of {total:,} actual readmissions:</p>
  <div class="detection-track">
    <div style="width:{detected_pct:.4f}%;background:#168B82"></div>
    <div style="width:{missed_pct:.4f}%;background:#D97757"></div>
  </div>
  <div class="legend">
    <span>
      <span class="dot" style="background:#168B82"></span>
      Identified: {detected:,} ({detected_pct:.1f}%)
    </span>
    <span>
      <span class="dot" style="background:#D97757"></span>
      Missed: {missed:,} ({missed_pct:.1f}%)
    </span>
  </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def horizontal_comparison(data, metric, title, color):
    if data is None:
        return

    if "Model" not in data.columns or metric not in data.columns:
        st.warning(f"Missing data for {title}.")
        return

    chart_data = (
        data[["Model", metric]]
        .dropna()
        .sort_values(metric, ascending=False)
        .copy()
    )

    chart = (
        alt.Chart(chart_data)
        .mark_bar(
            color=color,
            cornerRadiusEnd=5,
        )
        .encode(
            x=alt.X(
                f"{metric}:Q",
                title=title,
                scale=alt.Scale(zero=True),
            ),
            y=alt.Y(
                "Model:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelLimit=190),
            ),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip(
                    f"{metric}:Q",
                    format=".4f",
                    title=title,
                ),
            ],
        )
        .properties(height=245)
    )

    labels = (
        alt.Chart(chart_data)
        .mark_text(
            align="left",
            baseline="middle",
            dx=6,
            color="#1B3851",
        )
        .encode(
            x=alt.X(f"{metric}:Q"),
            y=alt.Y("Model:N", sort="-x"),
            text=alt.Text(f"{metric}:Q", format=".4f"),
        )
    )

    st.subheader(title)

    st.altair_chart(
        (chart + labels).properties(
            padding={"right": 55}
        ),
        width="stretch",
    )


def readable_feature(name):
    names = {
        "number_inpatient": "Previous Inpatient Visits",
        "total_previous_visits": "Total Previous Healthcare Visits",
        "number_diagnoses": "Number of Diagnoses",
        "time_in_hospital": "Hospital Length of Stay",
        "num_lab_procedures": "Number of Lab Procedures",
        "num_medications": "Number of Medications",
        "frequent_inpatient": "Frequent Inpatient Visits",
        "number_emergency": "Previous Emergency Visits",
        "num_procedures": "Number of Procedures",
        "number_outpatient": "Previous Outpatient Visits",
        "frequent_emergency": "Frequent Emergency Visits",
    }

    if name.startswith("num__"):
        raw = name.replace("num__", "", 1)
        return names.get(
            raw,
            raw.replace("_", " ").title(),
        )

    if name.startswith("cat__"):
        raw = name.replace("cat__", "", 1)

        if raw.startswith("discharge_disposition_id_"):
            code = raw.replace(
                "discharge_disposition_id_", ""
            )
            return f"Discharge Disposition — Code {code}"

        if raw.startswith("age_"):
            return "Age Group — " + raw[4:]

        if raw == "payer_code_Missing":
            return "Payer Code Missing"

        if raw.startswith("insulin_"):
            return "Insulin — " + raw[8:]

        if raw.startswith("diabetesMed_"):
            return "Diabetes Medication — " + raw[12:]

        if raw == "medical_specialty_Missing":
            return "Medical Specialty Missing"

        if raw.startswith("admission_source_id_"):
            return (
                "Admission Source — Code "
                + raw.replace("admission_source_id_", "")
            )

        return raw.replace("_", " ").title()

    return name.replace("_", " ").title()


# =====================================================
# 5. SIDEBAR
# =====================================================

st.sidebar.title("HRI | Intelligence")
st.sidebar.caption("Healthcare Predictive Analytics")

st.sidebar.divider()

page = st.sidebar.radio(
    "WORKSPACE",
    [
        "Executive Overview",
        "Risk Assessment",
        "Model Performance",
        "Explainable AI",
    ],
)

st.sidebar.divider()

st.sidebar.markdown("**MODEL INFORMATION**")
st.sidebar.caption("Selected model: Tuned CatBoost")
st.sidebar.caption("Prediction window: 30 days")
st.sidebar.caption("Research prototype")

st.sidebar.divider()
st.sidebar.caption("Not validated for clinical use")


# =====================================================
# 6. EXECUTIVE OVERVIEW
# =====================================================

if page == "Executive Overview":

    st.markdown(
        """
<div class="hero">
  <h1>Hospital Readmission Intelligence</h1>
  <p>Exploring 30-day hospital readmission prediction through machine learning and explainable AI.</p>
  <p>Healthcare Analytics | Predictive Modeling | Research Prototype</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    section_label("FINAL HELD-OUT TEST PERFORMANCE")

    if test_results:

        metrics = test_results["metrics"]

        a, b, c, d = st.columns(4)

        a.metric(
            "ROC-AUC",
            f"{metrics['roc_auc']:.4f}",
        )
        b.metric(
            "Accuracy",
            f"{metrics['accuracy']:.1%}",
        )
        c.metric(
            "Recall",
            f"{metrics['recall']:.1%}",
        )
        d.metric(
            "Precision",
            f"{metrics['precision']:.1%}",
        )

        st.caption(
            "Evaluation on 14,913 held-out encounters. "
            "Classification threshold: 0.14."
        )

        tn, fp = test_results["confusion_matrix"][0]
        fn, tp = test_results["confusion_matrix"][1]

        total = tn + fp + fn + tp

        st.subheader("Readmission Detection — Final Test")

        a, b, c = st.columns(3)

        a.metric(
            "Actual Readmissions",
            f"{tp + fn:,}",
        )
        b.metric(
            "Identified",
            f"{tp:,}",
        )
        c.metric(
            "Missed",
            f"{fn:,}",
        )

        detection_bar(tp, fn)

        recall_pct = 100 * tp / (tp + fn)

        st.markdown(
            f"""
<div class="insight">
  <h3>Executive Insight</h3>
  <p>At the selected threshold, the model identified <strong>{recall_pct:.1f}%</strong> of actual readmissions in the held-out test set.</p>
  <p>Across {total:,} encounters, it generated <strong>{fp:,} false-positive flags</strong> and missed <strong>{fn:,} readmissions</strong>.</p>
  <p>Validation analysis shows that lowering the threshold can increase detection, but substantially increases the number of encounters requiring review.</p>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("What Does This Project Demonstrate?")

        left, right = st.columns(2)

        with left:
            st.markdown(
                """
<div class="info-card">
  <h3>Predictive Modeling</h3>
  Uses historical hospital encounter information, diagnoses,
  medications, and previous healthcare utilization to
  rank encounters by readmission score.
</div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            st.markdown(
                """
<div class="info-card">
  <h3>Model Transparency</h3>
  Evaluates prediction quality, examines false alerts
  and missed readmissions, and uses SHAP to explain
  which features influence the model.
</div>
                """,
                unsafe_allow_html=True,
            )

        st.subheader("Evaluation Methodology")

        st.write(
            "The project uses historical diabetic hospital "
            "encounters, a patient-group-separated data split, "
            "multiple ML algorithms, hyperparameter tuning, "
            "and a held-out test evaluation."
        )

    else:
        st.error(
            "Final test results file is missing."
        )

    st.warning(
        "Research prototype only. Predictive performance "
        "is moderate, and the model has not been validated "
        "for clinical decisions."
    )

    st.divider()

    section_label("VALIDATION-BASED OPERATING SCENARIOS")

    st.info(
        "The interactive comparisons below use a different "
        "dataset: 14,911 validation encounters with 1,740 "
        "actual readmissions. Counts are expected to differ "
        "from the final test results above."
    )

    show_threshold_panel()


# =====================================================
# 7. RISK ASSESSMENT
# =====================================================

elif page == "Risk Assessment":

    st.title("Readmission Risk Assessment")

    st.write(
        "Demonstrate the saved CatBoost pipeline using "
        "synthetic or appropriately de-identified "
        "encounter records."
    )

    st.warning(
        "Software demonstration only. Do not upload "
        "identifiable patient information or use these "
        "outputs for patient care."
    )

    model, metadata = load_prediction_model()

    if model is None:
        st.error(
            "The trained model or metadata file is missing."
        )

    else:
        threshold = float(
            metadata["classification_threshold"]
        )

        a, b = st.columns(2)

        a.metric(
            "Saved Classification Threshold",
            f"{threshold:.2f}",
        )
        b.metric(
            "Required Input Features",
            "45",
        )

        st.caption(
            "Classification scores are not established "
            "clinical risk probabilities."
        )

        st.subheader("Step 1 — Download Sample Data")

        sample_path = DATA / "demo_encounters.csv"

        if sample_path.exists():

            st.download_button(
                label="Download Synthetic Sample CSV",
                data=sample_path.read_bytes(),
                file_name="demo_encounters.csv",
                mime="text/csv",
                help=(
                    "20 synthetic encounters generated "
                    "for software testing."
                ),
            )

            st.caption(
                "The sample contains synthetic combinations "
                "of model input values. It is not a clinically "
                "representative patient population."
            )

        else:
            st.info(
                "Demo CSV not found. Run "
                "python src/create_demo_data.py first."
            )

        st.subheader("Step 2 — Upload Encounter Data")

        uploaded = st.file_uploader(
            "Choose encounter CSV",
            type=["csv"],
        )

        if uploaded is not None:

            try:
                raw = pd.read_csv(uploaded)

            except Exception as error:
                st.error("Could not read the CSV.")
                st.code(str(error))
                st.stop()

            if raw.empty:
                st.error("The uploaded CSV is empty.")
                st.stop()

            if len(raw) > 10000:
                st.error(
                    "Limit this demonstration to "
                    "10,000 encounters."
                )
                st.stop()

            st.success(
                f"CSV loaded: {len(raw):,} encounters."
            )

            with st.expander("Preview Uploaded Data"):
                st.dataframe(
                    raw.head(10),
                    width="stretch",
                )

            st.subheader("Step 3 — Generate Predictions")

            if st.button(
                "Generate Risk Assessment",
                type="primary",
            ):

                try:

                    excluded = [
                        "encounter_id",
                        "patient_nbr",
                        "readmitted",
                        "readmitted_30_days",
                    ]

                    inputs = raw.drop(
                        columns=excluded,
                        errors="ignore",
                    )

                    engineered = {
                        "total_previous_visits",
                        "frequent_inpatient",
                        "frequent_emergency",
                    }

                    feature_names = list(
                        metadata["feature_names"]
                    )

                    required_raw = [
                        col
                        for col in feature_names
                        if col not in engineered
                    ]

                    missing = [
                        col
                        for col in required_raw
                        if col not in inputs.columns
                    ]

                    if missing:
                        st.error(
                            "Missing required columns: "
                            + ", ".join(missing)
                        )
                        st.stop()

                    inputs = inputs[
                        required_raw
                    ].copy()

                    preprocessor = model.named_steps[
                        "preprocessor"
                    ]

                    categorical_columns = list(
                        preprocessor.transformers_[1][2]
                    )

                    # Normalize input types before
                    # engineering numeric features.
                    for col in required_raw:

                        if col in categorical_columns:
                            inputs[col] = (
                                inputs[col]
                                .astype("string")
                                .fillna("Missing")
                                .replace("?", "Missing")
                                .astype(str)
                            )

                        else:
                            inputs[col] = pd.to_numeric(
                                inputs[col],
                                errors="coerce",
                            )

                    inputs = create_features(inputs)

                    inputs = inputs[feature_names]

                    for col in categorical_columns:
                        inputs[col] = (
                            inputs[col]
                            .astype("string")
                            .fillna("Missing")
                            .replace("?", "Missing")
                            .astype(str)
                        )

                    numeric_columns = [
                        col
                        for col in feature_names
                        if col not in categorical_columns
                    ]

                    for col in numeric_columns:
                        inputs[col] = pd.to_numeric(
                            inputs[col],
                            errors="coerce",
                        )

                    scores = model.predict_proba(
                        inputs
                    )[:, 1]

                    flags = scores >= threshold

                    output = pd.DataFrame({
                        "Encounter": np.arange(
                            1,
                            len(scores) + 1,
                        ),
                        "Model Score": scores.round(4),
                        "Status": np.where(
                            flags,
                            "Flagged",
                            "Not flagged",
                        ),
                    })

                    st.success(
                        "Prediction pipeline completed."
                    )

                    a, b, c = st.columns(3)

                    a.metric(
                        "Encounters Assessed",
                        f"{len(output):,}",
                    )

                    b.metric(
                        "Flagged Encounters",
                        f"{int(flags.sum()):,}",
                    )

                    c.metric(
                        "Not Flagged",
                        f"{int((~flags).sum()):,}",
                    )

                    st.subheader("Assessment Results")

                    st.dataframe(
                        output,
                        hide_index=True,
                        width="stretch",
                    )

                    st.download_button(
                        "Download Prediction Results",
                        data=output.to_csv(index=False),
                        file_name="readmission_predictions.csv",
                        mime="text/csv",
                    )

                    st.info(
                        "A flagged encounter has a model "
                        "score at or above the saved "
                        f"threshold of {threshold:.2f}. "
                        "This is not a diagnosis or "
                        "clinical recommendation."
                    )

                except Exception as error:
                    st.error(
                        "Prediction failed. "
                        "Review the uploaded input schema."
                    )

                    with st.expander(
                        "Technical Error Details"
                    ):
                        st.code(str(error))


# =====================================================
# 8. MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.title("Model Performance & Evaluation")

    st.write(
        "Compare the initial validation performance "
        "of five machine learning algorithms and "
        "review the selected CatBoost model's "
        "held-out test results."
    )

    section_label("INITIAL MODEL BENCHMARKS")

    if comparison is not None:

        st.subheader("Five-Model Comparison")

        display = comparison.copy()

        metric_columns = [
            col
            for col in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC_AUC",
                "PR_AUC",
            ]
            if col in display.columns
        ]

        st.dataframe(
            display.style.format(
                {col: "{:.4f}" for col in metric_columns}
            ),
            hide_index=True,
            width="stretch",
        )

        left, right = st.columns(2)

        with left:
            horizontal_comparison(
                comparison,
                "ROC_AUC",
                "ROC-AUC",
                "#167C91",
            )

        with right:
            horizontal_comparison(
                comparison,
                "PR_AUC",
                "PR-AUC",
                "#476FA9",
            )

        st.caption(
            "These are initial validation benchmarks, "
            "before later feature engineering and tuning. "
            "The differences between leading models "
            "are relatively small."
        )

        st.info(
            "CatBoost was selected for further evaluation "
            "after feature engineering, hyperparameter tuning, "
            "and classification-threshold analysis. "
            "The initial benchmark alone does not establish "
            "it as a statistically superior algorithm."
        )

    else:
        st.warning(
            "Model comparison file not found."
        )

    st.divider()

    section_label("SELECTED MODEL — FINAL HELD-OUT TEST")

    if test_results:

        metrics = test_results["metrics"]

        a, b, c = st.columns(3)

        a.metric(
            "ROC-AUC",
            f"{metrics['roc_auc']:.4f}",
        )
        b.metric(
            "PR-AUC",
            f"{metrics['pr_auc']:.4f}",
        )
        c.metric(
            "F1-score",
            f"{metrics['f1_score']:.4f}",
        )

        a, b, c = st.columns(3)

        a.metric(
            "Accuracy",
            f"{metrics['accuracy']:.1%}",
        )
        b.metric(
            "Recall",
            f"{metrics['recall']:.1%}",
        )
        c.metric(
            "Precision",
            f"{metrics['precision']:.1%}",
        )

        st.subheader("Final Test Confusion Matrix")

        tn, fp = test_results["confusion_matrix"][0]
        fn, tp = test_results["confusion_matrix"][1]

        matrix_df = pd.DataFrame(
            [
                [tn, fp],
                [fn, tp],
            ],
            index=[
                "Actual: No Readmission",
                "Actual: Readmission",
            ],
            columns=[
                "Predicted: No",
                "Predicted: Yes",
            ],
        )

        st.dataframe(
            matrix_df,
            width="stretch",
        )

        st.markdown(
            f"""
<div class="insight">
  <h3>Understanding Classification Errors</h3>
  <p><strong>{fn:,} false negatives:</strong> Actual readmissions that the model did not flag.</p>
  <p><strong>{fp:,} false positives:</strong> Encounters flagged by the model that did not result in readmission.</p>
  <p>Reducing missed readmissions generally increases the number of alerts requiring review.</p>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("How to Interpret These Metrics")

        st.write(
            "**ROC-AUC:** How well the model ranks "
            "readmitted encounters above non-readmitted ones."
        )

        st.write(
            "**PR-AUC:** Summarizes precision-recall "
            "performance across thresholds, particularly "
            "useful with an imbalanced outcome."
        )

        st.write(
            "**Recall:** Proportion of actual readmissions "
            "successfully flagged."
        )

        st.write(
            "**Precision:** Proportion of flagged encounters "
            "that actually resulted in readmission."
        )

        st.write(
            "**Accuracy:** Proportion of all encounters "
            "classified correctly. It should not be "
            "interpreted alone because readmissions "
            "are relatively uncommon."
        )

        st.caption(
            "The final test results are reported "
            "at the saved threshold of 0.14."
        )

    else:
        st.warning("Final test results not found.")


# =====================================================
# 9. EXPLAINABLE AI
# =====================================================

elif page == "Explainable AI":

    st.title("Explainable AI — Model Transparency")

    st.write(
        "Understand which encoded input features "
        "contribute most strongly to the model's "
        "predictions using SHAP analysis."
    )

    section_label("GLOBAL FEATURE IMPORTANCE")

    importance = load_csv(
        str(RESULTS / "shap_feature_importance.csv")
    )

    if importance is not None:

        required = {"feature", "mean_abs_shap"}

        if required.issubset(importance.columns):

            top = (
                importance
                .sort_values(
                    "mean_abs_shap",
                    ascending=False,
                )
                .head(10)
                .copy()
            )

            top["Feature"] = top["feature"].map(
                readable_feature
            )

            # Retain raw names in tooltips in case
            # two categories have similar labels.

            chart = (
                alt.Chart(top)
                .mark_bar(
                    color="#168B82",
                    cornerRadiusEnd=5,
                )
                .encode(
                    x=alt.X(
                        "mean_abs_shap:Q",
                        title="Mean absolute SHAP value",
                    ),
                    y=alt.Y(
                        "Feature:N",
                        sort="-x",
                        title=None,
                        axis=alt.Axis(labelLimit=330),
                    ),
                    tooltip=[
                        alt.Tooltip("Feature:N"),
                        alt.Tooltip(
                            "feature:N",
                            title="Encoded Feature",
                        ),
                        alt.Tooltip(
                            "mean_abs_shap:Q",
                            format=".4f",
                            title="Mean |SHAP|",
                        ),
                    ],
                )
                .properties(height=440)
            )

            st.subheader(
                "Top 10 Influential Features"
            )

            st.altair_chart(
                chart,
                width="stretch",
            )

            st.caption(
                "Global feature importance was calculated "
                "from 200 sampled validation encounters. "
                "The chart shows average absolute SHAP "
                "contributions in model-output units."
            )

            st.markdown(
                """
<div class="insight">
  <h3>What Did the Model Learn?</h3>
  <p>Previous inpatient encounters, discharge disposition, previous healthcare utilization, and the number of diagnoses are among the strongest contributors to model predictions.</p>
  <p>These findings describe which inputs the model relies on. They do not establish causes of readmission or indicate treatment priorities.</p>
</div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader(
                "Detailed Feature Importance"
            )

            table = importance.copy()

            table["Readable Feature"] = (
                table["feature"]
                .map(readable_feature)
            )

            st.dataframe(
                table[
                    [
                        "Readable Feature",
                        "feature",
                        "mean_abs_shap",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )

        else:
            st.error(
                "SHAP CSV is missing required columns."
            )

    else:
        st.warning(
            "SHAP results are unavailable. "
            "Run src/explain_model.py first."
        )

    with st.expander(
        "What is SHAP and how should we interpret it?"
    ):
        st.write(
            "SHAP attributes portions of a model's "
            "output to its input features. "
            "Global mean absolute SHAP values measure "
            "the average magnitude of those contributions."
        )

        st.write(
            "Larger values indicate greater model influence, "
            "but not whether a feature consistently increases "
            "or decreases a prediction."
        )

        st.write(
            "Because the pipeline uses one-hot encoding, "
            "some original variables appear as individual "
            "categories rather than a single grouped feature."
        )

    st.warning(
        "SHAP explains the behavior of this trained "
        "model. It is not evidence of causation "
        "or clinical effectiveness."
    )


# =====================================================
# 10. FOOTER
# =====================================================

st.divider()

st.caption(
    "Hospital Readmission Intelligence | "
    "Python • CatBoost • Scikit-learn • SHAP • Streamlit"
)

st.caption(
    "Historical research dataset | "
    "Not intended for clinical decision-making"
)
