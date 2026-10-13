
from pathlib import Path
from html import escape

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent

COMPARISON_PATH = (
    ROOT / "results" / "recall_target_comparison.csv"
)


@st.cache_data
def load_threshold_comparison():
    return pd.read_csv(COMPARISON_PATH)


def show_threshold_panel():

    st.header("Readmission Detection Scenarios")

    st.write(
        "Compare experimental operating thresholds "
        "and their impact on identifying readmissions "
        "and generating review alerts."
    )

    st.caption(
        "Validation dataset: 14,911 encounters, "
        "including 1,740 actual readmissions. "
        "These are not final test results."
    )

    if not COMPARISON_PATH.exists():
        st.warning("Threshold comparison CSV not found.")
        return

    df = load_threshold_comparison()

    options = df["Setting"].tolist()

    selected_name = st.selectbox(
        "Choose a detection scenario",
        options,
        index=options.index("70% Recall Target")
        if "70% Recall Target" in options else 0,
        key="threshold_scenario"
    )

    selected = df.loc[
        df["Setting"] == selected_name
    ].iloc[0]

    # ---------------------------------------
    # SELECTED METRICS
    # ---------------------------------------

    st.subheader("Selected Scenario")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Recall",
        f"{selected['Recall (%)']:.2f}%"
    )

    c2.metric(
        "Precision",
        f"{selected['Precision (%)']:.2f}%"
    )

    c3.metric(
        "Alert Burden",
        f"{selected['Alert Burden (%)']:.2f}%"
    )

    c4.metric(
        "Score Threshold",
        f"{selected['Threshold']:.3f}"
    )

    st.subheader("Impact on Readmissions")

    a, b, c, d = st.columns(4)

    detected = int(selected["Detected"])
    missed = int(selected["Missed"])
    false_alerts = int(selected["False Alerts"])
    flagged = int(selected["Total Flagged"])

    a.metric("Readmissions Identified", f"{detected:,}")
    b.metric("Readmissions Missed", f"{missed:,}")
    c.metric("False-Positive Alerts", f"{false_alerts:,}")
    d.metric("Total Encounters Flagged", f"{flagged:,}")

    # ---------------------------------------
    # DYNAMIC DETECTION BAR
    # ---------------------------------------

    total_readmissions = detected + missed

    detected_pct = (
        100 * detected / total_readmissions
        if total_readmissions else 0
    )

    missed_pct = 100 - detected_pct

    st.subheader("Readmission Detection Breakdown")

    st.markdown(
        f"""
<div style="background:white;padding:24px;border-radius:14px;
            border:1px solid #E1E9F1;margin-bottom:20px;">
    <p style="color:#334155;">
        {escape(selected_name)} —
        {total_readmissions:,} actual readmissions
    </p>
    <div style="display:flex;width:100%;height:32px;
                border-radius:9px;overflow:hidden;">
        <div style="width:{detected_pct:.4f}%;
                    background:#168B82;"></div>
        <div style="width:{missed_pct:.4f}%;
                    background:#D97757;"></div>
    </div>
    <p style="margin-top:16px;color:#334155;">
        <span style="color:#168B82;">■</span>
        Identified: {detected:,} ({detected_pct:.1f}%)
        &nbsp;&nbsp;&nbsp;
        <span style="color:#D97757;">■</span>
        Missed: {missed:,} ({missed_pct:.1f}%)
    </p>
</div>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------------------
    # DYNAMIC ALERT BURDEN BAR
    # ---------------------------------------

    st.subheader("Operational Alert Burden")

    burden_pct = float(selected["Alert Burden (%)"])
    unflagged_pct = 100 - burden_pct

    st.markdown(
        f"""
<div style="background:white;padding:24px;border-radius:14px;
            border:1px solid #E1E9F1;margin-bottom:20px;">
    <p style="color:#334155;">
        Percentage of all validation encounters flagged
    </p>
    <div style="display:flex;width:100%;height:32px;
                border-radius:9px;overflow:hidden;">
        <div style="width:{burden_pct:.4f}%;
                    background:#2563A6;"></div>
        <div style="width:{unflagged_pct:.4f}%;
                    background:#E2E8F0;"></div>
    </div>
    <p style="margin-top:16px;color:#334155;">
        <span style="color:#2563A6;">■</span>
        Flagged: {flagged:,} ({burden_pct:.1f}%)
        &nbsp;&nbsp;&nbsp;
        <span style="color:#64748B;">■</span>
        Not flagged: {14911 - flagged:,}
    </p>
</div>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------------------
    # COMPARISON TABLE
    # ---------------------------------------

    st.subheader("Compare All Operating Scenarios")

    st.dataframe(
        df,
        hide_index=True,
        width="stretch"
    )

    # ---------------------------------------
    # STAKEHOLDER INTERPRETATION
    # ---------------------------------------

    st.subheader("Operational Interpretation")

    st.info(
        f"Under the {selected_name} scenario, "
        f"the model identifies {detected:,} of "
        f"{total_readmissions:,} actual readmissions "
        f"({detected_pct:.1f}% recall). "
        f"It misses {missed:,} readmissions and generates "
        f"{false_alerts:,} false-positive alerts. "
        f"Overall, {burden_pct:.1f}% of encounters "
        f"would be flagged for review."
    )

    st.warning(
        "These are retrospective validation results, "
        "not clinically approved thresholds. "
        "The model has not been externally validated."
    )
