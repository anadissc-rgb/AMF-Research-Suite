import streamlit as st
import pandas as pd
import json
import asyncio

# Import your AMF pipeline
from amf.validation.pipeline import AmanuensisOrchestrator

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AMF Research Console",
    layout="wide"
)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("Amanuensis Model Framework (AMF)")
st.subheader(
    "Computational Manuscript Validation Console"
)

# ─────────────────────────────────────────────
# EVA INPUT
# ─────────────────────────────────────────────

st.markdown("## EVA Manuscript Input")

uploaded_text = st.text_area(
    "Paste EVA manuscript lines below:",
    height=250
)

# ─────────────────────────────────────────────
# SECTION SELECTOR
# ─────────────────────────────────────────────

section = st.selectbox(
    "Select Manuscript Section",
    [
        "Balneological",
        "Cosmological",
        "Logistical_FoldOut"
    ]
)

# ─────────────────────────────────────────────
# ANALYSIS BUTTON
# ─────────────────────────────────────────────

run_button = st.button("Run AMF Analysis")

# ─────────────────────────────────────────────
# PIPELINE EXECUTION
# ─────────────────────────────────────────────

if run_button:

    if not uploaded_text.strip():
        st.error("Please enter EVA manuscript text.")
        st.stop()

    lines = [
        line.strip()
        for line in uploaded_text.splitlines()
        if line.strip()
    ]

    corpus = {
        "folios": [
            {
                "folio_id": "GUI_INPUT",
                "section": section,
                "pipeline_type": "STREAMLIT_RUNTIME",
                "eva_lines": lines
            }
        ]
    }

    async def run_pipeline():

        orchestrator = AmanuensisOrchestrator()

        result = await orchestrator.run(
            corpus,
            {"folios": []}
        )

        return result

    with st.spinner("Running AMF pipeline..."):

        result = asyncio.run(run_pipeline())

    st.success("Analysis Complete")

    # ─────────────────────────────────────────
    # VALIDATION METRICS
    # ─────────────────────────────────────────

    st.markdown("## Validation Metrics")

    metrics = result["validation_metrics"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predictive Accuracy",
            metrics["predictive_testing_accuracy"]
        )

    with col2:
        st.metric(
            "Chi-Square",
            metrics["chi_square_statistic"]
        )

    with col3:
        st.metric(
            "p-Value",
            metrics["statistical_p_value"]
        )

    # ─────────────────────────────────────────
    # JSON OUTPUT
    # ─────────────────────────────────────────

    st.markdown("## AMF JSON Output")

    st.json(result)

    # ─────────────────────────────────────────
    # TABLE VIEW
    # ─────────────────────────────────────────

    rows = []

    for folio in result["linguistic_matrices"]:

        for row in folio["data_set"]:
            rows.append(row)

    if rows:

        st.markdown("## Token Analysis Table")

        df = pd.DataFrame(rows)

        st.dataframe(df)

        # CSV DOWNLOAD
        csv = df.to_csv(index=False)

        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="amf_analysis.csv",
            mime="text/csv"
        )

    # ─────────────────────────────────────────
    # RAW JSON DOWNLOAD
    # ─────────────────────────────────────────

    json_data = json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    )

    st.download_button(
        label="Download JSON",
        data=json_data,
        file_name="amf_output.json",
        mime="application/json"
    )
