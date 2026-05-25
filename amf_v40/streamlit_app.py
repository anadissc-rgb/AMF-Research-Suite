# ─────────────────────────────────────────────
# AMF STREAMLIT RESEARCH CONSOLE
# Amanuensis Model Framework (AMF)
# v4.0 GUI Prototype
# ─────────────────────────────────────────────

import streamlit as st
import pandas as pd
import json
import asyncio
import sys

from pathlib import Path

# ─────────────────────────────────────────────
# PYTHON PATH FIX
# ─────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent

sys.path.append(str(ROOT_DIR))

# ─────────────────────────────────────────────
# IMPORT AMF PIPELINE
# ─────────────────────────────────────────────

try:
    from amf.validation.pipeline import AmanuensisOrchestrator

except Exception as e:

    st.error("AMF import failed.")
    st.exception(e)
    st.stop()

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
# SIDEBAR
# ─────────────────────────────────────────────

st.sidebar.title("AMF Console")

st.sidebar.markdown("""
### Features
- DPAS Validation
- Statistical Analysis
- Symbolic Reconstruction
- Falsification Engine
- JSON Export
""")

st.sidebar.markdown("---")

st.sidebar.info(
    "AMF v4.0 — Proprietary Research Software"
)

# ─────────────────────────────────────────────
# EVA INPUT
# ─────────────────────────────────────────────

st.markdown("## EVA Manuscript Input")

uploaded_text = st.text_area(
    "Paste EVA manuscript lines below:",
    height=250,
    placeholder="qokeedy qokedy dal\notary shedy qokedy"
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
# RUN BUTTON
# ─────────────────────────────────────────────

run_button = st.button(
    "Run AMF Analysis",
    type="primary"
)

# ─────────────────────────────────────────────
# ANALYSIS EXECUTION
# ─────────────────────────────────────────────

if run_button:

    # Validate Input
    if not uploaded_text.strip():

        st.error(
            "Please enter EVA manuscript text."
        )

        st.stop()

    # Prepare EVA Lines
    lines = [
        line.strip()
        for line in uploaded_text.splitlines()
        if line.strip()
    ]

    # Build Corpus
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

    # Async Pipeline Wrapper
    async def run_pipeline():

        orchestrator = AmanuensisOrchestrator()

        result = await orchestrator.run(
            corpus,
            {"folios": []}
        )

        return result

    # Execute Pipeline
    with st.spinner(
        "Running AMF validation pipeline..."
    ):

        try:

            # Streamlit-safe async execution
            result = asyncio.run(
                run_pipeline()
            )

        except RuntimeError:

            # Fallback for active event loops
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                run_pipeline()
            )

    st.success("Analysis Complete")

    # ─────────────────────────────────────────
    # VALIDATION METRICS
    # ─────────────────────────────────────────

    st.markdown("## Validation Metrics")

    metrics = result.get(
        "validation_metrics",
        {}
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Predictive Accuracy",
            metrics.get(
                "predictive_testing_accuracy",
                "N/A"
            )
        )

    with col2:

        st.metric(
            "Chi-Square",
            metrics.get(
                "chi_square_statistic",
                "N/A"
            )
        )

    with col3:

        st.metric(
            "p-Value",
            metrics.get(
                "statistical_p_value",
                "N/A"
            )
        )

    # ─────────────────────────────────────────
    # JSON OUTPUT
    # ─────────────────────────────────────────

    st.markdown("## JSON Output")

    st.json(result)

    # ─────────────────────────────────────────
    # TABLE OUTPUT
    # ─────────────────────────────────────────

    rows = []

    for folio in result.get(
        "linguistic_matrices",
        []
    ):

        for row in folio.get(
            "data_set",
            []
        ):

            rows.append(row)

    if rows:

        st.markdown(
            "## Token Analysis Table"
        )

        df = pd.DataFrame(rows)

        st.dataframe(
            df,
            use_container_width=True
        )

        # CSV DOWNLOAD
        csv = df.to_csv(index=False)

        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="amf_analysis.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "No linguistic matrix rows generated."
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

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption(
    "AMF v4.0 • Computational Manuscript Validation Framework"
)
