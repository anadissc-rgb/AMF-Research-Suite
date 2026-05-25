# ─────────────────────────────────────────────
# AMF STREAMLIT RESEARCH CONSOLE
# Amanuensis Model Framework (AMF)
# v4.0 GUI Prototype
# ─────────────────────────────────────────────

import streamlit as st
import pandas as pd
import json
import tempfile
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

    from amf.validation.pipeline import run_pipeline

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
- Entropy Analysis
- Positional Analysis
- Markov Analysis
- DPAS Validation
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
# RUN BUTTON
# ─────────────────────────────────────────────

run_button = st.button(
    "Run AMF Analysis",
    type="primary"
)

# ─────────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────────

if run_button:

    if not uploaded_text.strip():

        st.error(
            "Please enter EVA manuscript text."
        )

        st.stop()

    try:

        # ─────────────────────────────────────
        # CREATE TEMP CORPUS FILE
        # ─────────────────────────────────────

        with tempfile.TemporaryDirectory() as tmpdir:

            tmpdir = Path(tmpdir)

            corpus_path = tmpdir / "gui_corpus.json"

            output_dir = tmpdir / "outputs"

            # SIMPLE EVA CORPUS FORMAT
            corpus_data = {
                "metadata": {
                    "transcription_version": "GUI_RUNTIME",
                    "uncertainty_line_count": 0
                },
                "records": []
            }

            for idx, line in enumerate(
                uploaded_text.splitlines()
            ):

                line = line.strip()

                if not line:
                    continue

                corpus_data["records"].append({
                    "folio": "GUI_INPUT",
                    "line_id": idx + 1,
                    "text": line,
                    "section": "GUI_RUNTIME"
                })

            # SAVE TEMP CORPUS
            with corpus_path.open(
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    corpus_data,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

            # ─────────────────────────────────
            # RUN AMF PIPELINE
            # ─────────────────────────────────

            with st.spinner(
                "Running AMF validation pipeline..."
            ):

                result = run_pipeline(
                    corpus_path=corpus_path,
                    output_dir=output_dir,
                    run_id="streamlit_gui_run"
                )

            st.success("Analysis Complete")

            # ─────────────────────────────────
            # VERIFIED RESULTS
            # ─────────────────────────────────

            verified = result.verified_results

            entropy = verified["entropy"]

            zipf = verified["zipf"]

            markov = verified["markov"]

            # ─────────────────────────────────
            # METRICS
            # ─────────────────────────────────

            st.markdown("## Validation Metrics")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Entropy",
                    round(
                        entropy["unigram_bits"],
                        4
                    )
                )

            with col2:

                st.metric(
                    "Zipf Alpha",
                    round(
                        zipf["alpha"],
                        4
                    )
                )

            with col3:

                st.metric(
                    "Markov Order",
                    markov["optimal_order_test"]
                )

            # ─────────────────────────────────
            # VERIFIED RESULTS
            # ─────────────────────────────────

            st.markdown("## Verified Results")

            st.json(verified)

            # ─────────────────────────────────
            # HYPOTHESIS RESULTS
            # ─────────────────────────────────

            st.markdown("## Hypothesis Results")

            st.json(
                result.hypothesis_results
            )

            # ─────────────────────────────────
            # WARNINGS
            # ─────────────────────────────────

            if result.warnings:

                st.markdown("## Warnings")

                for w in result.warnings:

                    st.warning(w)

            # ─────────────────────────────────
            # LIMITATIONS
            # ─────────────────────────────────

            st.markdown("## Limitations")

            for lim in result.limitations:

                st.info(lim)

            # ─────────────────────────────────
            # DATAFRAME OUTPUT
            # ─────────────────────────────────

            rows = []

            rows.append({
                "Entropy": entropy["unigram_bits"],
                "Zipf Alpha": zipf["alpha"],
                "Markov Order": markov["optimal_order_test"],
                "Token Count": entropy["token_count"]
            })

            df = pd.DataFrame(rows)

            st.markdown("## Summary Table")

            st.dataframe(
                df,
                use_container_width=True
            )

            # ─────────────────────────────────
            # CSV DOWNLOAD
            # ─────────────────────────────────

            csv = df.to_csv(index=False)

            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="amf_summary.csv",
                mime="text/csv"
            )

            # ─────────────────────────────────
            # JSON DOWNLOAD
            # ─────────────────────────────────

            json_output = json.dumps(
                {
                    "verified_results":
                        result.verified_results,

                    "hypothesis_results":
                        result.hypothesis_results,

                    "warnings":
                        result.warnings,

                    "limitations":
                        result.limitations
                },
                indent=2,
                ensure_ascii=False
            )

            st.download_button(
                label="Download JSON",
                data=json_output,
                file_name="amf_output.json",
                mime="application/json"
            )

    except Exception as e:

        st.error("Pipeline execution failed.")

        st.exception(e)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption(
    "AMF v4.0 • Computational Manuscript Validation Framework"
)
