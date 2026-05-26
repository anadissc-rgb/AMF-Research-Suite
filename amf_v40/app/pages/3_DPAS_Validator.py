# ─────────────────────────────────────────────
# AMF DPAS VALIDATOR
# ─────────────────────────────────────────────

import streamlit as st
import pandas as pd
import tempfile
import sys

from pathlib import Path

# ─────────────────────────────────────────────
# ROOT PATH FIX
# ─────────────────────────────────────────────

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parents[2]

sys.path.append(str(PROJECT_ROOT))

# ─────────────────────────────────────────────
# COMPONENTS
# ─────────────────────────────────────────────

from components.header import (
    render_header
)

from components.sidebar import (
    render_sidebar
)

from utils.state import (
    init_state
)

# ─────────────────────────────────────────────
# AMF IMPORT
# ─────────────────────────────────────────────

try:

    from amf.validation.pipeline import (
        run_pipeline
    )

except Exception as e:

    st.error(
        "AMF pipeline import failed."
    )

    st.exception(e)

    st.stop()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="DPAS Validator",
    layout="wide"
)

# ─────────────────────────────────────────────
# INITIALIZE
# ─────────────────────────────────────────────

init_state()

render_sidebar()

render_header()

# ─────────────────────────────────────────────
# PAGE TITLE
# ─────────────────────────────────────────────

st.title("DPAS Validator")

st.markdown("""
The Dynamic Procedural Alignment System
(DPAS) evaluates procedural slot
coverage across the corpus.

This module tests:

- Prefix slot coverage
- Gallows slot usage
- Core token structure
- Suffix completeness
- Constraint satisfaction
- Procedural consistency
""")

# ─────────────────────────────────────────────
# CORPUS CHECK
# ─────────────────────────────────────────────

if not st.session_state.corpus_path:

    st.warning(
        "Load a corpus first from Corpus Loader."
    )

    st.stop()

# ─────────────────────────────────────────────
# THRESHOLD CONTROL
# ─────────────────────────────────────────────

st.markdown("---")

threshold = st.slider(
    "DPAS Support Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.70,
    step=0.05
)

# ─────────────────────────────────────────────
# RUN BUTTON
# ─────────────────────────────────────────────

run_validation = st.button(
    "Run DPAS Validation",
    type="primary"
)

# ─────────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────────

if run_validation:

    try:

        with tempfile.TemporaryDirectory() as tmpdir:

            output_dir = Path(tmpdir)

            with st.spinner(
                "Running DPAS validation..."
            ):

                result = run_pipeline(
                    corpus_path=(
                        st.session_state.corpus_path
                    ),
                    output_dir=output_dir,
                    run_id="dpas_validator"
                )

            hypothesis = (
                result.hypothesis_results
            )

            dpas = hypothesis[
                "dpas_validation"
            ]

            st.success(
                "DPAS validation complete."
            )

            # ─────────────────────────────────
            # CORE METRICS
            # ─────────────────────────────────

            st.markdown("---")

            st.subheader("DPAS Metrics")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Coverage",
                    round(
                        dpas["coverage"] * 100,
                        2
                    )
                )

            with col2:

                st.metric(
                    "Threshold",
                    threshold
                )

            with col3:

                st.metric(
                    "Model Supported",
                    (
                        "YES"
                        if dpas[
                            "model_supported"
                        ]
                        else "NO"
                    )
                )

            # ─────────────────────────────────
            # INTERPRETATION
            # ─────────────────────────────────

            st.markdown("---")

            st.subheader(
                "Interpretation"
            )

            st.info(
                dpas["interpretation"]
            )

            # ─────────────────────────────────
            # SLOT USAGE
            # ─────────────────────────────────

            st.markdown("---")

            st.subheader("Slot Usage")

            slot_df = pd.DataFrame([
                {
                    "Slot": k,
                    "Usage": v
                }
                for k, v in (
                    dpas["slot_usage"]
                ).items()
            ])

            st.dataframe(
                slot_df,
                use_container_width=True
            )

            # ─────────────────────────────────
            # FAILURE CATEGORIES
            # ─────────────────────────────────

            st.markdown("---")

            st.subheader(
                "Failure Categories"
            )

            failure_df = pd.DataFrame([
                {
                    "Category": k,
                    "Count": v
                }
                for k, v in (
                    dpas[
                        "failure_categories"
                    ]
                ).items()
            ])

            st.dataframe(
                failure_df,
                use_container_width=True
            )

            # ─────────────────────────────────
            # STATUS ALERT
            # ─────────────────────────────────

            st.markdown("---")

            if dpas["coverage"] >= threshold:

                st.success("""
DPAS threshold satisfied.

Constraint model currently
SUPPORTED by this corpus.
""")

            else:

                st.error("""
DPAS threshold NOT satisfied.

Constraint model currently
NOT SUPPORTED by this corpus.
""")

            # ─────────────────────────────────
            # EXPORT
            # ─────────────────────────────────

            st.markdown("---")

            st.subheader("Export")

            export_df = pd.DataFrame([{

                "Coverage":
                    dpas["coverage"],

                "Threshold":
                    threshold,

                "Supported":
                    dpas["model_supported"]

            }])

            csv = export_df.to_csv(
                index=False
            )

            st.download_button(
                label="Download DPAS Report",
                data=csv,
                file_name="dpas_report.csv",
                mime="text/csv"
            )

    except Exception as e:

        st.error(
            "DPAS validation failed."
        )

        st.exception(e)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption("""
AMF DPAS Validator • Experimental
Constraint Validation Environment
""")
