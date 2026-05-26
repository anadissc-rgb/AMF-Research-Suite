# ─────────────────────────────────────────────
# AMF FALSIFICATION LAB
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
    page_title="Falsification Lab",
    layout="wide"
)

# ─────────────────────────────────────────────
# INITIALIZE
# ─────────────────────────────────────────────

init_state()

render_sidebar()

render_header()

# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────

st.title("Falsification Lab")

st.markdown("""
The Falsification Lab evaluates whether
the AMF framework survives adversarial
testing against the loaded corpus.

This module exposes:

- Critical failures
- Major failures
- Minor failures
- Pass/Fail distributions
- Statistical contradictions
- Constraint violations
- Evidentiary insufficiency
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
# RUN BUTTON
# ─────────────────────────────────────────────

run_tests = st.button(
    "Run Falsification Tests",
    type="primary"
)

# ─────────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────────

if run_tests:

    try:

        with tempfile.TemporaryDirectory() as tmpdir:

            output_dir = Path(tmpdir)

            with st.spinner(
                "Running falsification framework..."
            ):

                result = run_pipeline(
                    corpus_path=(
                        st.session_state.corpus_path
                    ),
                    output_dir=output_dir,
                    run_id="falsification_lab"
                )

            hypothesis = (
                result.hypothesis_results
            )

            falsification = hypothesis[
                "falsification"
            ]

            st.success(
                "Falsification analysis complete."
            )

            # ─────────────────────────────────
            # VERDICT
            # ─────────────────────────────────

            st.markdown("---")

            st.subheader("Overall Verdict")

            verdict = falsification[
                "overall_verdict"
            ]

            if verdict == "SUPPORTED":

                st.success(
                    f"VERDICT: {verdict}"
                )

            elif verdict == "PARTIAL":

                st.warning(
                    f"VERDICT: {verdict}"
                )

            else:

                st.error(
                    f"VERDICT: {verdict}"
                )

            # ─────────────────────────────────
            # FAILURE METRICS
            # ─────────────────────────────────

            st.markdown("---")

            st.subheader("Failure Metrics")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Critical Failures",
                    falsification[
                        "critical_failures"
                    ]
                )

            with col2:

                st.metric(
                    "Major Failures",
                    falsification[
                        "major_failures"
                    ]
                )

            with col3:

                st.metric(
                    "Minor Failures",
                    falsification[
                        "minor_failures"
                    ]
                )

            # ─────────────────────────────────
            # SUMMARY
            # ─────────────────────────────────

            st.markdown("---")

            st.subheader("Summary")

            st.info(
                falsification["summary"]
            )

            # ─────────────────────────────────
            # VERDICT TABLE
            # ─────────────────────────────────

            st.markdown("---")

            st.subheader(
                "Falsification Overview"
            )

            overview_df = pd.DataFrame([{

                "Verdict":
                    falsification[
                        "overall_verdict"
                    ],

                "Critical":
                    falsification[
                        "critical_failures"
                    ],

                "Major":
                    falsification[
                        "major_failures"
                    ],

                "Minor":
                    falsification[
                        "minor_failures"
                    ]

            }])

            st.dataframe(
                overview_df,
                use_container_width=True
            )

            # ─────────────────────────────────
            # INTERPRETATION
            # ─────────────────────────────────

            st.markdown("---")

            st.subheader(
                "Interpretation"
            )

            if (
                falsification[
                    "critical_failures"
                ] > 0
            ):

                st.error("""
One or more CRITICAL tests failed.

The AMF framework in its current
form is not supported by this
corpus.

Model revision is required before
interpretive claims proceed.
""")

            elif (
                falsification[
                    "major_failures"
                ] > 0
            ):

                st.warning("""
The framework shows substantial
structural weaknesses requiring
further validation.
""")

            else:

                st.success("""
No critical falsification detected
under current test conditions.
""")

            # ─────────────────────────────────
            # EXPORT
            # ─────────────────────────────────

            st.markdown("---")

            st.subheader("Export")

            csv = overview_df.to_csv(
                index=False
            )

            st.download_button(
                label="Download Falsification Report",
                data=csv,
                file_name=(
                    "falsification_report.csv"
                ),
                mime="text/csv"
            )

    except Exception as e:

        st.error(
            "Falsification framework failed."
        )

        st.exception(e)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption("""
AMF Falsification Lab • Experimental
Scientific Validation Environment
""")
