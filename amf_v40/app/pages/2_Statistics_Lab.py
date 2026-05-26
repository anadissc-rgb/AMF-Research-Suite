# ─────────────────────────────────────────────
# AMF STATISTICS LAB
# ─────────────────────────────────────────────

import streamlit as st
import pandas as pd
import json
import tempfile
import sys

import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

# ─────────────────────────────────────────────
# ROOT PATH FIX
# ─────────────────────────────────────────────

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parents[2]

sys.path.append(str(PROJECT_ROOT))

# ─────────────────────────────────────────────
# COMPONENT IMPORTS
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

    st.error("AMF pipeline import failed.")

    st.exception(e)

    st.stop()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Statistics Lab",
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

st.title("Statistics Lab")

st.markdown("""
Run statistical analysis on
loaded EVA corpora using:

- Entropy Analysis
- Zipf Analysis
- Markov Modeling
- Adjacency Statistics
- Falsification Framework
""")

# ─────────────────────────────────────────────
# CHECK CORPUS
# ─────────────────────────────────────────────

if not st.session_state.corpus_path:

    st.warning(
        "Load a corpus first from Corpus Loader."
    )

    st.stop()

# ─────────────────────────────────────────────
# RUN BUTTON
# ─────────────────────────────────────────────

run_analysis = st.button(
    "Run Statistical Analysis",
    type="primary"
)

# ─────────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────────

if run_analysis:

    try:

        with tempfile.TemporaryDirectory() as tmpdir:

            output_dir = Path(tmpdir)

            with st.spinner(
                "Running AMF statistical pipeline..."
            ):

                result = run_pipeline(
                    corpus_path=(
                        st.session_state.corpus_path
                    ),
                    output_dir=output_dir,
                    run_id="statistics_lab"
                )

            # ─────────────────────────────────
            # STORE RESULTS
            # ─────────────────────────────────

            st.session_state.analysis_result = (
                result
            )

            st.session_state.verified_results = (
                result.verified_results
            )

            st.session_state.hypothesis_results = (
                result.hypothesis_results
            )

            st.session_state.warnings = (
                result.warnings
            )

            st.session_state.limitations = (
                result.limitations
            )

            st.success(
                "Analysis completed successfully."
            )

            # ─────────────────────────────────
            # VERIFIED RESULTS
            # ─────────────────────────────────

            verified = result.verified_results

            entropy = verified["entropy"]

            zipf = verified["zipf"]

            markov = verified["markov"]

            adjacency = verified["adjacency"]

            # ─────────────────────────────────
            # CORE METRICS
            # ─────────────────────────────────

            st.markdown("---")

            st.subheader("Core Metrics")

            col1, col2, col3, col4 = st.columns(4)

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
                    markov[
                        "optimal_order_test"
                    ]
                )

            with col4:

                st.metric(
                    "Bigram Entropy",
                    round(
                        adjacency[
                            "bigram_entropy_bits"
                        ],
                        4
                    )
                )

            # ─────────────────────────────────
            # SUMMARY TABLE
            # ─────────────────────────────────

            st.markdown("---")

            st.subheader("Summary Table")

            summary_df = pd.DataFrame([{

                "Entropy":
                    entropy["unigram_bits"],

                "Token Count":
                    entropy["token_count"],

                "Type Count":
                    entropy["type_count"],

                "Zipf Alpha":
                    zipf["alpha"],

                "Markov Order":
                    markov["optimal_order_test"],

                "Bigram Entropy":
                    adjacency[
                        "bigram_entropy_bits"
                    ]

            }])

            st.dataframe(
                summary_df,
                use_container_width=True
            )

            # ─────────────────────────────────
            # VISUALIZATIONS
            # ─────────────────────────────────

            st.markdown("---")

            st.subheader(
                "Statistical Visualizations"
            )

            # ─────────────────────────────────
            # ENTROPY CHART
            # ─────────────────────────────────

            entropy_df = pd.DataFrame({

                "Metric": [
                    "Unigram",
                    "Bigram"
                ],

                "Bits": [

                    entropy[
                        "unigram_bits"
                    ],

                    entropy[
                        "bigram_conditional_bits"
                    ]

                ]

            })

            entropy_fig = px.bar(

                entropy_df,

                x="Metric",

                y="Bits",

                title="Entropy Analysis"

            )

            st.plotly_chart(
                entropy_fig,
                use_container_width=True
            )

            # ─────────────────────────────────
            # ZIPF CHART
            # ─────────────────────────────────

            zipf_df = pd.DataFrame({

                "Metric": [
                    "Alpha",
                    "R²"
                ],

                "Value": [

                    zipf["alpha"],

                    zipf["r_squared"]

                ]

            })

            zipf_fig = px.bar(

                zipf_df,

                x="Metric",

                y="Value",

                title="Zipf Distribution"

            )

            st.plotly_chart(
                zipf_fig,
                use_container_width=True
            )

            # ─────────────────────────────────
            # MARKOV VISUALIZATION
            # ─────────────────────────────────

            perplexity = markov[
                "perplexity_by_order"
            ]

            orders = []

            values = []

            for k, v in perplexity.items():

                if (
                    v["train"] != float("inf")
                    and v["train"] is not None
                ):

                    orders.append(int(k))

                    values.append(v["train"])

            markov_df = pd.DataFrame({

                "Order": orders,

                "Perplexity": values

            })

            markov_fig = px.line(

                markov_df,

                x="Order",

                y="Perplexity",

                markers=True,

                title="Markov Perplexity"

            )

            st.plotly_chart(
                markov_fig,
                use_container_width=True
            )

            # ─────────────────────────────────
            # TOKEN DISTRIBUTION
            # ─────────────────────────────────

            token_df = pd.DataFrame({

                "Category": [
                    "Tokens",
                    "Types"
                ],

                "Count": [

                    entropy[
                        "token_count"
                    ],

                    entropy[
                        "type_count"
                    ]

                ]

            })

            token_fig = px.pie(

                token_df,

                names="Category",

                values="Count",

                title="Token Distribution"

            )

            st.plotly_chart(
                token_fig,
                use_container_width=True
            )

            # ─────────────────────────────────
            # VERIFIED RESULTS
            # ─────────────────────────────────

            st.markdown("---")

            st.subheader(
                "Verified Results"
            )

            st.json(verified)

            # ─────────────────────────────────
            # HYPOTHESIS RESULTS
            # ─────────────────────────────────

            st.markdown("---")

            st.subheader(
                "Hypothesis Results"
            )

            st.json(
                result.hypothesis_results
            )

            # ─────────────────────────────────
            # WARNINGS
            # ─────────────────────────────────

            if result.warnings:

                st.markdown("---")

                st.subheader("Warnings")

                for warning in result.warnings:

                    st.warning(warning)

            # ─────────────────────────────────
            # LIMITATIONS
            # ─────────────────────────────────

            if result.limitations:

                st.markdown("---")

                st.subheader("Limitations")

                for limitation in (
                    result.limitations
                ):

                    st.info(limitation)

            # ─────────────────────────────────
            # EXPORTS
            # ─────────────────────────────────

            st.markdown("---")

            st.subheader("Export Results")

            # CSV EXPORT

            csv = summary_df.to_csv(
                index=False
            )

            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="amf_statistics.csv",
                mime="text/csv"
            )

            # JSON EXPORT

            json_output = json.dumps(
                verified,
                indent=2,
                ensure_ascii=False
            )

            st.download_button(
                label="Download JSON",
                data=json_output,
                file_name="amf_results.json",
                mime="application/json"
            )

    except Exception as e:

        st.error(
            "Pipeline execution failed."
        )

        st.exception(e)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption("""
AMF Statistics Lab • Experimental
Statistical Linguistics Environment
""")
