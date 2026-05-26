# ─────────────────────────────────────────────
# AMF STATISTICS LAB
# Unified Multi-Format Pipeline Execution
# ─────────────────────────────────────────────

import streamlit as st
import json
from pathlib import Path

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
# AMF PIPELINE IMPORT
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
# TITLE
# ─────────────────────────────────────────────

st.title("Statistics Lab")

st.markdown("""
The Statistics Lab performs:

- entropy analysis
- Zipf evaluation
- Markov modeling
- DPAS validation
- falsification testing
- manuscript statistical analysis
""")

# ─────────────────────────────────────────────
# SESSION CHECK
# ─────────────────────────────────────────────

if not st.session_state.corpus_path:

    st.warning(
        "No corpus loaded."
    )

    st.stop()

corpus_path = (
    st.session_state.corpus_path
)

uploaded_filename = (
    st.session_state.uploaded_filename
)

# ─────────────────────────────────────────────
# CORPUS STATUS
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Corpus Validation")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Source File",
        uploaded_filename
    )

with col2:

    st.metric(
        "Pipeline Format",
        "AMF JSON"
    )

# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

try:

    with open(

        corpus_path,

        "r",

        encoding="utf-8"

    ) as f:

        payload = json.load(f)

    st.success("""
Corpus successfully converted into
AMF-compatible JSON structure.

Pipeline ready.
""")

except Exception as e:

    st.error(
        "Corpus validation failed."
    )

    st.exception(e)

    st.stop()

# ─────────────────────────────────────────────
# CORPUS METADATA
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Corpus Metadata")

metadata = payload.get(
    "metadata",
    {}
)

records = payload.get(
    "records",
    []
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(

        "Tokens",

        metadata.get(
            "token_count",
            len(records)
        )

    )

with col2:

    st.metric(

        "Records",

        len(records)

    )

with col3:

    st.metric(

        "Source Files",

        len(
            metadata.get(
                "source_files",
                []
            )
        )

    )

# ─────────────────────────────────────────────
# CORPUS PREVIEW
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Corpus Preview")

preview_records = records[:20]

st.json(preview_records)

# ─────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────

st.markdown("---")

if st.button(
    "Run AMF Pipeline"
):

    try:

        with st.spinner(
            "Running AMF analysis..."
        ):

            result = run_pipeline(

                corpus_path=corpus_path,

                output_dir="outputs",

                run_id="streamlit_run"

            )

        # ─────────────────────────────────────
        # STORE RESULT
        # ─────────────────────────────────────

        st.session_state.analysis_result = (
            result
        )

        # ─────────────────────────────────────
        # DISPLAY
        # ─────────────────────────────────────

        st.success(
            "Analysis Complete"
        )

        # ─────────────────────────────────────
        # VERIFIED RESULTS
        # ─────────────────────────────────────

        if isinstance(result, dict):

            st.markdown("---")

            st.subheader(
                "Pipeline Result"
            )

            st.json(result)

            # ─────────────────────────────────
            # WARNINGS
            # ─────────────────────────────────

            warnings = result.get(
                "warnings",
                []
            )

            if warnings:

                st.markdown("---")

                st.subheader(
                    "Warnings"
                )

                for warning in warnings:

                    st.warning(warning)

            # ─────────────────────────────────
            # LIMITATIONS
            # ─────────────────────────────────

            limitations = result.get(
                "limitations",
                []
            )

            if limitations:

                st.markdown("---")

                st.subheader(
                    "Limitations"
                )

                for limitation in limitations:

                    st.info(
                        limitation
                    )

        else:

            st.write(result)

    except Exception as e:

        st.error(
            "Pipeline execution failed."
        )

        st.exception(e)

# ─────────────────────────────────────────────
# FUTURE AI LAYERS
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Future AI Analysis")

st.info("""
Future versions may support:

- semantic embeddings
- vector search
- AI-assisted manuscript analysis
- multilingual corpora
- OCR manuscript parsing
- adaptive token reasoning
- graph neural manuscript analysis
- autonomous hypothesis generation
""")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption("""
AMF Statistics Lab
Unified Computational Manuscript
Validation Environment
""")
