# ─────────────────────────────────────────────
# AMF STATISTICS LAB
# ─────────────────────────────────────────────

import streamlit as st
from pathlib import Path
import json

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
# FILE VALIDATION
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Corpus Validation")

file_extension = (
    uploaded_filename
    .split(".")[-1]
    .lower()
)

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "File",
        uploaded_filename
    )

with col2:

    st.metric(
        "Type",
        file_extension.upper()
    )

# ─────────────────────────────────────────────
# JSON ONLY
# ─────────────────────────────────────────────

if file_extension != "json":

    st.error("""
Statistics Lab currently supports
ONLY structured JSON corpora.

Please upload:

• EVA JSON corpus
• structured manuscript corpus
• AMF-compatible JSON dataset

TXT/PDF/CSV support will be added
in future AI ingestion layers.
""")

    st.stop()

# ─────────────────────────────────────────────
# JSON VALIDATION
# ─────────────────────────────────────────────

try:

    with open(
        corpus_path,
        "r",
        encoding="utf-8"
    ) as f:

        json.load(f)

    st.success(
        "JSON corpus validated."
    )

except Exception as e:

    st.error(
        "Invalid JSON corpus."
    )

    st.exception(e)

    st.stop()

# ─────────────────────────────────────────────
# RUN ANALYSIS
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

        # ─────────────────────────────────
        # STORE SESSION
        # ─────────────────────────────────

        st.session_state.analysis_result = (
            result
        )

        # ─────────────────────────────────
        # DISPLAY RESULT
        # ─────────────────────────────────

        st.success(
            "Analysis Complete"
        )

        st.markdown("---")

        st.subheader(
            "Pipeline Result"
        )

        st.json(result)

    except Exception as e:

        st.error(
            "Pipeline execution failed."
        )

        st.exception(e)

# ─────────────────────────────────────────────
# FUTURE FEATURES
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Future AI Analysis")

st.info("""
Future versions may support:

- PDF corpus ingestion
- OCR manuscript parsing
- semantic embeddings
- vector search
- adaptive manuscript analysis
- multilingual corpora
- AI-assisted reconstruction
""")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption("""
AMF Statistics Lab
Experimental Computational
Validation Environment
""")
