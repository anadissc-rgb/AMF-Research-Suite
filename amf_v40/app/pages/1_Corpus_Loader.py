# ─────────────────────────────────────────────
# AMF CORPUS LOADER
# ─────────────────────────────────────────────

import streamlit as st
import tempfile
import json

from pathlib import Path

from Components.header import (
    render_header
)

from Components.sidebar import (
    render_sidebar
)

from utils.state import (
    init_state
)

from utils.config import (
    SUPPORTED_FORMATS
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Corpus Loader",
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

st.title("Corpus Loader")

st.markdown("""
Upload EVA transcription corpora
for analysis within the
AMF Research Studio environment.
""")

# ─────────────────────────────────────────────
# FILE UPLOADER
# ─────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload Corpus File",
    type=SUPPORTED_FORMATS
)

# ─────────────────────────────────────────────
# FILE PROCESSING
# ─────────────────────────────────────────────

if uploaded_file:

    suffix = uploaded_file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{suffix}"
    ) as tmp:

        tmp.write(uploaded_file.read())

        tmp_path = tmp.name

    # STORE IN SESSION STATE
    st.session_state.corpus_path = tmp_path

    st.session_state.uploaded_filename = (
        uploaded_file.name
    )

    st.success(
        f"""
Corpus loaded successfully:

{uploaded_file.name}
"""
    )

    # ─────────────────────────────────────────
    # FILE INFO
    # ─────────────────────────────────────────

    st.markdown("---")

    st.subheader("Corpus Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Filename",
            uploaded_file.name
        )

    with col2:

        st.metric(
            "File Type",
            suffix.upper()
        )

    with col3:

        size_kb = round(
            uploaded_file.size / 1024,
            2
        )

        st.metric(
            "Size (KB)",
            size_kb
        )

    # ─────────────────────────────────────────
    # PREVIEW
    # ─────────────────────────────────────────

    st.markdown("---")

    st.subheader("Corpus Preview")

    try:

        content = uploaded_file.getvalue()

        decoded = content.decode(
            "utf-8",
            errors="ignore"
        )

        st.text_area(
            "Preview",
            decoded[:3000],
            height=300
        )

    except Exception as e:

        st.warning(
            "Preview unavailable."
        )

# ─────────────────────────────────────────────
# SESSION STATUS
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Current Session")

if st.session_state.uploaded_filename:

    st.info(
        f"""
Active Corpus:

{st.session_state.uploaded_filename}
"""
    )

else:

    st.warning(
        "No active corpus loaded."
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption("""
AMF Corpus Loader • Experimental
Corpus Ingestion Environment
""")
