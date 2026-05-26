# ─────────────────────────────────────────────
# AMF CORPUS LOADER
# Automatic Corpus Conversion Engine
# ─────────────────────────────────────────────

import streamlit as st
import json
import tempfile
import pandas as pd
from pathlib import Path
from datetime import datetime

from pypdf import PdfReader

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
# TITLE
# ─────────────────────────────────────────────

st.title("Corpus Loader")

st.markdown("""
Upload TXT, PDF, CSV, or JSON files.

AMF automatically converts uploaded
documents into pipeline-compatible
AMF JSON corpora.
""")

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def build_amf_json(text, source_name):

    tokens = text.split()

    records = []

    for i, token in enumerate(tokens):

        records.append({

            "token_id": i + 1,

            "token": token,

            "position": i + 1

        })

    corpus = {

        "metadata": {

            "title": source_name,

            "created_at":
                datetime.now().isoformat(),

            "source_files": [
                source_name
            ],

            "token_count":
                len(tokens)

        },

        "records": records

    }

    return corpus

# ─────────────────────────────────────────────
# FILE UPLOADER
# ─────────────────────────────────────────────

st.markdown("---")

uploaded_file = st.file_uploader(

    "Upload Research Corpus",

    type=[

        "txt",
        "json",
        "pdf",
        "csv"

    ]

)

# ─────────────────────────────────────────────
# FILE PROCESSING
# ─────────────────────────────────────────────

if uploaded_file is not None:

    file_name = uploaded_file.name

    extension = (
        file_name
        .split(".")[-1]
        .lower()
    )

    st.success(
        f"Loaded: {file_name}"
    )

    st.session_state.uploaded_filename = (
        file_name
    )

    extracted_text = ""

    # ─────────────────────────────────────────
    # TXT
    # ─────────────────────────────────────────

    if extension == "txt":

        extracted_text = (
            uploaded_file
            .read()
            .decode("utf-8")
        )

    # ─────────────────────────────────────────
    # PDF
    # ─────────────────────────────────────────

    elif extension == "pdf":

        reader = PdfReader(
            uploaded_file
        )

        for page in reader.pages:

            text = page.extract_text()

            if text:

                extracted_text += (
                    text + "\n"
                )

    # ─────────────────────────────────────────
    # CSV
    # ─────────────────────────────────────────

    elif extension == "csv":

        df = pd.read_csv(
            uploaded_file
        )

        extracted_text = " ".join(

            df.astype(str)
            .fillna("")
            .values
            .flatten()

        )

    # ─────────────────────────────────────────
    # JSON
    # ─────────────────────────────────────────

    elif extension == "json":

        try:

            data = json.load(
                uploaded_file
            )

            st.success(
                "AMF JSON corpus loaded directly."
            )

            with tempfile.NamedTemporaryFile(

                delete=False,
                suffix=".json",
                mode="w",
                encoding="utf-8"

            ) as tmp_file:

                json.dump(
                    data,
                    tmp_file,
                    indent=2
                )

                st.session_state.corpus_path = (
                    tmp_file.name
                )

            st.json(data)

            st.stop()

        except Exception as e:

            st.error(
                "Invalid JSON corpus."
            )

            st.exception(e)

            st.stop()

    # ─────────────────────────────────────────
    # BUILD AMF JSON
    # ─────────────────────────────────────────

    try:

        corpus = build_amf_json(

            extracted_text,

            file_name

        )

        # SAVE TEMP JSON

        with tempfile.NamedTemporaryFile(

            delete=False,
            suffix=".json",
            mode="w",
            encoding="utf-8"

        ) as tmp_file:

            json.dump(

                corpus,

                tmp_file,

                indent=2

            )

            st.session_state.corpus_path = (
                tmp_file.name
            )

        # ─────────────────────────────────────
        # DISPLAY
        # ─────────────────────────────────────

        st.markdown("---")

        st.subheader(
            "Converted AMF Corpus"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(

                "Tokens",

                corpus["metadata"][
                    "token_count"
                ]

            )

        with col2:

            st.metric(

                "Records",

                len(
                    corpus["records"]
                )

            )

        st.markdown("---")

        st.subheader(
            "Corpus Preview"
        )

        st.json(corpus)

        st.success("""
Automatic AMF corpus conversion
completed successfully.
""")

    except Exception as e:

        st.error(
            "Corpus conversion failed."
        )

        st.exception(e)

# ─────────────────────────────────────────────
# SESSION STATUS
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Session Status")

col1, col2 = st.columns(2)

with col1:

    st.metric(

        "Corpus Ready",

        (
            "YES"
            if st.session_state.corpus_path
            else "NO"
        )

    )

with col2:

    st.metric(

        "File",

        (
            st.session_state.uploaded_filename
            if st.session_state.uploaded_filename
            else "None"
        )

    )

# ─────────────────────────────────────────────
# FUTURE AI LAYERS
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Future AI Ingestion")

st.info("""
Future versions may support:

- OCR manuscript extraction
- semantic embeddings
- vector search
- multilingual corpora
- AI-assisted parsing
- manuscript segmentation
- codicological tagging
- adaptive corpus reconstruction
""")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption("""
AMF Corpus Loader
Automatic Corpus Conversion Engine
""")
