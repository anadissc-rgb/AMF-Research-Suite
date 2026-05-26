# ─────────────────────────────────────────────
# AMF CORPUS LOADER
# Schema-Compatible Automatic Conversion Engine
# ─────────────────────────────────────────────

import streamlit as st
import json
import tempfile
import pandas as pd

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
Upload manuscript corpora,
research papers,
EVA transcriptions,
PDF documents,
TXT files,
CSV datasets,
and AMF JSON corpora.

AMF automatically converts
uploaded documents into
pipeline-compatible JSON format.
""")

# ─────────────────────────────────────────────
# HELPER FUNCTION
# ─────────────────────────────────────────────

def build_amf_json(text, source_name):

    tokens = text.split()

    records = []

    for i, token in enumerate(tokens):

        records.append({

            # Minimal schema-safe structure

            "token_id": i + 1,

            "token": token,

            "position": i + 1

        })

    # ─────────────────────────────────────────
    # BACKEND-COMPATIBLE STRUCTURE
    # ─────────────────────────────────────────

    corpus = {

        "metadata": {

            "source_files": [
                source_name
            ]

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

    ],

    help="""
Supported formats:

• TXT
• JSON
• PDF
• CSV
"""

)

# ─────────────────────────────────────────────
# PROCESS FILE
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
    # TXT FILES
    # ─────────────────────────────────────────

    if extension == "txt":

        try:

            extracted_text = (

                uploaded_file
                .read()
                .decode("utf-8")

            )

        except Exception as e:

            st.error(
                "TXT processing failed."
            )

            st.exception(e)

            st.stop()

    # ─────────────────────────────────────────
    # PDF FILES
    # ─────────────────────────────────────────

    elif extension == "pdf":

        try:

            reader = PdfReader(
                uploaded_file
            )

            for page in reader.pages:

                text = page.extract_text()

                if text:

                    extracted_text += (
                        text + "\n"
                    )

        except Exception as e:

            st.error(
                "PDF extraction failed."
            )

            st.exception(e)

            st.stop()

    # ─────────────────────────────────────────
    # CSV FILES
    # ─────────────────────────────────────────

    elif extension == "csv":

        try:

            df = pd.read_csv(
                uploaded_file
            )

            extracted_text = " ".join(

                df.astype(str)
                .fillna("")
                .values
                .flatten()

            )

        except Exception as e:

            st.error(
                "CSV processing failed."
            )

            st.exception(e)

            st.stop()

    # ─────────────────────────────────────────
    # JSON FILES
    # ─────────────────────────────────────────

    elif extension == "json":

        try:

            data = json.load(
                uploaded_file
            )

            st.success(
                "AMF JSON corpus loaded directly."
            )

            # SAVE DIRECTLY

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

            st.markdown("---")

            st.subheader(
                "JSON Corpus Preview"
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
    # BUILD AMF JSON CORPUS
    # ─────────────────────────────────────────

    try:

        corpus = build_amf_json(

            extracted_text,

            file_name

        )

        # SAVE TEMP JSON FILE

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
        # DISPLAY SUMMARY
        # ─────────────────────────────────────

        st.markdown("---")

        st.subheader(
            "Converted AMF Corpus"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(

                "Tokens",

                len(
                    corpus["records"]
                )

            )

        with col2:

            st.metric(

                "Source Files",

                len(
                    corpus["metadata"][
                        "source_files"
                    ]
                )

            )

        st.markdown("---")

        st.subheader(
            "Corpus Preview"
        )

        st.json(

            corpus["records"][:20]

        )

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

        "Uploaded File",

        (
            st.session_state.uploaded_filename
            if st.session_state.uploaded_filename
            else "None"
        )

    )

# ─────────────────────────────────────────────
# FUTURE AI INGESTION
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Future AI Ingestion")

st.info("""
Future versions may support:

- OCR manuscript extraction
- semantic embeddings
- vector search
- multilingual parsing
- codicological segmentation
- AI-assisted corpus reconstruction
- adaptive manuscript intelligence
- graph neural manuscript learning
""")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption("""
AMF Corpus Loader
Schema-Compatible Automatic
Corpus Conversion Engine
""")
