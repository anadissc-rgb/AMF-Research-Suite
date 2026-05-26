# ─────────────────────────────────────────────
# AMF CORPUS LOADER
# Fully Backend-Compatible Conversion Engine
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
fully backend-compatible
pipeline-ready JSON corpora.
""")

# ─────────────────────────────────────────────
# HELPER FUNCTION
# ─────────────────────────────────────────────

def build_amf_json(text, source_name):

    # SPLIT INTO LINES

    lines = text.splitlines()

    records = []

    total_tokens = 0

    for i, line_text in enumerate(lines):

        clean_line = line_text.strip()

        if not clean_line:

            continue

        tokens = clean_line.split()

        total_tokens += len(tokens)

        # ─────────────────────────────────────
        # EXACT BACKEND-COMPATIBLE TOKEN RECORD
        # ─────────────────────────────────────

        record = {

            "folio":
                f"auto_folio_{i+1}",

            "section":
                "P",

            "line":
                str(i + 1),

            "transcriber":
                "AMF_AUTO",

            "raw_line":
                clean_line,

            "tokens":
                tokens

        }

        records.append(record)

    # ─────────────────────────────────────────
    # EXACT BACKEND-COMPATIBLE METADATA
    # ─────────────────────────────────────────

    corpus = {

        "metadata": {

            "source_files": [
                source_name
            ],

            "transcription_version":
                "AMF_AUTO_1.0",

            "loaded_at":
                "AUTO_GENERATED",

            "amf_version":
                "4.0",

            "record_count":
                len(records),

            "folio_count":
                len(records),

            "token_count":
                total_tokens,

            "uncertainty_line_count":
                0,

            "notes":
                "Automatically converted corpus"

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

            total_pages = len(
                reader.pages
            )

            for page in reader.pages:

                text = page.extract_text()

                if text:

                    extracted_text += (
                        text + "\n"
                    )

            st.markdown("---")

            st.subheader(
                "PDF Extraction Summary"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Pages",
                    total_pages
                )

            with col2:

                st.metric(
                    "Characters",
                    len(extracted_text)
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

            st.markdown("---")

            st.subheader(
                "CSV Preview"
            )

            st.dataframe(
                df.head(20),
                use_container_width=True
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
        # DISPLAY SUMMARY
        # ─────────────────────────────────────

        st.markdown("---")

        st.subheader(
            "Converted AMF Corpus"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(

                "Records",

                len(
                    corpus["records"]
                )

            )

        with col2:

            st.metric(

                "Tokens",

                corpus["metadata"][
                    "token_count"
                ]

            )

        with col3:

            st.metric(

                "Schema",

                "Backend Compatible"

            )

        st.markdown("---")

        st.subheader(
            "Corpus Preview"
        )

        st.json(

            corpus["records"][:10]

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
Fully Backend-Compatible
Automatic Corpus Conversion Engine
""")
