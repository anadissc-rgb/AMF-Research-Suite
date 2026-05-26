# ─────────────────────────────────────────────
# AMF CORPUS LOADER
# ─────────────────────────────────────────────

import streamlit as st
import json
import tempfile
from pathlib import Path

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
research files, EVA transcriptions,
PDF studies, and structured datasets
for AMF analysis.
""")

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
# FILE PROCESSING
# ─────────────────────────────────────────────

if uploaded_file is not None:

    file_name = uploaded_file.name

    file_extension = (
        file_name
        .split(".")[-1]
        .lower()
    )

    st.success(
        f"Loaded: {file_name}"
    )

    # ─────────────────────────────────────────
    # SESSION STORAGE
    # ─────────────────────────────────────────

    st.session_state.uploaded_filename = (
        file_name
    )

    # ─────────────────────────────────────────
    # TXT FILES
    # ─────────────────────────────────────────

    if file_extension == "txt":

        try:

            content = uploaded_file.read().decode(
                "utf-8"
            )

            st.markdown("---")

            st.subheader(
                "TXT Preview"
            )

            st.text_area(

                "Corpus Content",

                content,

                height=400

            )

            # SAVE TEMP FILE

            with tempfile.NamedTemporaryFile(

                delete=False,
                suffix=".txt"

            ) as tmp_file:

                tmp_file.write(
                    content.encode("utf-8")
                )

                st.session_state.corpus_path = (
                    tmp_file.name
                )

            st.success(
                "TXT corpus processed successfully."
            )

        except Exception as e:

            st.error(
                "TXT processing failed."
            )

            st.exception(e)

    # ─────────────────────────────────────────
    # JSON FILES
    # ─────────────────────────────────────────

    elif file_extension == "json":

        try:

            data = json.load(
                uploaded_file
            )

            st.markdown("---")

            st.subheader(
                "JSON Preview"
            )

            st.json(data)

            # SAVE TEMP FILE

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

            st.success(
                "JSON corpus processed successfully."
            )

        except Exception as e:

            st.error(
                "JSON processing failed."
            )

            st.exception(e)

    # ─────────────────────────────────────────
    # PDF FILES
    # ─────────────────────────────────────────

    elif file_extension == "pdf":

        try:

            reader = PdfReader(
                uploaded_file
            )

            extracted_text = ""

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
                "PDF Metadata"
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

            st.markdown("---")

            st.subheader(
                "Extracted PDF Text"
            )

            st.text_area(

                "PDF Content",

                extracted_text,

                height=500

            )

            # SAVE TEMP FILE

            with tempfile.NamedTemporaryFile(

                delete=False,
                suffix=".txt"

            ) as tmp_file:

                tmp_file.write(

                    extracted_text.encode(
                        "utf-8"
                    )

                )

                st.session_state.corpus_path = (
                    tmp_file.name
                )

            st.success(
                "PDF processed successfully."
            )

        except Exception as e:

            st.error(
                "PDF extraction failed."
            )

            st.exception(e)

    # ─────────────────────────────────────────
    # CSV FILES
    # ─────────────────────────────────────────

    elif file_extension == "csv":

        try:

            import pandas as pd

            df = pd.read_csv(
                uploaded_file
            )

            st.markdown("---")

            st.subheader(
                "CSV Preview"
            )

            st.dataframe(
                df,
                use_container_width=True
            )

            # SAVE TEMP FILE

            with tempfile.NamedTemporaryFile(

                delete=False,
                suffix=".csv"

            ) as tmp_file:

                df.to_csv(
                    tmp_file.name,
                    index=False
                )

                st.session_state.corpus_path = (
                    tmp_file.name
                )

            st.success(
                "CSV corpus processed successfully."
            )

        except Exception as e:

            st.error(
                "CSV processing failed."
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

        "Corpus Loaded",

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
# FUTURE INGESTION
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Future Corpus Intelligence")

st.info("""
Future versions may support:

- OCR manuscript extraction
- semantic embeddings
- vector database indexing
- multilingual parsing
- folio segmentation
- manuscript entity extraction
- AI-assisted corpus reconstruction
- automated codicological tagging
""")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption("""
AMF Corpus Loader
Experimental Research Corpus
Ingestion Environment
""")
