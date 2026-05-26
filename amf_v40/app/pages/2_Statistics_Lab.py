# ─────────────────────────────────────────────
# AMF STATISTICS LAB
# Safe Large-Corpus Execution Layer
# ─────────────────────────────────────────────

import streamlit as st
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

st.set_page_config(

    page_title="Statistics Lab",

    layout="wide"

)

init_state()

render_sidebar()

render_header()

# ─────────────────────────────────────────────

st.title("Statistics Lab")

st.markdown("""
Unified statistical execution
environment for AMF corpus analysis.
""")

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

st.markdown("---")

st.subheader("Corpus Status")

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
Corpus successfully validated.

Pipeline ready.
""")

except Exception as e:

    st.error(
        "Corpus validation failed."
    )

    st.exception(e)

    st.stop()

# ─────────────────────────────────────────────

st.markdown("---")

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
        "Records",
        len(records)
    )

with col2:

    st.metric(
        "Tokens",
        metadata.get(
            "token_count",
            0
        )
    )

with col3:

    st.metric(
        "Folios",
        metadata.get(
            "folio_count",
            0
        )
    )

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

        st.success(
            "Pipeline execution completed."
        )

        # SAFE RESULT DISPLAY

        st.markdown("---")

        st.subheader(
            "Pipeline Summary"
        )

        if isinstance(result, dict):

            safe_preview = {}

            for key, value in result.items():

                safe_preview[key] = (
                    str(value)[:500]
                )

            st.json(
                safe_preview
            )

            st.markdown("---")

            st.subheader(
                "Available Result Keys"
            )

            st.write(
                list(result.keys())
            )

        else:

            st.write(result)

    except Exception as e:

        st.error(
            "Pipeline execution failed."
        )

        st.exception(e)

# ─────────────────────────────────────────────

st.markdown("---")

st.caption("""
AMF Statistics Lab
Large-Corpus-Safe
Execution Environment
""")
