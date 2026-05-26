# ─────────────────────────────────────────────
# AMF RESEARCH STUDIO CORE
# ─────────────────────────────────────────────

import streamlit as st
import pandas as pd
from datetime import datetime

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
    page_title="Research Studio Core",
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

st.title("AMF Research Studio Core")

st.markdown("""
The Research Studio Core provides
centralized orchestration for:

- manuscript intelligence
- statistical workflows
- procedural reconstruction
- DPAS validation
- falsification analysis
- annotation management
- research session control
""")

# ─────────────────────────────────────────────
# SESSION OVERVIEW
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Active Session Overview")

col1, col2, col3, col4 = st.columns(4)

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
        "Analysis Available",
        (
            "YES"
            if st.session_state.analysis_result
            else "NO"
        )
    )

with col3:

    st.metric(
        "Warnings",
        len(st.session_state.warnings)
    )

with col4:

    st.metric(
        "Limitations",
        len(st.session_state.limitations)
    )

# ─────────────────────────────────────────────
# ACTIVE CORPUS
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Current Corpus")

if st.session_state.uploaded_filename:

    st.success(
        st.session_state.uploaded_filename
    )

else:

    st.warning(
        "No corpus loaded."
    )

# ─────────────────────────────────────────────
# MODULE STATUS
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Module Status")

modules = [

    {
        "Module": "Corpus Loader",
        "Status": "ACTIVE"
    },

    {
        "Module": "Statistics Lab",
        "Status": (
            "READY"
            if st.session_state.analysis_result
            else "WAITING"
        )
    },

    {
        "Module": "DPAS Validator",
        "Status": "READY"
    },

    {
        "Module": "Falsification Lab",
        "Status": "READY"
    },

    {
        "Module": "Node Alpha",
        "Status": "READY"
    },

    {
        "Module": "Node Beta",
        "Status": "READY"
    },

    {
        "Module": "Node Gamma",
        "Status": "READY"
    },

    {
        "Module": "Node Delta",
        "Status": "READY"
    },

    {
        "Module": "Manuscript Viewer",
        "Status": "READY"
    }

]

module_df = pd.DataFrame(modules)

st.dataframe(
    module_df,
    use_container_width=True
)

# ─────────────────────────────────────────────
# RESEARCH MEMORY
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Research Memory")

research_notes = st.text_area(

    "Session Research Notes",

    height=250,

    placeholder="""
Store observations,
cross-module findings,
procedural hypotheses,
codicological insights,
or validation notes.
"""

)

# ─────────────────────────────────────────────
# SESSION TIMELINE
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Session Timeline")

timeline_df = pd.DataFrame([{

    "Timestamp":
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

    "Corpus":
        (
            st.session_state.uploaded_filename
            if st.session_state.uploaded_filename
            else "None"
        ),

    "Analysis":
        (
            "Completed"
            if st.session_state.analysis_result
            else "Pending"
        )

}])

st.dataframe(
    timeline_df,
    use_container_width=True
)

# ─────────────────────────────────────────────
# QUICK ACCESS
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Research Workflow")

workflow = """

Corpus Loader
    ↓
Statistics Lab
    ↓
DPAS Validator
    ↓
Falsification Lab
    ↓
Node Alpha
    ↓
Node Beta
    ↓
Node Gamma
    ↓
Node Delta
    ↓
Advanced Manuscript Viewer

"""

st.code(workflow)

# ─────────────────────────────────────────────
# WARNINGS PANEL
# ─────────────────────────────────────────────

if st.session_state.warnings:

    st.markdown("---")

    st.subheader("Warnings")

    for warning in (
        st.session_state.warnings
    ):

        st.warning(warning)

# ─────────────────────────────────────────────
# LIMITATIONS PANEL
# ─────────────────────────────────────────────

if st.session_state.limitations:

    st.markdown("---")

    st.subheader("Limitations")

    for limitation in (
        st.session_state.limitations
    ):

        st.info(limitation)

# ─────────────────────────────────────────────
# EXPORT MEMORY
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Export Session Notes")

memory_df = pd.DataFrame([{

    "Timestamp":
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

    "Corpus":
        (
            st.session_state.uploaded_filename
            if st.session_state.uploaded_filename
            else "None"
        ),

    "Notes":
        research_notes

}])

csv = memory_df.to_csv(
    index=False
)

st.download_button(
    label="Download Session Notes",
    data=csv,
    file_name="amf_research_notes.csv",
    mime="text/csv"
)

# ─────────────────────────────────────────────
# FUTURE AI LAYER
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Future AI Integration")

st.info("""
Future versions may support:

- AI-assisted manuscript reasoning
- semantic token clustering
- OCR-assisted folio parsing
- graph neural manuscript analysis
- procedural prediction models
- glyph similarity engines
- adaptive research memory
- automated hypothesis generation
""")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption("""
AMF Research Studio Core
Experimental Computational Humanities
Operating Environment
""")
