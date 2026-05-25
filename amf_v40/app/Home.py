# ─────────────────────────────────────────────
# AMF RESEARCH STUDIO — HOME DASHBOARD
# ─────────────────────────────────────────────

import streamlit as st

from components.header import (
    render_header
)

from components.sidebar import (
    render_sidebar
)

from utils.state import (
    init_state
)

from utils.config import (
    APP_VERSION
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AMF Research Studio",
    layout="wide"
)

# ─────────────────────────────────────────────
# INITIALIZE SESSION STATE
# ─────────────────────────────────────────────

init_state()

# ─────────────────────────────────────────────
# RENDER COMPONENTS
# ─────────────────────────────────────────────

render_sidebar()

render_header()

# ─────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────

st.markdown("""
## Computational Humanities Platform

The Amanuensis Model Framework (AMF)
is a computational manuscript analysis
environment integrating:

- Statistical Linguistics
- DPAS Constraint Validation
- Markov Modeling
- Entropy Analysis
- Falsification Frameworks
- Procedural Reconstruction
- Manuscript Forensics
""")

# ─────────────────────────────────────────────
# DASHBOARD METRICS
# ─────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Framework",
        APP_VERSION
    )

with col2:

    corpus_loaded = (
        st.session_state.corpus_path
        is not None
    )

    st.metric(
        "Corpus Loaded",
        "YES" if corpus_loaded else "NO"
    )

with col3:

    st.metric(
        "Pipeline",
        "Operational"
    )

with col4:

    result_loaded = (
        st.session_state.analysis_result
        is not None
    )

    st.metric(
        "Analysis Ready",
        "YES" if result_loaded else "NO"
    )

# ─────────────────────────────────────────────
# ACTIVE SESSION
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Active Session")

if st.session_state.uploaded_filename:

    st.success(
        f"""
Loaded Corpus:

{st.session_state.uploaded_filename}
"""
    )

else:

    st.warning(
        "No corpus currently loaded."
    )

# ─────────────────────────────────────────────
# RESEARCH MODULES
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Research Modules")

module_data = [

    {
        "Module": "Corpus Loader",
        "Purpose": "Import EVA corpora"
    },

    {
        "Module": "Node Alpha",
        "Purpose": "Token isolation & shorthand mapping"
    },

    {
        "Module": "Node Beta",
        "Purpose": "Phonetic reconstruction"
    },

    {
        "Module": "Node Gamma",
        "Purpose": "Ayurvedic cross-reference"
    },

    {
        "Module": "Node Delta",
        "Purpose": "Curricular synthesis"
    },

    {
        "Module": "Statistics Lab",
        "Purpose": "Entropy, Zipf, Markov analysis"
    },

    {
        "Module": "DPAS Validator",
        "Purpose": "Constraint verification"
    },

    {
        "Module": "Falsification Lab",
        "Purpose": "Hypothesis testing"
    }

]

st.table(module_data)

# ─────────────────────────────────────────────
# SYSTEM STATUS
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("System Status")

st.info("""
AMF Research Studio is currently running
in experimental mode.

Statistical outputs are dependent on:

- corpus size
- transcription quality
- section classification
- DPAS rule completeness

Interpretive outputs remain hypothetical
until independently validated.
""")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption("""
AMF Research Studio • Experimental
Computational Humanities Environment
""")
