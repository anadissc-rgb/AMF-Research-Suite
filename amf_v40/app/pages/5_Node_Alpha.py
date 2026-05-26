# ─────────────────────────────────────────────
# AMF NODE ALPHA
# ─────────────────────────────────────────────

import streamlit as st
import pandas as pd

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
    page_title="Node Alpha",
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

st.title("Node Alpha")

st.markdown("""
Node Alpha performs token isolation,
glyph decomposition, shorthand mapping,
and procedural segmentation.

This layer represents the first-stage
interpretive engine of the
Amanuensis Model Framework.
""")

# ─────────────────────────────────────────────
# TOKEN INPUT
# ─────────────────────────────────────────────

st.markdown("---")

token = st.text_input(
    "Enter EVA Token",
    placeholder="qokeedy"
)

# ─────────────────────────────────────────────
# PROCEDURAL MAPPINGS
# ─────────────────────────────────────────────

TOKEN_MAP = {

    "qokeedy": {
        "meaning": "Heat-layer preparation",
        "prefix": "qo",
        "core": "kee",
        "suffix": "dy",
        "class": "thermal"
    },

    "qotedy": {
        "meaning": "Boil-down extraction",
        "prefix": "qo",
        "core": "te",
        "suffix": "dy",
        "class": "reduction"
    },

    "kchedy": {
        "meaning": "Press-extract operation",
        "prefix": "k",
        "core": "che",
        "suffix": "dy",
        "class": "compression"
    },

    "lchedy": {
        "meaning": "Strain-filter procedure",
        "prefix": "l",
        "core": "che",
        "suffix": "dy",
        "class": "filtration"
    }

}

# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────

if token:

    st.markdown("---")

    token_clean = token.strip().lower()

    if token_clean in TOKEN_MAP:

        result = TOKEN_MAP[token_clean]

        st.success(
            "Token recognized."
        )

        # ─────────────────────────────────
        # CORE METRICS
        # ─────────────────────────────────

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Prefix",
                result["prefix"]
            )

        with col2:

            st.metric(
                "Core",
                result["core"]
            )

        with col3:

            st.metric(
                "Suffix",
                result["suffix"]
            )

        # ─────────────────────────────────
        # INTERPRETATION
        # ─────────────────────────────────

        st.markdown("---")

        st.subheader(
            "Procedural Interpretation"
        )

        st.info(
            result["meaning"]
        )

        # ─────────────────────────────────
        # TOKEN CLASS
        # ─────────────────────────────────

        st.markdown("---")

        st.subheader("Token Classification")

        st.metric(
            "Functional Class",
            result["class"]
        )

        # ─────────────────────────────────
        # TOKEN TABLE
        # ─────────────────────────────────

        st.markdown("---")

        st.subheader(
            "Structural Breakdown"
        )

        df = pd.DataFrame([{

            "Token":
                token_clean,

            "Prefix":
                result["prefix"],

            "Core":
                result["core"],

            "Suffix":
                result["suffix"],

            "Meaning":
                result["meaning"],

            "Class":
                result["class"]

        }])

        st.dataframe(
            df,
            use_container_width=True
        )

    else:

        st.warning("""
Token not found in current
Node Alpha mapping database.

Future versions may support:

- fuzzy matching
- phonetic alignment
- DAWG traversal
- probabilistic decomposition
""")

# ─────────────────────────────────────────────
# THEORY NOTES
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Node Alpha Notes")

st.info("""
Node Alpha currently uses
experimental procedural mappings.

Interpretive outputs are hypothetical
and require independent validation.

This module does NOT claim direct
translation of the Voynich Manuscript.
""")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption("""
AMF Node Alpha • Experimental
Procedural Token Environment
""")
