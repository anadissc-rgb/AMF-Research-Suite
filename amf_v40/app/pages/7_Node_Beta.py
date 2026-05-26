# ─────────────────────────────────────────────
# AMF NODE BETA
# ─────────────────────────────────────────────

import streamlit as st
import pandas as pd
import re

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
    page_title="Node Beta",
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

st.title("Node Beta")

st.markdown("""
Node Beta performs phonetic
reconstruction and transliteration
hypothesis generation.

This module explores:

- consonant skeletons
- vowel normalization
- phonetic clustering
- transliteration hypotheses
- procedural phonology
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
# PHONETIC DATABASE
# ─────────────────────────────────────────────

PHONETIC_MAP = {

    "qokeedy": {

        "skeleton": "QKD",

        "phonetic":
            "ko-ke-di",

        "normalized":
            "kokedi",

        "class":
            "thermal procedural",

        "hypothesis":
            "Layered heating sequence"

    },

    "qotedy": {

        "skeleton": "QTD",

        "phonetic":
            "ko-te-di",

        "normalized":
            "kotedi",

        "class":
            "reduction procedural",

        "hypothesis":
            "Boiling reduction operation"

    },

    "kchedy": {

        "skeleton": "KCHD",

        "phonetic":
            "ke-che-di",

        "normalized":
            "kechedi",

        "class":
            "compression procedural",

        "hypothesis":
            "Pressing or extraction sequence"

    },

    "lchedy": {

        "skeleton": "LCHD",

        "phonetic":
            "le-che-di",

        "normalized":
            "lechedi",

        "class":
            "filtration procedural",

        "hypothesis":
            "Filtering or straining operation"

    }

}

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def extract_consonants(text):

    return "".join(

        re.findall(
            r"[^aeiou]",
            text.lower()
        )

    ).upper()

# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────

if token:

    st.markdown("---")

    token_clean = token.strip().lower()

    consonants = extract_consonants(
        token_clean
    )

    st.subheader("Phonetic Extraction")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Token",
            token_clean
        )

    with col2:

        st.metric(
            "Consonant Skeleton",
            consonants
        )

    # ─────────────────────────────────────────
    # DATABASE MATCH
    # ─────────────────────────────────────────

    if token_clean in PHONETIC_MAP:

        result = PHONETIC_MAP[
            token_clean
        ]

        st.success(
            "Phonetic match found."
        )

        # ─────────────────────────────────
        # CORE PHONETICS
        # ─────────────────────────────────

        st.markdown("---")

        st.subheader(
            "Phonetic Reconstruction"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Skeleton",
                result["skeleton"]
            )

        with col2:

            st.metric(
                "Normalized",
                result["normalized"]
            )

        with col3:

            st.metric(
                "Class",
                result["class"]
            )

        # ─────────────────────────────────
        # PHONETIC FORM
        # ─────────────────────────────────

        st.markdown("---")

        st.subheader(
            "Phonetic Form"
        )

        st.info(
            result["phonetic"]
        )

        # ─────────────────────────────────
        # HYPOTHESIS
        # ─────────────────────────────────

        st.markdown("---")

        st.subheader(
            "Procedural Hypothesis"
        )

        st.warning(
            result["hypothesis"]
        )

        # ─────────────────────────────────
        # TABLE
        # ─────────────────────────────────

        st.markdown("---")

        st.subheader(
            "Phonetic Breakdown"
        )

        df = pd.DataFrame([{

            "Token":
                token_clean,

            "Skeleton":
                result["skeleton"],

            "Normalized":
                result["normalized"],

            "Phonetic":
                result["phonetic"],

            "Class":
                result["class"],

            "Hypothesis":
                result["hypothesis"]

        }])

        st.dataframe(
            df,
            use_container_width=True
        )

    else:

        st.warning("""
No phonetic reconstruction
available in current Node Beta
database.

Future versions may support:

- fuzzy phonetic matching
- multilingual alignment
- Sanskrit comparison
- Arabic transliteration
- Latin procedural matching
""")

# ─────────────────────────────────────────────
# THEORY NOTES
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Node Beta Notes")

st.info("""
Node Beta provides hypothetical
phonetic reconstruction models.

Outputs do NOT represent verified
translation of the Voynich Manuscript.

Phonetic alignment remains
experimental and requires
independent validation.
""")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption("""
AMF Node Beta • Experimental
Phonetic Reconstruction Environment
""")
