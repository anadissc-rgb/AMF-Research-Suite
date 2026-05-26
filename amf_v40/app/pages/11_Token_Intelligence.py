# ─────────────────────────────────────────────
# AMF TOKEN INTELLIGENCE
# ─────────────────────────────────────────────

import streamlit as st
import pandas as pd
import difflib

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
    page_title="Token Intelligence",
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

st.title("AI-Assisted Token Intelligence")

st.markdown("""
The Token Intelligence Engine performs:

- fuzzy token matching
- structural similarity analysis
- procedural prediction
- phonetic clustering
- adaptive token reasoning
- semantic approximation
""")

# ─────────────────────────────────────────────
# TOKEN DATABASE
# ─────────────────────────────────────────────

TOKEN_DB = {

    "qokeedy": {
        "class": "thermal",
        "meaning": "Layered heating extraction"
    },

    "qotedy": {
        "class": "reduction",
        "meaning": "Boiling reduction process"
    },

    "kchedy": {
        "class": "compression",
        "meaning": "Press extraction sequence"
    },

    "lchedy": {
        "class": "filtration",
        "meaning": "Purification filtering"
    },

    "shedy": {
        "class": "supportive",
        "meaning": "Auxiliary preparation stage"
    },

    "qokain": {
        "class": "compound",
        "meaning": "Compound preparation"
    }

}

# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────

st.markdown("---")

query = st.text_input(
    "Enter EVA Token",
    placeholder="qokedy"
)

# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────

if query:

    st.markdown("---")

    query_clean = query.strip().lower()

    database_tokens = list(
        TOKEN_DB.keys()
    )

    # ─────────────────────────────────────────
    # SIMILARITY MATCHING
    # ─────────────────────────────────────────

    matches = difflib.get_close_matches(

        query_clean,

        database_tokens,

        n=5,

        cutoff=0.3

    )

    if matches:

        st.success(
            "Similarity matches found."
        )

        # ─────────────────────────────────
        # MATCH TABLE
        # ─────────────────────────────────

        rows = []

        for token in matches:

            similarity = round(

                difflib.SequenceMatcher(
                    None,
                    query_clean,
                    token
                ).ratio(),

                3

            )

            rows.append({

                "Matched Token":
                    token,

                "Similarity":
                    similarity,

                "Class":
                    TOKEN_DB[token]["class"],

                "Interpretation":
                    TOKEN_DB[token][
                        "meaning"
                    ]

            })

        df = pd.DataFrame(rows)

        # ─────────────────────────────────
        # TOP MATCH
        # ─────────────────────────────────

        best_match = rows[0]

        st.subheader("Best Match")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Token",
                best_match[
                    "Matched Token"
                ]
            )

        with col2:

            st.metric(
                "Similarity",
                best_match[
                    "Similarity"
                ]
            )

        with col3:

            st.metric(
                "Class",
                best_match[
                    "Class"
                ]
            )

        # ─────────────────────────────────
        # INTERPRETATION
        # ─────────────────────────────────

        st.markdown("---")

        st.subheader(
            "Predicted Interpretation"
        )

        st.info(
            best_match[
                "Interpretation"
            ]
        )

        # ─────────────────────────────────
        # FULL MATCHES
        # ─────────────────────────────────

        st.markdown("---")

        st.subheader(
            "Similarity Matches"
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        # ─────────────────────────────────
        # TOKEN NETWORK
        # ─────────────────────────────────

        st.markdown("---")

        st.subheader(
            "Token Intelligence Notes"
        )

        st.warning("""
Similarity scoring does NOT
constitute verified translation.

Matches represent probabilistic
structural proximity within the
experimental AMF token database.
""")

    else:

        st.error(
            "No similar tokens found."
        )

# ─────────────────────────────────────────────
# FUTURE AI ROADMAP
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Future Intelligence Layers")

st.info("""
Future versions may support:

- transformer embeddings
- semantic token clustering
- graph neural manuscript analysis
- probabilistic reconstruction
- OCR-assisted glyph extraction
- adaptive manuscript memory
- multilingual token embeddings
- neural procedural synthesis
""")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption("""
AMF Token Intelligence
Experimental Adaptive Manuscript
Reasoning Environment
""")
