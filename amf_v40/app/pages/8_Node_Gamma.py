# ─────────────────────────────────────────────
# AMF NODE GAMMA
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
    page_title="Node Gamma",
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

st.title("Node Gamma")

st.markdown("""
Node Gamma performs medicinal,
botanical, and procedural
cross-reference analysis.

This layer explores:

- herbal interpretation
- Ayurvedic parallels
- pharmaceutical procedures
- medicinal classifications
- botanical workflow alignment
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
# BOTANICAL DATABASE
# ─────────────────────────────────────────────

BOTANICAL_MAP = {

    "qokeedy": {

        "procedure":
            "Layered heating extraction",

        "botanical":
            "Herbal decoction",

        "ayurvedic":
            "Kwatha preparation",

        "classification":
            "Thermal medicinal process",

        "pharmacology":
            "Concentrated extraction"

    },

    "qotedy": {

        "procedure":
            "Reduction boiling",

        "botanical":
            "Medicinal resin reduction",

        "ayurvedic":
            "Sneha paka process",

        "classification":
            "Reduction therapy",

        "pharmacology":
            "Potency concentration"

    },

    "kchedy": {

        "procedure":
            "Press extraction",

        "botanical":
            "Oil extraction",

        "ayurvedic":
            "Taila preparation",

        "classification":
            "Compression process",

        "pharmacology":
            "Active compound isolation"

    },

    "lchedy": {

        "procedure":
            "Filtration sequence",

        "botanical":
            "Herbal purification",

        "ayurvedic":
            "Shodhana process",

        "classification":
            "Purification process",

        "pharmacology":
            "Impurity reduction"

    }

}

# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────

if token:

    st.markdown("---")

    token_clean = token.strip().lower()

    if token_clean in BOTANICAL_MAP:

        result = BOTANICAL_MAP[
            token_clean
        ]

        st.success(
            "Medicinal alignment found."
        )

        # ─────────────────────────────────
        # CORE METRICS
        # ─────────────────────────────────

        st.subheader(
            "Medicinal Classification"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Botanical Type",
                result["botanical"]
            )

        with col2:

            st.metric(
                "Classification",
                result["classification"]
            )

        # ─────────────────────────────────
        # PROCEDURE
        # ─────────────────────────────────

        st.markdown("---")

        st.subheader(
            "Procedural Mapping"
        )

        st.info(
            result["procedure"]
        )

        # ─────────────────────────────────
        # AYURVEDIC ALIGNMENT
        # ─────────────────────────────────

        st.markdown("---")

        st.subheader(
            "Ayurvedic Alignment"
        )

        st.success(
            result["ayurvedic"]
        )

        # ─────────────────────────────────
        # PHARMACOLOGY
        # ─────────────────────────────────

        st.markdown("---")

        st.subheader(
            "Pharmacological Interpretation"
        )

        st.warning(
            result["pharmacology"]
        )

        # ─────────────────────────────────
        # TABLE
        # ─────────────────────────────────

        st.markdown("---")

        st.subheader(
            "Medicinal Breakdown"
        )

        df = pd.DataFrame([{

            "Token":
                token_clean,

            "Procedure":
                result["procedure"],

            "Botanical":
                result["botanical"],

            "Ayurvedic":
                result["ayurvedic"],

            "Classification":
                result["classification"],

            "Pharmacology":
                result["pharmacology"]

        }])

        st.dataframe(
            df,
            use_container_width=True
        )

    else:

        st.warning("""
No botanical alignment found
in current Node Gamma database.

Future versions may support:

- plant taxonomy matching
- Ayurvedic corpus linking
- herbal ontology mapping
- procedural medicine alignment
- pharmacognostic comparison
""")

# ─────────────────────────────────────────────
# THEORY NOTES
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Node Gamma Notes")

st.info("""
Node Gamma provides hypothetical
medicinal and botanical alignment.

Outputs do NOT constitute verified
translation of the Voynich Manuscript.

Cross-reference interpretations remain
experimental and require independent
validation.
""")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption("""
AMF Node Gamma • Experimental
Medicinal Reconstruction Environment
""")
