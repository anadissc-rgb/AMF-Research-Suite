# ─────────────────────────────────────────────
# AMF NODE DELTA
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
    page_title="Node Delta",
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

st.title("Node Delta")

st.markdown("""
Node Delta performs procedural
synthesis and operational workflow
reconstruction.

This layer explores:

- instructional sequencing
- process reconstruction
- operational logic
- manuscript-to-workflow conversion
- procedural synthesis
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
# SYNTHESIS DATABASE
# ─────────────────────────────────────────────

DELTA_MAP = {

    "qokeedy": {

        "step":
            "Apply layered heating",

        "sequence":
            "Preparation → Heating → Reduction",

        "workflow":
            "Thermal extraction cycle",

        "instruction":
            "Gradually heat medicinal substrate in sequential layers.",

        "output":
            "Concentrated decoction"

    },

    "qotedy": {

        "step":
            "Perform reduction boiling",

        "sequence":
            "Heating → Reduction → Concentration",

        "workflow":
            "Liquid concentration cycle",

        "instruction":
            "Boil continuously until volume decreases significantly.",

        "output":
            "Reduced medicinal concentrate"

    },

    "kchedy": {

        "step":
            "Execute pressure extraction",

        "sequence":
            "Compression → Separation → Collection",

        "workflow":
            "Mechanical extraction cycle",

        "instruction":
            "Press botanical material to isolate active compounds.",

        "output":
            "Extracted medicinal oil"

    },

    "lchedy": {

        "step":
            "Perform filtration sequence",

        "sequence":
            "Purification → Filtering → Clarification",

        "workflow":
            "Purification cycle",

        "instruction":
            "Filter suspended impurities from processed solution.",

        "output":
            "Purified preparation"

    }

}

# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────

if token:

    st.markdown("---")

    token_clean = token.strip().lower()

    if token_clean in DELTA_MAP:

        result = DELTA_MAP[token_clean]

        st.success(
            "Procedural synthesis generated."
        )

        # ─────────────────────────────────
        # CORE METRICS
        # ─────────────────────────────────

        st.subheader(
            "Operational Synthesis"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Primary Step",
                result["step"]
            )

        with col2:

            st.metric(
                "Final Output",
                result["output"]
            )

        # ─────────────────────────────────
        # WORKFLOW
        # ─────────────────────────────────

        st.markdown("---")

        st.subheader(
            "Workflow Sequence"
        )

        st.info(
            result["sequence"]
        )

        # ─────────────────────────────────
        # PROCEDURAL CYCLE
        # ─────────────────────────────────

        st.markdown("---")

        st.subheader(
            "Procedural Cycle"
        )

        st.success(
            result["workflow"]
        )

        # ─────────────────────────────────
        # INSTRUCTION
        # ─────────────────────────────────

        st.markdown("---")

        st.subheader(
            "Instructional Reconstruction"
        )

        st.warning(
            result["instruction"]
        )

        # ─────────────────────────────────
        # TABLE
        # ─────────────────────────────────

        st.markdown("---")

        st.subheader(
            "Synthesis Breakdown"
        )

        df = pd.DataFrame([{

            "Token":
                token_clean,

            "Step":
                result["step"],

            "Sequence":
                result["sequence"],

            "Workflow":
                result["workflow"],

            "Instruction":
                result["instruction"],

            "Output":
                result["output"]

        }])

        st.dataframe(
            df,
            use_container_width=True
        )

    else:

        st.warning("""
No procedural synthesis available
in current Node Delta database.

Future versions may support:

- automated workflow generation
- process graph synthesis
- instructional AI reconstruction
- manuscript recipe modeling
- procedural simulation
""")

# ─────────────────────────────────────────────
# THEORY NOTES
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Node Delta Notes")

st.info("""
Node Delta provides hypothetical
procedural synthesis models.

Outputs do NOT constitute verified
translation of the Voynich Manuscript.

Workflow reconstructions remain
experimental and require independent
validation.
""")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption("""
AMF Node Delta • Experimental
Procedural Synthesis Environment
""")
