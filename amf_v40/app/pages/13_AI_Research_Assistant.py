# ─────────────────────────────────────────────
# AMF AI RESEARCH ASSISTANT
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
    page_title="AI Research Assistant",
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

st.title("AI Research Assistant")

st.markdown("""
The AI Research Assistant provides:

- cross-module guidance
- manuscript reasoning support
- token intelligence assistance
- workflow recommendations
- procedural synthesis guidance
- adaptive research orchestration
""")

# ─────────────────────────────────────────────
# KNOWLEDGE BASE
# ─────────────────────────────────────────────

KNOWLEDGE_BASE = {

    "entropy":
        """
Entropy analysis measures the
statistical unpredictability of
Voynich token distributions.
""",

    "zipf":
        """
Zipf analysis evaluates whether
token frequencies follow power-law
distribution behavior.
""",

    "dpas":
        """
DPAS validates procedural slot
coverage and constraint satisfaction
within the corpus.
""",

    "falsification":
        """
The falsification framework tests
whether AMF hypotheses survive
adversarial validation.
""",

    "qokeedy":
        """
qokeedy is currently modeled as a
thermal procedural token associated
with layered extraction workflows.
""",

    "kchedy":
        """
kchedy is associated with
compression or press-extraction
procedural sequences.
""",

    "graph":
        """
The knowledge graph visualizes
token relationships and procedural
clusters.
"""
}

# ─────────────────────────────────────────────
# QUERY INPUT
# ─────────────────────────────────────────────

st.markdown("---")

query = st.text_input(
    "Ask the Research Assistant",
    placeholder="""
Example:
What does qokeedy represent?
"""
)

# ─────────────────────────────────────────────
# PROCESS QUERY
# ─────────────────────────────────────────────

if query:

    st.markdown("---")

    query_clean = query.lower()

    matched = False

    for key, value in (
        KNOWLEDGE_BASE.items()
    ):

        if key in query_clean:

            matched = True

            st.subheader(
                "Assistant Response"
            )

            st.success(value)

            # ─────────────────────────────
            # CONTEXT PANEL
            # ─────────────────────────────

            st.markdown("---")

            st.subheader(
                "Suggested Modules"
            )

            module_rows = []

            if key in [
                "entropy",
                "zipf"
            ]:

                module_rows.append({

                    "Module":
                        "Statistics Lab",

                    "Purpose":
                        "Statistical analysis"
                })

            if key == "dpas":

                module_rows.append({

                    "Module":
                        "DPAS Validator",

                    "Purpose":
                        "Constraint analysis"
                })

            if key == "falsification":

                module_rows.append({

                    "Module":
                        "Falsification Lab",

                    "Purpose":
                        "Scientific testing"
                })

            if key in [
                "qokeedy",
                "kchedy"
            ]:

                module_rows.extend([

                    {

                        "Module":
                            "Node Alpha",

                        "Purpose":
                            "Structural analysis"
                    },

                    {

                        "Module":
                            "Node Beta",

                        "Purpose":
                            "Phonetic reconstruction"
                    },

                    {

                        "Module":
                            "Node Gamma",

                        "Purpose":
                            "Medicinal interpretation"
                    },

                    {

                        "Module":
                            "Node Delta",

                        "Purpose":
                            "Procedural synthesis"
                    }

                ])

            if key == "graph":

                module_rows.append({

                    "Module":
                        "Knowledge Graph",

                    "Purpose":
                        "Network intelligence"
                })

            if module_rows:

                module_df = pd.DataFrame(
                    module_rows
                )

                st.dataframe(
                    module_df,
                    use_container_width=True
                )

            # ─────────────────────────────
            # SESSION MEMORY
            # ─────────────────────────────

            st.markdown("---")

            st.subheader(
                "Session Memory"
            )

            memory_df = pd.DataFrame([{

                "Timestamp":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "Query":
                    query,

                "Matched Topic":
                    key

            }])

            st.dataframe(
                memory_df,
                use_container_width=True
            )

            break

    # ─────────────────────────────────────────
    # NO MATCH
    # ─────────────────────────────────────────

    if not matched:

        st.warning("""
No direct knowledge match found.

Future versions may support:

- semantic embeddings
- transformer reasoning
- adaptive manuscript memory
- neural procedural inference
- cross-folio intelligence
- graph-assisted reasoning
""")

# ─────────────────────────────────────────────
# QUICK ACCESS
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Suggested Research Queries")

queries = [

    "What is entropy analysis?",

    "Explain DPAS",

    "What does qokeedy represent?",

    "Explain falsification framework",

    "What is the knowledge graph?"

]

for q in queries:

    st.code(q)

# ─────────────────────────────────────────────
# AI ROADMAP
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Future AI Research Layers")

st.info("""
Future versions may support:

- transformer-based manuscript reasoning
- adaptive semantic embeddings
- OCR-assisted manuscript parsing
- AI procedural synthesis
- graph neural manuscript learning
- autonomous research guidance
- hypothesis generation engines
- manuscript memory systems
""")

# ─────────────────────────────────────────────
# THEORY NOTES
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Assistant Notes")

st.warning("""
The AI Research Assistant currently
uses experimental rule-based
research guidance.

Responses do NOT constitute verified
translation or historical certainty.
""")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption("""
AMF AI Research Assistant
Experimental Manuscript Intelligence
Orchestration Environment
""")
