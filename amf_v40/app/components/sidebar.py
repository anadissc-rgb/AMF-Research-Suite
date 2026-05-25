# ─────────────────────────────────────────────
# AMF SIDEBAR COMPONENT
# ─────────────────────────────────────────────

import streamlit as st

from app.utils.config import (
    AUTHOR,
    INSTITUTION
)

def render_sidebar():

    st.sidebar.title("AMF Navigation")

    st.sidebar.markdown("""
### Research Modules

- Corpus Loader
- Node Alpha
- Node Beta
- Node Gamma
- Node Delta
- Statistics Lab
- DPAS Validator
- Falsification Lab
- Research Export
""")

    st.sidebar.markdown("---")

    st.sidebar.caption(
        f"{AUTHOR}\n\n{INSTITUTION}"
    )
