# ─────────────────────────────────────────────
# AMF HEADER COMPONENT
# ─────────────────────────────────────────────

import streamlit as st

from utils.config import (
    APP_TITLE,
    APP_VERSION,
    APP_SUBTITLE
)

def render_header():

    st.title(APP_TITLE)

    st.caption(
        f"{APP_SUBTITLE} • {APP_VERSION}"
    )

    st.markdown("---")
