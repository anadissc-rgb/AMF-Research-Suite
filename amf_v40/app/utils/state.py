# ─────────────────────────────────────────────
# STREAMLIT SESSION STATE MANAGER
# ─────────────────────────────────────────────

import streamlit as st

def init_state():

    defaults = {

        "corpus_path": None,

        "uploaded_filename": None,

        "analysis_result": None,

        "verified_results": None,

        "hypothesis_results": None,

        "dpas_result": None,

        "falsification_result": None,

        "warnings": [],

        "limitations": []

    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value
