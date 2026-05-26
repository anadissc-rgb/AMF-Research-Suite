import streamlit as st
import sys

st.set_page_config(
    page_title="System Check",
    layout="wide"
)

st.title("AMF System Check")

st.markdown("---")

st.subheader("Python Runtime")

st.code(sys.version)

st.markdown("---")

packages = [

    "streamlit",
    "numpy",
    "pandas",
    "scipy",
    "plotly",
    "networkx",
    "PIL"

]

results = []

for package in packages:

    try:

        __import__(package)

        results.append({

            "Package": package,
            "Status": "OK"

        })

    except Exception as e:

        results.append({

            "Package": package,
            "Status": str(e)

        })

st.table(results)
