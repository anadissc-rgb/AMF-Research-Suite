# ─────────────────────────────────────────────
# AMF ADVANCED MANUSCRIPT VIEWER
# ─────────────────────────────────────────────

import streamlit as st
from PIL import Image
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
    page_title="Manuscript Viewer",
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

st.title("Advanced Manuscript Viewer")

st.markdown("""
The Advanced Manuscript Viewer provides
an annotation-aware workspace for:

- folio inspection
- codicological analysis
- token-region mapping
- manuscript intelligence workflows
- glyph annotation systems
""")

# ─────────────────────────────────────────────
# IMAGE UPLOAD
# ─────────────────────────────────────────────

st.markdown("---")

uploaded_image = st.file_uploader(
    "Upload Manuscript Folio",
    type=["png", "jpg", "jpeg"]
)

# ─────────────────────────────────────────────
# IMAGE PROCESSING
# ─────────────────────────────────────────────

if uploaded_image:

    image = Image.open(uploaded_image)

    st.success(
        "Folio image loaded successfully."
    )

    width, height = image.size

    # ─────────────────────────────────────────
    # METADATA
    # ─────────────────────────────────────────

    st.markdown("---")

    st.subheader("Folio Metadata")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Width",
            width
        )

    with col2:

        st.metric(
            "Height",
            height
        )

    with col3:

        st.metric(
            "Mode",
            image.mode
        )

    with col4:

        st.metric(
            "Aspect Ratio",
            round(width / height, 2)
        )

    # ─────────────────────────────────────────
    # ZOOM
    # ─────────────────────────────────────────

    st.markdown("---")

    zoom = st.slider(
        "Zoom Level",
        min_value=25,
        max_value=300,
        value=100,
        step=25
    )

    resized_width = int(
        width * (zoom / 100)
    )

    # ─────────────────────────────────────────
    # IMAGE DISPLAY
    # ─────────────────────────────────────────

    st.markdown("---")

    st.subheader("Folio Workspace")

    st.image(
        image,
        width=resized_width
    )

    # ─────────────────────────────────────────
    # TOKEN MAPPING
    # ─────────────────────────────────────────

    st.markdown("---")

    st.subheader("Token Mapping")

    token = st.text_input(
        "Associated EVA Token",
        placeholder="qokeedy"
    )

    col1, col2 = st.columns(2)

    with col1:

        x_coord = st.number_input(
            "X Coordinate",
            min_value=0,
            max_value=width,
            value=100
        )

    with col2:

        y_coord = st.number_input(
            "Y Coordinate",
            min_value=0,
            max_value=height,
            value=100
        )

    region_type = st.selectbox(

        "Region Type",

        [

            "Botanical",
            "Textual",
            "Pharmaceutical",
            "Astronomical",
            "Diagrammatic",
            "Unknown"

        ]

    )

    # ─────────────────────────────────────────
    # ANNOTATION
    # ─────────────────────────────────────────

    st.markdown("---")

    st.subheader("Annotation Notes")

    notes = st.text_area(
        "Research Annotation",
        height=200,
        placeholder="""
Enter codicological observations,
glyph relationships,
procedural hypotheses,
or manuscript analysis notes.
"""
    )

    # ─────────────────────────────────────────
    # ANNOTATION TABLE
    # ─────────────────────────────────────────

    st.markdown("---")

    st.subheader("Annotation Record")

    annotation_df = pd.DataFrame([{

        "Token":
            token,

        "X":
            x_coord,

        "Y":
            y_coord,

        "Region":
            region_type,

        "Notes":
            notes

    }])

    st.dataframe(
        annotation_df,
        use_container_width=True
    )

    # ─────────────────────────────────────────
    # EXPORT
    # ─────────────────────────────────────────

    st.markdown("---")

    st.subheader("Export Annotation")

    csv = annotation_df.to_csv(
        index=False
    )

    st.download_button(
        label="Download Annotation CSV",
        data=csv,
        file_name="amf_annotation.csv",
        mime="text/csv"
    )

    # ─────────────────────────────────────────
    # FUTURE OVERLAYS
    # ─────────────────────────────────────────

    st.markdown("---")

    st.subheader("Future Intelligence Layers")

    st.info("""
Future versions may support:

- glyph bounding boxes
- OCR integration
- AI-assisted token detection
- manuscript segmentation
- folio clustering
- visual token overlays
- region heatmaps
- codicological comparison
- graph-based annotation systems
""")

# ─────────────────────────────────────────────
# THEORY NOTES
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Viewer Notes")

st.info("""
The Advanced Manuscript Viewer
is designed as a manuscript research
workspace rather than a translation
engine.

Visual interpretation remains
experimental and requires
independent validation.
""")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption("""
AMF Advanced Manuscript Viewer
Experimental Codicological
Intelligence Environment
""")
