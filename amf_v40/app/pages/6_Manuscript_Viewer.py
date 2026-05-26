# ─────────────────────────────────────────────
# AMF MANUSCRIPT VIEWER
# ─────────────────────────────────────────────

import streamlit as st
from PIL import Image

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

st.title("Manuscript Viewer")

st.markdown("""
The Manuscript Viewer provides a
visual analysis workspace for
Voynich folios and manuscript imagery.

This module supports:

- Folio inspection
- Codicological analysis
- Visual annotation workflows
- Token-image alignment
- Diagram examination
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
# DISPLAY IMAGE
# ─────────────────────────────────────────────

if uploaded_image:

    image = Image.open(uploaded_image)

    st.success(
        "Folio image loaded successfully."
    )

    # ─────────────────────────────────────────
    # IMAGE METRICS
    # ─────────────────────────────────────────

    st.markdown("---")

    st.subheader("Image Information")

    width, height = image.size

    col1, col2, col3 = st.columns(3)

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

    # ─────────────────────────────────────────
    # ZOOM CONTROL
    # ─────────────────────────────────────────

    st.markdown("---")

    zoom = st.slider(
        "Zoom Level",
        min_value=25,
        max_value=200,
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

    st.subheader("Folio View")

    st.image(
        image,
        width=resized_width
    )

    # ─────────────────────────────────────────
    # TRANSCRIPTION PANEL
    # ─────────────────────────────────────────

    st.markdown("---")

    st.subheader(
        "Associated Transcription"
    )

    transcription = st.text_area(
        "EVA Transcription",
        height=200,
        placeholder="""
qokeedy qokedy dal

otary shedy qokedy

qotedy qokain shedy
"""
    )

    # ─────────────────────────────────────────
    # NOTES PANEL
    # ─────────────────────────────────────────

    st.markdown("---")

    st.subheader("Research Notes")

    notes = st.text_area(
        "Annotation Notes",
        height=200,
        placeholder="""
Enter codicological observations,
diagrammatic analysis,
glyph structure notes,
or procedural hypotheses.
"""
    )

    # ─────────────────────────────────────────
    # EXPORT PLACEHOLDER
    # ─────────────────────────────────────────

    st.markdown("---")

    st.subheader("Session Export")

    st.info("""
Future versions may support:

- annotation export
- token overlays
- glyph bounding boxes
- region tagging
- AI-assisted folio analysis
- codicological comparison
""")

# ─────────────────────────────────────────────
# VIEWER NOTES
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Viewer Notes")

st.info("""
The Manuscript Viewer is designed
as a research workspace rather than
a translation engine.

Visual interpretation remains
hypothetical and requires independent
validation.
""")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption("""
AMF Manuscript Viewer • Experimental
Codicological Analysis Environment
""")
