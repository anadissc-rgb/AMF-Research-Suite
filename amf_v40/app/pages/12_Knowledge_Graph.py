# ─────────────────────────────────────────────
# AMF KNOWLEDGE GRAPH
# ─────────────────────────────────────────────

import streamlit as st
import pandas as pd

try:

    import networkx as nx

except Exception as e:

    st.error(
        "networkx failed to load."
    )

    st.exception(e)

    st.stop()

try:

    import plotly.graph_objects as go

except Exception as e:

    st.error(
        "plotly failed to load."
    )

    st.exception(e)

    st.stop()
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
    page_title="Knowledge Graph",
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

st.title("Knowledge Graph & Token Networks")

st.markdown("""
The Knowledge Graph module visualizes:

- token relationships
- procedural clusters
- semantic proximity
- manuscript topology
- adaptive network structures
""")

# ─────────────────────────────────────────────
# TOKEN RELATIONSHIPS
# ─────────────────────────────────────────────

TOKEN_RELATIONS = [

    ("qokeedy", "qotedy"),
    ("qokeedy", "kchedy"),
    ("kchedy", "lchedy"),
    ("shedy", "qokeedy"),
    ("qokain", "qokeedy"),
    ("qokain", "qotedy"),
    ("lchedy", "shedy")

]

# ─────────────────────────────────────────────
# GRAPH BUILD
# ─────────────────────────────────────────────

G = nx.Graph()

G.add_edges_from(TOKEN_RELATIONS)

# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────

pos = nx.spring_layout(
    G,
    seed=42
)

# ─────────────────────────────────────────────
# EDGE TRACE
# ─────────────────────────────────────────────

edge_x = []
edge_y = []

for edge in G.edges():

    x0, y0 = pos[edge[0]]

    x1, y1 = pos[edge[1]]

    edge_x.extend([
        x0,
        x1,
        None
    ])

    edge_y.extend([
        y0,
        y1,
        None
    ])

edge_trace = go.Scatter(

    x=edge_x,

    y=edge_y,

    line=dict(
        width=1,
        color="#888"
    ),

    hoverinfo="none",

    mode="lines"

)

# ─────────────────────────────────────────────
# NODE TRACE
# ─────────────────────────────────────────────

node_x = []
node_y = []
node_text = []

for node in G.nodes():

    x, y = pos[node]

    node_x.append(x)

    node_y.append(y)

    node_text.append(node)

node_trace = go.Scatter(

    x=node_x,

    y=node_y,

    mode="markers+text",

    hoverinfo="text",

    text=node_text,

    textposition="top center",

    marker=dict(

        size=20,

        line_width=2

    )

)

# ─────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────

fig = go.Figure(

    data=[
        edge_trace,
        node_trace
    ],

    layout=go.Layout(

        title="AMF Token Relationship Network",

        showlegend=False,

        hovermode="closest",

        margin=dict(
            b=20,
            l=5,
            r=5,
            t=40
        ),

        xaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False
        ),

        yaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False
        )

    )

)

# ─────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Interactive Token Network")

st.plotly_chart(
    fig,
    use_container_width=True
)

# ─────────────────────────────────────────────
# NETWORK TABLE
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Relationship Matrix")

df = pd.DataFrame(

    TOKEN_RELATIONS,

    columns=[
        "Source Token",
        "Connected Token"
    ]

)

st.dataframe(
    df,
    use_container_width=True
)

# ─────────────────────────────────────────────
# NETWORK METRICS
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Network Metrics")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Nodes",
        G.number_of_nodes()
    )

with col2:

    st.metric(
        "Edges",
        G.number_of_edges()
    )

with col3:

    density = round(
        nx.density(G),
        3
    )

    st.metric(
        "Density",
        density
    )

# ─────────────────────────────────────────────
# CLUSTER ANALYSIS
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Cluster Analysis")

clusters = list(
    nx.connected_components(G)
)

cluster_rows = []

for i, cluster in enumerate(clusters):

    cluster_rows.append({

        "Cluster":
            f"Cluster {i+1}",

        "Tokens":
            ", ".join(cluster),

        "Size":
            len(cluster)

    })

cluster_df = pd.DataFrame(
    cluster_rows
)

st.dataframe(
    cluster_df,
    use_container_width=True
)

# ─────────────────────────────────────────────
# FUTURE GRAPH AI
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Future Graph Intelligence")

st.info("""
Future versions may support:

- graph neural networks
- semantic embeddings
- manuscript topology learning
- adaptive procedural clustering
- token influence propagation
- AI-assisted manuscript graphing
- dynamic relationship inference
""")

# ─────────────────────────────────────────────
# THEORY NOTES
# ─────────────────────────────────────────────

st.markdown("---")

st.subheader("Knowledge Graph Notes")

st.warning("""
Graph relationships currently represent
experimental procedural associations
within the AMF framework.

Graph proximity does NOT constitute
verified linguistic translation.
""")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")

st.caption("""
AMF Knowledge Graph
Experimental Manuscript Network
Intelligence Environment
""")
