# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # AMF v4.0 — Notebook 04: Adjacency & Markov Analysis
#
# **Purpose**: Examine token co-occurrence (PMI, bigrams) and how well
# Markov models of increasing order capture sequential structure.
#
# **Epistemic status — VERIFIED**: Bigram counts, PMI values, and Markov
# perplexity are reproducible measurements. Their interpretation
# is explicitly noted throughout.
#
# **Key question**: Does knowing the previous token help predict the next
# token, and by how much? If yes, sequential structure exists.

# %% [markdown]
# ## 1. Setup

# %%
import sys; sys.path.insert(0, "..")

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from amf.corpus.ingest import EVACorpus
from amf.stats.adjacency import compute_adjacency, save_adjacency_report
from amf.stats.markov import run_markov_analysis, save_markov_report

CORPUS_PATH = Path("../data/processed/corpus.json")
OUTPUT_DIR = Path("../outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## 2. Adjacency analysis

# %%
corpus = EVACorpus.from_json(CORPUS_PATH)
adj_report = compute_adjacency(corpus.records, top_n=30, min_pmi_count=5)
save_adjacency_report(adj_report, OUTPUT_DIR / "adjacency_report.json")

print(f"Total bigrams      : {adj_report.total_bigrams:,}")
print(f"Unique bigram types: {adj_report.unique_bigram_types:,}")
print(f"Bigram entropy     : {adj_report.bigram_entropy:.4f} bits")
print(f"\nTop 15 most frequent bigrams:")
for e in adj_report.top_bigrams[:15]:
    bg = " + ".join(e["bigram"])
    print(f"  {bg:35s}  n={e['count']:5,}")

# %% [markdown]
# ## 3. High-PMI token pairs
#
# High PMI pairs co-occur more than expected by their individual frequencies.
# These are the most "bound" bigrams — they may represent compound tokens,
# fixed phrases, or notation units.

# %%
print(f"\nTop 15 high-PMI pairs (min frequency={5}):")
print(f"{'Pair':35s}  {'PMI':>7}  {'Count':>7}")
print("-" * 55)
for e in adj_report.top_pmi_pairs[:15]:
    pair = " + ".join(e["bigram"])
    print(f"  {pair:35s}  {e['pmi']:>7.3f}  {e['count']:>7,}")

print(f"\n{adj_report.interpretation_note}")

# %% [markdown]
# ## 4. Token co-occurrence network (top 30 bigrams)
#
# Visualize the most frequent bigrams as a directed network.
# Node size = token frequency; edge weight = bigram frequency.

# %%
from collections import Counter

# Build frequency dict for node sizing
all_tokens = corpus.all_tokens(clean=True)
freq = Counter(all_tokens)

G = nx.DiGraph()
for e in adj_report.top_bigrams[:30]:
    src, tgt = e["bigram"]
    G.add_edge(src, tgt, weight=e["count"])

pos = nx.spring_layout(G, seed=42, k=2.5)
node_sizes = [freq.get(n, 1) * 0.5 for n in G.nodes()]
edge_weights = [d["weight"] / 100 for _, _, d in G.edges(data=True)]

fig, ax = plt.subplots(figsize=(12, 10))
nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                       node_color="#3D405B", alpha=0.8, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", ax=ax)
nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5,
                       edge_color="#B8956A", arrows=True,
                       arrowsize=15, ax=ax)
ax.set_title("Voynichese Token Co-occurrence Network\n"
             "(Top 30 bigrams; node size ∝ token frequency; edge weight ∝ bigram frequency)",
             fontsize=11)
ax.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "bigram_network.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Markov order analysis

# %%
markov_report = run_markov_analysis(corpus.records, max_order=4, smoothing=1.0)
save_markov_report(markov_report, OUTPUT_DIR / "markov_report.json")

print(f"\n{'Order':>6}  {'Perplexity':>14}  {'Contexts':>12}")
print("-" * 38)
for r in markov_report.results_by_order:
    print(f"  {r['order']:>4}  {r['train_perplexity']:>14.2f}  {r['unique_contexts']:>12,}")

print(f"\nPerplexity reduction by order:")
for k, v in markov_report.perplexity_reduction.items():
    print(f"  Order {k}: {v:.1%} reduction")

print(f"\nSuggested optimal order: {markov_report.optimal_order}")

# %% [markdown]
# ## 6. Perplexity curve plot

# %%
orders = [r["order"] for r in markov_report.results_by_order]
perplexities = [r["train_perplexity"] for r in markov_report.results_by_order]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(orders, perplexities, "o-", color="#3D405B", linewidth=2, markersize=8)
for o, pp in zip(orders, perplexities):
    ax.annotate(f"{pp:.1f}", (o, pp), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=9)
ax.set_xlabel("Markov order")
ax.set_ylabel("Perplexity (log₂ scale)")
ax.set_yscale("log")
ax.set_xticks(orders)
ax.set_title("Markov Model Perplexity vs. Order\n"
             "(CAUTION: in-sample estimates — see notebook note)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "markov_perplexity.png", dpi=150)
plt.show()

# %% [markdown]
# ## 7. Important caveat: in-sample perplexity

# %%
print("""
⚠ METHODOLOGICAL NOTE
======================
All Markov perplexity values above are TRAINING SET estimates —
the model is evaluated on the same data it was trained on.

Higher-order models always fit training data better regardless of
whether the underlying process has that structure. This means:

  - The perplexity curve above CANNOT be used to determine the
    true Markov order of the generating process
  - A proper analysis requires holding out 20% of lines as test data
    BEFORE training and evaluating on the held-out set

This is documented as a TODO in amf/stats/markov.py.

Despite this limitation, the ORDER of magnitude of perplexity values
(and the unigram → bigram reduction) are informative as lower bounds
on the true structure.
""")

# %% [markdown]
# ---
# **End of Notebook 04**
#
# Next: `05_dpas_validation.ipynb`
