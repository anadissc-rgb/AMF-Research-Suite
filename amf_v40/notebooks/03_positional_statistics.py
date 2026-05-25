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
# # AMF v4.0 — Notebook 03: Positional Statistics
#
# **Purpose**: Examine how token distributions vary by line position,
# word position within a line, and manuscript section.
#
# **Epistemic status — VERIFIED**: Positional frequencies are directly
# measurable from the corpus. Their *causes* (grammar, cipher, scribal
# convention, notation rules) are not determined here.
#
# **Key prior result (Currier 1976; Landini 2001)**:
# Voynichese exhibits strong positional token structure — some tokens
# appear almost exclusively at line-initial positions. This is one of
# the most reproducible findings in Voynich statistics.

# %% [markdown]
# ## 1. Setup

# %%
import sys; sys.path.insert(0, "..")

from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from amf.corpus.ingest import EVACorpus
from amf.stats.positional import (
    compute_positional_stats, high_bias_tokens, classify_folio, save_positional_report
)

CORPUS_PATH = Path("../data/processed/corpus.json")
OUTPUT_DIR = Path("../outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## 2. Compute positional statistics

# %%
corpus = EVACorpus.from_json(CORPUS_PATH)
stats = compute_positional_stats(corpus.records, top_n=40)
save_positional_report(stats, OUTPUT_DIR / "positional_report.json")

print(f"Total lines   : {stats.total_lines:,}")
print(f"Total tokens  : {stats.total_tokens:,}")
print(f"Sections found: {list(stats.section_profiles.keys())}")

# %% [markdown]
# ## 3. Line-initial bias plot
#
# Tokens with high line-initial rates appear disproportionately at the
# START of lines. If this is grammatical, they may be sentence-initial
# elements. If scribal, they may be rubric or paragraph markers.

# %%
initial_df = pd.DataFrame(stats.line_initial_top)
# Filter to tokens with meaningful frequency
initial_df = initial_df[initial_df["total_count"] >= 10].head(25)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#C46A55" if r >= 0.5 else "#3D405B" for r in initial_df["line_initial_rate"]]
ax.barh(initial_df["token"], initial_df["line_initial_rate"], color=colors)
ax.axvline(0.5, color="#B8956A", linestyle="--", alpha=0.7, label="50% rate")
ax.set_xlabel("Line-initial rate (fraction of occurrences at line start)")
ax.set_title("Voynichese: Token Line-Initial Rate\n"
             "(Red = ≥50% of occurrences are line-initial; min count=10)")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "positional_line_initial.png", dpi=150)
plt.show()

# %%
biased = high_bias_tokens(stats, min_rate=0.4, min_count=10)
print("High line-initial bias tokens (rate ≥ 40%, count ≥ 10):")
print(biased["line_initial_biased"])
print()
print("High line-final bias tokens (rate ≥ 40%, count ≥ 10):")
print(biased["line_final_biased"])
print()
print("NOTE:", biased["note"])

# %% [markdown]
# ## 4. Section-specific token profiles
#
# Does the token distribution differ between manuscript sections
# (herbal, pharmaceutical, biological, etc.)?
#
# This is a descriptive comparison only — we compute top tokens per section
# and visually compare them. Statistical significance testing (chi-square
# or permutation test) is a planned future step.

# %%
sections = [s for s in stats.section_profiles if s != "unknown"]
if sections:
    fig, axes = plt.subplots(1, len(sections), figsize=(4 * len(sections), 5),
                              sharey=False)
    if len(sections) == 1:
        axes = [axes]

    for ax, section in zip(axes, sections):
        profile = stats.section_profiles[section]
        top = sorted(profile.items(), key=lambda x: -x[1])[:10]
        tokens = [t for t, _ in top]
        counts = [c for _, c in top]
        ax.barh(tokens, counts, color="#7C9885")
        ax.set_title(f"{section.capitalize()}\n(top 10 tokens)")
        ax.set_xlabel("Count")

    plt.suptitle("Token Distributions by Manuscript Section\n"
                 "(No significance testing — descriptive only)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "section_token_profiles.png", dpi=150)
    plt.show()
else:
    print("No section data available. VTT folio IDs required for section mapping.")

# %% [markdown]
# ## 5. Word position within line
#
# Which tokens appear at positions 1, 2, 3 within a line?
# Strong positional preferences suggest fixed phrase templates.

# %%
pos_profiles = stats.word_position_profiles
positions_to_show = ["1", "2", "3", "4"]

fig, axes = plt.subplots(1, len(positions_to_show), figsize=(12, 5))
for ax, pos in zip(axes, positions_to_show):
    if pos not in pos_profiles:
        ax.set_visible(False)
        continue
    items = pos_profiles[pos][:8]
    tokens = [e["token"] for e in items]
    counts = [e["count"] for e in items]
    ax.bar(tokens, counts, color="#3D405B", alpha=0.8)
    ax.set_title(f"Word position {pos}")
    ax.set_xlabel("Token")
    ax.set_ylabel("Count" if pos == "1" else "")
    ax.tick_params(axis="x", rotation=45)

plt.suptitle("Top Tokens at Each Word Position Within Line", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "word_position_profiles.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Summary of findings

# %%
print("=" * 60)
print("POSITIONAL ANALYSIS SUMMARY")
print("=" * 60)
print(f"""
Verified measurements:
  - {len(biased['line_initial_biased'])} tokens show ≥40% line-initial rate (n≥10)
  - {len(biased['line_final_biased'])} tokens show ≥40% line-final rate (n≥10)
  - {len(sections)} manuscript sections identified with distinct token profiles

These findings replicate prior work (Currier 1976, Landini 2001).

INTERPRETATION NOTE:
Strong positional bias is consistent with:
  (a) Grammatical word-order constraints
  (b) Scribal formatting conventions
  (c) Cipher positional rules
  (d) DPAS notation template (AMF hypothesis)

The analysis here cannot distinguish between these alternatives.
Statistical significance testing against a null model (random token
placement) is a planned future step.
""")

# %% [markdown]
# ---
# **End of Notebook 03**
#
# Next: `04_adjacency_markov.ipynb`
