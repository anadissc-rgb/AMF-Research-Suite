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
# # AMF v4.0 — Notebook 01: Corpus Exploration
#
# **Purpose**: Load the EVA corpus and examine its basic structure.
#
# **Epistemic status**: This notebook produces *verified* statistical
# measurements — reproducible counts and distributions. No interpretive
# claims are made here.
#
# **Prerequisites**:
# 1. Download EVA transcription to `data/eva/` (see `data/README.md`)
# 2. Run corpus ingestion:
#    ```bash
#    python -m amf.corpus.ingest --input data/eva/ \
#        --version YOUR_VERSION --output data/processed/corpus.json
#    ```

# %% [markdown]
# ## 1. Setup

# %%
import sys
sys.path.insert(0, "..")   # so amf is importable from notebooks/

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from amf.corpus.ingest import EVACorpus
from amf.corpus.normalize import normalize_sequence, tokenize_to_units

# Reproducibility: fix random seed for any stochastic operations below
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

CORPUS_PATH = Path("../data/processed/corpus.json")

# %% [markdown]
# ## 2. Load corpus

# %%
if not CORPUS_PATH.exists():
    raise FileNotFoundError(
        f"Corpus not found at {CORPUS_PATH}. "
        "Run: python -m amf.corpus.ingest --input data/eva/ --output data/processed/corpus.json"
    )

corpus = EVACorpus.from_json(CORPUS_PATH)
print(corpus.summary())

# %% [markdown]
# ## 3. Basic counts

# %%
total_tokens = corpus.metadata.token_count
all_tokens = corpus.all_tokens(clean=False)
clean_tokens = corpus.all_tokens(clean=True)

token_types = Counter(clean_tokens)
print(f"Total tokens (raw)    : {len(all_tokens):,}")
print(f"Total tokens (clean)  : {len(clean_tokens):,}")
print(f"Unique types (clean)  : {len(token_types):,}")
print(f"Type-token ratio      : {len(token_types)/len(clean_tokens):.4f}")
print(f"\nTop 20 most frequent tokens:")
for tok, cnt in token_types.most_common(20):
    pct = 100 * cnt / len(clean_tokens)
    print(f"  {tok:20s}  {cnt:6,}  ({pct:.2f}%)")

# %% [markdown]
# ## 4. Token length distribution

# %%
# Measure length in EVA multigraph units (not raw characters)
lengths_in_units = [len(tokenize_to_units(t)) for t in clean_tokens]
length_counter = Counter(lengths_in_units)

fig, ax = plt.subplots(figsize=(8, 4))
lengths_sorted = sorted(length_counter.items())
ax.bar([x for x, _ in lengths_sorted],
       [y for _, y in lengths_sorted],
       color="#3D405B", alpha=0.8)
ax.set_xlabel("Token length (EVA units)")
ax.set_ylabel("Frequency")
ax.set_title("Voynichese Token Length Distribution\n(measured in EVA multigraph units)")
ax.axvline(np.mean(lengths_in_units), color="#C46A55", linestyle="--",
           label=f"Mean = {np.mean(lengths_in_units):.2f}")
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/token_length_distribution.png", dpi=150)
plt.show()

print(f"\nMean token length : {np.mean(lengths_in_units):.3f} units")
print(f"Std dev           : {np.std(lengths_in_units):.3f} units")
print(f"Min / Max         : {min(lengths_in_units)} / {max(lengths_in_units)}")

# %% [markdown]
# ## 5. Folio and section breakdown

# %%
folio_counts = Counter(r.folio for r in corpus.records)
section_counts = Counter(r.section for r in corpus.records)

print("Records by section type:")
for sec, cnt in sorted(section_counts.items(), key=lambda x: -x[1]):
    print(f"  {sec:5s}  {cnt:5,} lines")

print(f"\nTotal folios: {len(folio_counts)}")

# %% [markdown]
# ## 6. Uncertainty coverage

# %%
uncertain_lines = [r for r in corpus.records if r.has_uncertainty]
uncertain_pct = 100 * len(uncertain_lines) / len(corpus.records)
print(f"Lines with uncertainty markers: {len(uncertain_lines):,} ({uncertain_pct:.1f}%)")
print()
print("NOTE: Analyses using clean_tokens (uncertainty markers stripped) will")
print("treat {uncertain} characters as certain. This is documented but may")
print("introduce errors. Run with and without strip_uncertain to check sensitivity.")

# %% [markdown]
# ---
# **End of Notebook 01**
#
# Next: `02_entropy_analysis.ipynb` — Shannon entropy and Zipf fitting
