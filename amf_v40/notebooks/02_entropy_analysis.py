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
# # AMF v4.0 — Notebook 02: Entropy and Zipf Analysis
#
# **Purpose**: Compute Shannon entropy and Zipf distribution measures for
# the EVA corpus and compare against a frequency-matched random baseline.
#
# **Epistemic status — VERIFIED**: These are reproducible statistical
# measurements. Their *interpretation* (what they imply about the nature
# of Voynichese) is **not** settled here. See the interpretation notes
# throughout.
#
# **Literature context**:
# - Landini (2001) established that Voynichese entropy is consistent with
#   natural language and lower than random text with the same character
#   frequencies. This notebook reproduces and extends that analysis.
# - Montemurro & Zanette (2013) confirmed Zipf-like distribution with
#   exponent close to natural language values.

# %% [markdown]
# ## 1. Setup

# %%
import sys; sys.path.insert(0, "..")

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from amf.corpus.ingest import EVACorpus
from amf.stats.entropy import full_entropy_report, save_report

CORPUS_PATH = Path("../data/processed/corpus.json")
OUTPUT_DIR = Path("../outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## 2. Run entropy analysis

# %%
corpus = EVACorpus.from_json(CORPUS_PATH)
tokens = corpus.all_tokens(clean=True)   # clean = uncertainty markers stripped

print(f"Analysing {len(tokens):,} tokens ({len(set(tokens)):,} unique types)...")
report = full_entropy_report(tokens, label="voynich_eva")
save_report(report, OUTPUT_DIR / "entropy_report.json")

ent = report["entropy"]
zipf = report["zipf"]
baseline = report["random_baseline"]
comp = report["comparison"]

# %% [markdown]
# ## 3. Summary table

# %%
summary = pd.DataFrame({
    "Measure": [
        "Token count",
        "Unique types",
        "Type-token ratio",
        "Unigram entropy H(W) [bits]",
        "Normalised entropy H(W)/H_max",
        "Cond. entropy H(W|W-1) [bits]",
        "Redundancy",
        "Zipf exponent α",
        "Zipf R² (log-log fit)",
    ],
    "Voynich": [
        f"{ent['token_count']:,}",
        f"{ent['type_count']:,}",
        f"{ent['type_token_ratio']:.4f}",
        f"{ent['unigram_entropy']:.4f}",
        f"{ent['unigram_entropy_normalised']:.4f}",
        f"{ent['bigram_entropy']:.4f}",
        f"{ent['redundancy']:.4f}",
        f"{zipf['alpha']:.4f}",
        f"{zipf['r_squared']:.4f}",
    ],
    "Random baseline": [
        f"{baseline['token_count']:,}",
        f"{baseline['type_count']:,}",
        f"{baseline['type_token_ratio']:.4f}",
        f"{baseline['unigram_entropy']:.4f}",
        f"{baseline['unigram_entropy_normalised']:.4f}",
        f"{baseline['bigram_entropy']:.4f}",
        f"{baseline['redundancy']:.4f}",
        "N/A",
        "N/A",
    ],
})
print(summary.to_string(index=False))

# %% [markdown]
# ## 4. Rank-frequency (Zipf) plot

# %%
rf_pairs = zipf["rank_freq_pairs"]
ranks = np.array([p["rank"] for p in rf_pairs])
freqs = np.array([p["freq"] for p in rf_pairs])

# Fitted line
alpha = zipf["alpha"]
C_const = freqs[0] * (1 ** alpha)   # approximate normalization
fitted_freqs = C_const * (ranks ** -alpha)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Log-log plot
ax = axes[0]
ax.scatter(ranks, freqs, s=10, color="#3D405B", alpha=0.7, label="Observed")
ax.plot(ranks, fitted_freqs, color="#C46A55", linewidth=1.5,
        label=f"Zipf fit α={alpha:.3f}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Rank (log scale)")
ax.set_ylabel("Frequency (log scale)")
ax.set_title(f"Voynichese Rank-Frequency Distribution\nZipf α = {alpha:.3f}, R² = {zipf['r_squared']:.3f}")
ax.legend()

# Linear scale (first 50 ranks)
ax = axes[1]
ax.bar(ranks[:50], freqs[:50], color="#3D405B", alpha=0.8)
ax.set_xlabel("Rank (top 50 types)")
ax.set_ylabel("Frequency")
ax.set_title("Top 50 Token Frequencies (linear scale)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "zipf_analysis.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Interpretation (explicit)

# %%
print("=" * 60)
print("INTERPRETATION NOTE")
print("=" * 60)
print(f"""
Voynich unigram entropy: {ent['unigram_entropy']:.3f} bits
Random baseline entropy: {baseline['unigram_entropy']:.3f} bits
Δ (Voynich lower by)   : {comp['delta_bits']:.3f} bits

Zipf exponent α = {alpha:.3f}
Natural language range: approximately 0.8–1.2

{comp['interpretation']}

WHAT THIS DOES NOT TELL US:
These statistics are consistent with natural language, but also with:
  - A cipher over natural language
  - A constructed glossolalia with language-like statistics
  - A structured notational system (the AMF hypothesis)

No causal or semantic conclusion is drawn from these measurements.
""")

# %% [markdown]
# ---
# **End of Notebook 02**
#
# Next: `03_positional_statistics.ipynb`
