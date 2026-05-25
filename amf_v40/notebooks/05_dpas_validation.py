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
# # AMF v4.0 — Notebook 05: DPAS Constraint Validation
#
# **Purpose**: Test the DPAS (Discrete Positional Alignment System)
# constraint model against the full EVA corpus. This is the primary
# **falsifiability test** of AMF v4.0.
#
# **Epistemic status — HYPOTHESIS TEST**: The DPAS model makes a
# specific, testable prediction: ≥70% of corpus tokens should be
# classifiable as DPAS-valid. This notebook runs that test and
# reports the result without bias toward either outcome.
#
# **If the model fails**: Coverage < 70% means the DPAS template
# must be revised. The notebook documents failure modes.
#
# **If the model passes**: Coverage ≥ 70% supports (but does not
# prove) the DPAS hypothesis. Coverage alone is not sufficient
# evidence — the baseline coverage on non-Voynich text must also
# be established.

# %% [markdown]
# ## 1. Setup

# %%
import sys; sys.path.insert(0, "..")

from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from amf.corpus.ingest import EVACorpus
from amf.dpas.constraints import DPAS_TEMPLATE, COVERAGE_THRESHOLD, COVERAGE_NOTE
from amf.dpas.validator import (
    validate_corpus, validate_token, save_validation_report
)

CORPUS_PATH = Path("../data/processed/corpus.json")
OUTPUT_DIR = Path("../outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## 2. DPAS template summary

# %%
print("DPAS Template (provisional):")
print("=" * 60)
for slot in DPAS_TEMPLATE:
    status = "REQUIRED" if not slot.optional else "optional"
    print(f"\n  Slot: {slot.name.upper():12s}  [{status}]")
    print(f"  Permitted ({len(slot.permitted)} items): "
          f"{sorted(slot.permitted)[:8]}{'...' if len(slot.permitted)>8 else ''}")
    print(f"  Max repeat: {slot.max_repeat}")
    # First sentence of description only (keeping output concise)
    desc_short = slot.description.split(".")[0] + "."
    print(f"  Note: {desc_short}")

print(f"\n\nCoverage threshold: {COVERAGE_THRESHOLD:.0%}")
print(f"\n{COVERAGE_NOTE}")

# %% [markdown]
# ## 3. Run validation

# %%
corpus = EVACorpus.from_json(CORPUS_PATH)
tokens = corpus.all_tokens(clean=True)

print(f"Validating {len(tokens):,} tokens...")
report, full_results = validate_corpus(tokens, return_full_results=True)
save_validation_report(report, OUTPUT_DIR / "dpas_validation.json")

# %% [markdown]
# ## 4. Primary result: coverage

# %%
print("\n" + "=" * 60)
print("DPAS VALIDATION RESULT")
print("=" * 60)
print(f"  Total tokens    : {report.total_tokens:,}")
print(f"  Valid           : {report.valid_count:,} ({report.coverage:.1%})")
print(f"  Invalid         : {report.invalid_count:,}")
print(f"  Threshold       : {report.coverage_threshold:.0%}")
print(f"  Model supported : {'YES ✓' if report.model_supported else 'NO ✗ — SEE INTERPRETATION'}")
print()
print(f"INTERPRETATION:\n{report.interpretation}")

# %% [markdown]
# ## 5. Coverage visualization

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Pie chart
ax = axes[0]
labels = ["DPAS-valid", "Invalid"]
sizes = [report.valid_count, report.invalid_count]
colors = ["#7C9885", "#C46A55"]
explode = (0.05, 0)
ax.pie(sizes, labels=labels, colors=colors, explode=explode,
       autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})
ax.set_title(f"DPAS Coverage\n(threshold = {report.coverage_threshold:.0%})")

# Slot usage bar chart
ax = axes[1]
if report.slot_usage:
    slots = list(report.slot_usage.keys())
    usage = [report.slot_usage[s] for s in slots]
    ax.bar(slots, usage, color="#3D405B", alpha=0.8)
    ax.set_xlabel("DPAS Slot")
    ax.set_ylabel("Match count (valid tokens)")
    ax.set_title("Slot Usage in Valid Tokens\n(each valid token may use multiple slots)")
else:
    ax.text(0.5, 0.5, "No valid tokens found", ha="center", va="center",
            transform=ax.transAxes, fontsize=12)
    ax.set_title("Slot Usage (no data)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dpas_coverage.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Failure mode analysis

# %%
print("Failure categories:")
for reason, count in sorted(report.failure_reasons.items(), key=lambda x: -x[1]):
    pct = 100 * count / report.invalid_count if report.invalid_count > 0 else 0
    print(f"  {reason:35s}  {count:6,}  ({pct:.1f}%)")

print(f"\nMost common invalid tokens (top 20):")
print(f"{'Token':25s}  {'Count':>7}")
print("-" * 36)
for e in report.invalid_top[:20]:
    print(f"  {e['token']:25s}  {e['count']:>7,}")

# %% [markdown]
# ## 7. Per-section coverage breakdown

# %%
from amf.stats.positional import classify_folio

# Compute coverage per manuscript section
section_results = {}
for record in corpus.records:
    section = classify_folio(record.folio)
    if section not in section_results:
        section_results[section] = {"valid": 0, "invalid": 0}
    for tok in record.clean_tokens:
        result = validate_token(tok)
        if result.is_valid:
            section_results[section]["valid"] += 1
        else:
            section_results[section]["invalid"] += 1

print("\nDPAS coverage by manuscript section:")
print(f"{'Section':20s}  {'Valid':>8}  {'Invalid':>8}  {'Coverage':>10}")
print("-" * 54)
for section, counts in sorted(section_results.items()):
    total = counts["valid"] + counts["invalid"]
    cov = counts["valid"] / total if total > 0 else 0
    flag = "✓" if cov >= COVERAGE_THRESHOLD else "✗"
    print(f"  {section:20s}  {counts['valid']:>8,}  {counts['invalid']:>8,}  "
          f"{cov:>9.1%} {flag}")

# %%
# Visualize section coverage
if section_results:
    sections = [s for s in section_results if s != "unknown"]
    coverages = [
        section_results[s]["valid"] / (section_results[s]["valid"] + section_results[s]["invalid"])
        for s in sections
    ]
    colors = ["#7C9885" if c >= COVERAGE_THRESHOLD else "#C46A55" for c in coverages]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(sections, coverages, color=colors, alpha=0.85)
    ax.axhline(COVERAGE_THRESHOLD, color="#B8956A", linestyle="--",
               label=f"Threshold ({COVERAGE_THRESHOLD:.0%})")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("DPAS coverage")
    ax.set_title("DPAS Coverage by Manuscript Section\n"
                 "(Provisional — section classifications are interpretive)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dpas_section_coverage.png", dpi=150)
    plt.show()

# %% [markdown]
# ## 8. Falsifiability assessment

# %%
print("""
FALSIFIABILITY ASSESSMENT
==========================

The DPAS model was tested against the following pre-specified criterion:
  Coverage ≥ {:.0%} → model supported
  Coverage < {:.0%} → model requires revision

Result: {:.1%} → Model {}

Outstanding falsifiability requirements NOT yet met:
  1. Baseline coverage on non-Voynich text not established
     (Is 70%+ coverage trivially achievable for any text?)
  2. DPAS definitions were derived from the same corpus being tested
     (Circularity — see docs/limitations.md §4.3)
  3. No held-out test set was used (all data seen during definition)
  4. No independent replication by a second researcher

NEXT STEPS before claiming model support:
  - Test DPAS on 3 Latin texts and 3 constructed random texts
  - If baseline coverage ≈ Voynich coverage, the model adds no value
  - Pre-register slot definitions in a public registry before retesting
""".format(
    COVERAGE_THRESHOLD, COVERAGE_THRESHOLD,
    report.coverage,
    "SUPPORTED (with caveats)" if report.model_supported else "NOT SUPPORTED"
))

# %% [markdown]
# ---
# **End of Notebook 05**
#
# All five analysis notebooks complete.
# Full pipeline: `python -m amf.validation.pipeline --corpus data/processed/corpus.json`
