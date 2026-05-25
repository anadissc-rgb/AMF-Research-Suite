# AMF v4.0 — Future Work Roadmap

Prioritized list of required improvements before AMF v4.0 can
be considered research-ready. Items marked 🔴 are blockers for
any scholarly claim; 🟡 are important improvements; 🟢 are enhancements.

---

## 🔴 Priority 1: Eliminate Circularity Risk

**Issue**: DPAS slot definitions were derived from the same corpus
used to test coverage.

**Required action**:
1. Pre-register the current DPAS template publicly (OSF or Zenodo)
   before any further corpus analysis
2. Run a fresh coverage test against a second independent transcription
   (e.g. Takahashi transcription)
3. Report both results; if they diverge significantly, the model is
   not robust

**Estimated effort**: Medium (1–2 weeks)

---

## 🔴 Priority 2: Baseline Coverage on Non-Voynich Text

**Issue**: Without a baseline, we cannot determine whether the DPAS
coverage rate is distinctive to Voynichese or trivially achievable.

**Required action**:
1. Select 3 Latin medieval texts of comparable token count
2. Convert to EVA-like tokenization (apply the same pipeline)
3. Run DPAS validation on each
4. Report baseline coverage ± standard deviation
5. If Voynich coverage ≤ baseline mean + 2σ, the DPAS model is not
   distinctive

**Estimated effort**: Medium-high (2–4 weeks)

---

## 🔴 Priority 3: Train/Test Split for Markov Analysis

**Issue**: Current perplexity is in-sample (trained and evaluated on
same data).

**Required action**:
- Hold out 20% of lines (random split with fixed seed) before training
- Evaluate perplexity on held-out set only
- Report both train and test perplexity

**Estimated effort**: Low (< 1 day, code change in `amf/stats/markov.py`)

---

## 🟡 Priority 4: Confidence Intervals

**Issue**: All statistics are reported as point estimates without
confidence intervals.

**Required action**:
- Bootstrap (n=1000 resamples, seed=42) for entropy and Zipf estimates
- Report 95% bootstrap CI for all statistics
- Library: `scipy.stats.bootstrap`

**Estimated effort**: Medium (2–3 days)

---

## 🟡 Priority 5: Zipf MLE Fitting

**Issue**: Current log-log linear regression for Zipf α is biased.

**Required action**:
- Replace with `powerlaw` library MLE estimation
- Report both estimates and compare
- Reference: Clauset, Shalizi & Newman (2009)

**Estimated effort**: Low (< 1 day)

---

## 🟡 Priority 6: Inter-Transcription Comparison

**Issue**: All results depend on a single transcription source.

**Required action**:
- Run full pipeline on Zandbergen AND Takahashi transcriptions
- Compute: for each statistic, |result_Z - result_T| / mean
- Report discordant statistics (those that differ by >10%) as unreliable

**Estimated effort**: Medium (1 week, pending data availability)

---

## 🟢 Priority 7: Section Significance Testing

**Issue**: Section-specific token profiles are described but not
tested for statistical significance.

**Required action**:
- Chi-square test: are section token distributions significantly different?
- Report p-values with Bonferroni correction for multiple comparisons

---

## 🟢 Priority 8: DPAS Slot Revision Protocol

The DPAS slot definitions need a transparent revision process:

1. Any revision to slot definitions must be documented with:
   - Which corpus evidence motivated it
   - Which coverage test it was designed to improve
   - What it breaks (tokens that were valid and become invalid, vice versa)
2. All revisions must be versioned in `docs/dpas_spec.md`
3. Coverage must be re-reported after every revision

---

# AMF v4.0 — Independent Replication Protocol

This document specifies the steps an independent researcher must follow
to replicate AMF v4.0 results.

## Requirements

- Python 3.10+
- Docker (optional but recommended)
- EVA transcription files (freely available, see `data/README.md`)

## Step-by-step

```bash
# 1. Clone repository
git clone https://github.com/YOUR_ORG/amf_v40.git
cd amf_v40

# 2. Pin code version (use the exact commit of the results being replicated)
git checkout <COMMIT_HASH>

# 3. Set up environment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 4. Download EVA corpus to data/eva/
# Source: https://www.voynich.nu/transcr.html
# Record the exact download date as your transcription_version

# 5. Ingest corpus
python -m amf.corpus.ingest \
    --input data/eva/ \
    --version "Zandbergen-YYYY-MM-DD" \
    --output data/processed/corpus.json

# 6. Run full pipeline
python -m amf.validation.pipeline \
    --corpus data/processed/corpus.json \
    --output outputs/replication/ \
    --run-id independent-replication-1

# 7. Compare results
# Key metrics to compare (from pipeline_run_*.json > verified_results):
#   - entropy.unigram_bits
#   - zipf.alpha
#   - zipf.r_squared
#   - positional.line_initial_biased_tokens
# Key hypothesis test (from hypothesis_results):
#   - dpas_validation.coverage
#   - dpas_validation.model_supported
```

## Expected variation

Results may differ from the reference run if:
- A different EVA transcription version is used (expected, document it)
- A different Python or NumPy version produces floating-point differences
  (should be < 0.001 in entropy values)
- The corpus has been updated since the reference run

## Reporting discrepancies

If you obtain substantially different results, please:
1. Check transcription versions match
2. Check code commit hashes match
3. Open an issue at the repository with your `pipeline_run_*.json` output

This replication protocol is a living document. Additions welcome.
