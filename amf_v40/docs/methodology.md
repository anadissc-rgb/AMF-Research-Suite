# AMF v4.0 — Methodology

## Overview

AMF v4.0 is a computational framework for statistical and structural
analysis of the Voynich Manuscript EVA transcription. The methodology
is organized into four layers, each with defined epistemic status.

---

## Layer 1: Data Ingestion (Reproducible)

**Module**: `amf.corpus.ingest`

**Input**: EVA transcription files (`.vtt` interlinear format or plain text)

**Process**:
- Parse folio/section/line structure from VTT tags
- Tokenize on whitespace, normalizing dot and dash separators
- Record EVA uncertainty markers ({}, !, ?) without stripping
- Store transcription version in metadata for provenance

**Output**: JSON corpus with provenance metadata

**Epistemic status**: The token sequences produced are faithful
representations of the input transcription. All subsequent analyses
inherit the transcription's uncertainties.

---

## Layer 2: Statistical Analysis (Verified)

**Modules**: `amf.stats.*`

Results in this layer are directly computable from the corpus and
reproducible by any researcher with the same input data.

### 2.1 Entropy Analysis

Shannon unigram entropy H(W) measures the diversity of the token
distribution:

    H(W) = -Σ p(w) log₂ p(w)

Conditional bigram entropy H(W_i | W_{i-1}) estimates sequential
dependency. It is computed as H(bigrams) - H(unigrams) — an approximation
that underestimates true conditional entropy in short corpora.

**Limitation**: Entropy consistent with natural language is necessary
but not sufficient for the natural language hypothesis.

### 2.2 Zipf Analysis

Power-law exponent α fitted to rank-frequency data via log-log linear
regression. This is a standard but imperfect method; MLE fitting (via
the `powerlaw` library) should replace this in a future version.

### 2.3 Positional Analysis

Token frequency at line-initial and line-final positions, normalized
by overall token frequency to produce a positional rate measure.
Section classification is based on Zandbergen's codicological conventions
and is itself interpretive.

### 2.4 Adjacency and PMI

Within-line bigram and trigram frequencies. Pointwise Mutual Information
(PMI) computed for pairs with frequency ≥ 5 (Laplace smoothed).

### 2.5 Markov Analysis

Language models at orders 0–4 trained with Laplace smoothing.
Perplexity computed on TRAINING DATA (in-sample) — a known limitation.
Train/test splitting is a planned future step.

---

## Layer 3: DPAS Constraint Validation (Hypothesis)

**Module**: `amf.dpas`

The DPAS model is a **formal hypothesis** with one primary testable
prediction: that ≥70% of corpus tokens are classifiable as DPAS-valid.

### Slot Matching Algorithm

Token validation uses greedy left-to-right slot matching against
the template defined in `amf.dpas.constraints.DPAS_TEMPLATE`. A token
is valid iff:
1. All required slots are satisfied
2. No unconsumed characters remain after template exhaustion

The greedy algorithm may produce false negatives when alternative
segmentations would succeed. Coverage is therefore a **lower bound**.

### Coverage Threshold

The 70% threshold is provisional. It was chosen to be:
- Conservative enough to account for transcription noise (~10%)
  and model incompleteness (~20%)
- Strict enough to be scientifically meaningful

**Circularity risk**: The DPAS slot definitions were derived from
qualitative analysis of the same corpus used for validation. This is
a methodological weakness documented in `docs/limitations.md §4.3`.

---

## Layer 4: Interpretive Mapping (Speculative)

Any mapping from statistical patterns to semantic content (botanical
terminology, pharmaceutical instructions, etc.) is **speculative**.

The AMF framework records such mappings in structured form in the
pipeline output's `speculative_mappings` field. They are labelled
explicitly and include required validation steps before they could
be cited as findings.

---

## Reproducibility

All analyses are deterministic given:
1. The same input corpus (version recorded in metadata)
2. The same AMF code version (pinned in `pyproject.toml`)
3. The same random seed for baseline generation (fixed at 42, documented)

To reproduce any result:
```bash
# 1. Confirm corpus version
cat data/processed/corpus.json | python -c "import json,sys; m=json.load(sys.stdin)['metadata']; print(m['transcription_version'], m['amf_version'])"

# 2. Run full pipeline
python -m amf.validation.pipeline \
    --corpus data/processed/corpus.json \
    --output outputs/ \
    --run-id replication-$(date +%Y%m%d)
```

Results are saved to `outputs/pipeline_run_*.json` with full provenance.

---

## Baseline Comparisons (Planned)

To establish whether AMF statistics are distinctive to Voynichese
or trivially achievable, the following baseline comparisons are planned
but not yet implemented:

1. **Latin text baseline**: Apply the full pipeline to a Latin manuscript
   transcription of comparable length (e.g. Voynich-era Latin herbals)
2. **English baseline**: Apply to NLTK Brown corpus subset
3. **Random text baseline**: Already implemented (frequency-matched)
4. **Known cipher baseline**: Apply to a known historical cipher text

Without baselines 1 and 2, the absolute values of AMF statistics
cannot be interpreted as distinctive to Voynichese.
