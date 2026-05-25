# AMF v4.0 — Documented Limitations

This document is a first-class part of the AMF v4.0 framework.
It is maintained alongside the code, not as an afterthought.

Limitations are organized into four categories:
1. Corpus limitations (input data problems)
2. Statistical limitations (what the numbers can and cannot show)
3. Model limitations (DPAS constraint system)
4. Interpretive limitations (what no analysis can determine)

---

## 1. Corpus Limitations

### 1.1 Transcription Dependence
All AMF v4.0 analyses are derived from the EVA transcription, not from
direct manuscript analysis. Every transcription decision by the original
transcribers is assumed correct — errors propagate silently.

**Impact**: Unknown. Estimated 5–10% character-level uncertainty.

**Mitigation**: Run pipeline on multiple independent transcriptions and
report inter-transcription variance. Not yet implemented.

### 1.2 Single Transcription Source
AMF v4.0 alpha uses a single transcription source. Results are not
validated against alternative transcriptions (Takahashi, Currier, etc.).

**Impact**: All reported statistics may shift when a different
transcription is used.

**Required action before claiming robustness**: Run on ≥2 transcriptions
and compare outputs quantitatively.

### 1.3 Manuscript Damage and Lacunae
Some folios are damaged, faded, or missing. The EVA transcription marks
these with uncertainty characters ({}, !, ?). AMF v4.0 tracks these
markers but does not model the underlying missing information.

---

## 2. Statistical Limitations

### 2.1 Entropy Analysis
Shannon entropy of Voynichese is well-established (Landini 2001). AMF
v4.0 reproduces these measurements. However:

- Entropy consistent with natural language ≠ proof of natural language
- The same entropy range is achievable with: constructed glossolalia,
  cipher over natural language, constrained notation, or structured code
- Entropy analysis **cannot** distinguish between these hypotheses

### 2.2 Zipf Distribution Fitting
The Zipf exponent is estimated via log-log linear regression, which is
a standard but imperfect method (Clauset et al. 2009 criticise this
approach). The `powerlaw` library provides better estimation via maximum
likelihood — this should replace the current regression approach.

**Status**: TODO — switch to MLE-based Zipf fitting.

### 2.3 Small-Section Analyses
Pharmaceutical and recipe sections have fewer tokens than herbal/biological
sections. Section-specific statistics are less reliable for smaller sections.
Confidence intervals should be reported but are not yet computed.

**Status**: TODO — bootstrap confidence intervals for all statistics.

### 2.4 Positional Statistics
Positional token frequencies are reproducible measurements. Their
interpretation (that they reflect grammatical, scribal, or notational
structure) is a hypothesis. All three explanations are consistent with
the observed data.

---

## 3. Model Limitations (DPAS)

### 3.1 Slot Definitions Are Provisional
The DPAS slot definitions (in `amf/dpas/constraints.py`) were constructed
from qualitative analysis of frequent EVA token families. They have not
been:
- Derived from a principled linguistic analysis
- Validated against an independent dataset
- Tested for uniqueness (other slot definitions may achieve equal coverage)

### 3.2 Greedy Matching Ambiguity
The token validator uses greedy left-to-right slot matching. This may
mis-classify tokens where an alternative segmentation would satisfy the
template. The current implementation reports conservative (lower-bound)
coverage.

### 3.3 Coverage Threshold Is Arbitrary
The 70% coverage threshold was chosen conservatively. It is not derived
from a statistical power analysis or comparison to known languages. A
different threshold would yield different conclusions about model support.

### 3.4 No Cross-Language Validation
The DPAS model has not been tested on known languages or constructed
systems. We cannot determine whether a 70% coverage rate is unusual,
expected, or trivially achievable for any token sequence.

**Required validation**: Apply DPAS to EVA-transcribed Latin texts of
similar length to establish a baseline coverage rate.

---

## 4. Interpretive Limitations

### 4.1 The Framework Cannot Identify the Language
No statistical analysis of Voynichese can identify the underlying
language, script system, or content type. The AMF framework generates
hypotheses; it does not test them against ground truth.

### 4.2 The Framework Cannot Confirm or Refute Decipherment Claims
AMF v4.0 is a constraint-based analytical tool, not a decipherment
engine. Any mapping from Voynichese tokens to semantic content (botanical
terms, pharmaceutical instructions, etc.) remains speculative until
independently verified by a method that does not use the Voynich
Manuscript itself as its primary evidence.

### 4.3 Circular Interpretation Risk
There is an inherent risk that the DPAS slot definitions were tuned
(consciously or not) to match the corpus they were derived from. This
circular reasoning risk can only be mitigated by:
1. Pre-registering DPAS definitions before running coverage analysis
2. Testing against a held-out portion of the corpus
3. Testing against non-Voynich manuscripts

None of these mitigations have been implemented in AMF v4.0 alpha.

### 4.4 Publication Bias Risk
Positive results (DPAS coverage above threshold, entropy consistent with
natural language) are more likely to be reported than negative results.
AMF v4.0 explicitly commits to reporting all results including failures,
and to publishing the full pipeline output regardless of whether it
supports the DPAS hypothesis.

---

## Planned Mitigations (Future Work)

See `docs/future_work.md` for the roadmap. Priority items:
1. Multi-transcription comparative analysis
2. MLE-based Zipf fitting with confidence intervals
3. Bootstrap confidence intervals for all positional statistics
4. DPAS validation against non-Voynich manuscripts as baseline
5. Pre-registration of DPAS definitions before coverage analysis
6. Independent replication protocol (documented in `docs/replication.md`)
