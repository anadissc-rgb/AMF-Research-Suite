"""
amf.stats.adjacency
====================

Token Adjacency Analysis: Bigrams, Trigrams, and Transition Matrices
---------------------------------------------------------------------

Adjacency analysis examines which tokens co-occur as neighbors within
Voynich lines and which token sequences are frequent or rare.

WHAT THIS MEASURES
------------------
- Bigram (W_i, W_{i+1}) frequency: which pairs of adjacent tokens appear
  together more often than chance?
- Trigram frequency: extending to triples
- Transition probability matrix: P(W_{i+1} | W_i) — the probability of
  each token following each other token
- Mutual information: I(W_i; W_{i+1}) — which pairs are more dependent
  than their individual frequencies alone would predict?

WHAT THIS TELLS US
------------------
High mutual information between token pairs is consistent with:
  - Fixed phrase structure (grammatical patterns)
  - Scribal production formulas (recurring abbreviations)
  - Notational conventions (e.g., always precede ingredient with dosage marker)

It is NOT evidence for any specific linguistic or semantic interpretation.

BOUNDARY HANDLING
-----------------
Bigrams are computed WITHIN lines only — we do not form bigrams across
line boundaries. This is because:
  1. Line breaks may be semantically meaningful in the manuscript
  2. Cross-boundary bigrams could introduce spurious co-occurrences

This is a design choice. Its effect on results should be tested by
also computing cross-boundary bigrams and comparing.

COMPUTATIONAL NOTE
------------------
For a corpus of ~35,000 tokens and ~8,000 unique types, the full
transition matrix would be 8000×8000 ≈ 64M cells. This module uses
sparse representation (Counter dicts) rather than a dense numpy matrix
to keep memory manageable.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# N-gram extraction
# ---------------------------------------------------------------------------

def extract_ngrams(
    line_tokens: Sequence[Sequence[str]],
    n: int = 2,
) -> list[tuple[str, ...]]:
    """
    Extract n-grams from a sequence of token lines.

    Bigrams/trigrams are formed WITHIN lines only (not across line
    boundaries). See module docstring for rationale.

    Parameters
    ----------
    line_tokens
        A sequence of lines, where each line is a sequence of token strings.
        e.g. [["qokedy", "chedy", "daiin"], ["shol", "shory"]]
    n
        N-gram order. 2 = bigrams, 3 = trigrams.

    Returns
    -------
    list[tuple[str, ...]]
        All n-grams extracted across lines, as tuples of n strings.
    """
    ngrams = []
    for line in line_tokens:
        tokens = list(line)
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i + n]))
    return ngrams


# ---------------------------------------------------------------------------
# Mutual Information
# ---------------------------------------------------------------------------

def pairwise_mutual_information(
    bigrams: Sequence[tuple[str, str]],
    unigram_counts: Counter,
    total_unigrams: int,
) -> dict[tuple[str, str], float]:
    """
    Compute Pointwise Mutual Information (PMI) for all observed bigrams.

    PMI(w1, w2) = log2[ P(w1, w2) / (P(w1) * P(w2)) ]

    A high PMI pair appears together more often than expected by chance.
    A low (negative) PMI pair appears less often together than expected.

    PMI is sensitive to low-frequency events — pairs where either token
    is very rare will have noisy PMI estimates. The Positive PMI (PPMI)
    variant clamps negative values to zero and is used for downstream
    analyses.

    Parameters
    ----------
    bigrams
        All observed bigrams (from extract_ngrams with n=2).
    unigram_counts
        Counter of individual token frequencies.
    total_unigrams
        Total number of tokens (for probability normalisation).

    Returns
    -------
    dict mapping (w1, w2) → PMI value in bits.
    """
    bigram_counts = Counter(bigrams)
    total_bigrams = len(bigrams)

    pmi: dict[tuple[str, str], float] = {}

    for (w1, w2), bg_count in bigram_counts.items():
        p_w1 = unigram_counts[w1] / total_unigrams
        p_w2 = unigram_counts[w2] / total_unigrams
        p_w1w2 = bg_count / total_bigrams

        if p_w1 > 0 and p_w2 > 0 and p_w1w2 > 0:
            pmi[(w1, w2)] = math.log2(p_w1w2 / (p_w1 * p_w2))

    return pmi


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class AdjacencyReport:
    """
    Bigram/trigram adjacency analysis results.

    Attributes
    ----------
    total_bigrams : int
    total_trigrams : int
    unique_bigram_types : int
    unique_trigram_types : int
    top_bigrams : list[dict]
        Most frequent bigrams with counts.
    top_trigrams : list[dict]
        Most frequent trigrams with counts.
    top_pmi_pairs : list[dict]
        Highest PMI bigrams (filtered to min_count ≥ 5 to reduce noise).
        These represent pairs that co-occur more than chance predicts.
    low_pmi_pairs : list[dict]
        Lowest PMI bigrams (pairs that repel each other — co-occur less
        than expected by their individual frequencies).
    bigram_entropy : float
        Shannon entropy of the bigram distribution. Lower than unigram
        entropy indicates structured sequential dependencies.
    interpretation_note : str
    """
    total_bigrams: int
    total_trigrams: int
    unique_bigram_types: int
    unique_trigram_types: int
    top_bigrams: list[dict]
    top_trigrams: list[dict]
    top_pmi_pairs: list[dict]
    low_pmi_pairs: list[dict]
    bigram_entropy: float
    interpretation_note: str = (
        "Adjacency statistics measure token co-occurrence patterns. "
        "High-PMI pairs co-occur more than chance but this is consistent "
        "with grammatical structure, scribal formulas, or notational "
        "conventions. No semantic interpretation is implied."
    )


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def compute_adjacency(
    records: list,         # list[TokenRecord]
    top_n: int = 30,
    min_pmi_count: int = 5,
) -> AdjacencyReport:
    """
    Compute bigram/trigram adjacency statistics from corpus records.

    Parameters
    ----------
    records
        List of TokenRecord objects from amf.corpus.ingest.
    top_n
        Number of top entries to include in each ranked list.
    min_pmi_count
        Minimum bigram frequency for PMI computation. Pairs below this
        threshold are excluded from PMI ranking (noisy estimates).

    Returns
    -------
    AdjacencyReport
    """
    # Build list of lines (each line is a list of tokens)
    lines: list[list[str]] = [r.tokens for r in records if r.tokens]

    # Unigram baseline
    all_tokens = [tok for line in lines for tok in line]
    unigram_counts = Counter(all_tokens)
    total_unigrams = len(all_tokens)

    # Bigrams and trigrams (within-line only)
    logger.info("Extracting bigrams from %d lines...", len(lines))
    bigrams = extract_ngrams(lines, n=2)
    trigrams = extract_ngrams(lines, n=3)

    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(trigrams)

    logger.info(
        "Found %d bigrams (%d unique), %d trigrams (%d unique)",
        len(bigrams), len(bigram_counts),
        len(trigrams), len(trigram_counts),
    )

    # Bigram entropy
    bg_total = len(bigrams)
    bg_entropy = 0.0
    for count in bigram_counts.values():
        p = count / bg_total
        if p > 0:
            bg_entropy -= p * math.log2(p)

    # PMI computation (filtered by min_pmi_count)
    filtered_bigrams = [bg for bg, cnt in bigram_counts.items() if cnt >= min_pmi_count]
    pmi_values = pairwise_mutual_information(
        [bg for bg in bigrams if bigram_counts[bg] >= min_pmi_count],
        unigram_counts,
        total_unigrams,
    )

    # Sort by PMI
    pmi_sorted = sorted(pmi_values.items(), key=lambda x: x[1], reverse=True)
    top_pmi = [
        {
            "bigram": list(pair),
            "pmi": round(pmi, 4),
            "count": bigram_counts[pair],
        }
        for pair, pmi in pmi_sorted[:top_n]
        if pmi > 0  # PPMI — exclude negative PMI in top list
    ]
    low_pmi = [
        {
            "bigram": list(pair),
            "pmi": round(pmi, 4),
            "count": bigram_counts[pair],
        }
        for pair, pmi in pmi_sorted[-top_n:]
        if pmi < 0
    ]

    # Top frequency lists
    top_bigrams = [
        {"bigram": list(bg), "count": cnt}
        for bg, cnt in bigram_counts.most_common(top_n)
    ]
    top_trigrams = [
        {"trigram": list(tg), "count": cnt}
        for tg, cnt in trigram_counts.most_common(top_n)
    ]

    return AdjacencyReport(
        total_bigrams=len(bigrams),
        total_trigrams=len(trigrams),
        unique_bigram_types=len(bigram_counts),
        unique_trigram_types=len(trigram_counts),
        top_bigrams=top_bigrams,
        top_trigrams=top_trigrams,
        top_pmi_pairs=top_pmi,
        low_pmi_pairs=low_pmi,
        bigram_entropy=round(bg_entropy, 6),
    )


def save_adjacency_report(report: AdjacencyReport, output_path: str | Path) -> None:
    """Save adjacency report to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    logger.info("Adjacency report saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    from amf.corpus.ingest import EVACorpus

    parser = argparse.ArgumentParser(
        description="AMF v4.0 — Bigram/trigram adjacency analysis"
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default="outputs/adjacency_report.json")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--min-pmi-count", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    corpus = EVACorpus.from_json(args.corpus)
    report = compute_adjacency(
        corpus.records,
        top_n=args.top_n,
        min_pmi_count=args.min_pmi_count,
    )

    print(f"\nTotal bigrams     : {report.total_bigrams:,}")
    print(f"Unique bigram types: {report.unique_bigram_types:,}")
    print(f"Bigram entropy    : {report.bigram_entropy:.4f} bits")
    print(f"\nTop 10 bigrams:")
    for e in report.top_bigrams[:10]:
        print(f"  {' + '.join(e['bigram']):30s}  {e['count']}")
    print(f"\nTop 10 PMI pairs (min count={args.min_pmi_count}):")
    for e in report.top_pmi_pairs[:10]:
        print(f"  {' + '.join(e['bigram']):30s}  PMI={e['pmi']:.3f}  n={e['count']}")

    save_adjacency_report(report, args.output)
    print(f"\nFull report: {args.output}")


if __name__ == "__main__":
    main()
