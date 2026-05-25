"""
amf.stats.positional
====================

Positional Frequency Analysis of Voynichese Tokens
----------------------------------------------------

A well-established empirical finding in Voynich studies is that token
distributions are strongly **position-dependent**:

  - Certain tokens appear almost exclusively at the start of lines
  - Others appear almost exclusively at the end of lines
  - Some token families are associated with specific manuscript sections
    (herbal, pharmaceutical, astronomical, biological)

This positional structure is one of the most reproducible statistical
properties of the EVA corpus and is referenced in:

  Currier (1976) — original observation of two "languages" (Currier A/B)
  Landini (2001) — confirmation of positional glyph structure
  Bowern & Lindemann (2021) — phylogenetic analysis of positional tokens

WHAT THIS MODULE MEASURES
--------------------------
1. Line-initial and line-final token frequencies (and their ratios
   to overall frequency — high ratio = positionally biased)
2. Section-specific frequency distributions (does a token appear
   disproportionately in herbal vs. pharmaceutical folios?)
3. Word-position frequency (position 1, 2, 3... within a line)
4. Within-word positional glyph statistics (which EVA characters appear
   at character position 1, 2, 3... within a token)

INTERPRETATION CAUTION
-----------------------
Positional structure is consistent with:
  - Grammatical rules (word order in natural language)
  - Scribal formatting conventions (independent of content)
  - Cipher construction rules
  - The AMF DPAS hypothesis (position-governed notation)
Positional analysis alone cannot distinguish between these hypotheses.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section classification
# ---------------------------------------------------------------------------

# Approximate folio-to-section mapping based on codicological convention.
# Source: D'Imperio (1978) and Zandbergen section classifications.
# NOTE: These classifications are themselves interpretive — the manuscript
# has no explicit section headings. Sections are inferred from visual content.
SECTION_MAP: dict[str, str] = {
    # Herbal (f1–f66): folios beginning with f1–f66 (approximate)
    **{f"f{n}r": "herbal" for n in range(1, 67)},
    **{f"f{n}v": "herbal" for n in range(1, 67)},
    # Astronomical (f67–f73)
    **{f"f{n}r": "astronomical" for n in range(67, 74)},
    **{f"f{n}v": "astronomical" for n in range(67, 74)},
    # Cosmological (f68–f73)
    **{f"f{n}r": "cosmological" for n in range(68, 74)},
    **{f"f{n}v": "cosmological" for n in range(68, 74)},
    # Biological (f75–f84)
    **{f"f{n}r": "biological" for n in range(75, 85)},
    **{f"f{n}v": "biological" for n in range(75, 85)},
    # Pharmaceutical (f88–f102)
    **{f"f{n}r": "pharmaceutical" for n in range(88, 103)},
    **{f"f{n}v": "pharmaceutical" for n in range(88, 103)},
    # Recipes/Stars (f103–f116)
    **{f"f{n}r": "recipes" for n in range(103, 117)},
    **{f"f{n}v": "recipes" for n in range(103, 117)},
}

KNOWN_SECTIONS = frozenset(
    ["herbal", "astronomical", "cosmological", "biological", "pharmaceutical", "recipes"]
)


def classify_folio(folio: str) -> str:
    """
    Map a folio identifier to a manuscript section.

    Returns "unknown" if the folio cannot be classified.
    This mapping is approximate and based on codicological convention —
    see module docstring for source references.
    """
    return SECTION_MAP.get(folio.lower(), "unknown")


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class PositionalStats:
    """
    Positional frequency statistics for the corpus.

    Attributes
    ----------
    line_initial_top : list[tuple[str, int, float]]
        Top tokens appearing at the start of lines, with count and
        line-initial rate (fraction of their total occurrences that
        are line-initial).
    line_final_top : list[tuple[str, int, float]]
        Top tokens at end of lines.
    section_profiles : dict[str, dict[str, int]]
        For each manuscript section, the top-20 token frequencies.
    word_position_profiles : dict[int, list[tuple[str, int]]]
        For word positions 1–5, the top tokens at each position.
    char_position_profiles : dict[int, dict[str, int]]
        For character positions 1–8, the character frequency distribution.
    total_lines : int
    total_tokens : int
    """
    line_initial_top: list[dict]
    line_final_top: list[dict]
    section_profiles: dict[str, dict[str, int]]
    word_position_profiles: dict[str, list[dict]]
    char_position_profiles: dict[str, dict[str, int]]
    total_lines: int
    total_tokens: int
    interpretation_note: str = (
        "Positional token statistics are reproducible empirical measurements. "
        "Strong positional bias (high line-initial or line-final rates) is "
        "consistent with grammatical, scribal, or notational structure. "
        "It does not identify which type of structure is present."
    )


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_positional_stats(
    records: list,   # list[TokenRecord] — avoiding circular import
    top_n: int = 30,
) -> PositionalStats:
    """
    Compute all positional statistics from a corpus record list.

    Parameters
    ----------
    records
        List of TokenRecord objects (from amf.corpus.ingest).
    top_n
        Number of top tokens to return for each position category.

    Returns
    -------
    PositionalStats
    """
    total_lines = 0
    total_tokens = 0

    # Counters
    overall_count: Counter = Counter()
    line_initial_count: Counter = Counter()
    line_final_count: Counter = Counter()
    section_counts: dict[str, Counter] = defaultdict(Counter)
    word_pos_counts: dict[int, Counter] = defaultdict(Counter)   # position → counter
    char_pos_counts: dict[int, Counter] = defaultdict(Counter)   # char position → counter

    for record in records:
        tokens = record.tokens
        if not tokens:
            continue

        total_lines += 1
        total_tokens += len(tokens)
        section = classify_folio(record.folio)

        # Overall token counts
        overall_count.update(tokens)

        # Line-initial and line-final
        line_initial_count[tokens[0]] += 1
        line_final_count[tokens[-1]] += 1

        # Section-specific counts
        section_counts[section].update(tokens)

        # Word-position counts (1-indexed)
        for pos, tok in enumerate(tokens[:10], start=1):
            word_pos_counts[pos][tok] += 1

        # Character-position counts within tokens
        for tok in tokens:
            for char_pos, char in enumerate(tok[:12], start=1):
                char_pos_counts[char_pos][char] += 1

    # Build line-initial rate (what fraction of token occurrences are line-initial)
    def _initial_rate_entries(counter: Counter, top: int) -> list[dict]:
        results = []
        for tok, init_count in counter.most_common(top):
            total = overall_count.get(tok, 1)
            rate = init_count / total
            results.append({
                "token": tok,
                "line_initial_count": init_count,
                "total_count": total,
                "line_initial_rate": round(rate, 4),
            })
        return results

    def _final_rate_entries(counter: Counter, top: int) -> list[dict]:
        results = []
        for tok, final_count in counter.most_common(top):
            total = overall_count.get(tok, 1)
            rate = final_count / total
            results.append({
                "token": tok,
                "line_final_count": final_count,
                "total_count": total,
                "line_final_rate": round(rate, 4),
            })
        return results

    # Section profiles: top-20 per section
    section_profiles = {
        sec: dict(ctr.most_common(20))
        for sec, ctr in section_counts.items()
    }

    # Word position profiles: top-10 per position
    word_position_profiles = {
        str(pos): [
            {"token": tok, "count": cnt}
            for tok, cnt in ctr.most_common(10)
        ]
        for pos, ctr in word_pos_counts.items()
        if pos <= 5
    }

    # Character position profiles: all chars per position
    char_position_profiles = {
        str(pos): dict(ctr.most_common())
        for pos, ctr in char_pos_counts.items()
        if pos <= 8
    }

    logger.info(
        "Positional analysis complete: %d lines, %d tokens, %d sections",
        total_lines, total_tokens, len(section_counts),
    )

    return PositionalStats(
        line_initial_top=_initial_rate_entries(line_initial_count, top_n),
        line_final_top=_final_rate_entries(line_final_count, top_n),
        section_profiles=section_profiles,
        word_position_profiles=word_position_profiles,
        char_position_profiles=char_position_profiles,
        total_lines=total_lines,
        total_tokens=total_tokens,
    )


def high_bias_tokens(
    stats: PositionalStats,
    min_rate: float = 0.4,
    min_count: int = 10,
) -> dict[str, list[str]]:
    """
    Identify tokens with high positional bias.

    A token is 'line-initial biased' if ≥ min_rate of its occurrences
    appear at the start of lines and it appears at least min_count times.

    Parameters
    ----------
    stats
        PositionalStats from compute_positional_stats.
    min_rate
        Minimum positional rate to be considered biased.
    min_count
        Minimum total occurrences to be included.

    Returns
    -------
    dict with keys 'line_initial', 'line_final', each containing
    a list of token strings.
    """
    initial_biased = [
        e["token"] for e in stats.line_initial_top
        if e["line_initial_rate"] >= min_rate and e["total_count"] >= min_count
    ]
    final_biased = [
        e["token"] for e in stats.line_final_top
        if e["line_final_rate"] >= min_rate and e["total_count"] >= min_count
    ]

    return {
        "line_initial_biased": initial_biased,
        "line_final_biased": final_biased,
        "note": (
            f"Tokens with positional rate ≥ {min_rate} and count ≥ {min_count}. "
            "High positional bias may indicate grammatical, structural, or "
            "notational roles. DPAS constraint validation is needed to "
            "determine whether these positions are rule-governed."
        ),
    }


def save_positional_report(stats: PositionalStats, output_path: str | Path) -> None:
    """Save positional statistics to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2, ensure_ascii=False)
    logger.info("Positional report saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    from amf.corpus.ingest import EVACorpus

    parser = argparse.ArgumentParser(
        description="AMF v4.0 — Positional token frequency analysis"
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default="outputs/positional_report.json")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    corpus = EVACorpus.from_json(args.corpus)
    stats = compute_positional_stats(corpus.records, top_n=args.top_n)

    print(f"\nTotal lines: {stats.total_lines:,}")
    print(f"Total tokens: {stats.total_tokens:,}")
    print(f"Sections found: {list(stats.section_profiles.keys())}")
    print(f"\nTop line-initial tokens:")
    for e in stats.line_initial_top[:10]:
        print(f"  {e['token']:20s}  n={e['line_initial_count']:5d}  "
              f"rate={e['line_initial_rate']:.3f}")

    biased = high_bias_tokens(stats)
    print(f"\nHigh-bias line-initial tokens (rate≥0.4, n≥10): "
          f"{biased['line_initial_biased'][:15]}")

    save_positional_report(stats, args.output)
    print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()
