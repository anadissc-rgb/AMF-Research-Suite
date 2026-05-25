"""
amf.stats.significance
=======================

Statistical Significance Testing and Confidence Intervals
----------------------------------------------------------

This module provides proper inferential statistics for AMF v4.0 results.
All point estimates elsewhere in the framework gain uncertainty bounds here.

METHODS
-------
1. Bootstrap confidence intervals (percentile method)
   Used for: entropy, Zipf α, positional rates
   Resamples: N=1000 (configurable), seed always reported

2. Chi-square goodness-of-fit test
   Used for: section-level token distribution vs. corpus baseline
   Correction: Bonferroni for multiple comparisons

3. Permutation test for positional bias
   Used for: testing whether line-initial rates exceed chance
   Null: token positions within lines are random

WHY BOOTSTRAP
--------------
The EVA corpus is a single sample. We cannot treat its statistics as
population parameters. Bootstrap resampling of lines (not tokens, to
preserve within-line structure) gives confidence intervals that reflect
uncertainty from the finite corpus size.

IMPORTANT LIMITATION: Bootstrap CIs assume the EVA transcription is
a representative sample of the manuscript. If the manuscript has
systematic structure at the folio or section level not captured by
line-level resampling, bootstrap CIs will underestimate uncertainty.

SEED REPORTING
--------------
ALL random operations use a seed that is recorded in output metadata.
Results computed with different seeds should agree to within CI width.
If they don't, the CI width is too narrow (increase bootstrap_n).
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: list,
    statistic: Callable[[list], float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute a bootstrap percentile confidence interval for a statistic.

    RESAMPLING UNIT: The `data` argument should be a list of LINES
    (each line a list of tokens), not a flat token list. This preserves
    within-line structure during resampling.

    Parameters
    ----------
    data
        List of items to resample (e.g. list of token lists).
    statistic
        Function mapping list → float. Applied to each resample.
    n_resamples
        Number of bootstrap resamples. Higher = narrower CIs. Default 1000.
    confidence
        Confidence level (e.g. 0.95 = 95% CI).
    seed
        Random seed. Always recorded in output.

    Returns
    -------
    (point_estimate, lower_bound, upper_bound)

    Notes
    -----
    The percentile method is used (not BCa). BCa is more accurate for
    small samples but requires jacknife — a planned future improvement.
    """
    rng = np.random.default_rng(seed)
    n = len(data)

    point = statistic(data)

    bootstrap_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        resample = [data[j] for j in indices]
        bootstrap_stats[i] = statistic(resample)

    alpha = 1.0 - confidence
    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return round(point, 6), round(lower, 6), round(upper, 6)


# ---------------------------------------------------------------------------
# Statistics for bootstrap application
# ---------------------------------------------------------------------------

def _lines_to_flat(lines: list[list[str]]) -> list[str]:
    return [tok for line in lines for tok in line]


def _entropy_statistic(lines: list[list[str]]) -> float:
    from amf.stats.entropy import unigram_entropy
    return unigram_entropy(_lines_to_flat(lines))


def _zipf_alpha_statistic(lines: list[list[str]]) -> float:
    from amf.stats.entropy import zipf_fit
    tokens = _lines_to_flat(lines)
    result = zipf_fit(tokens)
    if "error" in result:
        return float("nan")
    return result["alpha"]


def _line_initial_rate_statistic(token: str) -> Callable:
    """Factory: returns a statistic for the line-initial rate of a specific token."""
    def stat(lines: list[list[str]]) -> float:
        total = sum(1 for line in lines for t in line if t == token)
        initial = sum(1 for line in lines if line and line[0] == token)
        return initial / total if total > 0 else 0.0
    return stat


# ---------------------------------------------------------------------------
# Chi-square section test
# ---------------------------------------------------------------------------

def chi_square_section_test(
    records: list,       # list[TokenRecord]
    alpha_level: float = 0.05,
) -> dict:
    """
    Chi-square goodness-of-fit test for each manuscript section.

    Tests whether each section's token distribution deviates significantly
    from the overall corpus distribution.

    H0: Section token frequencies match corpus-wide relative frequencies.
    H1: They differ.

    Bonferroni correction is applied for multiple comparisons.

    Parameters
    ----------
    records
        List of TokenRecord objects.
    alpha_level
        Family-wise error rate (default 0.05).

    Returns
    -------
    dict with per-section results and correction details.
    """
    from amf.stats.positional import classify_folio

    # Build corpus baseline
    all_tokens = [tok for rec in records for tok in rec.clean_tokens]
    corpus_counter = Counter(all_tokens)
    corpus_total = len(all_tokens)

    # Build per-section counts
    section_counters: dict[str, Counter] = {}
    for rec in records:
        sec = classify_folio(rec.folio)
        if sec not in section_counters:
            section_counters[sec] = Counter()
        section_counters[sec].update(rec.clean_tokens)

    sections = [s for s in section_counters if s != "unknown"]
    bonferroni_threshold = alpha_level / max(len(sections), 1)

    results = {
        "alpha_level": alpha_level,
        "bonferroni_threshold": round(bonferroni_threshold, 6),
        "n_sections_tested": len(sections),
        "sections": {},
        "sections_significant": 0,
        "interpretation": "",
    }

    # Try importing scipy for p-values
    try:
        from scipy.stats import chi2 as chi2_dist
        has_scipy = True
    except ImportError:
        has_scipy = False
        logger.warning(
            "scipy not available — chi-square p-values will be estimated. "
            "Install scipy for accurate p-values."
        )

    for sec in sections:
        sec_counter = section_counters[sec]
        sec_total = sum(sec_counter.values())

        if sec_total < 30:
            results["sections"][sec] = {
                "verdict": "INSUFFICIENT_DATA",
                "n_tokens": sec_total,
                "note": f"Only {sec_total} tokens — chi-square unreliable (need ≥30)",
            }
            continue

        # Restrict to tokens present in both corpus and section
        shared_tokens = [t for t in sec_counter if t in corpus_counter]
        if len(shared_tokens) < 5:
            results["sections"][sec] = {
                "verdict": "INSUFFICIENT_OVERLAP",
                "note": "Fewer than 5 shared token types",
            }
            continue

        chi2_stat = 0.0
        for tok in shared_tokens:
            observed = sec_counter[tok]
            expected = corpus_counter[tok] * sec_total / corpus_total
            if expected > 0.5:   # ignore cells with very small expected counts
                chi2_stat += (observed - expected) ** 2 / expected

        df = len(shared_tokens) - 1

        if has_scipy:
            p_value = float(1.0 - chi2_dist.cdf(chi2_stat, df=df))
        else:
            # Conservative approximation: flag as significant if chi2 >> df
            p_value = 0.001 if chi2_stat > df * 3 else 0.5

        significant = p_value < bonferroni_threshold

        if significant:
            results["sections_significant"] += 1

        results["sections"][sec] = {
            "n_tokens": sec_total,
            "shared_types": len(shared_tokens),
            "chi2_statistic": round(chi2_stat, 4),
            "degrees_of_freedom": df,
            "p_value": round(p_value, 6),
            "p_value_source": "scipy" if has_scipy else "estimated",
            "bonferroni_threshold": round(bonferroni_threshold, 6),
            "significant": significant,
            "verdict": "SIGNIFICANT" if significant else "NOT_SIGNIFICANT",
        }

    n_sig = results["sections_significant"]
    results["interpretation"] = (
        f"{n_sig}/{len(sections)} sections show token distributions "
        f"significantly different from the corpus baseline "
        f"(Bonferroni-corrected α = {bonferroni_threshold:.4f}). "
        + (
            "This supports section-specific token structure. "
            if n_sig >= 2 else
            "This does not support section-specific token structure at the "
            "current significance threshold."
        )
        + " NOTE: Section classification is itself interpretive — "
          "see docs/limitations.md §1.3."
    )

    return results


# ---------------------------------------------------------------------------
# Permutation test for positional bias
# ---------------------------------------------------------------------------

def permutation_test_positional_bias(
    records: list,
    token: str,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Permutation test: is a token's line-initial rate higher than chance?

    Under the null hypothesis (no positional structure), token positions
    within lines are random. We permute token positions within each line
    and measure how often the permuted line-initial rate equals or exceeds
    the observed rate.

    Parameters
    ----------
    records
        List of TokenRecord objects.
    token
        The EVA token to test.
    n_permutations
        Number of permutations (default 1000).
    seed
        Random seed.

    Returns
    -------
    dict with observed rate, null distribution stats, and p-value.
    """
    rng = np.random.default_rng(seed)

    lines = [rec.clean_tokens for rec in records if rec.clean_tokens]

    # Observed rate
    total = sum(t == token for line in lines for t in line)
    initial = sum(1 for line in lines if line and line[0] == token)
    if total == 0:
        return {"error": f"Token '{token}' not found in corpus"}
    observed_rate = initial / total

    # Null distribution: permute token positions within each line
    null_rates = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_initial = 0
        for line in lines:
            perm_line = list(line)
            rng.shuffle(perm_line)
            if perm_line and perm_line[0] == token:
                perm_initial += 1
        null_rates[i] = perm_initial / total

    p_value = float(np.mean(null_rates >= observed_rate))

    return {
        "token": token,
        "observed_initial_rate": round(observed_rate, 4),
        "total_occurrences": total,
        "observed_initial_count": initial,
        "null_mean": round(float(null_rates.mean()), 4),
        "null_std":  round(float(null_rates.std()),  4),
        "p_value":   round(p_value, 4),
        "significant_at_0.05": p_value < 0.05,
        "n_permutations": n_permutations,
        "seed": seed,
        "interpretation": (
            f"p = {p_value:.4f}. "
            + ("Token shows significantly higher line-initial rate than chance "
               "(reject null of random position)."
               if p_value < 0.05 else
               "Cannot reject null hypothesis of random token positioning "
               "at this significance level.")
        ),
    }


# ---------------------------------------------------------------------------
# Full significance report
# ---------------------------------------------------------------------------

@dataclass
class SignificanceReport:
    """Complete significance analysis output."""
    bootstrap_entropy_ci: dict
    bootstrap_zipf_ci: dict
    chi_square_sections: dict
    positional_permutation_tests: list[dict]
    bootstrap_config: dict
    interpretation_note: str = (
        "All CIs are 95% bootstrap percentile intervals. "
        "Chi-square tests use Bonferroni correction. "
        "P-values are two-tailed where applicable. "
        "Significance does not imply practical importance."
    )


def run_significance_analysis(
    records: list,
    n_bootstrap: int = 1000,
    seed: int = 42,
    tokens_to_test_positional: int = 5,
) -> SignificanceReport:
    """
    Run all significance analyses and return a structured report.

    Parameters
    ----------
    records
        List of TokenRecord objects.
    n_bootstrap
        Bootstrap resamples (default 1000).
    seed
        Master random seed (derived seeds used for each analysis).
    tokens_to_test_positional
        Number of top line-initial tokens to run permutation tests on.
    """
    from amf.stats.positional import compute_positional_stats

    lines = [rec.clean_tokens for rec in records if rec.clean_tokens]
    logger.info("Running significance analysis: %d lines, %d bootstrap resamples",
                len(lines), n_bootstrap)

    # Bootstrap entropy CI
    logger.info("  Bootstrap entropy CI...")
    h_point, h_lo, h_hi = bootstrap_ci(
        lines, _entropy_statistic, n_bootstrap, seed=seed
    )
    bootstrap_entropy = {
        "point_estimate": h_point,
        "ci_lower": h_lo,
        "ci_upper": h_hi,
        "confidence": 0.95,
        "n_resamples": n_bootstrap,
        "seed": seed,
        "unit": "bits",
    }

    # Bootstrap Zipf α CI
    logger.info("  Bootstrap Zipf α CI...")
    z_point, z_lo, z_hi = bootstrap_ci(
        lines, _zipf_alpha_statistic, n_bootstrap, seed=seed + 1
    )
    bootstrap_zipf = {
        "alpha_point_estimate": z_point,
        "ci_lower": z_lo,
        "ci_upper": z_hi,
        "confidence": 0.95,
        "n_resamples": n_bootstrap,
        "seed": seed + 1,
    }

    # Chi-square section tests
    logger.info("  Chi-square section tests...")
    chi2_results = chi_square_section_test(records)

    # Permutation tests for top line-initial tokens
    pos_stats = compute_positional_stats(records)
    top_initial_tokens = [
        e["token"] for e in pos_stats.line_initial_top[:tokens_to_test_positional]
        if e["total_count"] >= 10
    ]
    logger.info("  Permutation tests for %d tokens...", len(top_initial_tokens))
    perm_results = []
    for i, tok in enumerate(top_initial_tokens):
        result = permutation_test_positional_bias(
            records, tok, n_permutations=n_bootstrap, seed=seed + 100 + i
        )
        perm_results.append(result)

    return SignificanceReport(
        bootstrap_entropy_ci=bootstrap_entropy,
        bootstrap_zipf_ci=bootstrap_zipf,
        chi_square_sections=chi2_results,
        positional_permutation_tests=perm_results,
        bootstrap_config={
            "n_resamples": n_bootstrap,
            "master_seed": seed,
            "resample_unit": "lines",
            "ci_method": "percentile",
            "note": (
                "Resampling unit is manuscript lines (not individual tokens) "
                "to preserve within-line token structure."
            ),
        },
    )


def save_significance_report(report: SignificanceReport, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    logger.info("Significance report saved: %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    from amf.corpus.ingest import EVACorpus

    parser = argparse.ArgumentParser(
        description="AMF v4.0 — Bootstrap CIs and significance tests"
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default="outputs/significance_report.json")
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    corpus = EVACorpus.from_json(args.corpus)
    report = run_significance_analysis(
        corpus.records,
        n_bootstrap=args.bootstrap_n,
        seed=args.seed,
    )

    ent = report.bootstrap_entropy_ci
    zipf = report.bootstrap_zipf_ci
    chi2 = report.chi_square_sections

    print(f"\n{'='*55}")
    print("SIGNIFICANCE ANALYSIS REPORT")
    print(f"{'='*55}")
    print(f"\nEntropy (95% bootstrap CI, n={ent['n_resamples']}):")
    print(f"  H(W) = {ent['point_estimate']:.4f} bits  "
          f"[{ent['ci_lower']:.4f}, {ent['ci_upper']:.4f}]")
    print(f"\nZipf α (95% bootstrap CI, n={zipf['n_resamples']}):")
    print(f"  α = {zipf['alpha_point_estimate']:.4f}  "
          f"[{zipf['ci_lower']:.4f}, {zipf['ci_upper']:.4f}]")
    print(f"\nChi-square section tests "
          f"(Bonferroni α = {chi2['bonferroni_threshold']:.4f}):")
    for sec, res in chi2["sections"].items():
        if "verdict" in res:
            sig = "**SIGNIFICANT**" if res.get("significant") else ""
            p = res.get("p_value", "N/A")
            print(f"  {sec:20s}  p={p}  {sig}")
    print(f"\nSections significant: {chi2['sections_significant']}")
    print(f"\nPermutation tests (positional bias):")
    for pt in report.positional_permutation_tests:
        if "error" not in pt:
            sig = "* p<0.05" if pt["significant_at_0.05"] else ""
            print(f"  {pt['token']:20s}  rate={pt['observed_initial_rate']:.3f}  "
                  f"p={pt['p_value']:.4f}  {sig}")

    save_significance_report(report, args.output)
    print(f"\nFull report: {args.output}")


if __name__ == "__main__":
    main()
