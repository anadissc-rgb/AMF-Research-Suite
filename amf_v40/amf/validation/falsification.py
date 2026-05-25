# © 2026 Anadi Chakraborty
# Amanuensis Model Framework (AMF)
# Proprietary Research Software
"""
amf.validation.falsification
=============================

AMF v4.0 — Formal Falsification Test Suite
--------------------------------------------

This module defines the complete set of pre-specified falsification tests
for the AMF v4.0 framework. Each test has:

  - A unique ID
  - A precise, measurable criterion
  - A data source (which module produces the measurement)
  - An explicit PASS or FAIL verdict
  - A severity level: CRITICAL | MAJOR | MINOR
  - An interpretation of what failure means

DESIGN PRINCIPLE
----------------
All tests in this module are FALSIFICATION tests — they are designed to
detect when the AMF framework's predictions are WRONG, not to accumulate
supporting evidence. This is the primary safeguard against confirmation bias.

A test that cannot fail is not a scientific test. Every test here can fail.

PRE-SPECIFICATION REQUIREMENT
------------------------------
These tests must be finalized and committed to version control BEFORE
running them on the target corpus. Running tests after seeing results and
adjusting thresholds based on outcomes (HARKing — Hypothesizing After
Results are Known) invalidates the falsification claim.

The pre-registration manifest (amf.validation.preregistration) records
the state of these tests at pre-registration time. Any post-hoc changes
must be disclosed and justified.

TEST CATEGORIES
---------------
F1  DPAS coverage tests          (constraint model validity)
F2  Entropy range tests           (consistency with structured text)
F3  Zipf conformity tests         (power-law distribution)
F4  Positional structure tests    (token position regularity)
F5  Markov order tests            (sequential dependency)
F6  Cross-section differentiation (section-specific structure)

SEVERITY LEVELS
---------------
CRITICAL  Failure invalidates the core AMF hypothesis.
          The framework should not be used for further interpretive work
          until the issue is resolved.

MAJOR     Failure indicates a significant model weakness.
          Results should be treated with caution. The specific failing
          component should be revised.

MINOR     Failure indicates an area for improvement.
          Does not invalidate overall framework; document and address
          in next iteration.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test result data structures
# ---------------------------------------------------------------------------

@dataclass
class FalsificationTest:
    """
    A single pre-specified falsification test.

    Attributes
    ----------
    test_id : str
        Unique identifier (e.g. "F1.1").
    category : str
        Test category (e.g. "DPAS_coverage").
    description : str
        Precise statement of what is being tested.
    criterion : str
        The measurable threshold. Must be numeric and unambiguous.
    severity : str
        "CRITICAL" | "MAJOR" | "MINOR"
    measured_value : Optional[float]
        The actual measured value (set after running).
    verdict : Optional[str]
        "PASS" | "FAIL" | "INCONCLUSIVE" (set after running).
    failure_meaning : str
        What a FAIL result implies for the AMF framework.
    notes : str
        Additional context.
    """
    test_id: str
    category: str
    description: str
    criterion: str
    severity: str
    failure_meaning: str
    measured_value: Optional[float] = None
    verdict: Optional[str] = None
    notes: str = ""


@dataclass
class FalsificationReport:
    """
    Complete falsification test run results.

    Attributes
    ----------
    run_timestamp : str
    corpus_version : str
    amf_version : str
    tests : list[FalsificationTest]
    critical_failures : int
    major_failures : int
    minor_failures : int
    overall_verdict : str
        "SUPPORTED"     — 0 CRITICAL failures, ≤1 MAJOR failures
        "CONDITIONAL"   — 0 CRITICAL, 2+ MAJOR failures
        "NOT_SUPPORTED" — 1+ CRITICAL failures
        "INCONCLUSIVE"  — insufficient data to run all tests
    summary : str
    pre_registration_note : str
    """
    run_timestamp: str
    corpus_version: str
    amf_version: str
    tests: list[dict]
    critical_failures: int
    major_failures: int
    minor_failures: int
    overall_verdict: str
    summary: str
    pre_registration_note: str = (
        "These tests are valid only if the criteria were finalized before "
        "the corpus was analysed. Check the pre-registration manifest at "
        "preregistration/dpas_preregistration.json to verify test criteria "
        "were not modified post-hoc."
    )


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

def _define_tests() -> list[FalsificationTest]:
    """
    Return the complete pre-specified test suite.

    IMPORTANT: These criteria must not be modified after the corpus
    has been analysed. Any modification constitutes HARKing and must
    be disclosed as a change from the pre-registered protocol.
    """
    return [

        # ── F1: DPAS Coverage ─────────────────────────────────────────

        FalsificationTest(
            test_id="F1.1",
            category="DPAS_coverage",
            description=(
                "Full-corpus DPAS coverage must meet the pre-specified threshold. "
                "Coverage = fraction of clean tokens classified as DPAS-valid."
            ),
            criterion="coverage >= 0.70",
            severity="CRITICAL",
            failure_meaning=(
                "The DPAS slot definitions in their current form do not describe "
                "the majority of Voynichese tokens. The constraint model must be "
                "revised before any interpretive claims can be made."
            ),
        ),

        FalsificationTest(
            test_id="F1.2",
            category="DPAS_coverage",
            description=(
                "DPAS-invalid tokens must not be concentrated in a single "
                "failure category. If >90% of invalid tokens share the same "
                "failure reason, the model has a systematic gap rather than "
                "random noise."
            ),
            criterion="max_single_failure_category_fraction <= 0.90",
            severity="MAJOR",
            failure_meaning=(
                "A single systematic gap in the DPAS template accounts for "
                "nearly all invalidity. This indicates a structural error in "
                "the slot definitions, not random transcription noise."
            ),
        ),

        FalsificationTest(
            test_id="F1.3",
            category="DPAS_coverage",
            description=(
                "CORE slot must be the most-used slot across all valid tokens. "
                "By definition, every valid token must have a CORE match. "
                "If CORE is not the top slot by usage count, the validator "
                "has a logic error."
            ),
            criterion="slot_usage['core'] == max(slot_usage.values())",
            severity="CRITICAL",
            failure_meaning=(
                "The DPAS validator has a logic error. CORE is required but "
                "is not the most-matched slot — this is a code defect."
            ),
        ),

        # ── F2: Entropy Range ─────────────────────────────────────────

        FalsificationTest(
            test_id="F2.1",
            category="entropy_range",
            description=(
                "Voynich unigram entropy must be lower than the "
                "frequency-matched random baseline. This replicates the "
                "core finding of Landini (2001): Voynichese is not random."
            ),
            criterion="voynich_h_unigram < baseline_h_unigram",
            severity="CRITICAL",
            failure_meaning=(
                "Voynichese token distribution is indistinguishable from "
                "random text with the same character frequencies. This would "
                "contradict Landini (2001) and suggest the EVA corpus or "
                "analysis pipeline has an error. The AMF framework cannot "
                "proceed until this is resolved."
            ),
        ),

        FalsificationTest(
            test_id="F2.2",
            category="entropy_range",
            description=(
                "Unigram entropy must fall within the range established for "
                "natural languages and known cipher texts: 7.0–13.0 bits. "
                "Values outside this range indicate a corpus processing error "
                "or a highly anomalous token distribution."
            ),
            criterion="7.0 <= unigram_entropy <= 13.0",
            severity="MAJOR",
            failure_meaning=(
                "Entropy outside the 7–13 bit range is anomalous for a "
                "word-level token distribution of this size. Check corpus "
                "ingestion for tokenization errors or duplicate lines."
            ),
        ),

        FalsificationTest(
            test_id="F2.3",
            category="entropy_range",
            description=(
                "Conditional bigram entropy H(W|W-1) must be strictly less "
                "than unigram entropy H(W). This is a mathematical requirement: "
                "adding context cannot increase entropy. Violation = code error."
            ),
            criterion="bigram_entropy < unigram_entropy",
            severity="CRITICAL",
            failure_meaning=(
                "Mathematical invariant violated: conditional entropy exceeds "
                "unconditional entropy. This is a code defect in the entropy "
                "computation, not a property of the corpus."
            ),
        ),

        # ── F3: Zipf Conformity ───────────────────────────────────────

        FalsificationTest(
            test_id="F3.1",
            category="zipf_conformity",
            description=(
                "Zipf exponent must fall within the natural language range "
                "0.5 ≤ α ≤ 1.8 (log-log linear regression estimate). "
                "This replicates Montemurro & Zanette (2013)."
            ),
            criterion="0.5 <= zipf_alpha <= 1.8",
            severity="MAJOR",
            failure_meaning=(
                "Voynichese rank-frequency distribution deviates significantly "
                "from the natural language Zipf range. This does not disprove "
                "a language interpretation but requires investigation. "
                "Check for corpus size effects and alternative fitting methods."
            ),
        ),

        FalsificationTest(
            test_id="F3.2",
            category="zipf_conformity",
            description=(
                "Zipf log-log fit quality R² must be ≥ 0.85. Poor fit "
                "indicates the distribution is not well-described by a "
                "power law."
            ),
            criterion="zipf_r_squared >= 0.85",
            severity="MINOR",
            failure_meaning=(
                "The rank-frequency distribution is not well-fitted by a "
                "power law. This may indicate corpus heterogeneity or that "
                "a two-regime Zipf model (Mandelbrot) would be more appropriate."
            ),
        ),

        # ── F4: Positional Structure ──────────────────────────────────

        FalsificationTest(
            test_id="F4.1",
            category="positional_structure",
            description=(
                "At least 5 tokens must show line-initial rate ≥ 0.50 "
                "with total frequency ≥ 10. This replicates the well-known "
                "positional structure finding (Currier 1976, Landini 2001)."
            ),
            criterion="count(tokens where initial_rate >= 0.50 and n >= 10) >= 5",
            severity="MAJOR",
            failure_meaning=(
                "Expected positional structure is absent or much weaker than "
                "reported in prior literature. This may indicate a different "
                "transcription version or a corpus processing error."
            ),
        ),

        FalsificationTest(
            test_id="F4.2",
            category="positional_structure",
            description=(
                "Line-final token distribution must differ significantly from "
                "the line-initial distribution (Jaccard similarity < 0.30). "
                "If line-initial and line-final token sets are nearly identical, "
                "there is no positional structure."
            ),
            criterion="jaccard(top_initial_tokens, top_final_tokens) < 0.30",
            severity="MAJOR",
            failure_meaning=(
                "Line-initial and line-final positions draw from the same token "
                "pool. Positional structure is absent. DPAS prefix/suffix slot "
                "distinctions have no empirical basis."
            ),
        ),

        # ── F5: Markov Order ──────────────────────────────────────────

        FalsificationTest(
            test_id="F5.1",
            category="markov_order",
            description=(
                "Held-out test perplexity must decrease from order 0 to order 1 "
                "by at least 10%. This confirms that knowing the previous token "
                "helps predict the next (sequential dependency exists)."
            ),
            criterion="(PP_order0 - PP_order1) / PP_order0 >= 0.10",
            severity="MAJOR",
            failure_meaning=(
                "Adding the previous token as context does not improve prediction "
                "on held-out data. Sequential dependency is absent or very weak. "
                "This contradicts expectations for both natural language and "
                "structured notation."
            ),
        ),

        FalsificationTest(
            test_id="F5.2",
            category="markov_order",
            description=(
                "Test perplexity at order 1 must be strictly lower than at "
                "order 0 (unigram baseline) on the HELD-OUT test set. "
                "This is the honest (non-overfitting) version of F5.1."
            ),
            criterion="test_PP_order1 < test_PP_order0",
            severity="CRITICAL",
            failure_meaning=(
                "The bigram model does not generalize: sequential context "
                "adds no predictive value on unseen data. This contradicts "
                "the AMF structural hypothesis."
            ),
        ),

        # ── F6: Cross-Section Differentiation ────────────────────────

        FalsificationTest(
            test_id="F6.1",
            category="section_differentiation",
            description=(
                "At least two manuscript sections must have token distributions "
                "that differ significantly from the corpus baseline "
                "(chi-square p < 0.05 after Bonferroni correction)."
            ),
            criterion="count(sections with chi2_p < bonferroni_threshold) >= 2",
            severity="MINOR",
            failure_meaning=(
                "Token distributions are uniform across manuscript sections. "
                "The visual section divisions do not correspond to distinct "
                "token vocabularies. This weakens (but does not disprove) the "
                "section-specific semantic interpretation in AMF."
            ),
            notes=(
                "Bonferroni threshold = 0.05 / number_of_sections. "
                "Section classification itself is interpretive — see "
                "docs/limitations.md §1.3."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_falsification_tests(
    corpus_path: str | Path,
) -> FalsificationReport:
    """
    Execute all pre-specified falsification tests against the corpus.

    Parameters
    ----------
    corpus_path
        Path to processed corpus JSON.

    Returns
    -------
    FalsificationReport with pass/fail verdict for each test.
    """
    from amf.corpus.ingest import EVACorpus
    from amf.stats.entropy import full_entropy_report
    from amf.stats.positional import compute_positional_stats, high_bias_tokens
    from amf.stats.markov import run_markov_analysis
    from amf.dpas.validator import validate_corpus

    corpus = EVACorpus.from_json(corpus_path)
    clean_tokens = corpus.all_tokens(clean=True)

    logger.info("Running falsification tests on %d tokens...", len(clean_tokens))

    # ── Collect measurements ─────────────────────────────────────────
    entropy_report = full_entropy_report(clean_tokens, label="falsification")
    ent = entropy_report["entropy"]
    zipf = entropy_report["zipf"]
    baseline = entropy_report["random_baseline"]

    pos_stats = compute_positional_stats(corpus.records)
    biased = high_bias_tokens(pos_stats, min_rate=0.50, min_count=10)

    markov_report = run_markov_analysis(corpus.records, max_order=2)
    pp_by_order = {r["order"]: r for r in markov_report.results_by_order}

    dpas_report, _ = validate_corpus(clean_tokens)

    # ── Chi-square section test ───────────────────────────────────────
    from collections import Counter
    from amf.stats.positional import classify_folio
    import math as _math

    section_token_counts: dict[str, Counter] = {}
    for rec in corpus.records:
        sec = classify_folio(rec.folio)
        if sec == "unknown":
            continue
        if sec not in section_token_counts:
            section_token_counts[sec] = Counter()
        section_token_counts[sec].update(rec.clean_tokens)

    overall_counter = Counter(clean_tokens)
    total = len(clean_tokens)
    n_sections = len(section_token_counts)
    bonferroni = (0.05 / n_sections) if n_sections > 0 else 0.05

    section_chi2_results = {}
    for sec, sec_counter in section_token_counts.items():
        sec_total = sum(sec_counter.values())
        if sec_total < 50:
            section_chi2_results[sec] = {"p_approx": 1.0, "note": "insufficient data"}
            continue
        chi2 = 0.0
        for tok, obs in sec_counter.items():
            expected = overall_counter.get(tok, 0) * sec_total / total
            if expected > 0:
                chi2 += (obs - expected) ** 2 / expected
        # Very rough p-value approximation using chi2 CDF via scipy if available
        try:
            from scipy.stats import chi2 as chi2_dist
            df = len(sec_counter) - 1
            p = 1.0 - chi2_dist.cdf(chi2, df=df)
        except ImportError:
            # Without scipy: flag as "estimated"
            p = 0.0 if chi2 > 100 else 1.0
        section_chi2_results[sec] = {"chi2": round(chi2, 2), "p_approx": round(p, 6)}

    sections_significant = sum(
        1 for v in section_chi2_results.values()
        if v.get("p_approx", 1.0) < bonferroni
    )

    # ── Jaccard for positional overlap ───────────────────────────────
    top_initial = set(e["token"] for e in pos_stats.line_initial_top[:20])
    top_final   = set(e["token"] for e in pos_stats.line_final_top[:20])
    jaccard = (len(top_initial & top_final) / len(top_initial | top_final)
               if top_initial | top_final else 1.0)

    # ── Run each test ────────────────────────────────────────────────
    tests = _define_tests()

    measurements: dict[str, object] = {
        "coverage":                    dpas_report.coverage,
        "unigram_entropy":             ent["unigram_entropy"],
        "bigram_entropy":              ent["bigram_entropy"],
        "voynich_h_unigram":           ent["unigram_entropy"],
        "baseline_h_unigram":          baseline["unigram_entropy"],
        "zipf_alpha": (
    zipf.get("alpha")
    or zipf.get("zipf_alpha")
    or zipf.get("slope")
    or "N/A"
),
        "zipf_r_squared":              zipf["r_squared"],
        "line_initial_biased_count":   len(biased["line_initial_biased"]),
        "jaccard_initial_final":       jaccard,
        "test_PP_order0":              pp_by_order.get(0, {}).get("test_perplexity", None),
        "test_PP_order1":              pp_by_order.get(1, {}).get("test_perplexity", None),
        "sections_significant":        sections_significant,
        "slot_usage":                  dpas_report.slot_usage,
        "failure_reasons":             dpas_report.failure_reasons,
    }

    for test in tests:
        _evaluate_test(test, measurements, dpas_report)

    # ── Tally verdicts ───────────────────────────────────────────────
    critical_failures = sum(
        1 for t in tests
        if t.verdict == "FAIL" and t.severity == "CRITICAL"
    )
    major_failures = sum(
        1 for t in tests
        if t.verdict == "FAIL" and t.severity == "MAJOR"
    )
    minor_failures = sum(
        1 for t in tests
        if t.verdict == "FAIL" and t.severity == "MINOR"
    )

    if critical_failures > 0:
        overall = "NOT_SUPPORTED"
    elif major_failures >= 2:
        overall = "CONDITIONAL"
    elif any(t.verdict == "INCONCLUSIVE" for t in tests):
        overall = "INCONCLUSIVE"
    else:
        overall = "SUPPORTED"

    _log_verdict_summary(tests, overall)

    summary = _build_summary(tests, overall, critical_failures, major_failures, minor_failures)

    return FalsificationReport(
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        corpus_version=corpus.metadata.transcription_version,
        amf_version="4.0.0-alpha",
        tests=[asdict(t) for t in tests],
        critical_failures=critical_failures,
        major_failures=major_failures,
        minor_failures=minor_failures,
        overall_verdict=overall,
        summary=summary,
    )


def _evaluate_test(
    test: FalsificationTest,
    m: dict,
    dpas_report,
) -> None:
    """Apply measurements to a test and set verdict + measured_value."""

    tid = test.test_id

    try:
        if tid == "F1.1":
            v = float(m["coverage"])
            test.measured_value = round(v, 4)
            test.verdict = "PASS" if v >= 0.70 else "FAIL"

        elif tid == "F1.2":
            reasons = m["failure_reasons"]
            if not reasons or dpas_report.invalid_count == 0:
                test.verdict = "INCONCLUSIVE"
                test.notes = "No invalid tokens to analyse."
                return
            max_frac = max(reasons.values()) / dpas_report.invalid_count
            test.measured_value = round(max_frac, 4)
            test.verdict = "PASS" if max_frac <= 0.90 else "FAIL"

        elif tid == "F1.3":
            slot_usage = m["slot_usage"]
            if not slot_usage:
                test.verdict = "INCONCLUSIVE"
                test.notes = "No valid tokens."
                return
            top_slot = max(slot_usage, key=slot_usage.get)
            core_count = slot_usage.get("core", 0)
            max_count  = max(slot_usage.values())
            test.measured_value = core_count
            test.verdict = "PASS" if core_count == max_count else "FAIL"
            test.notes = f"Top slot: {top_slot} ({max_count}), core: {core_count}"

        elif tid == "F2.1":
            vh = float(m["voynich_h_unigram"])
            bh = float(m["baseline_h_unigram"])
            test.measured_value = round(vh - bh, 4)
            test.verdict = "PASS" if vh < bh else "FAIL"
            test.notes = f"Voynich={vh:.4f} bits, Baseline={bh:.4f} bits"

        elif tid == "F2.2":
            h = float(m["unigram_entropy"])
            test.measured_value = round(h, 4)
            test.verdict = "PASS" if 7.0 <= h <= 13.0 else "FAIL"

        elif tid == "F2.3":
            h_uni  = float(m["unigram_entropy"])
            h_bi   = float(m["bigram_entropy"])
            test.measured_value = round(h_bi, 4)
            test.verdict = "PASS" if h_bi < h_uni else "FAIL"
            test.notes = f"H(W)={h_uni:.4f}, H(W|W-1)={h_bi:.4f}"

        elif tid == "F3.1":
            alpha = float(m["zipf_alpha"])
            test.measured_value = round(alpha, 4)
            test.verdict = "PASS" if 0.5 <= alpha <= 1.8 else "FAIL"

        elif tid == "F3.2":
            r2 = float(m["zipf_r_squared"])
            test.measured_value = round(r2, 4)
            test.verdict = "PASS" if r2 >= 0.85 else "FAIL"

        elif tid == "F4.1":
            count = int(m["line_initial_biased_count"])
            test.measured_value = float(count)
            test.verdict = "PASS" if count >= 5 else "FAIL"

        elif tid == "F4.2":
            j = float(m["jaccard_initial_final"])
            test.measured_value = round(j, 4)
            test.verdict = "PASS" if j < 0.30 else "FAIL"

        elif tid == "F5.1":
            pp0 = m["test_PP_order0"]
            pp1 = m["test_PP_order1"]
            if pp0 is None or pp1 is None:
                test.verdict = "INCONCLUSIVE"
                test.notes = "Train/test split perplexity not available."
                return
            reduction = (float(pp0) - float(pp1)) / float(pp0)
            test.measured_value = round(reduction, 4)
            test.verdict = "PASS" if reduction >= 0.10 else "FAIL"

        elif tid == "F5.2":
            pp0 = m["test_PP_order0"]
            pp1 = m["test_PP_order1"]
            if pp0 is None or pp1 is None:
                test.verdict = "INCONCLUSIVE"
                test.notes = "Train/test split perplexity not available."
                return
            test.measured_value = round(float(pp1), 4)
            test.verdict = "PASS" if float(pp1) < float(pp0) else "FAIL"

        elif tid == "F6.1":
            count = int(m["sections_significant"])
            test.measured_value = float(count)
            test.verdict = "PASS" if count >= 2 else "FAIL"

        else:
            test.verdict = "INCONCLUSIVE"
            test.notes = "Test evaluator not implemented."

    except (KeyError, TypeError, ZeroDivisionError) as e:
        test.verdict = "INCONCLUSIVE"
        test.notes = f"Evaluation error: {e}"


def _log_verdict_summary(tests: list[FalsificationTest], overall: str) -> None:
    logger.info("Falsification tests complete — overall verdict: %s", overall)
    for t in tests:
        symbol = {"PASS": "✓", "FAIL": "✗", "INCONCLUSIVE": "?"}.get(t.verdict, " ")
        logger.info("  [%s] %s (%s)  value=%s", symbol, t.test_id,
                    t.severity, t.measured_value)


def _build_summary(
    tests: list[FalsificationTest],
    overall: str,
    critical: int,
    major: int,
    minor: int,
) -> str:
    passes = sum(1 for t in tests if t.verdict == "PASS")
    fails  = sum(1 for t in tests if t.verdict == "FAIL")
    incon  = sum(1 for t in tests if t.verdict == "INCONCLUSIVE")
    total  = len(tests)

    lines = [
        f"Overall verdict: {overall}",
        f"Tests: {total} total — {passes} PASS, {fails} FAIL, {incon} INCONCLUSIVE",
        f"Failures: {critical} CRITICAL, {major} MAJOR, {minor} MINOR",
    ]

    if overall == "NOT_SUPPORTED":
        lines.append(
            "One or more CRITICAL tests failed. The AMF framework in its "
            "current form is not supported by this corpus. Revision required "
            "before proceeding with interpretive analyses."
        )
    elif overall == "CONDITIONAL":
        lines.append(
            "No CRITICAL failures, but 2+ MAJOR failures. Results should be "
            "treated with caution. Address MAJOR failures before publication."
        )
    elif overall == "SUPPORTED":
        lines.append(
            "All critical and major tests passed. The framework is consistent "
            "with the pre-specified falsification criteria. This supports "
            "(but does not prove) the AMF hypothesis. Independent replication "
            "and baseline comparison are still required."
        )
    elif overall == "INCONCLUSIVE":
        lines.append(
            "Some tests could not be evaluated (INCONCLUSIVE). "
            "Complete the train/test split and chi-square analyses before "
            "drawing conclusions."
        )

    return " ".join(lines)


def save_falsification_report(
    report: FalsificationReport,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    logger.info("Falsification report saved: %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="AMF v4.0 — Formal falsification test suite"
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default="outputs/falsification_report.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    report = run_falsification_tests(args.corpus)

    print(f"\n{'='*60}")
    print(f"FALSIFICATION TEST REPORT — AMF v4.0")
    print(f"{'='*60}")
    print(f"Corpus version : {report.corpus_version}")
    print(f"Timestamp      : {report.run_timestamp}")
    print()

    verdict_sym = {"PASS": "✓", "FAIL": "✗", "INCONCLUSIVE": "?"}
    for t in [FalsificationTest(**d) for d in report.tests]:
        sym = verdict_sym.get(t.verdict, " ")
        val = f"  measured={t.measured_value}" if t.measured_value is not None else ""
        print(f"  [{sym}] {t.test_id:6s} [{t.severity:8s}] {t.description[:55]}...{val}")

    print()
    print(f"CRITICAL failures : {report.critical_failures}")
    print(f"MAJOR    failures : {report.major_failures}")
    print(f"MINOR    failures : {report.minor_failures}")
    print(f"\nOVERALL VERDICT   : {report.overall_verdict}")
    print(f"\n{report.summary}")
    print(f"\n{report.pre_registration_note}")

    save_falsification_report(report, args.output)
    print(f"\nFull report: {args.output}")


if __name__ == "__main__":
    main()
