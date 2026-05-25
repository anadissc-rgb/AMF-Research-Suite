"""
amf.dpas.validator
==================

DPAS Token Constraint Validator
---------------------------------

This module tests EVA tokens against the DPAS template defined in
amf.dpas.constraints and computes corpus-level validity statistics.

The validator is the primary FALSIFIABILITY mechanism of AMF v4.0.

If DPAS coverage (fraction of tokens classed as valid) is below
COVERAGE_THRESHOLD, the DPAS model in its current form is rejected.

APPROACH
--------
Token validation uses a greedy left-to-right slot matcher. This is
necessarily approximate because:
  1. EVA segmentation into multigraphs is ambiguous in some positions
  2. The slot definitions themselves are provisional
  3. Uncertain transcription characters may cause false negatives

All these limitations are reported in validation output.

The validator intentionally uses a CONSERVATIVE validity definition:
a token is valid only if it can be fully matched slot-by-slot with
no leftover characters. This makes the coverage measure a lower bound
on the true coverage under the hypothesis.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from amf.dpas.constraints import (
    COVERAGE_NOTE,
    COVERAGE_THRESHOLD,
    DPAS_TEMPLATE,
    DPASValidationResult,
    EVA_MULTIGRAPHS,
)

logger = logging.getLogger(__name__)

# Uncertainty markers in EVA transcription
_UNCERTAINTY_RE = re.compile(r"[{}!?]")


# ---------------------------------------------------------------------------
# Tokenization into EVA units
# ---------------------------------------------------------------------------

def _split_eva_units(token: str) -> list[str]:
    """
    Split an EVA token string into its constituent units.

    EVA uses digraphs and trigraphs (e.g. "ch", "sh", "cth") as single
    units. This function greedily segments a token from left to right,
    preferring longer multigraphs.

    This segmentation is AMBIGUOUS in some cases (e.g. "s" + "h" vs "sh").
    The greedy longest-match approach is used for consistency, but note that
    different segmentation choices will affect DPAS validation results.

    Parameters
    ----------
    token : str
        An EVA token string (possibly containing uncertainty markers).

    Returns
    -------
    list[str]
        List of EVA character units.
    """
    # Sorted by length descending for longest-match greedy scan
    multigraphs = sorted(EVA_MULTIGRAPHS, key=len, reverse=True)

    units = []
    i = 0
    while i < len(token):
        matched = False
        for mg in multigraphs:
            if token[i:i + len(mg)] == mg:
                units.append(mg)
                i += len(mg)
                matched = True
                break
        if not matched:
            units.append(token[i])
            i += 1

    return units


# ---------------------------------------------------------------------------
# Single token validation
# ---------------------------------------------------------------------------

def validate_token(token: str) -> DPASValidationResult:
    """
    Validate a single EVA token against the DPAS template.

    Parameters
    ----------
    token
        An EVA token string. May contain uncertainty markers ({}).

    Returns
    -------
    DPASValidationResult
    """
    has_uncertainty = bool(_UNCERTAINTY_RE.search(token))
    clean_token = _UNCERTAINTY_RE.sub("", token)

    if not clean_token:
        return DPASValidationResult(
            token=token,
            is_valid=False,
            matched_slots=[],
            failure_reason="Token is empty after removing uncertainty markers",
            confidence="low",
            notes="Token contains only uncertainty markers — cannot validate",
        )

    units = _split_eva_units(clean_token)
    remaining = list(units)
    matched_slots: list[str] = []

    for slot in DPAS_TEMPLATE:
        slot_matched = False
        repeat_count = 0

        # Try to consume units matching this slot's permitted set
        while remaining and repeat_count < slot.max_repeat:
            unit = remaining[0]
            if unit in slot.permitted:
                remaining.pop(0)
                repeat_count += 1
                slot_matched = True
            else:
                break

        if not slot.optional and not slot_matched:
            # Required slot not matched: token is invalid
            return DPASValidationResult(
                token=token,
                is_valid=False,
                matched_slots=matched_slots,
                failure_reason=(
                    f"Required slot '{slot.name}' not satisfied. "
                    f"Remaining units: {remaining}. "
                    f"Slot permits: {sorted(slot.permitted)}"
                ),
                confidence="low" if has_uncertainty else "medium",
                notes=(
                    "Contains uncertainty markers — may be a false negative."
                    if has_uncertainty else ""
                ),
            )

        if slot_matched:
            matched_slots.append(slot.name)

    # All slots processed. Check for unconsumed units.
    if remaining:
        return DPASValidationResult(
            token=token,
            is_valid=False,
            matched_slots=matched_slots,
            failure_reason=(
                f"Unconsumed units after template exhausted: {remaining}. "
                "These units do not fit any DPAS slot. "
                "This may indicate: (a) the slot definitions are incomplete, "
                "(b) the token is a multi-morpheme compound, "
                "(c) EVA segmentation ambiguity, or "
                "(d) the DPAS template is incorrect."
            ),
            confidence="low" if has_uncertainty else "medium",
            notes=(
                "Contains uncertainty markers." if has_uncertainty else ""
            ),
        )

    return DPASValidationResult(
        token=token,
        is_valid=True,
        matched_slots=matched_slots,
        failure_reason=None,
        confidence="medium" if has_uncertainty else "high",
        notes=(
            "Validation based on clean token (uncertainty markers stripped). "
            if has_uncertainty else ""
        ),
    )


# ---------------------------------------------------------------------------
# Corpus-level validation
# ---------------------------------------------------------------------------

@dataclass
class CorpusValidationReport:
    """
    Corpus-level DPAS validation statistics.

    Attributes
    ----------
    total_tokens : int
    valid_count : int
    invalid_count : int
    coverage : float
        Fraction of tokens classed as DPAS-valid.
    coverage_threshold : float
        The minimum acceptable coverage (COVERAGE_THRESHOLD).
    model_supported : bool
        True if coverage >= coverage_threshold.
        This is the primary falsifiability test.
    uncertainty_affected_count : int
        Tokens that contain EVA uncertainty markers. These may be
        false negatives — their (in)validity may be a transcription
        artefact rather than a genuine DPAS violation.
    invalid_top : list[dict]
        Most common invalid tokens and their frequencies.
    failure_reasons : dict[str, int]
        Aggregated failure reason categories and their counts.
    slot_usage : dict[str, int]
        How often each DPAS slot was matched across valid tokens.
    interpretation : str
    """
    total_tokens: int
    valid_count: int
    invalid_count: int
    coverage: float
    coverage_threshold: float
    model_supported: bool
    uncertainty_affected_count: int
    invalid_top: list[dict]
    failure_reasons: dict[str, int]
    slot_usage: dict[str, int]
    interpretation: str
    coverage_note: str = COVERAGE_NOTE


def validate_corpus(
    tokens: Sequence[str],
    return_full_results: bool = False,
) -> tuple[CorpusValidationReport, list[DPASValidationResult] | None]:
    """
    Validate all tokens in a corpus against the DPAS template.

    Parameters
    ----------
    tokens
        Flat list of EVA tokens.
    return_full_results
        If True, also return the full per-token result list.
        Warning: this can be large for full corpora.

    Returns
    -------
    (CorpusValidationReport, list[DPASValidationResult] | None)
    """
    tokens = list(tokens)
    n = len(tokens)
    logger.info("Validating %d tokens against DPAS template...", n)

    results: list[DPASValidationResult] = []
    valid_count = 0
    invalid_tokens: Counter = Counter()
    failure_category: Counter = Counter()
    slot_usage: Counter = Counter()
    uncertainty_count = 0

    for i, token in enumerate(tokens):
        if i % 10_000 == 0 and i > 0:
            logger.debug("  Validated %d/%d tokens", i, n)

        result = validate_token(token)
        results.append(result)

        if result.is_valid:
            valid_count += 1
            for slot in result.matched_slots:
                slot_usage[slot] += 1
        else:
            invalid_tokens[token] += 1
            # Categorise failure reasons
            reason = result.failure_reason or "unknown"
            if "Required slot" in reason:
                failure_category["missing_required_slot"] += 1
            elif "Unconsumed units" in reason:
                failure_category["unconsumed_units"] += 1
            elif "empty" in reason:
                failure_category["empty_token"] += 1
            else:
                failure_category["other"] += 1

        if result.notes and "uncertainty" in result.notes.lower():
            uncertainty_count += 1

    invalid_count = n - valid_count
    coverage = valid_count / n if n > 0 else 0.0
    model_supported = coverage >= COVERAGE_THRESHOLD

    # Build report
    invalid_top = [
        {"token": tok, "count": cnt}
        for tok, cnt in invalid_tokens.most_common(30)
    ]

    if model_supported:
        interpretation = (
            f"DPAS coverage = {coverage:.1%} ≥ threshold {COVERAGE_THRESHOLD:.0%}. "
            "The constraint model is consistent with the corpus at the current "
            "threshold. This supports (but does not prove) the DPAS hypothesis. "
            "Coverage should be re-evaluated as slot definitions are refined."
        )
    else:
        interpretation = (
            f"DPAS coverage = {coverage:.1%} < threshold {COVERAGE_THRESHOLD:.0%}. "
            "The DPAS model in its current form is NOT SUPPORTED by this corpus. "
            "Possible explanations: (1) slot definitions are incomplete or incorrect, "
            "(2) the corpus transcription introduces noise, "
            "(3) the DPAS hypothesis is wrong. "
            "The model must be revised before further interpretive work proceeds."
        )

    logger.info(
        "Validation complete: %d/%d valid (%.1f%%) — model %s",
        valid_count, n, coverage * 100,
        "SUPPORTED" if model_supported else "NOT SUPPORTED",
    )

    report = CorpusValidationReport(
        total_tokens=n,
        valid_count=valid_count,
        invalid_count=invalid_count,
        coverage=round(coverage, 6),
        coverage_threshold=COVERAGE_THRESHOLD,
        model_supported=model_supported,
        uncertainty_affected_count=uncertainty_count,
        invalid_top=invalid_top,
        failure_reasons=dict(failure_category),
        slot_usage=dict(slot_usage),
        interpretation=interpretation,
    )

    return report, (results if return_full_results else None)


def save_validation_report(
    report: CorpusValidationReport,
    output_path: str | Path,
) -> None:
    """Save DPAS validation report to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    logger.info("DPAS validation report saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    from amf.corpus.ingest import EVACorpus

    parser = argparse.ArgumentParser(
        description="AMF v4.0 — DPAS constraint validation"
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default="outputs/dpas_validation.json")
    parser.add_argument(
        "--clean", action="store_true",
        help="Strip uncertainty markers before validation"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    corpus = EVACorpus.from_json(args.corpus)
    tokens = corpus.all_tokens(clean=args.clean)

    report, _ = validate_corpus(tokens)

    print(f"\n{'='*50}")
    print("DPAS Validation Report")
    print(f"{'='*50}")
    print(f"Total tokens        : {report.total_tokens:,}")
    print(f"Valid               : {report.valid_count:,} ({report.coverage:.1%})")
    print(f"Invalid             : {report.invalid_count:,}")
    print(f"Coverage threshold  : {report.coverage_threshold:.0%}")
    print(f"Model supported     : {'YES' if report.model_supported else 'NO — see interpretation'}")
    print(f"\nFailure categories  : {report.failure_reasons}")
    print(f"Slot usage (valid)  : {report.slot_usage}")
    print(f"\n{report.interpretation}")

    save_validation_report(report, args.output)
    print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()
