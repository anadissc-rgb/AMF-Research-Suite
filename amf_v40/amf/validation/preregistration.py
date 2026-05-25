"""
amf.validation.preregistration
================================

DPAS Pre-Registration Manifest Generator
-----------------------------------------

Pre-registration is the practice of committing hypothesis tests, thresholds,
and analysis plans to a timestamped public record BEFORE collecting or
analysing data. This prevents HARKing (Hypothesizing After Results are Known)
and p-hacking.

For AMF v4.0, pre-registration applies specifically to the DPAS coverage test
(F1.1) and any other threshold-based claims. The mandate is:

    The DPAS slot definitions and coverage threshold must be finalized
    and recorded BEFORE running validate_corpus() on the target corpus.

This module generates a machine-readable JSON manifest that captures:
  - The exact DPAS slot definitions at the time of pre-registration
  - All falsification test criteria
  - The corpus to be tested (identified by version string, not content)
  - The AMF code commit hash (if available)
  - A timestamp

WHERE TO DEPOSIT THIS MANIFEST
-------------------------------
Acceptable pre-registration repositories:
  - OSF (Open Science Framework): https://osf.io/registries
  - Zenodo: https://zenodo.org (embargo-until-analysis-complete supported)
  - GitHub: commit the manifest file before running analysis
    (git commit -m "Pre-register DPAS v4.0 before corpus analysis")

IMPORTANT: A pre-registration manifest only has scientific value if it
is demonstrably timestamped BEFORE the analysis was run. Git commit
hashes are not sufficient alone (history can be rewritten). Use an
independent timestamping service (OSF, Zenodo) for peer-reviewed work.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manifest structure
# ---------------------------------------------------------------------------

@dataclass
class PreregistrationManifest:
    """
    Machine-readable pre-registration of the AMF v4.0 DPAS hypothesis.

    This object should be serialized to JSON and committed / uploaded to
    a timestamped repository BEFORE running the corpus analysis.

    Fields
    ------
    manifest_version : str
        Version of this manifest format.
    amf_version : str
        AMF framework version being pre-registered.
    created_at : str
        ISO 8601 timestamp of manifest creation.
    git_commit : Optional[str]
        Git commit hash of AMF code at registration time (if available).
    corpus_to_be_tested : str
        Corpus version identifier. Must match --version passed to amf ingest.
    primary_hypothesis : str
        Plain-language statement of the hypothesis being tested.
    primary_test : dict
        The single primary falsification criterion (F1.1).
    secondary_tests : list[dict]
        All other pre-specified tests.
    dpas_slot_definitions : list[dict]
        Exact slot definitions at registration time.
        Any post-hoc change to these definitions must be disclosed.
    analysis_plan : dict
        Step-by-step analysis plan with expected outputs.
    exclusion_criteria : list[str]
        Pre-specified conditions under which data or tokens will be excluded.
    deviations_disclosure : str
        Field to be filled in AFTER analysis if any deviations occurred.
    """
    manifest_version: str
    amf_version: str
    created_at: str
    corpus_to_be_tested: str
    primary_hypothesis: str
    primary_test: dict
    secondary_tests: list[dict]
    dpas_slot_definitions: list[dict]
    analysis_plan: dict
    exclusion_criteria: list[str]
    git_commit: Optional[str] = None
    deviations_disclosure: str = (
        "TO BE COMPLETED AFTER ANALYSIS: "
        "List any deviations from this pre-registration and justifications."
    )
    integrity_note: str = (
        "This manifest must be timestamped by an independent service "
        "(OSF, Zenodo, or equivalent) before corpus analysis begins. "
        "Git commit history alone is insufficient for peer-reviewed claims."
    )


# ---------------------------------------------------------------------------
# Manifest generator
# ---------------------------------------------------------------------------

def generate_preregistration(
    corpus_version: str = "SPECIFY_BEFORE_ANALYSIS",
    git_commit: Optional[str] = None,
) -> PreregistrationManifest:
    """
    Generate a pre-registration manifest from the current DPAS definitions.

    Parameters
    ----------
    corpus_version
        The transcription version string that will be used in the analysis.
        Must match the --version argument passed to `amf ingest`.
        Example: "Zandbergen-2024-01-15"
    git_commit
        Git commit hash (optional). Auto-detected if possible.

    Returns
    -------
    PreregistrationManifest
    """
    from amf.dpas.constraints import DPAS_TEMPLATE, COVERAGE_THRESHOLD

    # Try to get git commit
    if git_commit is None:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            git_commit = result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            git_commit = None

    # Serialize current slot definitions exactly
    slot_defs = []
    for slot in DPAS_TEMPLATE:
        slot_defs.append({
            "name": slot.name,
            "permitted": sorted(slot.permitted),
            "optional": slot.optional,
            "max_repeat": slot.max_repeat,
            "description": slot.description,
        })

    # Primary test (F1.1 — the most important)
    primary_test = {
        "test_id": "F1.1",
        "description": "DPAS coverage must meet pre-specified threshold",
        "criterion": f"coverage >= {COVERAGE_THRESHOLD}",
        "threshold": COVERAGE_THRESHOLD,
        "measurement": "fraction of clean corpus tokens classified as DPAS-valid",
        "verdict_rule": {
            "PASS": f"coverage >= {COVERAGE_THRESHOLD}",
            "FAIL": f"coverage < {COVERAGE_THRESHOLD}",
        },
        "severity": "CRITICAL",
        "failure_action": (
            "If FAIL: DPAS slot definitions must be revised before any "
            "further interpretive claims. The revised definitions must be "
            "pre-registered again before re-testing."
        ),
    }

    # Secondary tests (imported from falsification module)
    from amf.validation.falsification import _define_tests
    secondary = [
        {
            "test_id": t.test_id,
            "criterion": t.criterion,
            "severity": t.severity,
            "description": t.description,
        }
        for t in _define_tests()
        if t.test_id != "F1.1"
    ]

    analysis_plan = {
        "step_1": {
            "action": "amf ingest",
            "inputs": "EVA transcription files in data/eva/",
            "required_version_recorded": True,
            "note": "Version string must match corpus_to_be_tested in this manifest",
        },
        "step_2": {
            "action": "amf falsify",
            "uses_clean_tokens": True,
            "uncertainty_handling": (
                "Uncertainty markers stripped for DPAS validation. "
                "Analyses run with and without stripping for sensitivity check."
            ),
        },
        "step_3": {
            "action": "amf significance",
            "bootstrap_n": 1000,
            "seed": 42,
            "correction": "Bonferroni for multiple section comparisons",
        },
        "step_4": {
            "action": "amf pipeline",
            "outputs": [
                "entropy_report.json",
                "positional_report.json",
                "adjacency_report.json",
                "markov_report.json",
                "dpas_validation.json",
                "falsification_report.json",
                "significance_report.json",
            ],
        },
        "step_5": {
            "action": "amf report",
            "outputs": "reproducibility_report.md",
        },
        "reporting_rule": (
            "ALL results will be reported regardless of whether they support "
            "or contradict the AMF hypothesis. Selective reporting of "
            "positive results is not permitted."
        ),
    }

    exclusion_criteria = [
        "Tokens consisting entirely of uncertainty markers ({}, !, ?) are excluded",
        "Empty tokens (zero characters after normalization) are excluded",
        "Lines with zero tokens after parsing are excluded",
        "No exclusion is applied based on statistical outcome",
        "No folio or section is excluded based on coverage results",
        "No post-hoc exclusion is permitted after seeing coverage by section",
    ]

    return PreregistrationManifest(
        manifest_version="1.0",
        amf_version="4.0.0-alpha",
        created_at=datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit,
        corpus_to_be_tested=corpus_version,
        primary_hypothesis=(
            "Voynichese tokens are generated by a small set of positional "
            "slot constraints (the DPAS template). The primary test is that "
            f"≥{COVERAGE_THRESHOLD:.0%} of corpus tokens satisfy the template. "
            "This hypothesis is falsified if coverage falls below the threshold."
        ),
        primary_test=primary_test,
        secondary_tests=secondary,
        dpas_slot_definitions=slot_defs,
        analysis_plan=analysis_plan,
        exclusion_criteria=exclusion_criteria,
    )


def save_preregistration(
    manifest: PreregistrationManifest,
    output_path: str | Path,
) -> None:
    """Serialize manifest to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2, ensure_ascii=False)
    logger.info("Pre-registration manifest saved: %s", output_path)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  PRE-REGISTRATION MANIFEST GENERATED                         ║
╠══════════════════════════════════════════════════════════════╣
║  Saved to: {str(output_path):<51s}║
╠══════════════════════════════════════════════════════════════╣
║  NEXT STEPS (required for valid pre-registration):           ║
║                                                              ║
║  1. Upload this file to OSF (https://osf.io/registries)      ║
║     or Zenodo BEFORE running any corpus analysis             ║
║                                                              ║
║  2. Record the deposit URL and timestamp                      ║
║                                                              ║
║  3. Only then run: amf falsify --corpus <path>               ║
║                                                              ║
║  Git commits alone are NOT sufficient for peer review.        ║
╚══════════════════════════════════════════════════════════════╝
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "AMF v4.0 — Generate DPAS pre-registration manifest.\n"
            "Run this BEFORE analysing the corpus."
        )
    )
    parser.add_argument("--output", default="preregistration/dpas_preregistration.json")
    parser.add_argument(
        "--corpus-version",
        default="SPECIFY_BEFORE_ANALYSIS",
        help="Transcription version string (must match --version in amf ingest)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    manifest = generate_preregistration(corpus_version=args.corpus_version)
    save_preregistration(manifest, args.output)


if __name__ == "__main__":
    main()
