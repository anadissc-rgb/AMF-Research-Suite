"""
amf.validation.report
======================

Reproducibility Report Generator
-----------------------------------

Generates a structured Markdown report from a completed pipeline run.
The report separates verified measurements from hypotheses and speculative
claims, and includes all provenance information needed for replication.

The output is intended to accompany any publication or sharing of results.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_report(run_path: str | Path) -> str:
    """
    Generate a Markdown reproducibility report from a pipeline run JSON.

    Parameters
    ----------
    run_path
        Path to a pipeline_run_*.json file.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    run_path = Path(run_path)
    with run_path.open(encoding="utf-8") as f:
        run = json.load(f)

    now = datetime.now(timezone.utc).isoformat()
    vr = run.get("verified_results", {})
    hr = run.get("hypothesis_results", {})
    sp = run.get("speculative_mappings", {})
    meta = run.get("corpus_metadata", {})
    warnings = run.get("warnings", [])
    limitations = run.get("limitations", [])

    ent  = vr.get("entropy", {})
    zipf = vr.get("zipf", {})
    pos  = vr.get("positional", {})
    comp = vr.get("baseline_comparison", {})
    dpas = hr.get("dpas_validation", {})
    fals = hr.get("falsification", {})

    lines = [
        "# AMF v4.0 — Reproducibility Report",
        "",
        f"> Generated: {now}",
        f"> Run ID: {run.get('run_id', 'unknown')}",
        f"> AMF version: {run.get('amf_version', 'unknown')}",
        "",
        "---",
        "",
        "## Epistemic Statement",
        "",
        f"> {run.get('epistemic_statement', '')}",
        "",
        "---",
        "",
        "## Corpus Provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Transcription version | `{meta.get('transcription_version', 'unknown')}` |",
        f"| Source files | {len(meta.get('source_files', []))} file(s) |",
        f"| Total records (lines) | {meta.get('record_count', 0):,} |",
        f"| Total tokens | {meta.get('token_count', 0):,} |",
        f"| Folios | {meta.get('folio_count', 0):,} |",
        f"| Lines with uncertainty markers | {meta.get('uncertainty_line_count', 0):,} |",
        f"| Loaded at | {meta.get('loaded_at', 'unknown')} |",
        "",
        "---",
        "",
        "## Section 1: Verified Results",
        "",
        "_These measurements are reproducible from the same corpus and code._",
        "_Any researcher can verify them by running `amf pipeline --corpus <same corpus>`._",
        "",
        "### 1.1 Entropy",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Unigram entropy H(W) | **{ent.get('unigram_bits', 'N/A')} bits** |",
        f"| Normalised entropy H(W)/H_max | {ent.get('normalised_entropy', 'N/A')} |",
        f"| Conditional entropy H(W\\|W-1) | {ent.get('bigram_conditional_bits', 'N/A')} bits |",
        f"| Redundancy | {ent.get('redundancy', 'N/A')} |",
        f"| Token count | {ent.get('token_count', 0):,} |",
        f"| Type count | {ent.get('type_count', 0):,} |",
        "",
        "**Baseline comparison:**",
        "",
        f"- Voynich H(W): {comp.get('voynich_h_unigram', 'N/A')} bits",
        f"- Random baseline H(W): {comp.get('baseline_h_unigram', 'N/A')} bits",
        f"- Δ: {comp.get('delta_bits', 'N/A')} bits (positive = Voynich lower than random)",
        f"- {comp.get('interpretation', '')}",
        "",
        "### 1.2 Zipf Distribution",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Zipf exponent α | **{zipf.get('alpha', 'N/A')}** |",
        f"| Log-log fit R² | {zipf.get('r_squared', 'N/A')} |",
        "",
        "> Natural language range: α ≈ 0.8–1.2. "
        "Consistency with this range is necessary but not sufficient for a "
        "natural language interpretation.",
        "",
        "### 1.3 Positional Structure",
        "",
        f"- Total lines analysed: {pos.get('total_lines', 0):,}",
        f"- Manuscript sections identified: "
        f"{', '.join(pos.get('sections_found', ['none']))}",
        "",
        "**Line-initial biased tokens** (≥40% of occurrences at line start, n≥10):",
        "",
        f"```\n{pos.get('line_initial_biased_tokens', [])}\n```",
        "",
        "---",
        "",
        "## Section 2: Hypothesis Results",
        "",
        "_These results test specific model predictions. They require independent "
        "replication and are not verified measurements._",
        "",
        "### 2.1 DPAS Constraint Validation",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Coverage | **{dpas.get('coverage', 'N/A'):.1%}** |",
        f"| Threshold | {dpas.get('threshold', 'N/A'):.0%} |",
        f"| Model supported | {'**YES**' if dpas.get('model_supported') else '**NO — see interpretation**'} |",
        "",
        f"> {dpas.get('interpretation', '')}",
        "",
        "**Slot usage (valid tokens):**",
        "",
        "| Slot | Match count |",
        "|---|---|",
    ]

    for slot, count in (dpas.get("slot_usage") or {}).items():
        lines.append(f"| {slot} | {count:,} |")

    # ── 2.2 Falsification verdict ─────────────────────────────────────
    fals_verdict  = fals.get("overall_verdict", "N/A")
    fals_critical = fals.get("critical_failures", "N/A")
    fals_major    = fals.get("major_failures",    "N/A")
    fals_minor    = fals.get("minor_failures",    "N/A")
    fals_summary  = fals.get("summary", "")

    lines += [
        "",
        "### 2.2 Falsification Test Results",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Overall verdict | **{fals_verdict}** |",
        f"| CRITICAL failures | {fals_critical} |",
        f"| MAJOR failures | {fals_major} |",
        f"| MINOR failures | {fals_minor} |",
        "",
        f"> {fals_summary}",
        "",
        "> **Pre-registration note**: These results are valid only if the "
        "falsification criteria were committed before the corpus was analysed. "
        "Verify against preregistration/dpas_preregistration.json.",
        "",
    ]

    # ── 1.4 Adjacency + 1.5 Markov + 1.6 Significance ────────────────
    adj = vr.get("adjacency", {})
    mk  = vr.get("markov",    {})
    sig = vr.get("significance", {})

    lines += [
        "---",
        "",
        "### 1.4 Adjacency Statistics",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Total bigrams | {adj.get('total_bigrams', 'N/A')} |",
        f"| Unique bigram types | {adj.get('unique_bigram_types', 'N/A')} |",
        f"| Bigram entropy | {adj.get('bigram_entropy_bits', 'N/A')} bits |",
        "",
        "### 1.5 Markov Order Analysis",
        "",
        f"- Train/test split honest: **{mk.get('split_config', {}).get('honest_split', 'N/A')}** "
        f"(seed={mk.get('split_config', {}).get('seed', 'N/A')})",
        f"- Optimal order (train): {mk.get('optimal_order_train', 'N/A')}",
        f"- Optimal order (test/honest): **{mk.get('optimal_order_test', 'N/A')}**",
        "",
        "| Order | Train PP | Test PP |",
        "|---|---|---|",
    ]
    for o, pp in (mk.get("perplexity_by_order") or {}).items():
        lines.append(f"| {o} | {pp.get('train','N/A')} | {pp.get('test','N/A')} |")

    lines += [
        "",
        "> Test perplexity is computed on held-out lines not seen during training.",
        "",
    ]

    if sig:
        ent_ci  = sig.get("entropy_95ci",    ["N/A", "N/A"])
        zipf_ci = sig.get("zipf_alpha_95ci", ["N/A", "N/A"])
        n_sig   = sig.get("sections_significant", "N/A")
        lines += [
            "### 1.6 Significance (Bootstrap 95% CIs)",
            "",
            f"- Entropy H(W): [{ent_ci[0]}, {ent_ci[1]}] bits",
            f"- Zipf alpha: [{zipf_ci[0]}, {zipf_ci[1]}]",
            f"- Sections significantly different from corpus baseline: {n_sig}",
            "",
        ]

    lines += [
        "---",
        "",
        "## Section 3: Speculative Mappings",
        "",
        f"_{sp.get('note', '')}_",
        "",
    ]

    for claim in sp.get("amf_framework_claims", []):
        lines += [
            f"**{claim.get('claim', '')}**",
            "",
            f"- Basis: {claim.get('basis', '')}",
            f"- Status: _{claim.get('status', '')}_",
            f"- Required validation: {claim.get('required_validation', '')}",
            "",
        ]

    lines += [
        "---",
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        for w in warnings:
            lines.append(f"- ⚠ {w}")
    else:
        lines.append("_No warnings._")

    lines += [
        "",
        "## Documented Limitations",
        "",
    ]
    for lim in limitations:
        lines.append(f"- {lim}")

    lines += [
        "",
        "---",
        "",
        "## Replication Instructions",
        "",
        "```bash",
        "# 1. Check out the same code version",
        f"git checkout {run.get('amf_version', 'HEAD')}",
        "",
        "# 2. Obtain the same EVA corpus",
        f"# Transcription version: {meta.get('transcription_version', 'unknown')}",
        "# Source: https://www.voynich.nu/transcr.html",
        "",
        "# 3. Ingest",
        "amf ingest --input data/eva/ \\",
        f"    --version \"{meta.get('transcription_version', 'unknown')}\" \\",
        "    --output data/processed/corpus.json",
        "",
        "# 4. Reproduce",
        f"amf pipeline --corpus data/processed/corpus.json \\",
        f"    --run-id replication-$(date +%Y%m%d)",
        "```",
        "",
        "Results should match to within floating-point precision on the same "
        "corpus. Differences indicate either a different corpus version or "
        "a code change — both should be documented.",
        "",
        "---",
        "",
        "_AMF v4.0 — Proprietary. All rights reserved._",
        "_This report is an output of the AMF research framework, not a "
        "peer-reviewed publication._",
    ]

    return "\n".join(lines)


def save_report(report_text: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    logger.info("Reproducibility report saved: %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse, glob

    parser = argparse.ArgumentParser(
        description="AMF v4.0 — Generate reproducibility report from pipeline run"
    )
    parser.add_argument("--run", required=True,
                        help="Path to pipeline_run_*.json (glob patterns accepted)")
    parser.add_argument("--output", default="outputs/reproducibility_report.md")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Resolve glob
    paths = sorted(glob.glob(args.run))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.run}")
    run_path = paths[-1]   # Use most recent if multiple matches
    if len(paths) > 1:
        logger.warning("Multiple run files matched; using most recent: %s", run_path)

    text = generate_report(run_path)
    save_report(text, args.output)
    print(f"Report saved to: {args.output}")
    print(f"\nPreview (first 30 lines):")
    for line in text.splitlines()[:30]:
        print(line)


if __name__ == "__main__":
    main()
