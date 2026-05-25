# © 2026 Anadi Chakraborty
# Amanuensis Model Framework (AMF)
# Proprietary Research Software
"""
amf.validation.pipeline
========================

AMF v4.0 — End-to-End Validation Pipeline
-------------------------------------------

Orchestrates the complete analysis sequence:

    1. Corpus ingestion check
    2. Entropy analysis          → outputs/entropy_report.json
    3. Positional analysis       → outputs/positional_report.json
    4. Adjacency analysis        → outputs/adjacency_report.json
    5. Markov analysis           → outputs/markov_report.json
    6. DPAS validation           → outputs/dpas_validation.json
    7. Significance tests        → outputs/significance_report.json
    8. Falsification tests       → outputs/falsification_report.json
    9. Reproducibility report    → outputs/reproducibility_report.md
   10. Pipeline run manifest     → outputs/pipeline_run_<id>.json

SEPARATION OF CLAIMS
--------------------
Outputs are sorted into three epistemic tiers:

    VERIFIED      Reproducible statistical measurements.
                  Any researcher with the same corpus can compute these.

    HYPOTHESIS    Model-specific predictions that require validation.
                  Supported or falsified by pre-specified criteria.

    SPECULATIVE   Interpretive mappings beyond statistical support.
                  Clearly labelled; not to be cited as findings.

This separation is enforced structurally — all three tiers are distinct
JSON keys in the pipeline run manifest.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline run container
# ---------------------------------------------------------------------------

@dataclass
class PipelineRun:
    """Complete pipeline run output, all tiers, full provenance."""
    run_id:               str
    timestamp:            str
    corpus_metadata:      dict
    verified_results:     dict
    hypothesis_results:   dict
    speculative_mappings: dict
    warnings:             list[str]
    limitations:          list[str]
    amf_version:          str = "4.0.0-alpha"
    epistemic_statement:  str = (
        "This pipeline output separates computationally verified measurements "
        "from interpretive hypotheses and speculative claims. Only "
        "'verified_results' makes reproducible empirical claims. All other "
        "sections are explicitly provisional and require independent validation "
        "before being cited in scholarly work."
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    corpus_path:   str | Path,
    output_dir:    str | Path,
    run_id:        Optional[str] = None,
    test_fraction: float = 0.20,
    seed:          int   = 42,
    bootstrap_n:   int   = 500,
    skip_significance: bool = False,
) -> PipelineRun:
    """
    Execute the full AMF v4.0 analysis pipeline.

    Parameters
    ----------
    corpus_path
        Path to processed corpus JSON (output of amf.corpus.ingest).
    output_dir
        Directory for all pipeline outputs.
    run_id
        Optional identifier for this run (default: timestamp-based).
    test_fraction
        Held-out fraction for Markov test perplexity (default 0.20).
    seed
        Master random seed for all stochastic operations.
    bootstrap_n
        Bootstrap resamples for significance analysis (default 500).
        Increase to 1000+ for publication-quality CIs.
    skip_significance
        If True, skip bootstrap significance analysis (faster for dev runs).
    """
    from amf.corpus.ingest          import EVACorpus
    from amf.stats.entropy          import full_entropy_report, save_report
    from amf.stats.positional       import compute_positional_stats, high_bias_tokens
    from amf.stats.positional       import save_positional_report
    from amf.stats.adjacency        import compute_adjacency, save_adjacency_report
    from amf.stats.markov           import run_markov_analysis, save_markov_report
    from amf.dpas.validator         import validate_corpus, save_validation_report
    from amf.validation.falsification  import run_falsification_tests, save_falsification_report
    from amf.validation.report      import generate_report, save_report as save_md_report

    corpus_path = Path(corpus_path)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    run_id    = run_id or f"run_{timestamp[:19].replace(':', '-')}"

    logger.info("═" * 60)
    logger.info("AMF v4.0 pipeline  run_id=%s", run_id)
    logger.info("═" * 60)

    warnings:    list[str] = []
    limitations: list[str] = []

    # ── 1. Load corpus ──────────────────────────────────────────────
    logger.info("[1/8] Loading corpus from %s", corpus_path)
    corpus       = EVACorpus.from_json(corpus_path)
    tokens       = corpus.all_tokens(clean=False)
    tokens_clean = corpus.all_tokens(clean=True)

    if corpus.metadata.transcription_version == "unknown":
        warnings.append(
            "Corpus transcription version is 'unknown'. "
            "Results cannot be precisely attributed to a specific EVA release. "
            "Use --version when ingesting."
        )

    unc_frac = (corpus.metadata.uncertainty_line_count
                / max(len(corpus.records), 1))
    if unc_frac > 0.15:
        warnings.append(
            f"{unc_frac:.1%} of lines contain EVA uncertainty markers. "
            "Statistical results may be affected. Consider --skip-uncertain."
        )

    limitations += [
        "All results derived from the EVA transcription, not the manuscript directly.",
        "Entropy and Zipf statistics cannot distinguish natural language, "
        "cipher, glossolalia, or structured notation.",
        "DPAS slot definitions are provisional and were derived from the same "
        "corpus being tested (circularity risk — see docs/limitations.md §4.3).",
        "Section classifications are codicological conventions, not ground truth.",
    ]

    # ── 2. Entropy ──────────────────────────────────────────────────
    logger.info("[2/8] Entropy analysis")
    entropy_report = full_entropy_report(tokens_clean, label="voynich_eva_clean")
    save_report(entropy_report, output_dir / "entropy_report.json")

    # ── 3. Positional ────────────────────────────────────────────────
    logger.info("[3/8] Positional analysis")
    pos_stats = compute_positional_stats(corpus.records)
    biased    = high_bias_tokens(pos_stats)
    save_positional_report(pos_stats, output_dir / "positional_report.json")

    # ── 4. Adjacency ─────────────────────────────────────────────────
    logger.info("[4/8] Adjacency analysis")
    adj_report = compute_adjacency(corpus.records, top_n=30, min_pmi_count=5)
    save_adjacency_report(adj_report, output_dir / "adjacency_report.json")

    # ── 5. Markov (with train/test split) ────────────────────────────
    logger.info("[5/8] Markov order analysis (train/test split, seed=%d)", seed)
    markov_report = run_markov_analysis(
        corpus.records,
        max_order=4,
        smoothing=1.0,
        test_fraction=test_fraction,
        seed=seed,
    )
    save_markov_report(markov_report, output_dir / "markov_report.json")

    pp_by_order = {r["order"]: r for r in markov_report.results_by_order}

    # ── 6. DPAS validation ───────────────────────────────────────────
    logger.info("[6/8] DPAS constraint validation")
    dpas_report, _ = validate_corpus(tokens_clean)
    save_validation_report(dpas_report, output_dir / "dpas_validation.json")

    if not dpas_report.model_supported:
        warnings.append(
            f"DPAS coverage ({dpas_report.coverage:.1%}) is below threshold "
            f"({dpas_report.coverage_threshold:.0%}). "
            "The DPAS constraint model is NOT supported. Revise slot definitions."
        )

    # ── 7. Significance tests ────────────────────────────────────────
    sig_summary: dict = {}
    if not skip_significance:
        logger.info("[7/8] Significance tests (bootstrap n=%d, seed=%d)",
                    bootstrap_n, seed)
        try:
            from amf.stats.significance import (
                run_significance_analysis, save_significance_report
            )
            sig_report = run_significance_analysis(
                corpus.records, n_bootstrap=bootstrap_n, seed=seed
            )
            save_significance_report(
                sig_report, output_dir / "significance_report.json"
            )
            sig_summary = {
                "entropy_95ci": [
                    sig_report.bootstrap_entropy_ci["ci_lower"],
                    sig_report.bootstrap_entropy_ci["ci_upper"],
                ],
                "zipf_alpha_95ci": [
                    sig_report.bootstrap_zipf_ci["ci_lower"],
                    sig_report.bootstrap_zipf_ci["ci_upper"],
                ],
                "sections_significant": (
                    sig_report.chi_square_sections.get("sections_significant", 0)
                ),
            }
        except Exception as e:
            warnings.append(f"Significance analysis failed: {e}. Skipped.")
            logger.warning("Significance analysis failed: %s", e)
    else:
        logger.info("[7/8] Significance tests skipped (--skip-significance)")
        warnings.append("Significance tests were skipped. Run `amf significance` separately.")

    # ── 8. Falsification tests ───────────────────────────────────────
    logger.info("[8/8] Falsification tests")
    falsification_report = run_falsification_tests(corpus_path)
    save_falsification_report(
        falsification_report, output_dir / "falsification_report.json"
    )

    if falsification_report.critical_failures > 0:
        warnings.append(
            f"{falsification_report.critical_failures} CRITICAL falsification "
            f"test(s) failed. Overall verdict: {falsification_report.overall_verdict}. "
            "See outputs/falsification_report.json."
        )

    # ── Assemble pipeline run manifest ───────────────────────────────
    ent  = entropy_report["entropy"]
    zipf = entropy_report["zipf"]

    verified_results = {
        "note": "Reproducible from the same corpus and code version.",
        "entropy": {
            "unigram_bits":             ent["unigram_entropy"],
            "bigram_conditional_bits":  ent["bigram_entropy"],
            "normalised_entropy":       ent["unigram_entropy_normalised"],
            "redundancy":               ent["redundancy"],
            "token_count":              ent["token_count"],
            "type_count":               ent["type_count"],
        },
        "zipf": {
            "alpha":     zipf["alpha"],
            "r_squared": zipf["r_squared"],
        },
        "positional": {
            "total_lines":                  pos_stats.total_lines,
            "sections_found":               list(pos_stats.section_profiles.keys()),
            "line_initial_biased_tokens":   biased["line_initial_biased"][:10],
            "line_final_biased_tokens":     biased["line_final_biased"][:10],
        },
        "adjacency": {
            "total_bigrams":       adj_report.total_bigrams,
            "unique_bigram_types": adj_report.unique_bigram_types,
            "bigram_entropy_bits": adj_report.bigram_entropy,
        },
        "markov": {
            "split_config": markov_report.split_config,
            "optimal_order_train": markov_report.optimal_order_by_train_pp,
            "optimal_order_test":  markov_report.optimal_order_by_test_pp,
            "perplexity_by_order": {
                str(r["order"]): {
                    "train": r["train_perplexity"],
                    "test":  r["test_perplexity"],
                }
                for r in markov_report.results_by_order
            },
            "reduction_test": markov_report.perplexity_reduction_test,
        },
        "baseline_comparison": entropy_report["comparison"],
        "significance": sig_summary,
    }

    hypothesis_results = {
        "note": (
            "Interpretive results requiring independent validation. "
            "Consistent with the AMF DPAS hypothesis but not proven."
        ),
        "dpas_validation": {
            "coverage":        dpas_report.coverage,
            "model_supported": dpas_report.model_supported,
            "threshold":       dpas_report.coverage_threshold,
            "slot_usage":      dpas_report.slot_usage,
            "failure_categories": dpas_report.failure_reasons,
            "interpretation":  dpas_report.interpretation,
        },
        "falsification": {
            "overall_verdict":    falsification_report.overall_verdict,
            "critical_failures":  falsification_report.critical_failures,
            "major_failures":     falsification_report.major_failures,
            "minor_failures":     falsification_report.minor_failures,
            "summary":            falsification_report.summary,
        },
    }

    speculative_mappings = {
        "note": (
            "SPECULATIVE claims only. Not supported by statistical analysis "
            "above. Not to be cited as findings without independent validation."
        ),
        "amf_framework_claims": [
            {
                "claim": "Voynichese may encode botanical/pharmaceutical notation",
                "basis": "Section-specific token distribution and visual manuscript content",
                "status": "SPECULATIVE — no statistical support established",
                "required_validation": (
                    "Section-specific token frequencies must differ significantly "
                    "from corpus baseline in a pre-registered test."
                ),
            },
            {
                "claim": "DPAS slot structure may reflect scribal production rules",
                "basis": "Token positional regularity and low conditional entropy",
                "status": "SPECULATIVE — consistent with hypothesis but not proven",
                "required_validation": (
                    "DPAS coverage > threshold AND cross-section consistency test "
                    "AND independent replication on a second transcription."
                ),
            },
        ],
    }

    pipeline_run = PipelineRun(
        run_id=run_id,
        timestamp=timestamp,
        corpus_metadata=asdict(corpus.metadata),
        verified_results=verified_results,
        hypothesis_results=hypothesis_results,
        speculative_mappings=speculative_mappings,
        warnings=warnings,
        limitations=limitations,
    )

    run_path = output_dir / f"pipeline_run_{run_id}.json"
    with run_path.open("w") as f:
        json.dump(asdict(pipeline_run), f, indent=2)
    logger.info("Pipeline run manifest: %s", run_path)

    # ── Reproducibility report ───────────────────────────────────────
    try:
        report_text = generate_report(run_path)
        save_md_report(report_text, output_dir / "reproducibility_report.md")
    except Exception as e:
        warnings.append(f"Reproducibility report generation failed: {e}")

    logger.info("═" * 60)
    logger.info(
        "Pipeline complete — verdict: %s",
        falsification_report.overall_verdict,
    )
    logger.info("═" * 60)

    return pipeline_run


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="AMF v4.0 — Full validation pipeline"
    )
    parser.add_argument("--corpus",        required=True)
    parser.add_argument("--output",        default="outputs/")
    parser.add_argument("--run-id",        default=None)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--bootstrap-n",   type=int,   default=500)
    parser.add_argument("--skip-significance", action="store_true")
    parser.add_argument("--log-level",     default="INFO",
                        choices=["DEBUG","INFO","WARNING","ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run = run_pipeline(
        args.corpus,
        args.output,
        run_id=args.run_id,
        test_fraction=args.test_fraction,
        seed=args.seed,
        bootstrap_n=args.bootstrap_n,
        skip_significance=args.skip_significance,
    )

    print(f"\n{'═'*60}")
    print(f"AMF v4.0 Pipeline Run: {run.run_id}")
    print(f"{'═'*60}")

    if run.warnings:
        print(f"\n⚠  Warnings ({len(run.warnings)}):")
        for w in run.warnings:
            print(f"   {w}")

    vr = run.verified_results
    print(f"\nVERIFIED RESULTS:")
    print(f"  Entropy H(W)    : {vr['entropy']['unigram_bits']:.4f} bits")
    print(f"  Zipf α          : {vr['zipf']['alpha']:.4f}  (R²={vr['zipf']['r_squared']:.4f})")
    print(f"  Bigram entropy  : {vr['adjacency']['bigram_entropy_bits']:.4f} bits")

    mk = vr["markov"]
    print(f"  Markov optimal  : order={mk['optimal_order_test']} (by test PP)")
    for o, pp in mk["perplexity_by_order"].items():
        tpp = f"{pp['test']:.2f}" if pp['test'] else "N/A"
        print(f"    order {o}: train={pp['train']:.2f}  test={tpp}")

    hr = run.hypothesis_results
    dpas = hr["dpas_validation"]
    fals = hr["falsification"]
    print(f"\nHYPOTHESIS RESULTS:")
    print(f"  DPAS coverage   : {dpas['coverage']:.1%}  "
          f"(supported={dpas['model_supported']})")
    print(f"  Falsification   : {fals['overall_verdict']}  "
          f"(critical={fals['critical_failures']} "
          f"major={fals['major_failures']} "
          f"minor={fals['minor_failures']})")

    print(f"\n{run.epistemic_statement}")
    print(f"\nLimitations:")
    for lim in run.limitations:
        print(f"  - {lim}")


if __name__ == "__main__":
    main()
