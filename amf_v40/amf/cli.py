# © 2026 Anadi Chakraborty
# Amanuensis Model Framework (AMF)
# Proprietary Research Software
"""
amf.cli
=======

AMF v4.0 — Unified Command-Line Interface
-------------------------------------------

All pipeline operations are available through this single entry point.

USAGE
-----
    amf <command> [options]

COMMANDS
--------
    ingest          Parse EVA transcription files into processed corpus JSON
    entropy         Shannon entropy, Zipf, and character-level analysis
    positional      Line-initial/final and section-specific token statistics
    adjacency       Bigram/trigram frequencies and PMI pairs
    markov          Markov order analysis with train/test split
    dpas            DPAS constraint validation (primary falsifiability test)
    falsify         Run all falsification tests and report pass/fail
    significance    Chi-square and bootstrap significance tests
    pipeline        Run the complete analysis pipeline end-to-end
    report          Generate a structured reproducibility report
    preregister     Write a pre-registration manifest for DPAS hypothesis
    render          Generate manuscript background images

EXAMPLES
--------
    # Full pipeline from raw transcription to report
    amf ingest --input data/eva/ --version Zandbergen-2024-01
    amf pipeline --corpus data/processed/corpus.json
    amf report --run outputs/pipeline_run_*.json

    # Individual analyses
    amf entropy   --corpus data/processed/corpus.json
    amf dpas      --corpus data/processed/corpus.json
    amf falsify   --corpus data/processed/corpus.json

    # Pre-register DPAS before testing (prevents circularity)
    amf preregister --output preregistration/dpas_v4.json

REPRODUCIBILITY
---------------
Every command writes its output with embedded provenance metadata
(corpus version, AMF version, timestamp, random seeds). Run IDs
can be specified with --run-id to facilitate cross-run comparison.

All random operations use fixed seeds (default 42). The seed is
always recorded in output metadata and must be reported in any
publication using these results.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> int:
    from amf.corpus.ingest import main as _main
    sys.argv = ["amf-ingest",
                "--input", args.input,
                "--output", args.output,
                "--version", args.version,
                "--log-level", args.log_level]
    if args.skip_uncertain:
        sys.argv.append("--skip-uncertain")
    _main()
    return 0


def cmd_entropy(args: argparse.Namespace) -> int:
    from amf.stats.entropy import main as _main
    sys.argv = ["amf-entropy",
                "--corpus", args.corpus,
                "--output", args.output,
                "--label", args.label,
                "--log-level", args.log_level]
    if args.clean:
        sys.argv.append("--clean")
    _main()
    return 0


def cmd_positional(args: argparse.Namespace) -> int:
    from amf.stats.positional import main as _main
    sys.argv = ["amf-positional",
                "--corpus", args.corpus,
                "--output", args.output,
                "--top-n", str(args.top_n)]
    _main()
    return 0


def cmd_adjacency(args: argparse.Namespace) -> int:
    from amf.stats.adjacency import main as _main
    sys.argv = ["amf-adjacency",
                "--corpus", args.corpus,
                "--output", args.output,
                "--top-n", str(args.top_n),
                "--min-pmi-count", str(args.min_pmi_count)]
    _main()
    return 0


def cmd_markov(args: argparse.Namespace) -> int:
    from amf.stats.markov import main as _main
    sys.argv = ["amf-markov",
                "--corpus", args.corpus,
                "--output", args.output,
                "--max-order", str(args.max_order),
                "--test-fraction", str(args.test_fraction),
                "--seed", str(args.seed)]
    _main()
    return 0


def cmd_dpas(args: argparse.Namespace) -> int:
    from amf.dpas.validator import main as _main
    sys.argv = ["amf-dpas",
                "--corpus", args.corpus,
                "--output", args.output]
    if args.clean:
        sys.argv.append("--clean")
    _main()
    return 0


def cmd_falsify(args: argparse.Namespace) -> int:
    from amf.validation.falsification import main as _main
    sys.argv = ["amf-falsify",
                "--corpus", args.corpus,
                "--output", args.output]
    _main()
    return 0


def cmd_significance(args: argparse.Namespace) -> int:
    from amf.stats.significance import main as _main
    sys.argv = ["amf-significance",
                "--corpus", args.corpus,
                "--output", args.output,
                "--bootstrap-n", str(args.bootstrap_n),
                "--seed", str(args.seed)]
    _main()
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    from amf.validation.pipeline import main as _main
    sys.argv = ["amf-pipeline",
                "--corpus", args.corpus,
                "--output", args.output,
                "--log-level", args.log_level,
                "--test-fraction", str(getattr(args, "test_fraction", 0.20)),
                "--seed", str(getattr(args, "seed", 42)),
                "--bootstrap-n", str(getattr(args, "bootstrap_n", 500))]
    if args.run_id:
        sys.argv += ["--run-id", args.run_id]
    if getattr(args, "skip_significance", False):
        sys.argv.append("--skip-significance")
    _main()
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from amf.validation.report import main as _main
    sys.argv = ["amf-report",
                "--run", args.run,
                "--output", args.output]
    _main()
    return 0


def cmd_preregister(args: argparse.Namespace) -> int:
    from amf.validation.preregistration import main as _main
    sys.argv = ["amf-preregister",
                "--output", args.output]
    _main()
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from amf.render.generate_bg import main as _main
    sys.argv = ["amf-render", args.output]
    _main()
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amf",
        description=(
            "AMF v4.0 — Amanuensis Model Framework\n"
            "Computational analysis of Voynichese (EVA corpus)\n\n"
            "EPISTEMIC STATUS: Experimental framework. No decipherment claim is made."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--version", action="version", version="AMF v4.0.0-alpha"
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # ── ingest ────────────────────────────────────────────────────────
    p = sub.add_parser("ingest", help="Parse EVA transcription into corpus JSON")
    p.add_argument("--input",  required=True, help="EVA directory or .txt file")
    p.add_argument("--output", default="data/processed/corpus.json")
    p.add_argument("--version", dest="version", default="unknown",
                   help="Transcription version string (record for reproducibility)")
    p.add_argument("--skip-uncertain", action="store_true")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    p.set_defaults(func=cmd_ingest)

    # ── entropy ───────────────────────────────────────────────────────
    p = sub.add_parser("entropy", help="Shannon entropy, Zipf, and character analysis")
    p.add_argument("--corpus", required=True)
    p.add_argument("--output", default="outputs/entropy_report.json")
    p.add_argument("--label",  default="voynich_eva")
    p.add_argument("--clean",  action="store_true",
                   help="Strip uncertainty markers (documented in output)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    p.set_defaults(func=cmd_entropy)

    # ── positional ────────────────────────────────────────────────────
    p = sub.add_parser("positional", help="Line-initial/final and section token statistics")
    p.add_argument("--corpus", required=True)
    p.add_argument("--output", default="outputs/positional_report.json")
    p.add_argument("--top-n",  type=int, default=30)
    p.set_defaults(func=cmd_positional)

    # ── adjacency ─────────────────────────────────────────────────────
    p = sub.add_parser("adjacency", help="Bigram/trigram frequencies and PMI")
    p.add_argument("--corpus", required=True)
    p.add_argument("--output", default="outputs/adjacency_report.json")
    p.add_argument("--top-n",  type=int, default=30)
    p.add_argument("--min-pmi-count", type=int, default=5)
    p.set_defaults(func=cmd_adjacency)

    # ── markov ────────────────────────────────────────────────────────
    p = sub.add_parser("markov", help="Markov order analysis with train/test split")
    p.add_argument("--corpus", required=True)
    p.add_argument("--output", default="outputs/markov_report.json")
    p.add_argument("--max-order", type=int, default=4)
    p.add_argument("--test-fraction", type=float, default=0.20,
                   help="Fraction of lines held out for test perplexity (default 0.20)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for train/test split (default 42, always reported)")
    p.set_defaults(func=cmd_markov)

    # ── dpas ──────────────────────────────────────────────────────────
    p = sub.add_parser("dpas", help="DPAS constraint validation (falsifiability test)")
    p.add_argument("--corpus", required=True)
    p.add_argument("--output", default="outputs/dpas_validation.json")
    p.add_argument("--clean",  action="store_true")
    p.set_defaults(func=cmd_dpas)

    # ── falsify ───────────────────────────────────────────────────────
    p = sub.add_parser(
        "falsify",
        help="Run all pre-specified falsification tests with pass/fail verdicts",
    )
    p.add_argument("--corpus", required=True)
    p.add_argument("--output", default="outputs/falsification_report.json")
    p.set_defaults(func=cmd_falsify)

    # ── significance ──────────────────────────────────────────────────
    p = sub.add_parser(
        "significance",
        help="Bootstrap confidence intervals and chi-square tests",
    )
    p.add_argument("--corpus", required=True)
    p.add_argument("--output", default="outputs/significance_report.json")
    p.add_argument("--bootstrap-n", type=int, default=1000,
                   help="Bootstrap resamples for CIs (default 1000)")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_significance)

    # ── pipeline ──────────────────────────────────────────────────────
    p = sub.add_parser("pipeline", help="Full end-to-end analysis pipeline")
    p.add_argument("--corpus", required=True)
    p.add_argument("--output", default="outputs/")
    p.add_argument("--run-id", default=None)
    p.add_argument("--test-fraction", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap-n", type=int, default=500)
    p.add_argument("--skip-significance", action="store_true",
                   help="Skip bootstrap significance tests (faster dev runs)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    p.set_defaults(func=cmd_pipeline)

    # ── report ────────────────────────────────────────────────────────
    p = sub.add_parser("report", help="Generate structured reproducibility report")
    p.add_argument("--run",    required=True,
                   help="Path to pipeline_run_*.json (or glob pattern)")
    p.add_argument("--output", default="outputs/reproducibility_report.md")
    p.set_defaults(func=cmd_report)

    # ── preregister ───────────────────────────────────────────────────
    p = sub.add_parser(
        "preregister",
        help="Write DPAS pre-registration manifest (run BEFORE corpus validation)",
    )
    p.add_argument("--output", default="preregistration/dpas_preregistration.json")
    p.set_defaults(func=cmd_preregister)

    # ── render ────────────────────────────────────────────────────────
    p = sub.add_parser("render", help="Generate manuscript background images")
    p.add_argument("--output", default="outputs/assets/")
    p.set_defaults(func=cmd_render)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main CLI entry point.

    Registered in pyproject.toml as:
        [project.scripts]
        amf = "amf.cli:main"
    """
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger("amf.cli")
    logger.debug("AMF v4.0 CLI — command: %s", args.command)

    try:
        rc = args.func(args)
        sys.exit(rc)
    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
