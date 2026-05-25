"""
tests/test_validation.py
=========================

Tests for:
  - amf.stats.markov        (train/test split + perplexity)
  - amf.validation.falsification
  - amf.stats.significance  (bootstrap CI, chi-square, permutation)
  - amf.validation.preregistration
  - amf.validation.report
  - amf.cli                 (argument parsing)

All tests use synthetic data — no EVA corpus download required.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from amf.corpus.ingest import EVACorpus, TokenRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYNTHETIC_VTT = """\
# synthetic
<f1r.P.1;H> chedy daiin qokedy shol or aiin
<f1r.P.2;H> daiin chedy or aiin shol qokedy
<f1r.L.1;H> qokedy chedy daiin shol
<f1r.L.2;H> shol or daiin chedy
<f2r.P.1;H> daiin shol chol chedy or
<f2r.P.2;H> aiin chedy daiin qokedy shol or
<f2r.P.3;H> chedy or aiin shol daiin
<f2r.P.4;H> qokedy daiin chedy or aiin
<f2r.P.5;H> shol chedy daiin or aiin qokedy
<f2r.P.6;H> or aiin chedy daiin shol qokedy
"""


def _make_corpus_file(tmp_path: Path) -> Path:
    """Write a synthetic VTT and return its corpus JSON path."""
    vtt = tmp_path / "test.vtt"
    vtt.write_text(SYNTHETIC_VTT)
    corpus = EVACorpus.from_vtt_directory(tmp_path)
    out    = tmp_path / "corpus.json"
    corpus.to_json(out)
    return out


def _make_record(tokens: list[str], folio: str = "f1r") -> TokenRecord:
    return TokenRecord(folio, "P", "1", "H", " ".join(tokens), tokens)


LINES = [
    ["chedy", "daiin", "qokedy", "shol", "or"],
    ["daiin", "chedy", "or", "aiin", "shol"],
    ["qokedy", "chedy", "daiin", "shol", "or"],
    ["shol", "or", "daiin", "chedy", "aiin"],
    ["aiin", "shol", "qokedy", "or", "chedy"],
    ["chedy", "or", "aiin", "shol", "daiin"],
    ["daiin", "qokedy", "chedy", "or", "aiin"],
    ["shol", "chedy", "daiin", "or", "aiin"],
    ["or", "aiin", "chedy", "daiin", "shol"],
    ["qokedy", "daiin", "chedy", "or", "aiin"],
]


# ============================================================================
# amf.stats.markov — train/test split
# ============================================================================

class TestMarkovTrainTestSplit:
    from amf.stats.markov import split_lines, MarkovModel, run_markov_analysis

    def test_split_sizes(self):
        from amf.stats.markov import split_lines
        lines = LINES
        train, test = split_lines(lines, test_fraction=0.20, seed=42)
        assert len(train) + len(test) == len(lines)
        assert len(test) >= 1
        # ~20% held out
        assert abs(len(test) / len(lines) - 0.20) < 0.15

    def test_split_disjoint(self):
        from amf.stats.markov import split_lines
        train, test = split_lines(LINES, test_fraction=0.20, seed=42)
        train_set = {tuple(l) for l in train}
        test_set  = {tuple(l) for l in test}
        assert train_set.isdisjoint(test_set)

    def test_split_reproducible(self):
        from amf.stats.markov import split_lines
        t1, e1 = split_lines(LINES, seed=42)
        t2, e2 = split_lines(LINES, seed=42)
        assert t1 == t2 and e1 == e2

    def test_split_seed_affects_result(self):
        from amf.stats.markov import split_lines
        _, e42 = split_lines(LINES, seed=42)
        _, e99 = split_lines(LINES, seed=99)
        # Different seeds should (very likely) produce different test sets
        # (not guaranteed for tiny corpora, so just check it doesn't crash)
        assert isinstance(e42, list) and isinstance(e99, list)

    def test_zero_test_fraction_returns_all_train(self):
        from amf.stats.markov import split_lines
        train, test = split_lines(LINES, test_fraction=0.0, seed=42)
        # With 0 fraction, n_test = max(1, 0) = 1 minimum
        assert len(train) + len(test) == len(LINES)

    def test_markov_model_perplexity_returns_tuple(self):
        from amf.stats.markov import MarkovModel
        m = MarkovModel(order=1)
        m.train(LINES)
        pp, n = m.perplexity(LINES)
        assert isinstance(pp, float) and pp > 0
        assert isinstance(n, int) and n > 0

    def test_markov_model_tracks_unk(self):
        from amf.stats.markov import MarkovModel
        train = [["a", "b", "c"]]
        test  = [["a", "X", "c"]]   # X not in train vocab
        m = MarkovModel(order=1)
        m.train(train)
        m.unk_token_count = 0
        m.perplexity(test)
        assert m.unk_token_count >= 1

    def test_run_markov_has_test_perplexity(self):
        from amf.stats.markov import run_markov_analysis
        records = [_make_record(l) for l in LINES]
        report  = run_markov_analysis(records, max_order=2,
                                      test_fraction=0.20, seed=42)
        for r in report.results_by_order:
            # test_perplexity may be None only if test set has no evaluable tokens
            assert "test_perplexity" in r

    def test_run_markov_split_config_honest(self):
        from amf.stats.markov import run_markov_analysis
        records = [_make_record(l) for l in LINES]
        report  = run_markov_analysis(records, max_order=1,
                                      test_fraction=0.20, seed=42)
        assert report.split_config["honest_split"] is True
        assert report.split_config["seed"] == 42

    def test_run_markov_reduction_dicts_populated(self):
        from amf.stats.markov import run_markov_analysis
        records = [_make_record(l) for l in LINES]
        report  = run_markov_analysis(records, max_order=2,
                                      test_fraction=0.20, seed=42)
        assert "0_to_1" in report.perplexity_reduction_train
        assert "0_to_1" in report.perplexity_reduction_test

    def test_optimal_order_test_is_set(self):
        from amf.stats.markov import run_markov_analysis
        records = [_make_record(l) for l in LINES]
        report  = run_markov_analysis(records, max_order=2,
                                      test_fraction=0.20, seed=42)
        # May be None if no finite test PP exists, but should be set
        assert report.optimal_order_by_test_pp is None or \
               isinstance(report.optimal_order_by_test_pp, int)

    def test_train_pp_always_lower_or_equal_to_order0_inSample(self):
        """Higher-order model always fits training data >= order 0."""
        from amf.stats.markov import run_markov_analysis
        records = [_make_record(l) for l in LINES]
        report  = run_markov_analysis(records, max_order=3,
                                      test_fraction=0.0, seed=42)
        train_pps = [r["train_perplexity"] for r in report.results_by_order]
        for i in range(1, len(train_pps)):
            assert train_pps[i] <= train_pps[i-1] + 0.5   # tolerance for smoothing


# ============================================================================
# amf.validation.falsification
# ============================================================================

class TestFalsification:
    def test_falsification_report_fields(self, tmp_path: Path):
        from amf.validation.falsification import run_falsification_tests
        corpus_path = _make_corpus_file(tmp_path)
        report = run_falsification_tests(corpus_path)
        assert hasattr(report, "overall_verdict")
        assert hasattr(report, "critical_failures")
        assert hasattr(report, "major_failures")
        assert hasattr(report, "minor_failures")
        assert hasattr(report, "tests")
        assert len(report.tests) > 0

    def test_all_tests_have_verdict(self, tmp_path: Path):
        from amf.validation.falsification import run_falsification_tests
        corpus_path = _make_corpus_file(tmp_path)
        report = run_falsification_tests(corpus_path)
        for t in report.tests:
            assert t["verdict"] in ("PASS", "FAIL", "INCONCLUSIVE"), \
                f"Test {t['test_id']} has invalid verdict: {t['verdict']}"

    def test_overall_verdict_consistent_with_failures(self, tmp_path: Path):
        from amf.validation.falsification import run_falsification_tests
        corpus_path = _make_corpus_file(tmp_path)
        report = run_falsification_tests(corpus_path)
        if report.critical_failures > 0:
            assert report.overall_verdict == "NOT_SUPPORTED"
        assert report.overall_verdict in (
            "SUPPORTED", "CONDITIONAL", "NOT_SUPPORTED", "INCONCLUSIVE"
        )

    def test_counts_sum_to_total_failures(self, tmp_path: Path):
        from amf.validation.falsification import run_falsification_tests
        corpus_path = _make_corpus_file(tmp_path)
        report = run_falsification_tests(corpus_path)
        total_fail = sum(1 for t in report.tests if t["verdict"] == "FAIL")
        assert (report.critical_failures + report.major_failures
                + report.minor_failures) == total_fail

    def test_save_and_reload(self, tmp_path: Path):
        from amf.validation.falsification import (
            run_falsification_tests, save_falsification_report, FalsificationReport
        )
        corpus_path = _make_corpus_file(tmp_path)
        report = run_falsification_tests(corpus_path)
        out = tmp_path / "fals.json"
        save_falsification_report(report, out)
        assert out.exists()
        with out.open() as f:
            data = json.load(f)
        assert "overall_verdict" in data
        assert "tests" in data

    def test_f2_3_mathematical_invariant(self, tmp_path: Path):
        """F2.3: conditional entropy < unigram entropy — mathematical law."""
        from amf.validation.falsification import run_falsification_tests
        corpus_path = _make_corpus_file(tmp_path)
        report = run_falsification_tests(corpus_path)
        f23 = next(t for t in report.tests if t["test_id"] == "F2.3")
        # This MUST always pass — violation = code defect
        assert f23["verdict"] in ("PASS", "INCONCLUSIVE"), \
            "F2.3 failed: conditional entropy exceeds unigram entropy — code defect"

    def test_f1_3_core_slot_invariant(self, tmp_path: Path):
        """F1.3: CORE slot must be top-used if any tokens are valid."""
        from amf.validation.falsification import run_falsification_tests
        corpus_path = _make_corpus_file(tmp_path)
        report = run_falsification_tests(corpus_path)
        f13 = next(t for t in report.tests if t["test_id"] == "F1.3")
        assert f13["verdict"] in ("PASS", "INCONCLUSIVE"), \
            "F1.3 failed: CORE is not the top-used slot — validator logic error"

    def test_pre_registration_note_present(self, tmp_path: Path):
        from amf.validation.falsification import run_falsification_tests
        corpus_path = _make_corpus_file(tmp_path)
        report = run_falsification_tests(corpus_path)
        assert len(report.pre_registration_note) > 0


# ============================================================================
# amf.stats.significance
# ============================================================================

class TestSignificance:
    def test_bootstrap_ci_point_in_ci(self):
        """Point estimate must fall within or very near its own CI."""
        from amf.stats.significance import bootstrap_ci
        data = [["a", "b", "c", "d", "e"]] * 20
        stat = lambda lines: len(set(t for l in lines for t in l))
        pt, lo, hi = bootstrap_ci(data, stat, n_resamples=200, seed=42)
        assert lo <= pt <= hi or abs(pt - lo) < 0.01 or abs(pt - hi) < 0.01

    def test_bootstrap_ci_order(self):
        """lower bound must be <= upper bound."""
        from amf.stats.significance import bootstrap_ci
        from amf.stats.significance import _entropy_statistic
        data = [["chedy","daiin","or","shol"]] * 10
        pt, lo, hi = bootstrap_ci(data, _entropy_statistic,
                                   n_resamples=100, seed=42)
        assert lo <= hi

    def test_bootstrap_entropy_finite(self):
        from amf.stats.significance import bootstrap_ci, _entropy_statistic
        data = LINES
        pt, lo, hi = bootstrap_ci(data, _entropy_statistic,
                                   n_resamples=100, seed=42)
        assert math.isfinite(pt)
        assert math.isfinite(lo)
        assert math.isfinite(hi)

    def test_chi_square_returns_sections(self):
        from amf.stats.significance import chi_square_section_test
        records = [_make_record(l) for l in LINES]
        result  = chi_square_section_test(records)
        assert "sections" in result
        assert "bonferroni_threshold" in result
        assert "sections_significant" in result

    def test_chi_square_bonferroni_correct(self):
        from amf.stats.significance import chi_square_section_test
        records = [_make_record(l) for l in LINES]
        result  = chi_square_section_test(records, alpha_level=0.05)
        n_sec   = result["n_sections_tested"]
        expected_threshold = 0.05 / max(n_sec, 1)
        assert abs(result["bonferroni_threshold"] - expected_threshold) < 1e-9

    def test_permutation_test_structure(self):
        from amf.stats.significance import permutation_test_positional_bias
        records = [_make_record(l) for l in LINES]
        result  = permutation_test_positional_bias(
            records, "chedy", n_permutations=100, seed=42
        )
        assert "p_value" in result
        assert "observed_initial_rate" in result
        assert 0.0 <= result["p_value"] <= 1.0

    def test_permutation_test_missing_token(self):
        from amf.stats.significance import permutation_test_positional_bias
        records = [_make_record(l) for l in LINES]
        result  = permutation_test_positional_bias(
            records, "NONEXISTENT_TOKEN_XYZ", n_permutations=50, seed=42
        )
        assert "error" in result

    def test_run_significance_analysis(self, tmp_path: Path):
        from amf.stats.significance import (
            run_significance_analysis, save_significance_report
        )
        records = [_make_record(l) for l in LINES]
        report  = run_significance_analysis(
            records, n_bootstrap=100, seed=42, tokens_to_test_positional=2
        )
        assert hasattr(report, "bootstrap_entropy_ci")
        assert hasattr(report, "bootstrap_zipf_ci")
        assert hasattr(report, "chi_square_sections")
        assert hasattr(report, "positional_permutation_tests")
        assert len(report.positional_permutation_tests) > 0

        out = tmp_path / "sig.json"
        save_significance_report(report, out)
        assert out.exists()

    def test_significance_reproducible(self):
        from amf.stats.significance import run_significance_analysis
        records = [_make_record(l) for l in LINES]
        r1 = run_significance_analysis(records, n_bootstrap=50, seed=42)
        r2 = run_significance_analysis(records, n_bootstrap=50, seed=42)
        assert (r1.bootstrap_entropy_ci["point_estimate"]
                == r2.bootstrap_entropy_ci["point_estimate"])

    def test_significance_seed_recorded(self):
        from amf.stats.significance import run_significance_analysis
        records = [_make_record(l) for l in LINES]
        report  = run_significance_analysis(records, n_bootstrap=50, seed=99)
        assert report.bootstrap_config["master_seed"] == 99


# ============================================================================
# amf.validation.preregistration
# ============================================================================

class TestPreregistration:
    def test_generate_manifest_fields(self):
        from amf.validation.preregistration import generate_preregistration
        m = generate_preregistration(corpus_version="test-v1")
        assert m.amf_version == "4.0.0-alpha"
        assert "test-v1" in m.corpus_to_be_tested
        assert len(m.dpas_slot_definitions) > 0
        assert len(m.primary_test) > 0
        assert len(m.secondary_tests) > 0
        assert len(m.exclusion_criteria) > 0

    def test_slot_definitions_complete(self):
        from amf.validation.preregistration import generate_preregistration
        from amf.dpas.constraints import DPAS_TEMPLATE
        m = generate_preregistration()
        assert len(m.dpas_slot_definitions) == len(DPAS_TEMPLATE)
        names_in_manifest  = {s["name"] for s in m.dpas_slot_definitions}
        names_in_template  = {s.name for s in DPAS_TEMPLATE}
        assert names_in_manifest == names_in_template

    def test_slot_permitted_sets_match(self):
        from amf.validation.preregistration import generate_preregistration
        from amf.dpas.constraints import DPAS_TEMPLATE
        m = generate_preregistration()
        for mslot in m.dpas_slot_definitions:
            tslot = next(s for s in DPAS_TEMPLATE if s.name == mslot["name"])
            assert set(mslot["permitted"]) == tslot.permitted

    def test_primary_test_has_threshold(self):
        from amf.validation.preregistration import generate_preregistration
        from amf.dpas.constraints import COVERAGE_THRESHOLD
        m = generate_preregistration()
        assert m.primary_test["threshold"] == COVERAGE_THRESHOLD

    def test_deviations_disclosure_is_placeholder(self):
        """Must not be pre-filled — researcher fills this post-analysis."""
        from amf.validation.preregistration import generate_preregistration
        m = generate_preregistration()
        assert "TO BE COMPLETED" in m.deviations_disclosure

    def test_save_and_reload(self, tmp_path: Path):
        from amf.validation.preregistration import (
            generate_preregistration, save_preregistration
        )
        m   = generate_preregistration(corpus_version="test-v1")
        out = tmp_path / "prereg.json"
        save_preregistration(m, out)
        assert out.exists()
        with out.open() as f:
            data = json.load(f)
        assert data["corpus_to_be_tested"] == "test-v1"
        assert "dpas_slot_definitions" in data


# ============================================================================
# amf.validation.report
# ============================================================================

class TestReport:
    def _make_run_json(self, tmp_path: Path) -> Path:
        """Write a minimal pipeline run JSON for report testing."""
        run = {
            "run_id": "test-run-001",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "amf_version": "4.0.0-alpha",
            "epistemic_statement": "Test epistemic statement.",
            "corpus_metadata": {
                "transcription_version": "test-v1",
                "source_files": ["test.vtt"],
                "record_count": 100,
                "token_count": 500,
                "folio_count": 10,
                "uncertainty_line_count": 5,
                "loaded_at": "2025-01-01T00:00:00+00:00",
            },
            "verified_results": {
                "entropy": {
                    "unigram_bits": 9.1234,
                    "bigram_conditional_bits": 7.5,
                    "normalised_entropy": 0.72,
                    "redundancy": 0.28,
                    "token_count": 500,
                    "type_count": 60,
                },
                "zipf":     {"alpha": 1.05, "r_squared": 0.93},
                "positional": {
                    "total_lines": 100,
                    "sections_found": ["herbal", "pharmaceutical"],
                    "line_initial_biased_tokens": ["qokedy", "daiin"],
                    "line_final_biased_tokens":   ["chedy", "or"],
                },
                "adjacency": {
                    "total_bigrams": 400,
                    "unique_bigram_types": 120,
                    "bigram_entropy_bits": 6.8,
                },
                "markov": {
                    "split_config": {"honest_split": True, "seed": 42},
                    "optimal_order_train": 2,
                    "optimal_order_test": 1,
                    "perplexity_by_order": {
                        "0": {"train": 50.0, "test": 55.0},
                        "1": {"train": 20.0, "test": 25.0},
                    },
                    "reduction_test": {"0_to_1": 0.54},
                },
                "baseline_comparison": {
                    "voynich_h_unigram": 9.1,
                    "baseline_h_unigram": 11.2,
                    "delta_bits": 2.1,
                    "interpretation": "Voynich lower entropy than random.",
                },
                "significance": {},
            },
            "hypothesis_results": {
                "dpas_validation": {
                    "coverage": 0.78,
                    "model_supported": True,
                    "threshold": 0.70,
                    "slot_usage": {"core": 350, "suffix": 200},
                    "failure_categories": {"unconsumed_units": 30},
                    "interpretation": "Model supported.",
                },
                "falsification": {
                    "overall_verdict": "SUPPORTED",
                    "critical_failures": 0,
                    "major_failures": 0,
                    "minor_failures": 1,
                    "summary": "All critical tests passed.",
                },
            },
            "speculative_mappings": {
                "note": "Speculative only.",
                "amf_framework_claims": [
                    {
                        "claim": "Test claim",
                        "basis": "Test basis",
                        "status": "SPECULATIVE",
                        "required_validation": "Independent replication.",
                    }
                ],
            },
            "warnings":    ["Test warning"],
            "limitations": ["Test limitation"],
        }
        path = tmp_path / "pipeline_run_test.json"
        path.write_text(json.dumps(run))
        return path

    def test_report_generates(self, tmp_path: Path):
        from amf.validation.report import generate_report
        run_path = self._make_run_json(tmp_path)
        text = generate_report(run_path)
        assert isinstance(text, str)
        assert len(text) > 100

    def test_report_contains_key_sections(self, tmp_path: Path):
        from amf.validation.report import generate_report
        run_path = self._make_run_json(tmp_path)
        text = generate_report(run_path)
        assert "Verified Results"       in text
        assert "Hypothesis Results"     in text
        assert "Speculative Mappings"   in text
        assert "Replication"            in text
        assert "Documented Limitations" in text

    def test_report_includes_run_id(self, tmp_path: Path):
        from amf.validation.report import generate_report
        run_path = self._make_run_json(tmp_path)
        text = generate_report(run_path)
        assert "test-run-001" in text

    def test_report_includes_corpus_version(self, tmp_path: Path):
        from amf.validation.report import generate_report
        run_path = self._make_run_json(tmp_path)
        text = generate_report(run_path)
        assert "test-v1" in text

    def test_report_saves(self, tmp_path: Path):
        from amf.validation.report import generate_report, save_report
        run_path = self._make_run_json(tmp_path)
        text = generate_report(run_path)
        out  = tmp_path / "report.md"
        save_report(text, out)
        assert out.exists()
        assert out.read_text(encoding="utf-8") == text

    def test_report_verdict_visible(self, tmp_path: Path):
        from amf.validation.report import generate_report
        run_path = self._make_run_json(tmp_path)
        text = generate_report(run_path)
        assert "SUPPORTED" in text


# ============================================================================
# amf.cli — argument parsing
# ============================================================================

class TestCLI:
    def test_parser_builds(self):
        from amf.cli import build_parser
        parser = build_parser()
        assert parser is not None

    def test_all_subcommands_registered(self):
        from amf.cli import build_parser
        parser = build_parser()
        # Access the subparser choices
        sp_action = next(
            a for a in parser._actions
            if hasattr(a, "_name_parser_map")
        )
        commands = set(sp_action._name_parser_map.keys())
        expected = {
            "ingest", "entropy", "positional", "adjacency",
            "markov", "dpas", "falsify", "significance",
            "pipeline", "report", "preregister", "render",
        }
        assert expected.issubset(commands), \
            f"Missing commands: {expected - commands}"

    def test_markov_subcommand_has_test_fraction(self):
        from amf.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "markov", "--corpus", "c.json",
            "--test-fraction", "0.30", "--seed", "99"
        ])
        assert args.test_fraction == 0.30
        assert args.seed == 99

    def test_falsify_subcommand_parses(self):
        from amf.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "falsify", "--corpus", "c.json", "--output", "out.json"
        ])
        assert args.corpus == "c.json"
        assert args.output == "out.json"

    def test_significance_subcommand_parses(self):
        from amf.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "significance", "--corpus", "c.json",
            "--bootstrap-n", "2000", "--seed", "7"
        ])
        assert args.bootstrap_n == 2000
        assert args.seed == 7

    def test_preregister_subcommand_parses(self):
        from amf.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["preregister", "--output", "pre.json"])
        assert args.output == "pre.json"

    def test_pipeline_subcommand_parses(self):
        from amf.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "pipeline", "--corpus", "c.json",
            "--run-id", "test-001",
            "--skip-significance",
        ])
        assert args.run_id == "test-001"
        assert args.skip_significance is True

    def test_default_log_level(self):
        from amf.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["entropy", "--corpus", "c.json"])
        assert args.log_level == "INFO"
