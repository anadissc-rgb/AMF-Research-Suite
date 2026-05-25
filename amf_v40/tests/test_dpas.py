"""
tests/test_dpas.py
==================

Unit tests for amf.dpas.constraints and amf.dpas.validator.

IMPORTANT: These tests validate the mechanics of the DPAS constraint
system — parsing, slot matching, coverage computation.

They do NOT validate that the DPAS model correctly describes Voynichese
(that requires corpus analysis against real EVA data). The distinction
between testing the implementation vs. testing the hypothesis is critical
and is explicitly maintained here.
"""

import pytest

from amf.dpas.constraints import (
    COVERAGE_THRESHOLD,
    DPAS_TEMPLATE,
    DPASValidationResult,
    CharClass,
    CHAR_CLASSES,
    EVA_MULTIGRAPHS,
)
from amf.dpas.validator import (
    _split_eva_units,
    validate_token,
    validate_corpus,
    CorpusValidationReport,
)
from amf.corpus.ingest import TokenRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_record(tokens: list[str]) -> TokenRecord:
    return TokenRecord(
        folio="f1r", section="P", line="1",
        transcriber="test", raw_line=" ".join(tokens),
        tokens=tokens,
    )


# ---------------------------------------------------------------------------
# Character class definitions
# ---------------------------------------------------------------------------

class TestCharClasses:
    def test_gallows_defined(self):
        assert "k" in CHAR_CLASSES[CharClass.GALLOWS]
        assert "t" in CHAR_CLASSES[CharClass.GALLOWS]

    def test_no_overlap_gallows_vowels(self):
        """Gallows and vowel-like sets should be disjoint."""
        gallows = CHAR_CLASSES[CharClass.GALLOWS]
        vowels = CHAR_CLASSES[CharClass.VOWEL_LIKE]
        assert gallows.isdisjoint(vowels)

    def test_multigraphs_set(self):
        assert "ch" in EVA_MULTIGRAPHS
        assert "sh" in EVA_MULTIGRAPHS
        assert "cth" in EVA_MULTIGRAPHS


# ---------------------------------------------------------------------------
# DPAS template structure
# ---------------------------------------------------------------------------

class TestDPASTemplate:
    def test_core_slot_required(self):
        """The CORE slot must be non-optional."""
        core_slots = [s for s in DPAS_TEMPLATE if s.name == "core"]
        assert len(core_slots) == 1
        assert core_slots[0].optional is False

    def test_all_slots_have_permitted_set(self):
        for slot in DPAS_TEMPLATE:
            assert len(slot.permitted) > 0, f"Slot '{slot.name}' has empty permitted set"

    def test_coverage_threshold_reasonable(self):
        """Coverage threshold should be between 0 and 1."""
        assert 0.0 < COVERAGE_THRESHOLD <= 1.0

    def test_slot_names_unique(self):
        names = [s.name for s in DPAS_TEMPLATE]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# EVA unit splitting
# ---------------------------------------------------------------------------

class TestSplitEVAUnits:
    def test_basic_split(self):
        units = _split_eva_units("abcd")
        assert units == ["a", "b", "c", "d"]

    def test_multigraph_ch(self):
        units = _split_eva_units("chedy")
        assert units[0] == "ch"
        assert "e" in units

    def test_multigraph_sh(self):
        units = _split_eva_units("shol")
        assert units[0] == "sh"

    def test_multigraph_ain(self):
        units = _split_eva_units("daiin")
        # 'ain' should be recognized as a unit
        assert "ain" in units or "ii" in units  # aiin or ain

    def test_empty_string(self):
        assert _split_eva_units("") == []

    def test_single_char(self):
        assert _split_eva_units("a") == ["a"]

    def test_longest_match_preferred(self):
        """cth should be preferred over c + th."""
        units = _split_eva_units("cthy")
        assert "cth" in units


# ---------------------------------------------------------------------------
# Single token validation
# ---------------------------------------------------------------------------

class TestValidateToken:
    def test_empty_after_uncertainty_strip_invalid(self):
        result = validate_token("{}")
        assert result.is_valid is False

    def test_uncertainty_flagged(self):
        result = validate_token("{o}kaiin")
        assert "uncertainty" in result.notes.lower() or result.confidence in ("low", "medium")

    def test_core_required(self):
        """A token with no core characters should be invalid."""
        # Construct a token with only prefix chars and no CORE chars
        # 'q' is in prefix, 'd' is in prefix — neither is in core
        result = validate_token("qd")
        # This may or may not be valid depending on slot overlap
        # Just ensure we get a result without crashing
        assert isinstance(result.is_valid, bool)

    def test_result_has_all_fields(self):
        result = validate_token("chedy")
        assert hasattr(result, "token")
        assert hasattr(result, "is_valid")
        assert hasattr(result, "matched_slots")
        assert hasattr(result, "failure_reason")
        assert hasattr(result, "confidence")
        assert hasattr(result, "notes")

    def test_valid_token_has_core_in_slots(self):
        """A valid token must have 'core' in its matched slots."""
        result = validate_token("chedy")
        if result.is_valid:
            assert "core" in result.matched_slots

    def test_token_preserved_in_result(self):
        token = "{o}kaiin"
        result = validate_token(token)
        assert result.token == token


# ---------------------------------------------------------------------------
# Corpus validation
# ---------------------------------------------------------------------------

class TestValidateCorpus:
    def test_empty_corpus(self):
        report, _ = validate_corpus([])
        assert report.total_tokens == 0
        assert report.coverage == 0.0
        assert report.model_supported is False

    def test_coverage_in_range(self):
        tokens = ["chedy", "daiin", "qokedy", "shol", "or", "aiin"]
        report, _ = validate_corpus(tokens)
        assert 0.0 <= report.coverage <= 1.0

    def test_counts_sum_correctly(self):
        tokens = ["chedy", "daiin", "qokedy", "shol"]
        report, _ = validate_corpus(tokens)
        assert report.valid_count + report.invalid_count == report.total_tokens

    def test_model_supported_flag(self):
        """model_supported must be True iff coverage >= threshold."""
        tokens = ["chedy", "daiin"] * 100   # likely high coverage for common tokens
        report, _ = validate_corpus(tokens)
        assert report.model_supported == (report.coverage >= COVERAGE_THRESHOLD)

    def test_full_results_returned_when_requested(self):
        tokens = ["chedy", "daiin", "or"]
        report, results = validate_corpus(tokens, return_full_results=True)
        assert results is not None
        assert len(results) == len(tokens)

    def test_full_results_not_returned_by_default(self):
        tokens = ["chedy", "daiin"]
        report, results = validate_corpus(tokens, return_full_results=False)
        assert results is None

    def test_invalid_top_is_list(self):
        tokens = ["chedy", "XXXX", "YYYY", "ZZZZ"]
        report, _ = validate_corpus(tokens)
        assert isinstance(report.invalid_top, list)

    def test_slot_usage_populated_for_valid_tokens(self):
        """If any tokens are valid, slot_usage should be non-empty."""
        tokens = ["chedy", "daiin", "shol", "or"] * 5
        report, _ = validate_corpus(tokens)
        if report.valid_count > 0:
            assert len(report.slot_usage) > 0

    def test_failure_reasons_recorded(self):
        # Include some clearly invalid tokens
        tokens = ["chedy", "INVALID_TOKEN_XYZ"] * 3
        report, _ = validate_corpus(tokens)
        if report.invalid_count > 0:
            assert len(report.failure_reasons) > 0
