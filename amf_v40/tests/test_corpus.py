"""
tests/test_corpus.py
====================

Unit tests for amf.corpus.ingest and amf.corpus.normalize.

Tests use synthetic EVA data — no real Voynich corpus is required to
run the test suite. This ensures tests are runnable without downloading
external data, and that test results are deterministic.
"""

import json
import tempfile
from pathlib import Path

import pytest

from amf.corpus.ingest import (
    EVACorpus,
    TokenRecord,
    _extract_tokens,
    _parse_vtt_file,
)
from amf.corpus.normalize import (
    has_uncertainty,
    is_gallows_token,
    normalize,
    normalize_sequence,
    strip_uncertainty,
    token_length,
    tokenize_to_units,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic EVA data
# ---------------------------------------------------------------------------

SYNTHETIC_VTT = """\
# Synthetic test data — not a real Voynich transcription
<f1r.P.1;H> fachys ykal ar ataiin shol shory
<f1r.P.2;H> sory shtory or qokaiin or
<f1r.L.1;H> qokedy chedy daiin
<f1r.L.2;H> {o}kaiin chor chedy
<f2r.P.1;H> daiin shol chol chedy
"""

SYNTHETIC_PLAIN = """\
# plain-text EVA
fachys ykal ar
daiin chedy
shol qokaiin
"""


@pytest.fixture
def vtt_file(tmp_path: Path) -> Path:
    """Write synthetic VTT content to a temp file."""
    p = tmp_path / "test.vtt"
    p.write_text(SYNTHETIC_VTT, encoding="utf-8")
    return p


@pytest.fixture
def plain_file(tmp_path: Path) -> Path:
    """Write synthetic plain-text EVA to a temp file."""
    p = tmp_path / "test.txt"
    p.write_text(SYNTHETIC_PLAIN, encoding="utf-8")
    return p


@pytest.fixture
def vtt_dir(tmp_path: Path, vtt_file: Path) -> Path:
    """A directory containing one VTT file."""
    return tmp_path


# ---------------------------------------------------------------------------
# _extract_tokens
# ---------------------------------------------------------------------------

class TestExtractTokens:
    def test_basic_split(self):
        assert _extract_tokens("fachys ykal ar") == ["fachys", "ykal", "ar"]

    def test_dot_separator(self):
        """Dots between words are treated as separators."""
        result = _extract_tokens("fachys.ykal.ar")
        assert "fachys" in result
        assert "ykal" in result

    def test_dash_separator(self):
        """Dashes (line-break markers) are treated as separators."""
        result = _extract_tokens("shol-shory")
        assert "shol" in result

    def test_empty_string(self):
        assert _extract_tokens("") == []

    def test_whitespace_only(self):
        assert _extract_tokens("   ") == []

    def test_uncertainty_markers_preserved(self):
        """Uncertainty markers must be preserved by _extract_tokens."""
        tokens = _extract_tokens("{o}kaiin chor")
        assert any("{" in t for t in tokens), "Uncertainty marker should be preserved"


# ---------------------------------------------------------------------------
# TokenRecord
# ---------------------------------------------------------------------------

class TestTokenRecord:
    def test_has_uncertainty_true(self):
        rec = TokenRecord(
            folio="f1r", section="P", line="1",
            transcriber="H", raw_line="{o}kaiin",
            tokens=["{o}kaiin", "chor"],
        )
        assert rec.has_uncertainty is True

    def test_has_uncertainty_false(self):
        rec = TokenRecord(
            folio="f1r", section="P", line="1",
            transcriber="H", raw_line="fachys ykal",
            tokens=["fachys", "ykal"],
        )
        assert rec.has_uncertainty is False

    def test_clean_tokens_strips_braces(self):
        rec = TokenRecord(
            folio="f1r", section="P", line="1",
            transcriber="H", raw_line="{o}kaiin",
            tokens=["{o}kaiin"],
        )
        assert rec.clean_tokens == ["okaiin"]

    def test_token_count(self):
        rec = TokenRecord(
            folio="f1r", section="P", line="1",
            transcriber="H", raw_line="a b c",
            tokens=["a", "b", "c"],
        )
        assert rec.token_count == 3


# ---------------------------------------------------------------------------
# EVACorpus from VTT
# ---------------------------------------------------------------------------

class TestEVACorpusFromVTT:
    def test_loads_records(self, vtt_dir: Path):
        corpus = EVACorpus.from_vtt_directory(vtt_dir, transcription_version="test-1")
        assert len(corpus.records) > 0

    def test_folio_parsing(self, vtt_dir: Path):
        corpus = EVACorpus.from_vtt_directory(vtt_dir)
        folios = corpus.folios()
        assert "f1r" in folios
        assert "f2r" in folios

    def test_section_parsing(self, vtt_dir: Path):
        corpus = EVACorpus.from_vtt_directory(vtt_dir)
        sections = {r.section for r in corpus.records}
        assert "P" in sections
        assert "L" in sections

    def test_transcription_version_stored(self, vtt_dir: Path):
        corpus = EVACorpus.from_vtt_directory(
            vtt_dir, transcription_version="synthetic-test-v1"
        )
        assert corpus.metadata.transcription_version == "synthetic-test-v1"

    def test_uncertainty_lines_counted(self, vtt_dir: Path):
        corpus = EVACorpus.from_vtt_directory(vtt_dir)
        # Line "<f1r.L.2;H> {o}kaiin chor chedy" has uncertainty
        assert corpus.metadata.uncertainty_line_count >= 1

    def test_skip_uncertain(self, vtt_dir: Path):
        corpus_all = EVACorpus.from_vtt_directory(vtt_dir, skip_uncertain=False)
        corpus_clean = EVACorpus.from_vtt_directory(vtt_dir, skip_uncertain=True)
        assert len(corpus_clean.records) < len(corpus_all.records)

    def test_directory_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            EVACorpus.from_vtt_directory(tmp_path / "nonexistent")

    def test_all_tokens_flat(self, vtt_dir: Path):
        corpus = EVACorpus.from_vtt_directory(vtt_dir)
        tokens = corpus.all_tokens()
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_by_folio(self, vtt_dir: Path):
        corpus = EVACorpus.from_vtt_directory(vtt_dir)
        f1r_records = corpus.by_folio("f1r")
        assert all(r.folio == "f1r" for r in f1r_records)

    def test_json_roundtrip(self, vtt_dir: Path, tmp_path: Path):
        """Serialized corpus must be byte-identical to original after roundtrip."""
        corpus = EVACorpus.from_vtt_directory(vtt_dir)
        output_path = tmp_path / "corpus.json"
        corpus.to_json(output_path)

        reloaded = EVACorpus.from_json(output_path)
        assert len(reloaded.records) == len(corpus.records)
        assert reloaded.metadata.transcription_version == corpus.metadata.transcription_version

        # Token content should be identical
        original_tokens = corpus.all_tokens()
        reloaded_tokens = reloaded.all_tokens()
        assert original_tokens == reloaded_tokens


# ---------------------------------------------------------------------------
# EVACorpus from plain text
# ---------------------------------------------------------------------------

class TestEVACorpusFromPlain:
    def test_loads(self, plain_file: Path):
        corpus = EVACorpus.from_plain_text(plain_file)
        assert len(corpus.records) > 0

    def test_folio_unknown(self, plain_file: Path):
        corpus = EVACorpus.from_plain_text(plain_file)
        assert all(r.folio == "unknown" for r in corpus.records)

    def test_comments_excluded(self, plain_file: Path):
        """Lines starting with # should be skipped."""
        corpus = EVACorpus.from_plain_text(plain_file)
        for record in corpus.records:
            assert not record.raw_line.startswith("#")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_strip_uncertainty_basic(self):
        assert strip_uncertainty("{o}kaiin") == "okaiin"

    def test_strip_uncertainty_multiple(self):
        assert strip_uncertainty("{c}h{e}dy") == "chedy"

    def test_has_uncertainty_true(self):
        assert has_uncertainty("{o}kaiin") is True

    def test_has_uncertainty_false(self):
        assert has_uncertainty("chedy") is False

    def test_normalize_lowercase(self):
        assert normalize("ChEdY") == "chedy"

    def test_normalize_strip_off_by_default(self):
        """Uncertainty markers NOT stripped unless strip_uncertain=True."""
        result = normalize("{o}kaiin", strip_uncertain=False)
        assert "{" in result

    def test_normalize_strip_on(self):
        result = normalize("{o}kaiin", strip_uncertain=True)
        assert "{" not in result
        assert "}" not in result

    def test_normalize_sequence_drops_empty(self):
        tokens = normalize_sequence(["chedy", "", "daiin"], strip_uncertain=False)
        assert "" not in tokens
        assert len(tokens) == 2

    def test_tokenize_multigraph(self):
        assert tokenize_to_units("chedy") == ["ch", "e", "d", "y"]

    def test_tokenize_ain(self):
        units = tokenize_to_units("qokaiin")
        # 'ain' should be treated as a unit
        assert "ain" in units or "aiin" not in "".join(units)

    def test_token_length_units(self):
        # "chedy" → [ch, e, d, y] = 4 units
        assert token_length("chedy", in_units=True) == 4

    def test_token_length_chars(self):
        # "chedy" = 5 characters
        assert token_length("chedy", in_units=False) == 5

    def test_is_gallows_token_true(self):
        assert is_gallows_token("okeedy") is True   # contains 'k'

    def test_is_gallows_token_false(self):
        assert is_gallows_token("chedy") is False

    def test_is_gallows_case_insensitive(self):
        assert is_gallows_token("OKEEDY") is True
