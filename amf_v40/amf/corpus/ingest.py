# © 2026 Anadi Chakraborty
# Amanuensis Model Framework (AMF)
# Proprietary Research Software
"""
amf.corpus.ingest
=================

EVA Corpus Ingestion and Parsing
---------------------------------

This module loads and parses Voynich Manuscript EVA (European Voynich
Alphabet) transcriptions into structured Python data for downstream
statistical analysis.

SUPPORTED INPUT FORMATS
-----------------------
1.  Interlinear format (.vtt): The primary format used by the Zandbergen
    transcription (https://www.voynich.nu/transcr.html). Lines are tagged
    with folio, section, line identifiers and contain EVA tokens separated
    by spaces. Uncertain characters are marked with braces {x}.

2.  Plain-text EVA: One word per line or space-separated tokens, no
    structural metadata. Used by some secondary transcriptions.

EPISTEMIC NOTE
--------------
The EVA transcription system itself involves transcriber interpretation.
Different transcribers disagree on ~5–10% of characters. This module
records the transcription source version in output metadata so that
downstream analyses can be attributed to specific transcription decisions.

USAGE
-----
    from amf.corpus.ingest import EVACorpus

    corpus = EVACorpus.from_vtt_directory("data/eva/")
    print(corpus.summary())

    # Iterate tokens by folio section
    for record in corpus.records:
        print(record.folio, record.section, record.tokens)

CLI
---
    python -m amf.corpus.ingest --input data/eva/ --output data/processed/

"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EVA character inventory
# ---------------------------------------------------------------------------

# All characters recognized in the standard EVA transcription scheme.
# Source: Landini & Zandbergen EVA specification.
EVA_CHARS: frozenset[str] = frozenset(
    "abcdefghijklmnopqrstuvwxyz"          # Basic EVA alphabet
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"          # Uppercase variants (rare)
)

# Uncertain/damaged characters in the interlinear format
UNCERTAINTY_MARKERS = frozenset("{}")

# Token separator in interlinear format
TOKEN_SEP = " "

# Interlinear line-tag pattern: <f1r.P.1;H> or <f1r.C.1;C>
# Groups: folio, section_type, line_number, transcriber_id
_VTT_TAG_PATTERN = re.compile(
    r"<([^.]+)\.([^.]+)\.([^;>]+)(?:;([^>]*))?>"
)

# Gallows characters (structurally prominent tall glyphs in EVA)
GALLOWS = frozenset(["k", "t", "p", "f"])

# Bench characters (characters with descenders)
BENCH = frozenset(["g", "d"])


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TokenRecord:
    """
    A single transcribed line from the Voynich Manuscript.

    Attributes
    ----------
    folio : str
        Manuscript folio identifier (e.g. "f1r", "f67r2").
    section : str
        Section type within the folio. Common values:
        "P" = paragraph body, "L" = label, "C" = circular/radial text,
        "H" = header.
    line : str
        Line number within the section (e.g. "1", "2").
    transcriber : str
        Transcriber identifier from the interlinear file header.
        Important for attribution and reproducibility.
    raw_line : str
        The original unparsed line string (preserved for auditing).
    tokens : list[str]
        Space-separated EVA tokens. Uncertainty markers ({}) have been
        retained rather than stripped, so callers can decide how to
        handle uncertain characters.
    has_uncertainty : bool
        True if any token in this line contains { or } markers,
        indicating transcriber uncertainty.
    """
    folio: str
    section: str
    line: str
    transcriber: str
    raw_line: str
    tokens: list[str]
    has_uncertainty: bool = field(init=False)

    def __post_init__(self) -> None:
        self.has_uncertainty = any(
            c in tok for tok in self.tokens for c in UNCERTAINTY_MARKERS
        )

    @property
    def token_count(self) -> int:
        """Number of tokens (words) in this line."""
        return len(self.tokens)

    @property
    def clean_tokens(self) -> list[str]:
        """
        Tokens with uncertainty markers stripped.

        WARNING: This discards information. Use only for analyses where
        uncertain characters would introduce noise. Always document when
        using this property vs. the raw `tokens` field.
        """
        return [re.sub(r"[{}]", "", tok) for tok in self.tokens]


@dataclass
class CorpusMetadata:
    """
    Provenance and version information for a loaded corpus.

    Included in all output files so that results can be attributed
    to specific transcription decisions.
    """
    source_files: list[str]
    transcription_version: str
    loaded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    amf_version: str = "4.0.0-alpha"
    record_count: int = 0
    folio_count: int = 0
    token_count: int = 0
    uncertainty_line_count: int = 0
    notes: str = (
        "EVA transcription data. Statistical results depend on transcription "
        "version. Interpretive claims derived from this corpus are hypothetical."
    )


# ---------------------------------------------------------------------------
# Corpus class
# ---------------------------------------------------------------------------

class EVACorpus:
    """
    Container for a parsed EVA transcription corpus.

    Parameters
    ----------
    records : list[TokenRecord]
        All parsed line records.
    metadata : CorpusMetadata
        Provenance information.

    Example
    -------
    >>> corpus = EVACorpus.from_vtt_directory("data/eva/")
    >>> print(corpus.summary())
    >>> tokens = corpus.all_tokens()
    """

    def __init__(
        self,
        records: list[TokenRecord],
        metadata: CorpusMetadata,
    ) -> None:
        self.records = records
        self.metadata = metadata

    # ------------------------------------------------------------------
    # Class-level constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_vtt_directory(
        cls,
        directory: str | Path,
        transcription_version: str = "unknown",
        skip_uncertain: bool = False,
    ) -> "EVACorpus":
        """
        Load all .vtt (interlinear) files from a directory.

        Parameters
        ----------
        directory
            Path to directory containing .vtt transcription files.
        transcription_version
            Human-readable version string (e.g. "Zandbergen-2024-01").
            Include this to make outputs reproducible.
        skip_uncertain
            If True, lines containing uncertainty markers are excluded.
            Default False: all lines are included and uncertainty is
            flagged in the `has_uncertainty` field.

        Returns
        -------
        EVACorpus
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

        vtt_files = sorted(directory.glob("**/*.vtt"))
        if not vtt_files:
            # Fall back to .txt files
            vtt_files = sorted(directory.glob("**/*.txt"))

        if not vtt_files:
            raise FileNotFoundError(
                f"No .vtt or .txt files found in {directory}. "
                "Download the EVA transcription from voynich.nu and place "
                "files in the data/eva/ directory."
            )

        logger.info("Loading %d transcription file(s) from %s", len(vtt_files), directory)

        all_records: list[TokenRecord] = []
        for path in vtt_files:
            records = _parse_vtt_file(path, skip_uncertain=skip_uncertain)
            all_records.extend(records)
            logger.debug("  %s: %d lines", path.name, len(records))

        folios = {r.folio for r in all_records}
        total_tokens = sum(r.token_count for r in all_records)
        uncertain_lines = sum(1 for r in all_records if r.has_uncertainty)

        metadata = CorpusMetadata(
            source_files=[str(p) for p in vtt_files],
            transcription_version=transcription_version,
            record_count=len(all_records),
            folio_count=len(folios),
            token_count=total_tokens,
            uncertainty_line_count=uncertain_lines,
        )

        logger.info(
            "Corpus loaded: %d records, %d folios, %d tokens, "
            "%d lines with uncertainty",
            len(all_records), len(folios), total_tokens, uncertain_lines,
        )

        return cls(all_records, metadata)

    @classmethod
    def from_plain_text(
        cls,
        path: str | Path,
        transcription_version: str = "unknown",
    ) -> "EVACorpus":
        """
        Load a plain-text EVA file (one word per line or space-separated).

        For files without structural metadata, folio/section/line are
        set to "unknown". Use from_vtt_directory when structural
        information is available.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        tokens_flat = text.split()
        tokens_flat = [t for t in tokens_flat if t]

        # Treat entire file as one "record" per line of text
        lines = text.splitlines()
        records = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            records.append(TokenRecord(
                folio="unknown",
                section="unknown",
                line=str(i),
                transcriber="unknown",
                raw_line=line,
                tokens=toks,
            ))

        metadata = CorpusMetadata(
            source_files=[str(path)],
            transcription_version=transcription_version,
            record_count=len(records),
            folio_count=0,
            token_count=sum(r.token_count for r in records),
            uncertainty_line_count=sum(1 for r in records if r.has_uncertainty),
        )

        return cls(records, metadata)

    # ------------------------------------------------------------------
    # Iteration and access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[TokenRecord]:
        return iter(self.records)

    def by_folio(self, folio: str) -> list[TokenRecord]:
        """Return all records for a given folio identifier."""
        return [r for r in self.records if r.folio == folio]

    def by_section(self, section: str) -> list[TokenRecord]:
        """Return all records for a given section type."""
        return [r for r in self.records if r.section == section]

    def all_tokens(self, clean: bool = False) -> list[str]:
        """
        Flat list of all tokens across the entire corpus.

        Parameters
        ----------
        clean
            If True, uncertainty markers are stripped. See
            TokenRecord.clean_tokens for caveats.
        """
        if clean:
            return [tok for r in self.records for tok in r.clean_tokens]
        return [tok for r in self.records for tok in r.tokens]

    def folios(self) -> list[str]:
        """Sorted list of unique folio identifiers."""
        return sorted({r.folio for r in self.records})

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self, path: str | Path) -> None:
        """
        Serialize corpus to JSON, including provenance metadata.

        The JSON schema:
        {
          "metadata": { ... CorpusMetadata fields ... },
          "records": [ { ... TokenRecord fields ... }, ... ]
        }
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "metadata": asdict(self.metadata),
            "records": [asdict(r) for r in self.records],
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info("Corpus saved to %s", path)

    @classmethod
    def from_json(cls, path: str | Path) -> "EVACorpus":
        """Load a previously serialized corpus from JSON."""
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)

        metadata = CorpusMetadata(**payload["metadata"])

        # has_uncertainty is a computed field (set in __post_init__),
        # not a constructor parameter — strip it before unpacking.
        records = [
            TokenRecord(**{k: v for k, v in r.items() if k != "has_uncertainty"})
            for r in payload["records"]
        ]
        return cls(records, metadata)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable corpus summary."""
        lines_with_uncertainty = sum(1 for r in self.records if r.has_uncertainty)
        unique_types = len(set(self.all_tokens(clean=True)))
        lines = [
            "AMF v4.0 — EVA Corpus Summary",
            "=" * 40,
            f"Transcription version : {self.metadata.transcription_version}",
            f"Loaded at             : {self.metadata.loaded_at}",
            f"Source files          : {len(self.metadata.source_files)}",
            f"Records (lines)       : {len(self.records):,}",
            f"Folios                : {self.metadata.folio_count:,}",
            f"Total tokens          : {self.metadata.token_count:,}",
            f"Unique token types    : {unique_types:,}",
            f"Lines with uncertainty: {lines_with_uncertainty:,} "
            f"({100*lines_with_uncertainty/max(len(self.records),1):.1f}%)",
            "",
            "EPISTEMIC NOTE: Token counts depend on transcription version.",
            "Interpretations derived from this corpus are hypothetical.",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal parsing functions
# ---------------------------------------------------------------------------

def _parse_vtt_file(
    path: Path,
    skip_uncertain: bool = False,
) -> list[TokenRecord]:
    """
    Parse a single .vtt interlinear transcription file.

    The interlinear format uses angle-bracket tags to identify
    folio/section/line, followed by space-separated EVA tokens.

    Example line from a .vtt file:
        <f1r.P.1;H> fachys ykal ar ataiin shol shory
        <f1r.P.2;H> sory shtory or qokaiin or

    The ';H' suffix is the transcriber identifier. Not all files
    use this convention — we handle missing transcriber IDs gracefully.
    """
    records: list[TokenRecord] = []
    current_folio = "unknown"
    current_section = "unknown"
    current_line = "0"
    current_transcriber = "unknown"

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Some older files use Latin-1
        text = path.read_text(encoding="latin-1")

    for raw_line in text.splitlines():
        raw_line = raw_line.strip()

        # Skip empty lines and comments
        if not raw_line or raw_line.startswith("#"):
            continue

        # Detect tag + content on same line: "<f1r.P.1;H> token1 token2"
        tag_match = _VTT_TAG_PATTERN.match(raw_line)
        if tag_match:
            current_folio = tag_match.group(1)
            current_section = tag_match.group(2)
            current_line = tag_match.group(3)
            current_transcriber = tag_match.group(4) or "unknown"

            # Tokens follow the tag on the same line
            remainder = raw_line[tag_match.end():].strip()
            tokens = _extract_tokens(remainder)
        else:
            # No tag on this line — continuation or plain-text format
            tokens = _extract_tokens(raw_line)

        if not tokens:
            continue

        record = TokenRecord(
            folio=current_folio,
            section=current_section,
            line=current_line,
            transcriber=current_transcriber,
            raw_line=raw_line,
            tokens=tokens,
        )

        if skip_uncertain and record.has_uncertainty:
            continue

        records.append(record)

    return records


def _extract_tokens(text: str) -> list[str]:
    """
    Split a text string into EVA tokens.

    Tokens are separated by spaces. Special characters:
    - '.' separates words within a run (treated as token separator)
    - '-' is a line-break marker (treated as token separator)
    - '!' marks damaged/unreadable sections (token retained as-is)
    - '?' marks uncertain word boundaries

    Empty strings after splitting are removed.
    """
    # Normalize line-break markers and word separators to spaces
    text = text.replace(".", " ").replace("-", " ")
    tokens = text.split()
    # Keep non-empty tokens; preserve uncertainty markers intact
    return [t for t in tokens if t and not t.isspace()]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI: ingest EVA corpus and save processed JSON."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AMF v4.0 — EVA corpus ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m amf.corpus.ingest --input data/eva/ --output data/processed/corpus.json
  python -m amf.corpus.ingest --input data/eva/ --version Zandbergen-2024-01
        """,
    )
    parser.add_argument(
        "--input", required=True,
        help="Directory containing .vtt transcription files, or path to a single .txt file",
    )
    parser.add_argument(
        "--output", default="data/processed/corpus.json",
        help="Output path for processed JSON corpus (default: data/processed/corpus.json)",
    )
    parser.add_argument(
        "--version", default="unknown",
        help="Transcription version string for provenance metadata",
    )
    parser.add_argument(
        "--skip-uncertain", action="store_true",
        help="Exclude lines containing EVA uncertainty markers",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    input_path = Path(args.input)
    if input_path.is_dir():
        corpus = EVACorpus.from_vtt_directory(
            input_path,
            transcription_version=args.version,
            skip_uncertain=args.skip_uncertain,
        )
    else:
        corpus = EVACorpus.from_plain_text(input_path, args.version)

    print(corpus.summary())
    corpus.to_json(args.output)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
