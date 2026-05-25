"""
amf.corpus.normalize
====================

EVA Token Normalization
------------------------

EVA transcriptions contain variant spellings and encoding inconsistencies
across transcription sources and time periods. This module provides
normalization functions to produce a canonical form suitable for
consistent statistical analysis.

NORMALIZATION IS LOSSY
-----------------------
Every normalization decision discards information. For example:
- Stripping uncertainty markers ({}) discards transcriber doubt signals
- Case-folding loses rare uppercase distinctions
- Multigraph normalization assumes a specific segmentation

Each normalization step is documented with what information it discards
and when it is appropriate to use.

APPROACH: ALWAYS PRESERVE ORIGINALS
--------------------------------------
This module never modifies the source TokenRecord objects. Normalized
forms are produced as new strings or new lists. Callers should always
retain the original tokens alongside normalized forms for audit purposes.

REFERENCES
----------
Zandbergen, R. (2004). The Voynich Manuscript — EVA description.
  https://www.voynich.nu/extra/eva.html
"""

from __future__ import annotations

import re
from typing import Sequence


# ---------------------------------------------------------------------------
# Uncertainty marker handling
# ---------------------------------------------------------------------------

_UNCERTAINTY_RE = re.compile(r"[{}]")
_DAMAGE_RE = re.compile(r"[!?]")


def strip_uncertainty(token: str) -> str:
    """
    Remove EVA uncertainty markers ({ and }) from a token.

    The braces in the interlinear format enclose characters that the
    transcriber was uncertain about. Stripping them treats those
    characters as certain, which may introduce errors.

    USE WITH CAUTION. Always document when this function is applied.

    Parameters
    ----------
    token : str
        An EVA token string, possibly containing { } markers.

    Returns
    -------
    str
        Token with { and } removed; uncertain characters retained.

    Examples
    --------
    >>> strip_uncertainty("{o}kaiin")
    'okaiin'
    >>> strip_uncertainty("ch{e}dy")
    'chedy'
    """
    return _UNCERTAINTY_RE.sub("", token)


def strip_damage_markers(token: str) -> str:
    """
    Remove damage markers (! and ?) from a token.

    ! marks unreadable/damaged sections; ? marks uncertain word boundaries.
    Stripping these treats damaged sections as absent rather than unknown.

    Parameters
    ----------
    token : str

    Returns
    -------
    str
    """
    return _DAMAGE_RE.sub("", token)


def has_uncertainty(token: str) -> bool:
    """True if the token contains any EVA uncertainty marker."""
    return bool(_UNCERTAINTY_RE.search(token) or _DAMAGE_RE.search(token))


# ---------------------------------------------------------------------------
# Case normalization
# ---------------------------------------------------------------------------

def to_lowercase(token: str) -> str:
    """
    Fold token to lowercase EVA.

    Uppercase EVA characters appear rarely and their significance is
    debated. Most corpus analyses use lowercase only.

    This discards any information encoded in case distinctions.

    Returns
    -------
    str
    """
    return token.lower()


# ---------------------------------------------------------------------------
# Multigraph normalization
# ---------------------------------------------------------------------------

# Standard EVA multigraphs, sorted longest-first for greedy matching.
# These are the units that should be treated as atomic in analysis.
_MULTIGRAPHS_SORTED = sorted(
    ["cth", "ckh", "cph", "cfh", "eee", "iii",
     "ch", "sh", "ee", "ii", "ol", "or", "al", "ar",
     "ain", "aiin", "aiiin"],
    key=len,
    reverse=True,
)


def tokenize_to_units(token: str) -> list[str]:
    """
    Segment an EVA token into its constituent character units.

    Uses greedy longest-match to prefer multigraphs over individual
    characters, consistent with standard EVA practice.

    The segmentation is DETERMINISTIC but not unique — some tokens
    could be segmented differently. This function always applies
    the same greedy rule, so results are reproducible.

    Parameters
    ----------
    token : str
        Clean EVA token (uncertainty markers should be stripped first
        if character-level analysis is intended).

    Returns
    -------
    list[str]
        List of EVA units (individual characters and/or multigraphs).

    Examples
    --------
    >>> tokenize_to_units("chedy")
    ['ch', 'e', 'd', 'y']
    >>> tokenize_to_units("qokaiin")
    ['q', 'o', 'k', 'ain']  # note 'ain' treated as unit
    """
    units = []
    i = 0
    while i < len(token):
        matched = False
        for mg in _MULTIGRAPHS_SORTED:
            if token[i:i + len(mg)] == mg:
                units.append(mg)
                i += len(mg)
                matched = True
                break
        if not matched:
            units.append(token[i])
            i += 1
    return units


def units_to_token(units: list[str]) -> str:
    """Reconstruct a token string from its unit list."""
    return "".join(units)


# ---------------------------------------------------------------------------
# Canonical normalization pipeline
# ---------------------------------------------------------------------------

def normalize(
    token: str,
    strip_uncertain: bool = False,
    lowercase: bool = True,
) -> str:
    """
    Apply the canonical AMF v4.0 normalization pipeline to a token.

    Steps (in order):
    1. Lowercase (optional, default True)
    2. Strip uncertainty markers (optional, default False)

    Parameters
    ----------
    token : str
    strip_uncertain : bool
        Strip { } and ! ? markers. Default False — markers are preserved
        to maintain visibility of transcription uncertainty.
    lowercase : bool
        Fold to lowercase. Default True.

    Returns
    -------
    str
        Normalized token.
    """
    result = token
    if lowercase:
        result = to_lowercase(result)
    if strip_uncertain:
        result = strip_uncertainty(result)
        result = strip_damage_markers(result)
    return result


def normalize_sequence(
    tokens: Sequence[str],
    strip_uncertain: bool = False,
    lowercase: bool = True,
) -> list[str]:
    """
    Apply normalize() to a sequence of tokens.

    Tokens that become empty after normalization are excluded from
    the output — their removal is logged at WARNING level.

    Parameters
    ----------
    tokens : Sequence[str]
    strip_uncertain : bool
    lowercase : bool

    Returns
    -------
    list[str]
        Normalized tokens, with empty-after-normalization tokens removed.
    """
    import logging
    logger = logging.getLogger(__name__)

    result = []
    dropped = 0
    for tok in tokens:
        norm = normalize(tok, strip_uncertain=strip_uncertain, lowercase=lowercase)
        if norm:
            result.append(norm)
        else:
            dropped += 1

    if dropped > 0:
        logger.warning(
            "normalize_sequence: %d token(s) dropped (empty after normalization). "
            "This may indicate transcription artefacts.", dropped
        )

    return result


# ---------------------------------------------------------------------------
# Token classification helpers
# ---------------------------------------------------------------------------

def token_length(token: str, in_units: bool = True) -> int:
    """
    Compute token length, either in characters or in EVA units.

    Parameters
    ----------
    token : str
        A normalized EVA token.
    in_units : bool
        If True (default), count EVA multigraph units.
        If False, count raw characters.

    Returns
    -------
    int
    """
    if in_units:
        return len(tokenize_to_units(strip_uncertainty(token)))
    return len(strip_uncertainty(token))


def is_gallows_token(token: str) -> bool:
    """
    Return True if the token contains a gallows character (k, t, p, f).

    Gallows characters are structurally prominent tall EVA glyphs.
    Their presence is used in several DPAS slot analyses.
    """
    clean = strip_uncertainty(token.lower())
    units = tokenize_to_units(clean)
    gallows = {"k", "t", "p", "f", "ck", "ct", "cp", "cf", "sk", "st", "sp", "sf"}
    return any(u in gallows for u in units)
