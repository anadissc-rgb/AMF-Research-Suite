# AMF v4.0 — Glossary of Terms

Terms are defined as used within the AMF v4.0 framework.
Where a term has an established meaning in the broader literature,
that meaning is noted alongside the AMF-specific usage.

---

**AMF** — Amanuensis Model Framework. The overall computational
framework for Voynichese analysis. The name reflects the hypothesis
that Voynichese was produced by a scribe (amanuensis) following
procedural compositional rules.

**Coverage** — The fraction of EVA corpus tokens that satisfy the DPAS
template. The primary falsifiability metric of AMF v4.0.

**DPAS** — Discrete Positional Alignment System. The formal constraint
model that defines which token sequences are "well-formed" under AMF.
A *hypothesis*, not a proven grammar.

**EVA** — European Voynich Alphabet. The standard transcription
alphabet for the Voynich Manuscript, defined by Landini and Zandbergen.

**Folio** — A single leaf (two pages) of the Voynich Manuscript.
Identified by number and recto/verso (e.g. f1r = folio 1, recto side).

**Gallows characters** — EVA characters k, t, p, f. Structurally tall
glyphs that appear with restricted positional distributions in the corpus.

**Interlinear format (.vtt)** — The transcription format used by the
Zandbergen EVA transcription. Structural tags identify folio/section/line.

**Multigraph** — An EVA character unit consisting of multiple letters
treated as a single glyph (e.g. "ch", "sh", "cth").

**PMI** — Pointwise Mutual Information. A measure of how much more often
two tokens co-occur as bigrams than their individual frequencies predict.

**Positional bias** — A token is positionally biased if it appears
at a specific line position (initial, final) more often than its overall
frequency would predict.

**Slot** — One positional element in the DPAS template (PREFIX, GALLOWS,
CORE, SUFFIX). Each slot has a defined permitted character set.

**Token** — A single EVA "word" — a sequence of EVA characters separated
by whitespace in the transcription.

**Token family** — A group of tokens sharing a common structural pattern
(e.g. all tokens matching the CORE="ch" + SUFFIX="y" pattern). AMF uses
token families as analysis units.

**Type** — A unique token string. Contrast with *token* (an occurrence).
Type-token ratio measures lexical diversity.

**Voynichese** — The unidentified writing system of the Voynich
Manuscript. Used neutrally to refer to the script and its tokens
without implying any linguistic interpretation.
