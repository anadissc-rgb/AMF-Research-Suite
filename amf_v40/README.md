# AMF v4.0 — Amanuensis Model Framework

**A computational procedural interpretation framework for Voynichese technical notation**

[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Status: Experimental](https://img.shields.io/badge/status-experimental-orange.svg)]()

---

## ⚠ Epistemic Status

This framework is **experimental**. No claim is made that the Voynich Manuscript has
been deciphered. AMF v4.0 is a constraint-based computational model that proposes
procedural interpretations of Voynichese token structure. All outputs are **hypotheses
subject to independent validation**, not translations.

Specifically:
- Statistical results are reproducible and verifiable
- Interpretive mappings are **explicitly labelled as speculative**
- No confidence claims are made without quantitative grounding
- Limitations are documented inline throughout the codebase

---

## What AMF v4.0 Is

AMF v4.0 is a **digital humanities research tool** that applies:

1. **Corpus-scale statistical analysis** of the Voynich EVA transcription
2. **Constraint-based tokenization** via the DPAS (Discrete Positional Alignment
   System) model
3. **Entropy and Zipf analysis** benchmarked against natural languages and ciphers
4. **Positional and adjacency statistics** at word, line, page, and section level
5. **Procedural notation hypothesis**: the conjecture that Voynichese encodes
   structured technical content (possibly botanical/pharmaceutical) via a scribe
   following fixed compositional rules

AMF v4.0 is **not**:
- A proven decipherment
- A translation system
- A claim about the manuscript's language or origin
- A finished product

---

## Repository Structure

```
amf_v40/
├── amf/                        # Core library
│   ├── corpus/
│   │   ├── ingest.py           # EVA corpus loading & parsing
│   │   ├── tokenize.py         # Voynichese tokenization
│   │   └── normalize.py        # EVA normalization utilities
│   ├── stats/
│   │   ├── entropy.py          # Shannon entropy, conditional entropy
│   │   ├── zipf.py             # Zipf/Mandelbrot distribution fitting
│   │   ├── positional.py       # Positional frequency analysis
│   │   ├── adjacency.py        # Bigram/trigram adjacency matrices
│   │   └── markov.py           # Markov chain order analysis
│   ├── dpas/
│   │   ├── constraints.py      # DPAS constraint definitions
│   │   └── validator.py        # Token constraint validation
│   ├── render/
│   │   └── generate_bg.py      # Manuscript background renderer
│   └── validation/
│       └── pipeline.py         # End-to-end validation pipeline
├── notebooks/
│   ├── 01_corpus_exploration.ipynb
│   ├── 02_entropy_analysis.ipynb
│   ├── 03_zipf_analysis.ipynb
│   ├── 04_positional_statistics.ipynb
│   └── 05_dpas_validation.ipynb
├── data/
│   └── README.md               # Data sourcing & licensing instructions
├── docs/
│   ├── methodology.md          # Full methodology description
│   ├── dpas_spec.md            # DPAS formal specification
│   ├── limitations.md          # Documented limitations
│   └── glossary.md             # AMF terminology glossary
├── tests/                      # Unit + integration tests
├── scripts/                    # CLI utilities
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

---

## Author

**Anadi Chakraborty**
Junior Assistant, Srikishan Sarda College

---



### 1. Clone & set up environment

```bash
git clone https://github.com/anadiChakraborty/amf_v40.git
cd amf_v40

# Option A: pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Option B: Docker
docker compose up --build
```

### 2. Obtain the EVA corpus

The Voynich Manuscript EVA transcription is available under open access:

- **Primary source**: [Zandbergen EVA transcription](https://www.voynich.nu/transcr.html)
  (interlinear format, `.vtt` files)
- **Alternative**: [voynich-text on GitHub](https://github.com/jedahan/voynich-text)
  (plain-text EVA format)

Place transcription files in `data/eva/`. See `data/README.md` for format details.

### 3. Run statistical analysis

```bash
# Corpus ingestion
python -m amf.corpus.ingest --input data/eva/ --output data/processed/

# Entropy analysis
python -m amf.stats.entropy --corpus data/processed/corpus.json

# Full pipeline
python -m amf.validation.pipeline --config config/default.yaml
```

### 4. Launch notebooks

```bash
jupyter lab notebooks/
```

---

## Reproducibility

All analyses are deterministic given the same input corpus. Random seeds are fixed
and documented. Results vary across EVA transcription versions — the version used is
always logged in output metadata.

See `docs/methodology.md` for full reproducibility instructions.

---

## Limitations

Key limitations are documented in `docs/limitations.md`. In brief:

- The EVA transcription itself contains transcriber uncertainty
- DPAS constraints are *hypothesized*, not linguistically proven
- Statistical similarity to natural language is necessary but not sufficient evidence
- Interpretive mappings from token families to semantic domains are speculative
- The framework cannot currently distinguish between: natural language, cipher,
  constructed notation, or glossolalia

---

## License

Proprietary. All rights reserved. See [LICENSE](LICENSE).

Use, reproduction, or distribution of any part of this repository without
the prior written permission of the copyright holder is prohibited.
For licensing or research-use inquiries, contact Anadi Chakraborty,
Junior Assistant, Srikishan Sarda College.

---

## Citation

If you use AMF v4.0 in research, please cite:

```bibtex
@software{amf_v40,
  author = {Chakraborty, Anadi},
  title  = {AMF v4.0: Amanuensis Model Framework for Voynichese Analysis},
  year   = {2025},
  institution = {Srikishan Sarda College},
  note   = {Experimental computational humanities framework.
            No decipherment claim is made.},
  url    = {https://github.com/anadiChakraborty/amf_v40}
}
```
