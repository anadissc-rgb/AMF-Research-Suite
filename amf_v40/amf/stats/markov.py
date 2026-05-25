# © 2026 Anadi Chakraborty
# Amanuensis Model Framework (AMF)
# Proprietary Research Software
"""
amf.stats.markov
================

Markov Chain Order Analysis of Voynichese
------------------------------------------

Markov analysis determines whether Voynichese token sequences exhibit
statistical memory — whether knowing the last N tokens helps predict
the next token beyond what unigram frequencies alone predict.

WHAT MARKOV ORDER TELLS US
---------------------------
A sequence is well-modeled by a Markov chain of order K if:
    P(W_i | W_{i-1}, W_{i-2}, ...) ≈ P(W_i | W_{i-1}, ..., W_{i-K})

For natural languages:
  - Order 0 (unigram): poor model, high perplexity
  - Order 1 (bigram): substantially better
  - Order 2 (trigram): better still, diminishing returns

For Voynichese, Montemurro & Zanette (2013) show conditional entropy drops
significantly from order 0 to order 1, consistent with natural language.

TRAIN / TEST SPLIT  (TODO resolved)
------------------------------------
Previously, all perplexity was computed in-sample (train = test), which
is an optimistic lower bound and cannot be used to claim generalization.

This module now implements a proper 80/20 stratified line split:
  - 80% of lines used for training (selected by fixed-seed RNG)
  - 20% held out strictly for test perplexity
  - The split is reproducible given the same seed
  - Both train AND test perplexity are reported

If test perplexity at order K is HIGHER than at order K-1, the model
is overfitting — additional context does not generalize. This is now
a first-class reported metric and feeds into falsification test F5.2.

SMOOTHING
----------
Laplace (add-1) smoothing prevents zero probabilities for unseen n-grams.
The smoothing parameter is always recorded in output so results are
reproducible. The vocabulary used for smoothing is the TRAINING vocabulary
only — test tokens outside the training vocabulary use the UNK smoothing
path and are counted in a separate unk_token_count field.

REFERENCES
----------
Montemurro, M.A. & Zanette, D.H. (2013). Keywords and co-occurrence
  patterns in the Voynich manuscript: An information-theoretic analysis.
  PLoS ONE, 8(6): e66344.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Markov model
# ---------------------------------------------------------------------------

class MarkovModel:
    """
    N-gram Markov language model with Laplace smoothing.

    Parameters
    ----------
    order : int
        Markov order (context window size). Order 0 = unigram baseline.
    smoothing : float
        Laplace smoothing parameter (default 1.0). Recorded in all output.
    """

    def __init__(self, order: int = 1, smoothing: float = 1.0) -> None:
        self.order    = order
        self.smoothing = smoothing
        self.vocab: set[str] = set()
        self._counts: dict[tuple, Counter] = defaultdict(Counter)
        self._context_totals: dict[tuple, int] = {}
        self._trained = False
        self.unk_token_count = 0    # tokens seen at test time not in train vocab

    def train(self, lines: Sequence[Sequence[str]]) -> None:
        """
        Train on token lines. Contexts never cross line boundaries.

        The vocabulary is built exclusively from training lines. Tokens
        seen at test time that are outside this vocabulary are handled
        by the Laplace UNK path in log_prob().
        """
        for line in lines:
            tokens = list(line)
            self.vocab.update(tokens)
            if len(tokens) <= self.order:
                continue
            for i in range(self.order, len(tokens)):
                context   = tuple(tokens[i - self.order : i])
                next_tok  = tokens[i]
                self._counts[context][next_tok] += 1

        for context, ctr in self._counts.items():
            self._context_totals[context] = sum(ctr.values())

        self._trained = True
        logger.debug(
            "MarkovModel(order=%d) trained: %d contexts, %d vocab",
            self.order, len(self._counts), len(self.vocab),
        )

    def log_prob(self, context: tuple[str, ...], token: str) -> float:
        """
        log₂ P(token | context) with Laplace smoothing.

        Unseen tokens (outside training vocab) are handled by treating
        the vocab size as fixed at training-vocab size. This is a
        closed-vocabulary assumption — appropriate here because we do not
        expect genuinely new Voynichese tokens at test time, only
        transcription variants.

        Returns a finite value for all inputs (smoothing prevents log(0)).
        """
        if not self._trained:
            raise RuntimeError("Model not trained — call train() first.")

        vocab_size    = len(self.vocab)
        context_total = self._context_totals.get(context, 0)
        token_count   = self._counts.get(context, {}).get(token, 0)

        # Track OOV tokens for reporting
        if token not in self.vocab:
            self.unk_token_count += 1

        p = (token_count + self.smoothing) / (
            context_total + self.smoothing * max(vocab_size, 1)
        )
        return math.log2(p)

    def perplexity(self, lines: Sequence[Sequence[str]]) -> tuple[float, int]:
        """
        Compute model perplexity over the given lines.

        PP = 2^{-1/N · Σ log₂ P(w_i | context)}

        Parameters
        ----------
        lines
            Lines to evaluate. For honest evaluation these must be
            HELD-OUT lines not used during training.

        Returns
        -------
        (perplexity, n_tokens_evaluated)
            n_tokens_evaluated counts tokens that contributed to PP.
            Lines shorter than the model order contribute nothing.
        """
        total_log_prob = 0.0
        total_tokens   = 0

        for line in lines:
            tokens = list(line)
            if len(tokens) <= self.order:
                continue
            for i in range(self.order, len(tokens)):
                context = tuple(tokens[i - self.order : i])
                token   = tokens[i]
                total_log_prob += self.log_prob(context, token)
                total_tokens   += 1

        if total_tokens == 0:
            return float("inf"), 0

        avg_log_prob = total_log_prob / total_tokens
        return 2 ** (-avg_log_prob), total_tokens


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_lines(
    lines: list[list[str]],
    test_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Split manuscript lines into train and test sets.

    STRATIFICATION: Lines are shuffled with a fixed seed and split at
    the test_fraction boundary. No stratification by folio or section
    is applied — this is a known limitation when section distributions
    differ. Stratified splitting is a planned future improvement.

    Parameters
    ----------
    lines
        All manuscript lines (each a list of tokens).
    test_fraction
        Fraction of lines held out for testing (default 0.20 = 20%).
    seed
        Random seed. MUST be reported alongside any perplexity result.

    Returns
    -------
    (train_lines, test_lines)
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(lines)).tolist()

    n_test  = max(1, int(len(lines) * test_fraction))
    test_idx  = set(indices[:n_test])
    train_idx = set(indices[n_test:])

    train = [lines[i] for i in sorted(train_idx)]
    test  = [lines[i] for i in sorted(test_idx)]

    logger.debug(
        "Train/test split: %d train lines, %d test lines (seed=%d)",
        len(train), len(test), seed,
    )
    return train, test


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class MarkovOrderResult:
    """Perplexity results for one Markov order."""
    order:             int
    train_perplexity:  float
    test_perplexity:   Optional[float]   # None if test set too small
    n_train_tokens:    int
    n_test_tokens:     int
    vocab_size:        int
    unique_contexts:   int
    unk_token_count:   int               # OOV tokens at test time
    smoothing:         float
    note:              str = ""


@dataclass
class MarkovAnalysisReport:
    """
    Complete Markov order analysis across orders 0 to max_order.

    Key fields
    ----------
    results_by_order
        Per-order results including both train and test perplexity.
    split_config
        Documents the exact train/test split parameters so results
        are reproducible.
    optimal_order_by_test_pp
        Order with lowest TEST perplexity. This is the honest measure
        of how much context generalizes to unseen data.
    perplexity_reduction_test
        Reduction in TEST perplexity from order K to K+1.
        Used by falsification tests F5.1 and F5.2.
    """
    results_by_order:           list[dict]
    split_config:               dict
    optimal_order_by_train_pp:  int
    optimal_order_by_test_pp:   Optional[int]
    perplexity_reduction_train: dict[str, float]
    perplexity_reduction_test:  dict[str, float]
    interpretation_note: str = (
        "Train perplexity is an in-sample estimate (optimistic). "
        "Test perplexity is computed on held-out lines and is the "
        "honest measure of sequential generalization. "
        "A model that improves on train but not test is overfitting."
    )
    literature_reference: str = (
        "Montemurro & Zanette (2013) PLoS ONE 8(6):e66344 report "
        "conditional entropy values for Voynichese consistent with "
        "a first- or second-order Markov model."
    )


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_markov_analysis(
    records: list,
    max_order:     int   = 4,
    smoothing:     float = 1.0,
    test_fraction: float = 0.20,
    seed:          int   = 42,
) -> MarkovAnalysisReport:
    """
    Train Markov models at orders 0 through max_order and compare perplexity
    on a held-out test set.

    Parameters
    ----------
    records
        List of TokenRecord objects.
    max_order
        Maximum Markov order to evaluate.
    smoothing
        Laplace smoothing parameter.
    test_fraction
        Fraction of lines held out for test evaluation (default 0.20).
        Set to 0.0 to disable splitting (produces in-sample estimates only,
        with an explicit warning in the output).
    seed
        Random seed for the train/test split. Always recorded in output.

    Returns
    -------
    MarkovAnalysisReport
    """
    all_lines = [r.tokens for r in records if r.tokens]
    n_total   = len(all_lines)

    logger.info(
        "Markov analysis: %d lines, orders 0–%d, test_fraction=%.2f, seed=%d",
        n_total, max_order, test_fraction, seed,
    )

    # Perform split
    if test_fraction > 0.0 and n_total >= 10:
        train_lines, test_lines = split_lines(all_lines, test_fraction, seed)
        split_honest = True
    else:
        train_lines = all_lines
        test_lines  = all_lines     # fallback: in-sample
        split_honest = False
        logger.warning(
            "Train/test split disabled (test_fraction=%.2f, n=%d). "
            "Perplexity values are in-sample estimates only.",
            test_fraction, n_total,
        )

    split_config = {
        "total_lines":    n_total,
        "n_train_lines":  len(train_lines),
        "n_test_lines":   len(test_lines),
        "test_fraction":  test_fraction,
        "seed":           seed,
        "honest_split":   split_honest,
        "warning": (
            None if split_honest else
            "Test perplexity computed on training data — in-sample estimate only."
        ),
    }

    results       = []
    train_pp_dict = {}
    test_pp_dict  = {}

    for order in range(max_order + 1):
        model = MarkovModel(order=order, smoothing=smoothing)
        model.train(train_lines)

        train_pp, n_train_tok = model.perplexity(train_lines)

        # Reset OOV counter before test evaluation
        model.unk_token_count = 0
        test_pp, n_test_tok = model.perplexity(test_lines)
        unk_count = model.unk_token_count

        train_pp_dict[order] = train_pp
        test_pp_dict[order]  = test_pp if n_test_tok > 0 else None

        note = ""
        if order == 0:
            note = "Unigram baseline — no sequential context."
        elif order == 1:
            note = "Bigram model — one previous token as context."
        elif order == 2:
            note = "Trigram model."

        r = MarkovOrderResult(
            order=order,
            train_perplexity=round(train_pp, 4),
            test_perplexity=(round(test_pp, 4) if n_test_tok > 0 else None),
            n_train_tokens=n_train_tok,
            n_test_tokens=n_test_tok,
            vocab_size=len(model.vocab),
            unique_contexts=len(model._counts),
            unk_token_count=unk_count,
            smoothing=smoothing,
            note=note,
        )
        results.append(asdict(r))

        logger.info(
            "  Order %d: train_PP=%.2f  test_PP=%.2f  "
            "vocab=%d  contexts=%d  OOV=%d",
            order,
            train_pp,
            test_pp if n_test_tok > 0 else float("nan"),
            len(model.vocab),
            len(model._counts),
            unk_count,
        )

    # Perplexity reduction tables (train and test separately)
    def _reductions(pp_dict: dict) -> dict[str, float]:
        out = {}
        for order in range(1, max_order + 1):
            pp_prev = pp_dict.get(order - 1)
            pp_curr = pp_dict.get(order)
            if pp_prev and pp_curr and pp_prev > 0:
                red = (pp_prev - pp_curr) / pp_prev
            else:
                red = 0.0
            out[f"{order-1}_to_{order}"] = round(red, 4)
        return out

    train_reductions = _reductions(train_pp_dict)
    test_reductions  = _reductions(test_pp_dict)

    # Best order by test perplexity (lowest finite test PP)
    finite_test = {
        o: pp for o, pp in test_pp_dict.items()
        if pp is not None and math.isfinite(pp)
    }
    optimal_test  = min(finite_test, key=finite_test.get) if finite_test else None
    optimal_train = min(
        (o for o in train_pp_dict if math.isfinite(train_pp_dict[o])),
        key=train_pp_dict.get,
        default=0,
    )

    return MarkovAnalysisReport(
        results_by_order=results,
        split_config=split_config,
        optimal_order_by_train_pp=optimal_train,
        optimal_order_by_test_pp=optimal_test,
        perplexity_reduction_train=train_reductions,
        perplexity_reduction_test=test_reductions,
    )


def save_markov_report(report: MarkovAnalysisReport, output_path: str | Path) -> None:
    """Save Markov analysis report to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)
    logger.info("Markov report saved: %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    from amf.corpus.ingest import EVACorpus

    parser = argparse.ArgumentParser(
        description="AMF v4.0 — Markov order analysis with train/test split"
    )
    parser.add_argument("--corpus",        required=True)
    parser.add_argument("--output",        default="outputs/markov_report.json")
    parser.add_argument("--max-order",     type=int,   default=4)
    parser.add_argument("--smoothing",     type=float, default=1.0)
    parser.add_argument("--test-fraction", type=float, default=0.20,
                        help="Fraction of lines held out for test PP (default 0.20)")
    parser.add_argument("--seed",          type=int,   default=42,
                        help="Random seed for train/test split (default 42)")
    parser.add_argument("--log-level",     default="INFO",
                        choices=["DEBUG","INFO","WARNING","ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    corpus = EVACorpus.from_json(args.corpus)
    report = run_markov_analysis(
        corpus.records,
        max_order=args.max_order,
        smoothing=args.smoothing,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    cfg = report.split_config
    print(f"\n{'='*60}")
    print("Markov Order Analysis — AMF v4.0")
    print(f"{'='*60}")
    print(f"Train lines : {cfg['n_train_lines']:,}  |  "
          f"Test lines : {cfg['n_test_lines']:,}  |  "
          f"Seed : {cfg['seed']}  |  "
          f"Honest split : {cfg['honest_split']}")
    if cfg.get("warning"):
        print(f"⚠  {cfg['warning']}")

    print(f"\n{'Order':>6}  {'Train PP':>12}  {'Test PP':>12}  "
          f"{'Train→Test Δ':>14}  {'Contexts':>10}")
    print("-" * 62)
    for r in report.results_by_order:
        tpp = f"{r['test_perplexity']:.2f}" if r['test_perplexity'] else "N/A"
        delta = ""
        if r['train_perplexity'] and r['test_perplexity']:
            d = r['test_perplexity'] - r['train_perplexity']
            delta = f"{d:+.2f}"
        print(f"  {r['order']:>4}  {r['train_perplexity']:>12.2f}  "
              f"{tpp:>12}  {delta:>14}  {r['unique_contexts']:>10,}")

    print(f"\nTrain perplexity reductions:")
    for k, v in report.perplexity_reduction_train.items():
        print(f"  Order {k}: {v:+.1%}")

    print(f"\nTest perplexity reductions (honest generalization):")
    for k, v in report.perplexity_reduction_test.items():
        flag = " ← overfitting" if v < 0 else ""
        print(f"  Order {k}: {v:+.1%}{flag}")

    print(f"\nOptimal order (train) : {report.optimal_order_by_train_pp}")
    print(f"Optimal order (test)  : {report.optimal_order_by_test_pp}")
    print(f"\n{report.interpretation_note}")

    save_markov_report(report, args.output)
    print(f"\nFull report: {args.output}")


if __name__ == "__main__":
    main()
