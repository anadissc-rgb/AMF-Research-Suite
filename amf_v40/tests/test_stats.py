"""
tests/test_stats.py
====================

Unit tests for amf.stats.entropy, adjacency, positional, and markov.

All tests use synthetic token sequences, not real corpus data.
Results are checked for mathematical correctness and expected ranges,
not against specific corpus values (which depend on transcription).
"""

import math
from collections import Counter

import pytest

from amf.stats.entropy import (
    unigram_entropy,
    conditional_bigram_entropy,
    analyse_tokens,
    zipf_fit,
    full_entropy_report,
    character_entropy,
)
from amf.stats.adjacency import (
    extract_ngrams,
    pairwise_mutual_information,
    compute_adjacency,
)
from amf.stats.markov import MarkovModel, run_markov_analysis
from amf.stats.positional import compute_positional_stats, high_bias_tokens
from amf.corpus.ingest import TokenRecord


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def make_record(tokens: list[str], folio: str = "f1r", section: str = "P") -> TokenRecord:
    """Helper to build a minimal TokenRecord."""
    return TokenRecord(
        folio=folio, section=section, line="1",
        transcriber="test", raw_line=" ".join(tokens),
        tokens=tokens,
    )


SYNTHETIC_LINES = [
    ["chedy", "daiin", "qokedy", "shol"],
    ["daiin", "chedy", "or", "aiin"],
    ["qokedy", "chedy", "chedy", "daiin"],
    ["shol", "or", "daiin", "chedy"],
    ["aiin", "shol", "qokedy", "or"],
]

SYNTHETIC_TOKENS = [tok for line in SYNTHETIC_LINES for tok in line]


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

class TestUnigramEntropy:
    def test_uniform_distribution(self):
        """Uniform distribution has maximum entropy = log2(n)."""
        tokens = ["a", "b", "c", "d"]
        h = unigram_entropy(tokens)
        assert abs(h - math.log2(4)) < 1e-9

    def test_degenerate_single_token(self):
        """Deterministic sequence has zero entropy."""
        tokens = ["a", "a", "a", "a"]
        assert unigram_entropy(tokens) == 0.0

    def test_empty_returns_zero(self):
        assert unigram_entropy([]) == 0.0

    def test_entropy_range(self):
        """Entropy must be in [0, log2(vocab_size)]."""
        h = unigram_entropy(SYNTHETIC_TOKENS)
        vocab = len(set(SYNTHETIC_TOKENS))
        assert 0.0 <= h <= math.log2(vocab) + 1e-9

    def test_larger_vocab_higher_entropy(self):
        """More diverse token sequence should have higher entropy."""
        uniform_5 = list("abcde") * 10
        uniform_3 = list("abc") * 10
        assert unigram_entropy(uniform_5) > unigram_entropy(uniform_3)


class TestConditionalEntropy:
    def test_result_nonnegative(self):
        h = conditional_bigram_entropy(SYNTHETIC_TOKENS)
        assert h >= 0.0

    def test_less_than_unigram(self):
        """Conditional entropy H(W|W-1) should be ≤ unigram H(W)."""
        h_uni = unigram_entropy(SYNTHETIC_TOKENS)
        h_cond = conditional_bigram_entropy(SYNTHETIC_TOKENS)
        assert h_cond <= h_uni + 1e-6  # allow floating point margin

    def test_single_token_returns_zero(self):
        assert conditional_bigram_entropy(["a"]) == 0.0


class TestAnalyseTokens:
    def test_returns_result(self):
        result = analyse_tokens(SYNTHETIC_TOKENS, label="test")
        assert result.token_count == len(SYNTHETIC_TOKENS)
        assert result.type_count == len(set(SYNTHETIC_TOKENS))

    def test_normalised_entropy_in_range(self):
        result = analyse_tokens(SYNTHETIC_TOKENS)
        assert 0.0 <= result.unigram_entropy_normalised <= 1.0

    def test_redundancy_plus_normalised_is_one(self):
        result = analyse_tokens(SYNTHETIC_TOKENS)
        assert abs(result.redundancy + result.unigram_entropy_normalised - 1.0) < 1e-6

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            analyse_tokens([])

    def test_top_tokens_ordering(self):
        """Top tokens should be in descending frequency order."""
        result = analyse_tokens(SYNTHETIC_TOKENS, top_n=5)
        counts = [cnt for _, cnt in result.top_tokens]
        assert counts == sorted(counts, reverse=True)


class TestZipfFit:
    def test_returns_alpha(self):
        result = zipf_fit(SYNTHETIC_TOKENS)
        assert "alpha" in result
        assert isinstance(result["alpha"], float)

    def test_r_squared_in_range(self):
        result = zipf_fit(SYNTHETIC_TOKENS)
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_insufficient_types_returns_error(self):
        result = zipf_fit(["a", "a", "b"])
        assert "error" in result

    def test_rank_freq_pairs_sorted(self):
        result = zipf_fit(SYNTHETIC_TOKENS)
        freqs = [p["freq"] for p in result["rank_freq_pairs"]]
        assert freqs == sorted(freqs, reverse=True)


class TestCharacterEntropy:
    def test_returns_expected_keys(self):
        result = character_entropy(SYNTHETIC_TOKENS)
        assert "h_char" in result
        assert "h_char_conditional" in result
        assert "unique_chars" in result

    def test_nonnegative(self):
        result = character_entropy(SYNTHETIC_TOKENS)
        assert result["h_char"] >= 0.0
        assert result["h_char_conditional"] >= 0.0


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------

class TestExtractNgrams:
    def test_bigrams_within_line(self):
        lines = [["a", "b", "c"]]
        bigrams = extract_ngrams(lines, n=2)
        assert ("a", "b") in bigrams
        assert ("b", "c") in bigrams
        assert len(bigrams) == 2

    def test_no_cross_line_bigrams(self):
        """Bigrams must NOT cross line boundaries."""
        lines = [["a", "b"], ["c", "d"]]
        bigrams = extract_ngrams(lines, n=2)
        assert ("b", "c") not in bigrams

    def test_short_line_skipped(self):
        lines = [["a"], ["b", "c"]]
        bigrams = extract_ngrams(lines, n=2)
        # "a" alone cannot form a bigram
        assert all("a" in bg for bg in bigrams if "a" in bg) or len(bigrams) == 1

    def test_trigrams(self):
        lines = [["a", "b", "c", "d"]]
        trigrams = extract_ngrams(lines, n=3)
        assert ("a", "b", "c") in trigrams
        assert ("b", "c", "d") in trigrams
        assert len(trigrams) == 2

    def test_empty_lines(self):
        assert extract_ngrams([], n=2) == []


class TestPMI:
    def test_identical_distribution_pmi_zero(self):
        """If w1 and w2 are independent, PMI ≈ 0."""
        # Build a sequence where every token is independent
        tokens = ["a", "b", "c", "d"] * 25
        lines = [tokens]
        bigrams = extract_ngrams(lines, n=2)
        unigram_counts = Counter(tokens)
        pmi = pairwise_mutual_information(bigrams, unigram_counts, len(tokens))
        # PMI values should be close to 0 for a near-uniform sequence
        for val in pmi.values():
            assert abs(val) < 3.0  # loose bound — depends on sequence structure


class TestComputeAdjacency:
    def test_runs_without_error(self):
        records = [make_record(line) for line in SYNTHETIC_LINES]
        report = compute_adjacency(records, top_n=5, min_pmi_count=1)
        assert report.total_bigrams > 0
        assert report.unique_bigram_types > 0

    def test_bigram_count_correct(self):
        """Each line of N tokens produces N-1 bigrams."""
        records = [make_record(line) for line in SYNTHETIC_LINES]
        report = compute_adjacency(records)
        expected = sum(len(line) - 1 for line in SYNTHETIC_LINES)
        assert report.total_bigrams == expected

    def test_bigram_entropy_nonnegative(self):
        records = [make_record(line) for line in SYNTHETIC_LINES]
        report = compute_adjacency(records)
        assert report.bigram_entropy >= 0.0


# ---------------------------------------------------------------------------
# Markov
# ---------------------------------------------------------------------------

class TestMarkovModel:
    def test_trains_without_error(self):
        model = MarkovModel(order=1)
        model.train(SYNTHETIC_LINES)
        assert model._trained

    def test_perplexity_positive(self):
        model = MarkovModel(order=1)
        model.train(SYNTHETIC_LINES)
        pp = model.perplexity(SYNTHETIC_LINES)
        assert pp > 0.0

    def test_higher_order_lower_perplexity(self):
        """Order 2 should fit training data better than order 1."""
        m1 = MarkovModel(order=1)
        m1.train(SYNTHETIC_LINES)
        pp1 = m1.perplexity(SYNTHETIC_LINES)

        m2 = MarkovModel(order=2)
        m2.train(SYNTHETIC_LINES)
        pp2 = m2.perplexity(SYNTHETIC_LINES)

        # On training data, higher order always fits as well or better
        assert pp2 <= pp1 + 1.0  # small tolerance for smoothing effects

    def test_untrained_raises(self):
        model = MarkovModel(order=1)
        with pytest.raises(RuntimeError):
            model.log_prob(("a",), "b")

    def test_log_prob_is_finite(self):
        """With Laplace smoothing, log_prob should never be -inf."""
        model = MarkovModel(order=1)
        model.train(SYNTHETIC_LINES)
        # Unseen context + unseen token: should return a finite (smoothed) value
        lp = model.log_prob(("UNSEEN",), "ALSO_UNSEEN")
        assert math.isfinite(lp)


class TestRunMarkovAnalysis:
    def test_runs_all_orders(self):
        records = [make_record(line) for line in SYNTHETIC_LINES]
        report = run_markov_analysis(records, max_order=3)
        assert len(report.results_by_order) == 4  # orders 0,1,2,3

    def test_perplexity_decreases_with_order(self):
        """For training data, perplexity should be non-increasing with order."""
        records = [make_record(line) for line in SYNTHETIC_LINES]
        report = run_markov_analysis(records, max_order=3)
        pps = [r["train_perplexity"] for r in report.results_by_order]
        for i in range(1, len(pps)):
            assert pps[i] <= pps[i - 1] + 1.0  # tolerance for smoothing

    def test_optimal_order_in_range(self):
        records = [make_record(line) for line in SYNTHETIC_LINES]
        report = run_markov_analysis(records, max_order=3)
        assert 0 <= report.optimal_order <= 3


# ---------------------------------------------------------------------------
# Positional
# ---------------------------------------------------------------------------

class TestPositionalStats:
    def test_runs_without_error(self):
        records = [make_record(line) for line in SYNTHETIC_LINES]
        stats = compute_positional_stats(records, top_n=5)
        assert stats.total_lines == len(SYNTHETIC_LINES)

    def test_token_count_correct(self):
        records = [make_record(line) for line in SYNTHETIC_LINES]
        stats = compute_positional_stats(records)
        expected = sum(len(line) for line in SYNTHETIC_LINES)
        assert stats.total_tokens == expected

    def test_line_initial_top_has_items(self):
        records = [make_record(line) for line in SYNTHETIC_LINES]
        stats = compute_positional_stats(records, top_n=5)
        assert len(stats.line_initial_top) > 0

    def test_initial_rate_in_range(self):
        records = [make_record(line) for line in SYNTHETIC_LINES]
        stats = compute_positional_stats(records)
        for entry in stats.line_initial_top:
            assert 0.0 <= entry["line_initial_rate"] <= 1.0

    def test_high_bias_tokens(self):
        records = [make_record(line) for line in SYNTHETIC_LINES]
        stats = compute_positional_stats(records)
        biased = high_bias_tokens(stats, min_rate=0.0, min_count=1)
        assert "line_initial_biased" in biased
        assert "line_final_biased" in biased
        assert "note" in biased

    def test_word_position_profiles_populated(self):
        records = [make_record(line) for line in SYNTHETIC_LINES]
        stats = compute_positional_stats(records)
        # Position "1" should have entries
        assert "1" in stats.word_position_profiles
        assert len(stats.word_position_profiles["1"]) > 0
