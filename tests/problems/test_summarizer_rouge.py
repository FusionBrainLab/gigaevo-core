"""Tests for word-level ROUGE-L F1."""

from __future__ import annotations

import pytest

from problems.chains.summarizer.rouge import rouge_l_f1


def test_identical_is_one():
    assert rouge_l_f1("the cat sat", "the cat sat") == pytest.approx(1.0)


def test_disjoint_is_zero():
    assert rouge_l_f1("alpha beta", "gamma delta") == 0.0


def test_partial_overlap():
    assert rouge_l_f1("the cat sat", "the cat") == pytest.approx(0.8)


def test_empty_strings_are_zero():
    assert rouge_l_f1("", "the cat") == 0.0
    assert rouge_l_f1("the cat", "") == 0.0
    assert rouge_l_f1("", "") == 0.0


def test_case_insensitive():
    assert rouge_l_f1("The Cat", "the cat") == pytest.approx(1.0)


def test_subsequence_not_substring():
    assert rouge_l_f1("a x b y c", "a b c") == pytest.approx(0.75)


def test_russian_tokens():
    assert rouge_l_f1("кот сидел на крыше", "кот сидел на крыше") == pytest.approx(1.0)
