from unittest.mock import Mock

import lumis.evaluators.conciseness_and_clarity_analyzer as module
from lumis.evaluators.conciseness_and_clarity_analyzer import TextConcisenessAnalyzer as Conciseness
from lumis.evaluators.relevancy import TextRelevancyAnalyzer as Relevance

import pytest


@pytest.mark.parametrize("a,b,expected", [("cat cat dog", "cat dog dog", 0.8), ("Same!", "same", 1), ("cat", "dog", 0), ("", "dog", 0), ("", "", 0)])
def test_cosine_word_frequency_contract(a, b, expected):
    assert Relevance.cosine_similarity(a, b) == pytest.approx(expected)


@pytest.mark.parametrize("a,b,expected", [("A cat!", "a dog", 1 / 3), ("", "", 0), ("cat cat", "cat", 1)])
def test_jaccard(a, b, expected):
    assert Relevance.jaccard_similarity(a, b) == pytest.approx(expected)
    assert Conciseness.jaccard(a, b) == pytest.approx(expected)


@pytest.mark.parametrize("a,b,distance", [("kitten", "sitting", 3), ("", "abc", 3), ("abc", "", 3), ("same", "same", 0)])
def test_edit_distance(a, b, distance):
    assert Relevance.levenshtein_distance(a, b) == distance


def test_ngram_and_keywords():
    assert Relevance.ngram_similarity("abcd", "abce") == 0.5
    assert Relevance.ngram_similarity("", "") == 0
    assert Relevance.keyword_relevance("A cat", "cat here", ["cat"])
    assert not Relevance.keyword_relevance("A cat", "dog here", ["cat"])


@pytest.mark.parametrize("method", ["cosine", "jaccard", "levenshtein", "ngram"])
def test_relevance_dispatch_and_threshold(method):
    assert Relevance.is_relevant("same text", "same text", method=method, threshold=0.99)
    assert not Relevance.is_relevant("abc", "xyz", method=method, threshold=0.1)
    with pytest.raises(ValueError, match="Unknown method"):
        Relevance.is_relevant("a", "b", method="bogus")


def test_sentence_lengths_ignore_trailing_empty_sentences():
    assert Conciseness.measure_lengths("One two. Three four!") == {"average_sentence_length": 2, "average_word_length": 3.75}
    assert Conciseness.measure_lengths("...") == {"average_sentence_length": 0, "average_word_length": 0}
    assert Conciseness.is_too_verbose("one two three", max_avg_sentence_length=2)
    assert Conciseness.is_too_verbose("a and b or c", max_clauses=2)
    assert not Conciseness.is_too_verbose("short")


def test_compression_and_tokenization():
    assert Conciseness.compression_ratio("") == 0
    assert Conciseness.compression_ratio("repeat " * 100) < 0.1
    assert Conciseness.tokenize_sentence("One, ONE! Two?") == {"one", "two"}


def test_density_uses_content_tags_without_nltk_download(monkeypatch):
    monkeypatch.setattr(module, "word_tokenize", Mock(return_value=["the", "red", "cat", "runs"]))
    monkeypatch.setattr(module, "pos_tag", Mock(return_value=[("the", "DT"), ("red", "JJ"), ("cat", "NN"), ("runs", "VB")]))
    assert Conciseness.information_density("the red cat runs") == 0.75
    monkeypatch.setattr(module, "word_tokenize", Mock(return_value=[]))
    assert Conciseness.information_density("") == 0


def test_readability_weighting(monkeypatch):
    monkeypatch.setattr(module.textstat, "flesch_reading_ease", Mock(return_value=80))
    monkeypatch.setattr(module.textstat, "flesch_kincaid_grade", Mock(return_value=3))
    assert Conciseness.composite_readability_score("text", alpha=5) == 65


def test_explicit_resource_install_only_downloads_expected_data(monkeypatch):
    import nltk

    download = Mock()
    monkeypatch.setattr(nltk, "download", download)
    Conciseness.install_requirements()
    assert [c.args[0] for c in download.call_args_list] == ["punkt", "punkt_tab", "averaged_perceptron_tagger_eng"]
