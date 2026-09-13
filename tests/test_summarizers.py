import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

from lumis.nlp.spacy_summarizer import SpacySummarizer
from lumis.nlp.summarizer import Summarizer

import pytest
import torch


class Batch(dict):
    def to(self, device):
        return self


@pytest.fixture
def summarizer(monkeypatch):
    module = ModuleType("transformers")
    tokenizer = MagicMock()

    def tokenize(text, truncation=False, max_length=None, **kwargs):
        count = len(text.split())
        if truncation:
            count = min(count, max_length)
        return Batch(input_ids=torch.ones((1, count), dtype=torch.long), attention_mask=torch.ones((1, count), dtype=torch.long))

    tokenizer.side_effect = tokenize
    tokenizer.encode.side_effect = lambda text, **kwargs: text.split()
    tokenizer.decode.return_value = "brief"
    model = Mock()
    model.to.return_value = model
    model.generate.return_value = [[1, 2]]
    module.AutoTokenizer = SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
    module.AutoModelForSeq2SeqLM = SimpleNamespace(from_pretrained=Mock(return_value=model))
    monkeypatch.setitem(sys.modules, "transformers", module)
    instance = Summarizer()
    instance.max_model_length = 4
    yield instance
    instance.executor.shutdown(wait=True)


async def test_short_and_long_summary_use_token_limit(summarizer):
    assert await summarizer.summarize("short text") == "brief"
    assert await summarizer.summarize("one two three four five six") == "brief"
    assert summarizer.model.generate.call_args.args[0].shape == (1, 4)
    assert summarizer.model.generate.call_args.kwargs["do_sample"] is False


async def test_long_summary_splits_then_reduces(summarizer):
    assert await summarizer.summarize("one two. three four. five six.", split_long_text=True) == "brief"
    assert summarizer.model.generate.call_count >= 3


def test_sentence_chunking_and_empty_text(summarizer):
    chunks = summarizer._split_text_into_chunks("one two. three four. five six.")
    assert chunks == ["one two. three four.", "five six."]
    assert summarizer._split_text_into_chunks("") == []


@pytest.mark.parametrize("name,max_length,min_length,beams", [("facebook/bart-large-cnn", 150, 40, 4), ("allenai/led-base-16384", 512, 50, 4), ("sshleifer/distilbart-cnn-12-6", 130, 30, 1)])
def test_model_generation_settings(summarizer, name, max_length, min_length, beams):
    summarizer.model_name = name
    inputs = Batch(input_ids=torch.tensor([[1]]), attention_mask=torch.tensor([[1]]))
    assert summarizer._generate_summary(inputs) == "brief"
    kwargs = summarizer.model.generate.call_args.kwargs
    assert (kwargs["max_length"], kwargs["min_length"], kwargs["num_beams"]) == (max_length, min_length, beams)


async def test_summary_propagates_model_failure(summarizer):
    summarizer.model.generate.side_effect = ValueError("model failed")
    with pytest.raises(ValueError, match="model failed"):
        await summarizer.summarize("text")


@pytest.fixture
def spacy_summarizer(monkeypatch):
    first = SimpleNamespace(text="First.", start_char=0)
    last = SimpleNamespace(text="Last.", start_char=10)
    rank = Mock()
    rank.summary.return_value = [last, first]
    doc = SimpleNamespace(sents=[first, last], _=SimpleNamespace(textrank=rank))
    nlp = Mock(return_value=doc)
    nlp.pipe_names = []
    module = ModuleType("spacy")
    module.load = Mock(return_value=nlp)
    monkeypatch.setitem(sys.modules, "spacy", module)
    monkeypatch.setitem(sys.modules, "pytextrank", ModuleType("pytextrank"))
    instance = SpacySummarizer()
    yield instance, rank
    instance.executor.shutdown(wait=True)


async def test_spacy_summary_restores_original_order(spacy_summarizer):
    instance, rank = spacy_summarizer
    assert await instance.summarize("First. Last.", summary_ratio=0.5) == "First. Last."
    rank.summary.assert_called_once_with(limit_sentences=1)
    instance.nlp.add_pipe.assert_called_once_with("textrank", last=True)


async def test_spacy_empty_summary(spacy_summarizer):
    instance, rank = spacy_summarizer
    rank.summary.return_value = []
    assert await instance.summarize("") == ""
