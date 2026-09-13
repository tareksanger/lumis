from unittest.mock import AsyncMock, Mock

from lumis.core.document import Document
import lumis.nlp.semantic_parser as module
from lumis.nlp.semantic_parser import SemanticParser

import numpy as np
import pytest


@pytest.fixture
def parser(monkeypatch):
    monkeypatch.setattr(module, "word_tokenize", str.split)
    embedder = Mock()
    embedder.embed.side_effect = lambda sentences: np.ones((len(sentences), 2))
    embedder.similarity.return_value = 1.0
    return SemanticParser(embedder, max_tokens=3)


def test_sentence_split_preserves_boundaries_and_cleans_controls(parser):
    tokenizer = Mock()
    tokenizer.span_tokenize.return_value = [(0, 4), (5, 9)]
    assert parser.split_by_sentence("One.\nTwo.", tokenizer) == ["One. ", "Two."]


def test_parse_limits_tokens_and_preserves_metadata(parser):
    document = Document(doc_id="doc", content="one two three four five", metadata={"source": "test"})
    chunks = parser.parse_document(document)
    assert [chunk.content for chunk in chunks] == ["one two three", "four five"]
    assert [chunk.metadata["token_count"] for chunk in chunks] == [3, 2]
    assert all(chunk.parent_id == "doc" and chunk.metadata["source"] == "test" for chunk in chunks)
    assert document.metadata == {"source": "test"}


def test_semantic_groups_do_not_join_words_between_sentences(parser):
    parser.max_tokens = 20
    chunks = parser.parse("First sentence. Second sentence.")
    assert chunks[0].content == "First sentence. Second sentence."


def test_breakpoints_create_separate_groups(parser):
    sentences = [{"sentence": text, "embedding": np.ones(2)} for text in ["one", "two", "three"]]
    parser.breakpoint_percentile_threshold = 50
    chunks = parser._build_node_chunks(sentences, [0.1, 0.9])
    assert [chunk.content for chunk in chunks] == ["one two", "three"]


def test_empty_document_yields_no_chunks(parser):
    assert parser.parse("") == []


async def test_async_parse_variants(parser):
    assert [c.content for c in await parser.aparse("one two")] == ["one two"]
    assert [c.content for c in await parser._aparse("one")] == ["one"]
    doc = Document(content="text", doc_id="id")
    assert (await parser.aparse_document(doc))[0].parent_id == "id"
    assert len(await parser.aparse_documents([doc, doc])) == 2


async def test_batch_retains_successful_documents(parser):
    good = parser.parse("good")
    parser.aparse_document = AsyncMock(side_effect=[ValueError("bad"), good])
    assert await parser.aparse_documents([Document(content="bad"), Document(content="good")]) == good


def test_sync_batch_is_explicitly_unsupported(parser):
    with pytest.raises(NotImplementedError):
        parser.parse_documents([])
