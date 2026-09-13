from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from lumis.core.document import Chunk
from lumis.nlp.vector_similarity_retriever import VectorSimilarityRetriever

import numpy as np
import pytest


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_retrieval_ranks_and_limits(asynchronous):
    results = [np.array([2.0, 0]), np.array([[0.0, 3], [4.0, 0], [-2.0, 0]])]
    embedding = SimpleNamespace(embed=Mock(side_effect=results), aembed=AsyncMock(side_effect=results))
    retriever = VectorSimilarityRetriever(embedding)
    chunks = [Chunk(content=name) for name in ["orthogonal", "same", "opposite"]]
    found = await retriever.aextract_relevant_chunks("query", chunks, k=2) if asynchronous else retriever.extract_relevant_chunks("query", chunks, k=2)
    assert [c.content for _, c in found] == ["same", "orthogonal"]
    assert [score for score, _ in found] == pytest.approx([1, 0])


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_empty_chunks_and_zero_k_skip_embedding(asynchronous):
    embedding = SimpleNamespace(embed=Mock(), aembed=AsyncMock())
    retriever = VectorSimilarityRetriever(embedding)
    method = retriever.aextract_relevant_chunks if asynchronous else retriever.extract_relevant_chunks
    for chunks, k in [([], 5), ([Chunk(content="a")], 0)]:
        result = method("query", chunks, k=k)
        assert (await result if asynchronous else result) == []
    embedding.embed.assert_not_called()
    embedding.aembed.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_zero_vectors_have_finite_similarity(asynchronous):
    vectors = [np.zeros(2), np.zeros((1, 2))]
    embedding = SimpleNamespace(embed=Mock(side_effect=vectors), aembed=AsyncMock(side_effect=vectors))
    retriever = VectorSimilarityRetriever(embedding)
    chunks = [Chunk(content="empty")]
    result = await retriever.aextract_relevant_chunks("", chunks) if asynchronous else retriever.extract_relevant_chunks("", chunks)
    assert result[0][0] == 0.0
