from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from lumis.core.document import Chunk, Document
import lumis.tools.search.vector_search_retrieval_engine as module
from lumis.tools.search.vector_search_retrieval_engine import VectorSearchRetrievalEngine

import pytest


@pytest.fixture
def engine():
    return VectorSearchRetrievalEngine(
        embedding=Mock(),
        search_engine_client=SimpleNamespace(search=AsyncMock()),
        web_scrapper=SimpleNamespace(batch_fetch_content=AsyncMock()),
        semantic_parser=SimpleNamespace(aparse_document=AsyncMock()),
        similarity_retriever=SimpleNamespace(aextract_relevant_chunks=AsyncMock()),
    )


def test_default_collaborators_construct_without_undefined_arguments(monkeypatch):
    for name in ["SearchEngineClient", "WebScrapper", "SemanticParser", "VectorSimilarityRetriever"]:
        monkeypatch.setattr(module, name, Mock())
    engine = VectorSearchRetrievalEngine(embedding=Mock())
    module.WebScrapper.assert_called_once_with()
    assert engine.default_k == 5


async def test_search_orchestrates_provider_scraper_parser_ranking(engine):
    results = [{"url": "https://example.org/a", "title": "A"}]
    doc = Document(content="article", metadata={"url": results[0]["url"]})
    chunks = [Chunk(content="one"), Chunk(content="two")]
    engine.search_engine_client.search.return_value = results
    engine.scraper.batch_fetch_content.return_value = [doc]
    engine.parser.aparse_document.return_value = chunks
    engine.retriever.aextract_relevant_chunks.return_value = [(1.0, chunks[1])]
    assert await engine.search("q", topic="news", search_engine="google", max_results=2, k=1) == [chunks[1]]
    engine.search_engine_client.search.assert_awaited_once_with("q", topic="news", max_results=2, search_engine="google")
    engine.scraper.batch_fetch_content.assert_awaited_once_with(["https://example.org/a"], results)
    engine.retriever.aextract_relevant_chunks.assert_awaited_once_with("q", chunks, 1)


@pytest.mark.parametrize("failure", [False, True])
async def test_missing_search_results_stop_pipeline(engine, failure):
    engine.search_engine_client.search.return_value = []
    if failure:
        engine.search_engine_client.search.side_effect = RuntimeError("offline")
    assert await engine.search("q") == []
    engine.scraper.batch_fetch_content.assert_not_called()


async def test_documents_without_urls_and_parse_failures_are_skipped(engine):
    engine.scraper.batch_fetch_content.return_value = [Document(content="missing"), Document(content="bad", metadata={"url": "bad"}), Document(content="good", metadata={"url": "good"})]
    chunks = [Chunk(content="success")]
    engine.parser.aparse_document.side_effect = [RuntimeError("invalid"), chunks]
    assert await engine._collect_chunks(["bad", "good"], [{}, {}]) == chunks
    assert engine.parser.aparse_document.await_count == 2


async def test_empty_documents_and_empty_similarity_results(engine):
    engine.search_engine_client.search.return_value = [{"url": "https://example.org"}]
    engine.scraper.batch_fetch_content.return_value = []
    assert await engine.search("q") == []
    engine.retriever.aextract_relevant_chunks.assert_not_called()
    engine.scraper.batch_fetch_content.return_value = [Document(content="a", metadata={"url": "url"})]
    engine.parser.aparse_document.return_value = [Chunk(content="a")]
    engine.retriever.aextract_relevant_chunks.return_value = []
    assert await engine.search("q") == []
