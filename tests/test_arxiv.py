"""arXiv behavior tested with fake optional SDK, HTTP, and PDF boundaries."""

import asyncio
from datetime import datetime
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from lumis.core.document import Document
from lumis.tools.search import arxiv as module

import pytest


@pytest.fixture
def searcher(monkeypatch):
    client = SimpleNamespace(results=Mock(return_value=[]))
    sdk = SimpleNamespace(
        Client=Mock(return_value=client),
        Search=Mock(side_effect=lambda **kwargs: kwargs),
        SortCriterion=SimpleNamespace(SubmittedDate="submitted"),
        SortOrder=SimpleNamespace(Descending="descending"),
    )
    monkeypatch.setitem(sys.modules, "arxiv", sdk)
    monkeypatch.setattr(module.httpx, "AsyncClient", Mock(return_value=SimpleNamespace(get=AsyncMock(), aclose=AsyncMock())))
    return module.ArxivSearcher(max_results=3, max_concurrent_pdfs=2)


def paper(identifier="2401.00001", pdf_url="https://example.test/paper.pdf"):
    return SimpleNamespace(
        get_short_id=lambda: identifier,
        title="A paper",
        summary="The abstract",
        authors=[SimpleNamespace(name="Ada"), SimpleNamespace(name="Alan")],
        published=datetime(2024, 1, 2),
        updated=datetime(2024, 2, 3),
        pdf_url=pdf_url,
        categories=["cs.AI", "cs.LG"],
        primary_category="cs.AI",
        comment=None,
        journal_ref="Journal 1",
        doi="10.1234/example",
    )


async def test_search_preserves_date_query_and_default_options(searcher):
    query = "cat:cs.AI AND submittedDate:[202401010000 TO 202401312359]"
    assert await searcher.search(query) == []
    searcher._arxiv.Search.assert_called_once_with(query=query, max_results=3, sort_by="submitted", sort_order="descending", id_list=[])


@pytest.mark.parametrize("limit", [0, 1, 7])
async def test_search_respects_explicit_options_and_zero_limit(searcher, limit):
    await searcher.search("q", max_results=limit, sort_by="relevance", sort_order="ascending", id_list=["2401.00001"])
    searcher._arxiv.Search.assert_called_once_with(query="q", max_results=limit, sort_by="relevance", sort_order="ascending", id_list=["2401.00001"])


async def test_search_maps_paper_fields_without_loading_pdf(searcher, monkeypatch):
    searcher.client.results.return_value = [paper()]
    read = AsyncMock()
    monkeypatch.setattr(searcher, "_read_pdf", read)
    (result,) = await searcher.search("q")
    assert result == module.ArxivResult(
        title="A paper",
        authors=["Ada", "Alan"],
        abstract="The abstract",
        pdf_url="https://example.test/paper.pdf",
        arxiv_id="2401.00001",
        published_date="2024-01-02",
        updated_date="2024-02-03",
        categories=["cs.AI", "cs.LG"],
        primary_category="cs.AI",
        comment=None,
        journal_ref="Journal 1",
        doi="10.1234/example",
    )
    read.assert_not_awaited()


async def test_cached_search_result_gains_pdf_once_and_get_document_reuses_it(searcher, monkeypatch):
    searcher.client.results.return_value = [paper()]
    read = AsyncMock(return_value=Document(content="full paper", metadata={"title": "PDF title"}))
    monkeypatch.setattr(searcher, "_read_pdf", read)
    (initial,) = await searcher.search("q")
    (populated,) = await searcher.search("q", read_pdfs=True)
    (repeated,) = await searcher.search("q", read_pdfs=True)
    assert initial is populated is repeated
    assert populated.content == "full paper"
    assert populated.metadata == {"title": "PDF title"}
    read.assert_awaited_once_with("https://example.test/paper.pdf")
    calls = searcher.client.results.call_count
    assert await searcher.get_document("2401.00001") is populated
    assert searcher.client.results.call_count == calls


@pytest.mark.parametrize("cached", [False, True])
async def test_pdf_failure_keeps_search_metadata_and_allows_retry(searcher, monkeypatch, cached):
    searcher.client.results.return_value = [paper()]
    read = AsyncMock(side_effect=[RuntimeError("PDF failed"), Document(content="recovered")])
    monkeypatch.setattr(searcher, "_read_pdf", read)
    if cached:
        await searcher.search("q")
    (result,) = await searcher.search("q", read_pdfs=True)
    assert result.title == "A paper"
    assert result.content is None
    assert await searcher.get_document(result.arxiv_id) is result
    assert result.content == "recovered"


async def test_missing_pdf_url_skips_pdf_loading(searcher, monkeypatch):
    searcher.client.results.return_value = [paper(pdf_url=None)]
    read = AsyncMock()
    monkeypatch.setattr(searcher, "_read_pdf", read)
    (result,) = await searcher.search("q", read_pdfs=True)
    assert result.content is None
    read.assert_not_awaited()


async def test_get_document_searches_by_id_and_populates_new_pdf(searcher, monkeypatch):
    searcher.client.results.return_value = [paper()]
    monkeypatch.setattr(searcher, "_read_pdf", AsyncMock(return_value=Document(content="text")))
    result = await searcher.get_document("2401.00001")
    assert result.content == "text"
    assert searcher._arxiv.Search.call_args.kwargs["query"] == "id:2401.00001"
    assert searcher._arxiv.Search.call_args.kwargs["id_list"] == ["2401.00001"]
    assert searcher._arxiv.Search.call_args.kwargs["max_results"] == 1


async def test_missing_document_returns_none(searcher):
    assert await searcher.get_document("missing") is None


async def test_search_errors_propagate_but_get_document_returns_none(searcher):
    searcher.client.results.side_effect = RuntimeError("upstream failed")
    with pytest.raises(RuntimeError, match="upstream failed"):
        await searcher.search("q")
    assert await searcher.get_document("missing") is None


def pdf_stub(monkeypatch, *, texts=("Page one", "Page two"), metadata=None, xmp=None):
    class PDF:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __len__(self):
            return len(texts)

        def __iter__(self):
            return iter([SimpleNamespace(number=i, get_text=Mock(return_value=text)) for i, text in enumerate(texts)])

    pdf = PDF()
    pdf.metadata = metadata
    pdf.xmp_metadata = xmp
    monkeypatch.setattr(module.pymupdf, "open", Mock(return_value=pdf))


async def test_pdf_extracts_pages_and_trimmed_metadata(searcher, monkeypatch):
    response = SimpleNamespace(content=b"fake pdf", raise_for_status=Mock())
    searcher._http_client.get.return_value = response
    pdf_stub(monkeypatch, metadata={"title": " Title ", "subject": " Description "})
    result = await searcher._read_pdf("https://example.test/paper.pdf")
    assert result.content == "Page one\fPage two\f"
    assert result.metadata == {"title": "Title", "description": "Description"}
    response.raise_for_status.assert_called_once()


@pytest.mark.parametrize(
    "xmp, expected",
    [
        (
            '<root xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            "<dc:description><rdf:Alt><rdf:li> XMP summary </rdf:li></rdf:Alt></dc:description></root>",
            "XMP summary",
        ),
        ("malformed xml", "Page one..."),
        (None, "Page one..."),
    ],
)
async def test_pdf_uses_xmp_or_text_fallback_description(searcher, monkeypatch, xmp, expected):
    searcher._http_client.get.return_value = SimpleNamespace(content=b"pdf", raise_for_status=Mock())
    pdf_stub(monkeypatch, texts=("Page one",), xmp=xmp)
    result = await searcher._read_pdf("url")
    assert result.metadata == {"description": expected}


async def test_empty_pdf_returns_none(searcher, monkeypatch):
    searcher._http_client.get.return_value = SimpleNamespace(content=b"pdf", raise_for_status=Mock())
    pdf_stub(monkeypatch, texts=("  ",))
    assert await searcher._read_pdf("url") is None


async def test_failed_download_does_not_parse_pdf(searcher, monkeypatch):
    searcher._http_client.get.return_value = SimpleNamespace(content=b"bad response", raise_for_status=Mock(side_effect=RuntimeError("HTTP failed")))
    parse = Mock()
    monkeypatch.setattr(module.pymupdf, "open", parse)
    with pytest.raises(RuntimeError, match="HTTP failed"):
        await searcher._read_pdf("url")
    parse.assert_not_called()


async def test_pdf_concurrency_is_bounded_and_slots_released_after_error(searcher, monkeypatch):
    release = asyncio.Event()
    saturated = asyncio.Event()
    active = 0
    peak = 0

    async def download(url):
        nonlocal active, peak
        active += 1
        peak = max(active, peak)
        if active == 2:
            saturated.set()
        try:
            await release.wait()
            if url == "bad":
                raise ValueError("bad PDF")
            return Document(content=url)
        finally:
            active -= 1

    monkeypatch.setattr(searcher, "_download_and_parse_pdf", download)
    tasks = [asyncio.create_task(searcher._read_pdf(url)) for url in ["bad", "a", "b", "c"]]
    try:
        await asyncio.wait_for(saturated.wait(), 2)
        await asyncio.sleep(0)
        assert peak == 2
    finally:
        release.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 2)
    assert isinstance(results[0], ValueError)
    assert [result.content for result in results[1:]] == ["a", "b", "c"]
    assert peak == 2
    assert active == 0


async def test_close_releases_http_client(searcher):
    await searcher.close()
    searcher._http_client.aclose.assert_awaited_once()


def test_missing_optional_sdk_has_actionable_install_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "arxiv", None)
    with pytest.raises(ImportError, match=r"pip install lumis-ai\[search\]"):
        module.ArxivSearcher()


@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_pdf_concurrency_is_rejected(searcher, limit):
    with pytest.raises(ValueError, match="max_concurrent_pdfs"):
        module.ArxivSearcher(max_concurrent_pdfs=limit)
