"""Scraper tests with injected HTTP clients and small local HTML/PDF documents."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from lumis.core.document import Document
from lumis.tools.scraper import WebScrapper

import httpx
import pymupdf
import pytest


@pytest.fixture
def scraper():
    return WebScrapper()


def http_scraper(response=None, error=None):
    client = SimpleNamespace(get=AsyncMock(return_value=response, side_effect=error))
    context = AsyncMock()
    context.__aenter__.return_value = client
    return WebScrapper(http_client_factory=lambda: context), client


def html_response(body, status=200, content_type="text/html"):
    return httpx.Response(status, text=body, headers={"content-type": content_type}, request=httpx.Request("GET", "https://example.org/final"))


def test_custom_headers_do_not_leak_between_instances():
    baseline = WebScrapper.HEADERS.copy()
    first = WebScrapper(default_headers={"x-test-isolated": "one"})
    second = WebScrapper()
    assert first.default_headers["x-test-isolated"] == "one"
    assert "x-test-isolated" not in second.default_headers
    assert WebScrapper.HEADERS == baseline


def test_none_headers_use_defaults():
    assert WebScrapper(default_headers=None).default_headers == WebScrapper.HEADERS


async def test_fetch_html_uses_redirect_url_params_and_metadata():
    body = "<html><head><title>Example paper</title></head><body><article><p>" + "A scientific finding with detailed supporting evidence. " * 8 + "</p></article></body></html>"
    scraper, client = http_scraper(html_response(body))
    result = await scraper.fetch_content("example.org/start", metadata={"rank": 2}, params={"page": "1"})
    client.get.assert_awaited_once_with(url="https://example.org/start", headers=scraper.default_headers, params={"page": "1"}, follow_redirects=True)
    assert len(result) == 1
    assert result[0].metadata["url"] == "https://example.org/final"
    assert result[0].metadata["rank"] == 2
    assert result[0].metadata["title"] == "Example paper"
    assert "scientific finding" in result[0].content


@pytest.mark.parametrize(
    "response",
    [
        html_response("Failure", status=500),
        html_response("Unsupported", content_type="application/json"),
        html_response("No type", content_type=""),
        html_response("   "),
    ],
)
async def test_unusable_http_response_returns_empty_documents(response):
    scraper, _ = http_scraper(response)
    assert await scraper.fetch_content("https://example.org") == []


async def test_transport_failure_returns_empty_documents():
    scraper, _ = http_scraper(error=httpx.ConnectError("Offline"))
    assert await scraper.fetch_content("https://example.org") == []


async def test_parse_failure_returns_empty_documents(monkeypatch):
    scraper, _ = http_scraper(html_response("<p>Body</p>"))
    monkeypatch.setattr(scraper, "parse_html", AsyncMock(side_effect=ValueError("Unreadable")))
    assert await scraper.fetch_content("https://example.org") == []


async def test_fetch_dispatches_pdf_bytes(monkeypatch):
    response = httpx.Response(200, content=b"mock-pdf", headers={"content-type": "application/pdf"}, request=httpx.Request("GET", "https://example.org/file.pdf"))
    scraper, _ = http_scraper(response)
    document = Document(content="PDF content", metadata={})
    parser = AsyncMock(return_value=document)
    monkeypatch.setattr(scraper, "parse_pdf", parser)
    assert await scraper.fetch_content("https://example.org/file.pdf") == [document]
    parser.assert_awaited_once_with(b"mock-pdf", {"url": "https://example.org/file.pdf"})


def test_cleanup_extracts_text_metadata_and_resolved_favicon(scraper):
    html = """<html><head><title>Study title</title>
    <meta name="date" content="2026-01-02"><meta name="author" content="Ada">
    <meta name="description" content="A study"><link rel="icon" href="/favicon.ico">
    <script>secret_script</script><style>secret_style</style></head><body><article>
    <p>This research has a substantial body of evidence and explains the study results in detail.</p>
    <a href="/paper?ref=home#summary">Paper</a></article></body></html>"""
    title, body, links, metadata = scraper.cleanup_html("https://example.org/start", html)
    assert title == "Study title"
    assert "research" in body
    assert "secret_script" not in body
    assert "secret_style" not in body
    assert links == ["https://example.org/paper"]
    assert metadata == {"publication_date": "2026-01-02T00:00:00", "author": "Ada", "description": "A study", "logo": "https://example.org/favicon.ico"}


@pytest.mark.parametrize("marker", ["captcha", "Cloudflare"])
def test_cleanup_detects_challenge_pages(scraper, marker):
    assert scraper.cleanup_html("https://example.org", f"<html><p>{marker} challenge</p></html>") == (None, None, None, None)


def test_invalid_publication_date_is_omitted_and_open_graph_logo_wins(scraper):
    _, _, _, metadata = scraper.cleanup_html(
        "https://example.org",
        """<html><head>
    <meta name="date" content="not-a-date"><meta property="og:image" content="https://images.example/logo.png">
    <link rel="icon" href="/favicon.ico"></head><body><p>A readable article with enough text to be extracted.</p></body></html>""",
    )
    assert metadata["publication_date"] is None
    assert metadata["logo"] == "https://images.example/logo.png"


async def test_parse_html_merges_metadata_and_deduplicates_links(scraper, monkeypatch):
    monkeypatch.setattr(scraper, "cleanup_html", Mock(return_value=("Title", "Body", ["https://example.org/a"] * 2, {"author": "Ada", "description": None})))
    document = await scraper.parse_html("https://example.org", "html", {"description": "Keep description", "rank": 1})
    assert document.content == "Body"
    assert document.metadata == {"description": "Keep description", "rank": 1, "title": "Title", "author": "Ada", "content_length": 4, "links": ["https://example.org/a"]}


async def test_parse_html_challenge_returns_empty_content(scraper):
    result = await scraper.parse_html("https://example.org", "<p>captcha</p>")
    assert result.content == ""
    assert result.metadata == {}


async def test_parse_pdf_extracts_pages_title_and_description(scraper):
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "First page")
        pdf.new_page().insert_text((72, 72), "Second page")
        pdf.set_metadata({"title": "Research PDF", "subject": "A detailed study"})
        data = pdf.tobytes()
    result = await scraper.parse_pdf(data, {"rank": 1})
    assert "First page" in result.content
    assert "\f" in result.content
    assert "Second page" in result.content
    assert result.metadata == {"rank": 1, "title": "Research PDF", "description": "A detailed study"}


async def test_parse_pdf_uses_text_as_description_when_metadata_absent(scraper):
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Plain PDF text")
        data = pdf.tobytes()
    result = await scraper.parse_pdf(data)
    assert result.metadata["description"] == "Plain PDF text..."


async def test_parse_empty_pdf_returns_none(scraper):
    with pymupdf.open() as pdf:
        pdf.new_page()
        data = pdf.tobytes()
    assert await scraper.parse_pdf(data) is None


async def test_parse_invalid_pdf_raises(scraper):
    with pytest.raises(pymupdf.FileDataError):
        await scraper.parse_pdf(b"not a pdf")


def test_internal_links_require_matching_host_and_drop_queries_fragments(scraper):
    links = [
        "/paper?x=1#top",
        "https://example.org/paper#other",
        "//example.org/second",
        "mailto:a@example.org",
        "javascript:void(0)",
        "https://evil.example/example.org/page",
        "https://example.org.evil.example/page",
        "",
        None,
    ]
    assert set(scraper._process_links("https://example.org/index", links)) == {"https://example.org/paper", "https://example.org/second"}


@pytest.mark.parametrize(("url", "expected"), [("example.org/a", "example.org"), ("http://example.org:8080/a", "example.org:8080")])
def test_parse_domain_accepts_schemeless_and_explicit_urls(scraper, url, expected):
    assert scraper._parse_domain(url) == expected


def test_clean_string_replaces_control_characters(scraper):
    assert scraper._clean_string("a\nb\tc\rd\fe") == "a b c d e"


async def test_batch_rejects_mismatched_metadata_before_fetching(scraper, monkeypatch):
    fetch = AsyncMock()
    monkeypatch.setattr(scraper, "fetch_content", fetch)
    with pytest.raises(ValueError, match="number of URLs"):
        await scraper.batch_fetch_content(["https://example.org"], [])
    fetch.assert_not_awaited()


async def test_empty_batch_returns_empty_list(scraper):
    assert await scraper.batch_fetch_content([]) == []


async def test_batch_preserves_order_flattens_documents_and_isolates_failure(scraper, monkeypatch):
    docs = [Document(content="First", metadata={}), Document(content="Last", metadata={})]
    fetch = AsyncMock(side_effect=[[docs[0]], RuntimeError("One failure"), [docs[1]]])
    monkeypatch.setattr(scraper, "fetch_content", fetch)
    assert await scraper.batch_fetch_content(["first", "bad", "last"], [{"n": 1}, {"n": 2}, {"n": 3}]) == docs
    assert [call.args for call in fetch.await_args_list] == [("first", {"n": 1}), ("bad", {"n": 2}), ("last", {"n": 3})]


async def test_batch_enforces_concurrency_limit(monkeypatch):
    scraper = WebScrapper(max_concurrency=2)
    active = peak = 0
    release = asyncio.Event()
    reached_limit = asyncio.Event()

    async def fetch(url, metadata):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 2:
            reached_limit.set()
        await release.wait()
        active -= 1
        return []

    monkeypatch.setattr(scraper, "fetch_content", fetch)
    task = asyncio.create_task(scraper.batch_fetch_content(["one", "two", "three"]))
    try:
        await asyncio.wait_for(reached_limit.wait(), timeout=1)
        assert active == 2
    finally:
        release.set()
        await task
    assert peak == 2
