from datetime import datetime
from enum import Enum
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

from lumis.agents.research_agent.agent import create_research_agent, simple_prompt_guardrail
import lumis.agents.research_agent.tools.search as module
from lumis.agents.research_agent.types import ResearchAgentResponse
from lumis.tools.search.arxiv import ArxivResult

import pytest


def paper(title, content):
    return ArxivResult(
        arxiv_id=title,
        title=title,
        authors=["Author"],
        abstract="abstract",
        pdf_url="https://example.org/paper",
        published_date=datetime(2025, 1, 1),
        updated_date=datetime(2025, 1, 1),
        categories=["cs.AI"],
        primary_category="cs.AI",
        content=content,
        comment=None,
        journal_ref=None,
        doi=None,
    )


@pytest.fixture
def arxiv_search(monkeypatch):
    arxiv = ModuleType("arxiv")
    arxiv.SortCriterion = Enum("SortCriterion", {"SubmittedDate": "submittedDate"})
    arxiv.SortOrder = Enum("SortOrder", {"Descending": "descending"})
    monkeypatch.setitem(sys.modules, "arxiv", arxiv)
    client = SimpleNamespace(search=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(module, "ArxivSearcher", Mock(return_value=client))
    return client


async def test_arxiv_summaries_stay_attached_to_papers_with_content(monkeypatch, arxiv_search):
    arxiv_search.search.return_value = [paper("missing", None), paper("present", "full text")]
    summarize = AsyncMock(return_value="summary")
    monkeypatch.setattr(module, "_read_arxiv_pdf", summarize)
    results = await module.search_arxiv("query", max_results=2)
    assert results[0]["content"] is None
    assert results[1]["content"] == "summary"
    assert summarize.await_count == 1
    arxiv_search.close.assert_awaited_once()


async def test_arxiv_summary_failure_and_search_failure_cleanup(monkeypatch, arxiv_search):
    arxiv_search.search.return_value = [paper("bad", "text")]
    monkeypatch.setattr(module, "_read_arxiv_pdf", AsyncMock(side_effect=ValueError("failed")))
    assert (await module.search_arxiv("query"))[0]["content"] is None
    arxiv_search.search.side_effect = RuntimeError("search failed")
    with pytest.raises(RuntimeError, match="search failed"):
        await module.search_arxiv("query")
    assert arxiv_search.close.await_count == 2


@pytest.mark.parametrize("with_semaphore", [False, True])
async def test_read_pdf_delegates_summary(monkeypatch, with_semaphore):
    import asyncio

    content = paper("paper", "text")
    summary = AsyncMock(return_value="summary")
    monkeypatch.setattr(module, "_generate_summary", summary)
    semaphore = asyncio.Semaphore(1) if with_semaphore else None
    assert await module._read_arxiv_pdf("query", content, semaphore=semaphore) == "summary"
    summary.assert_awaited_once_with("query", content, "concise")


@pytest.mark.parametrize("present", [False, True])
async def test_summary_and_web_search_use_mocked_gemini(monkeypatch, present):
    gemini = SimpleNamespace(
        generate_content=AsyncMock(return_value=SimpleNamespace(text="summary") if present else None),
        extract_response_sources_and_answer=AsyncMock(return_value=SimpleNamespace(model_dump=lambda: {"answer": "yes"})),
    )
    monkeypatch.setattr(module, "Gemini", Mock(return_value=gemini))
    assert await module._generate_summary("query", paper("paper", "content"), "concise") == ("summary" if present else None)
    assert "content" in gemini.generate_content.call_args.kwargs["contents"]
    assert await module.web_search("question") == {"answer": "yes"}
    gemini.generate_content.assert_awaited_with(contents="question", use_search=True)
    gemini.generate_content.side_effect = RuntimeError("offline")
    assert await module.web_search("question") is None


async def test_wiki_results_are_normalized_and_capped(monkeypatch):
    page = SimpleNamespace(title="Title", summary="summary", url="url", content="content", categories=["category"])
    wiki = SimpleNamespace(search=AsyncMock(return_value=[page]))
    monkeypatch.setattr(module, "wiki", wiki)
    assert await module.wiki_search("q", num_results=10) == [vars(page)]
    wiki.search.assert_awaited_once_with("q", 5)


def test_research_agent_factory_wires_tools_hooks_and_output():
    callback = Mock()
    agent = create_research_agent(model="gpt-4o-mini", on_end_callback=callback)
    assert agent.model == "gpt-4o-mini"
    assert agent.output_type is ResearchAgentResponse
    assert {t.name for t in agent.tools} == {"search_arxiv", "wiki_search", "web_search"}
    assert agent.input_guardrails == [simple_prompt_guardrail]


@pytest.mark.parametrize("text,blocked", [("What is a star?", False), ("Explain everything about physics", True), ("word " * 51, True)])
def test_guardrail_flags_overbroad_inputs(text, blocked):
    output = simple_prompt_guardrail.guardrail_function(None, None, text)
    assert output.tripwire_triggered is blocked
