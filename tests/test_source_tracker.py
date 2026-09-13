"""Source tracking and hook contracts with injected, deterministic embeddings."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from lumis.agents.research_agent.source_models import SourceType
from lumis.agents.research_agent.source_tracker import ResearchAgentHooks, ResearchSourceTracker
from lumis.agents.research_agent.types import ResearchAgentResponse

import numpy as np
import pytest


def source(title="paper", url="https://example.com/paper", **kwargs):
    return SourceType(title=title, url=url, summary="summary", source_type="web", **kwargs)


@pytest.fixture
def tracker():
    return ResearchSourceTracker(embedding_model=SimpleNamespace(aembed=AsyncMock()))


@pytest.fixture
def hooks():
    return ResearchAgentHooks()


def test_source_registration_uses_url_or_title_and_replaces_duplicate_key(tracker):
    first = source()
    replacement = source(title="updated")
    untitled_url = source(title="offline", url=None)
    tracker.add_source(first)
    tracker.add_source(replacement)
    tracker.add_source(untitled_url)
    assert tracker.get_source_by_url(first.url) is replacement
    assert tracker.get_source_by_url("missing") is None
    assert tracker.sources["offline"] is untitled_url
    assert tracker.get_sources() == [replacement.model_dump(), untitled_url.model_dump()]


def test_trackers_have_independent_mutable_defaults(tracker):
    other = ResearchSourceTracker(embedding_model=tracker.embedding_model)
    tracker.add_source(source())
    tracker.insights.append(ResearchAgentResponse(response="insight"))
    assert other.sources == {}
    assert other.insights == []


def test_clear_removes_sources_and_query(tracker):
    tracker.add_source(source())
    tracker.query = "old query"
    tracker.clear()
    tracker.clear()
    assert tracker.sources == {}
    assert tracker.query == ""


async def test_empty_tracker_never_embeds(tracker):
    assert await tracker.find_relevant_sources("anything") == []
    tracker.embedding_model.aembed.assert_not_awaited()


async def test_explicit_url_match_has_full_confidence_and_skips_embedding(tracker):
    item = source()
    tracker.add_source(item)
    matches = await tracker.find_relevant_sources(f"See {item.url}.")
    assert len(matches) == 1
    assert matches[0].source is item
    assert matches[0].confidence == 1
    assert matches[0].match_type == "explicit"
    tracker.embedding_model.aembed.assert_not_awaited()


async def test_semantic_matches_use_cosine_threshold_and_sort_by_confidence(tracker):
    items = [source(title=name, url=f"https://example.com/{name}") for name in ["threshold", "best", "opposite"]]
    for item in items:
        tracker.add_source(item)
    tracker.embedding_model.aembed.side_effect = [np.array([2.0, 0.0]), np.array([[3.0, 4.0], [5.0, 0.0], [-2.0, 0.0]])]
    matches = await tracker.find_relevant_sources("query", threshold=0.6)
    assert [match.source for match in matches] == [items[1], items[0]]
    assert [match.confidence for match in matches] == pytest.approx([1, 0.6])
    assert all(match.match_type == "semantic" for match in matches)
    assert tracker.embedding_model.aembed.await_args_list[0].args == ("query",)
    assert tracker.embedding_model.aembed.await_args_list[1].args == ([f"{item.title} summary" for item in items],)


async def test_semantic_batch_excludes_explicit_sources(tracker):
    explicit, semantic = source(), source(title="other", url=None)
    tracker.add_source(explicit)
    tracker.add_source(semantic)
    tracker.embedding_model.aembed.side_effect = [np.array([1.0, 0.0]), np.array([[4.0, 3.0]])]
    matches = await tracker.find_relevant_sources(explicit.url)
    assert [match.source for match in matches] == [explicit, semantic]
    assert [match.match_type for match in matches] == ["explicit", "semantic"]
    assert tracker.embedding_model.aembed.await_args_list[1].args == (["other summary"],)


async def test_semantic_match_below_threshold_is_omitted(tracker):
    tracker.add_source(source())
    tracker.embedding_model.aembed.side_effect = [np.array([1.0, 0.0]), np.array([[0.0, 1.0]])]
    assert await tracker.find_relevant_sources("query", threshold=0.5) == []


async def test_on_start_resets_source_state_and_timestamp(tracker, hooks):
    tracker.add_source(source())
    tracker.timestamp = datetime(2000, 1, 1)
    wrapper = SimpleNamespace(context=tracker)
    before = datetime.now()
    await hooks.on_start(wrapper, Mock())
    assert tracker.sources == {}
    assert tracker.query
    assert before <= tracker.timestamp <= datetime.now()


async def test_on_start_ignores_unrelated_context(hooks):
    context = SimpleNamespace(context={"keep": True})
    await hooks.on_start(context, Mock())
    assert context.context == {"keep": True}


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_on_end_saves_insight_before_callback(tracker, asynchronous):
    wrapper, agent = SimpleNamespace(context=tracker), Mock()
    output = ResearchAgentResponse(response="answer")

    def callback(context, actual_agent, actual_output):
        assert tracker.insights == [output]
        assert (context, actual_agent, actual_output) == (wrapper, agent, output)

    handler = AsyncMock(side_effect=callback) if asynchronous else Mock(side_effect=callback)
    await ResearchAgentHooks(handler).on_end(wrapper, agent, output)
    handler.assert_called_once_with(wrapper, agent, output)
    if asynchronous:
        handler.assert_awaited_once()


async def test_on_end_ignores_unstructured_output_but_calls_callback(tracker):
    callback = Mock()
    await ResearchAgentHooks(callback).on_end(SimpleNamespace(context=tracker), Mock(), "raw output")
    assert tracker.insights == []
    callback.assert_called_once()


async def test_on_end_supports_unrelated_context_and_no_callback(hooks):
    await hooks.on_end(SimpleNamespace(context={}), Mock(), ResearchAgentResponse(response="answer"))


@pytest.mark.parametrize(
    "item,expected",
    [
        ({}, (None, None)),
        ({"published_date": "2024-01-02T03:04:05", "updated_date": "2025-02-03"}, (datetime(2024, 1, 2, 3, 4, 5), datetime(2025, 2, 3))),
        ({"published_date": None, "updated_date": "invalid"}, (None, None)),
    ],
)
def test_arxiv_date_parsing_tolerates_missing_and_invalid_dates(hooks, item, expected):
    assert hooks._parse_arxiv_dates(item) == expected


def test_arxiv_handler_keeps_paper_details_and_metadata(hooks):
    item = {"url": "https://arxiv.org/abs/1234", "title": "Paper", "abstract": "Finding", "categories": ["cs.AI"], "authors": ["Author"], "published_date": "2024-01-01", "metadata": {"custom": 7}}
    result = hooks._handle_arxiv_source(item, "search_arxiv")
    assert result.source_type == "arxiv"
    assert result.summary == "Finding"
    assert result.keywords == ["cs.AI"]
    assert result.credibility == 0.9
    assert result.metadata["authors"] == ["Author"]
    assert result.metadata["published_date"] == datetime(2024, 1, 1)
    assert result.metadata["custom"] == 7


@pytest.mark.parametrize(
    "handler,expected_type,summary_field,extra",
    [
        ("_handle_wiki_source", "wiki", "summary", {"content": "body", "categories": ["science"]}),
        ("_handle_openai_web_source", "web", "description", {"raw_content": "body"}),
        ("_handle_gemini_source", "gemini", "summary", {"content": "body", "segments": ["part"], "confidence": 0.8, "metadata": {"extra": True}}),
    ],
)
def test_source_handlers_preserve_type_summary_and_content(hooks, handler, expected_type, summary_field, extra):
    result = getattr(hooks, handler)({"title": "Title", "url": "https://example.com", summary_field: "Summary", **extra}, "tool")
    assert result.source_type == expected_type
    assert result.summary == "Summary"
    assert result.metadata["content"] == "body"
    if expected_type == "gemini":
        assert result.metadata["segments"] == ["part"]
        assert result.metadata["confidence"] == 0.8
        assert result.metadata["extra"] is True


@pytest.mark.parametrize("handler", ["_handle_arxiv_source", "_handle_wiki_source", "_handle_openai_web_source", "_handle_gemini_source"])
def test_handlers_drop_invalid_sources(hooks, handler):
    assert getattr(hooks, handler)({"title": None}, "tool") is None


@pytest.mark.parametrize(
    "tool_name,expected", [("search_arxiv", "arxiv"), ("wiki_search", "wiki"), ("web_search", "gemini"), ("gemini_search", "gemini"), ("openai_search", "web"), ("openai_web_search", "web")]
)
def test_process_source_routes_tool_names(tracker, hooks, tool_name, expected):
    hooks._process_source(SimpleNamespace(context=tracker), {"title": "Title", "description": "Description"}, tool_name)
    assert tracker.sources["Title"].source_type == expected


def test_process_source_ignores_unknown_tools_and_unrelated_context(tracker, hooks):
    hooks._process_source(SimpleNamespace(context=tracker), {"title": "Title"}, "unknown")
    hooks._process_source(SimpleNamespace(context={}), {"title": "Title"}, "wiki_search")
    assert tracker.sources == {}


@pytest.mark.parametrize(
    "result",
    [
        [{"title": "Title"}, "ignored", 42],
        {"title": "Title"},
        {"results": [{"title": "Title"}, None]},
        {"answer": "Answer", "sources": [{"title": "Title"}, None]},
    ],
)
async def test_tool_end_handles_supported_envelopes(tracker, hooks, result):
    await hooks.on_tool_end(SimpleNamespace(context=tracker), Mock(), SimpleNamespace(name="web_search"), result)
    assert list(tracker.sources) == ["Title"]


@pytest.mark.parametrize("result", [None, "plain text", {"unrelated": 1}])
async def test_tool_end_ignores_non_source_results(tracker, hooks, result):
    await hooks.on_tool_end(SimpleNamespace(context=tracker), Mock(), SimpleNamespace(name="web_search"), result)
    assert tracker.sources == {}


async def test_tool_end_logs_processing_errors_without_interrupting_agent(tracker, hooks, monkeypatch, caplog):
    monkeypatch.setattr(hooks, "_process_source", Mock(side_effect=RuntimeError("bad source")))
    await hooks.on_tool_end(SimpleNamespace(context=tracker), Mock(), SimpleNamespace(name="web_search"), {"title": "Title"})
    assert "bad source" in caplog.text
