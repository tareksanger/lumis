"""Wikipedia SDK boundaries are replaced; no HTTP or rate-limit waits occur."""

import asyncio
from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

from lumis.tools.search import wiki as module

import pytest


@pytest.fixture
def sdk(monkeypatch):
    fake = SimpleNamespace(**{name: Mock() for name in ("set_user_agent", "set_rate_limiting", "set_lang", "search", "page", "random", "suggest", "geosearch", "summary")})
    monkeypatch.setattr(module, "wikipedia", fake)
    return fake


@pytest.fixture
def searcher(sdk):
    instance = module.WikipediaSearcher(user_agent="test-agent", rate_limit=False, max_workers=2)
    yield instance
    instance._executor.shutdown(wait=True)
    instance._get_page_sync.cache_clear()
    instance._get_summary_sync.cache_clear()


def test_initialization_configures_sdk(sdk, searcher):
    sdk.set_user_agent.assert_called_once_with("test-agent")
    sdk.set_rate_limiting.assert_called_once_with(False, timedelta(milliseconds=50))


async def test_page_cache_is_language_specific(sdk, searcher):
    sdk.page.side_effect = [SimpleNamespace(title="English"), SimpleNamespace(title="French")]
    english = await searcher.get_page_details("Earth")
    assert await searcher.get_page_details("Earth") is english
    french = await searcher.get_page_details("Earth", lang="fr")
    assert french.title == "French"
    assert sdk.page.call_count == 2
    assert [call.args for call in sdk.set_lang.call_args_list] == [("en",), ("fr",)]


@pytest.mark.parametrize("error", [module.PageError("missing"), module.DisambiguationError("Mercury", ["planet", "element"]), RuntimeError("upstream")])
async def test_page_errors_return_none(sdk, searcher, error):
    sdk.page.side_effect = error
    assert await searcher.get_page_details("missing") is None


async def test_search_preserves_sdk_order_and_filters_missing_pages(sdk, searcher):
    sdk.search.return_value = ["first", "missing", "last"]
    pages = {"first": SimpleNamespace(title="first"), "last": SimpleNamespace(title="last")}

    def page(title):
        if title == "missing":
            raise module.PageError(title)
        return pages[title]

    sdk.page.side_effect = page
    assert await searcher.search("query", num_results=3, lang="de") == list(pages.values())
    sdk.search.assert_called_once_with("query", 3)


@pytest.mark.parametrize(
    "method, args, sdk_method, expected",
    [
        ("search", ("q",), "search", []),
        ("get_random_pages", (), "random", []),
        ("get_suggestion", ("q",), "suggest", None),
        ("geosearch", (1, 2), "geosearch", []),
        ("search_with_suggestion", ("q",), "search", ([], None)),
    ],
)
async def test_public_sdk_failures_return_empty_result(sdk, searcher, method, args, sdk_method, expected):
    getattr(sdk, sdk_method).side_effect = RuntimeError("unavailable")
    assert await getattr(searcher, method)(*args) == expected


async def test_empty_search_skips_page_fetches(sdk, searcher):
    sdk.search.return_value = []
    assert await searcher.search("q") == []
    sdk.page.assert_not_called()


@pytest.mark.parametrize("titles", ["single", ["first", "second"]])
async def test_random_pages_normalizes_single_title_and_list(sdk, searcher, titles):
    sdk.random.return_value = titles
    sdk.page.side_effect = lambda title: SimpleNamespace(title=title)
    expected = [titles] if isinstance(titles, str) else titles
    assert [p.title for p in await searcher.get_random_pages(len(expected))] == expected
    sdk.random.assert_called_once_with(len(expected))


@pytest.mark.parametrize("value", ["correction", None])
async def test_suggestion_preserves_sdk_result(sdk, searcher, value):
    sdk.suggest.return_value = value
    assert await searcher.get_suggestion("typo", lang="es") == value
    sdk.suggest.assert_called_once_with("typo")
    sdk.set_lang.assert_called_once_with("es")


async def test_geosearch_passes_coordinates_radius_and_limit(sdk, searcher):
    sdk.geosearch.return_value = ["nearby"]
    page = SimpleNamespace(title="nearby")
    sdk.page.return_value = page
    lat, lon = Decimal("43.65"), Decimal("-79.38")
    assert await searcher.geosearch(lat, lon, title="Toronto", radius=2500, num_results=2) == [page]
    sdk.geosearch.assert_called_once_with(latitude=lat, longitude=lon, title="Toronto", results=2, radius=2500)


async def test_empty_geosearch_skips_page_fetches(sdk, searcher):
    sdk.geosearch.return_value = []
    assert await searcher.geosearch(0, 0) == []
    sdk.page.assert_not_called()


@pytest.mark.parametrize("titles", [[], ["one"]])
async def test_search_with_suggestion_retains_suggestion_even_without_pages(sdk, searcher, titles):
    sdk.search.return_value = (titles, "correction")
    sdk.page.side_effect = lambda title: SimpleNamespace(title=title)
    pages, suggestion = await searcher.search_with_suggestion("typo", num_results=2)
    assert [p.title for p in pages] == titles
    assert suggestion == "correction"
    sdk.search.assert_called_once_with("typo", results=2, suggestion=True)


async def test_summary_cache_separates_format_and_language(sdk, searcher):
    sdk.summary.side_effect = ["short", "long", "French"]
    assert await searcher.get_summary("Earth", sentences=1, chars=20, auto_suggest=False) == "short"
    assert await searcher.get_summary("Earth", sentences=1, chars=20, auto_suggest=False) == "short"
    assert await searcher.get_summary("Earth", sentences=2) == "long"
    assert await searcher.get_summary("Earth", sentences=2, lang="fr") == "French"
    assert sdk.summary.call_count == 3
    assert sdk.summary.call_args_list[0].kwargs == {"sentences": 1, "chars": 20, "auto_suggest": False}


@pytest.mark.parametrize("error", [module.PageError("missing"), module.DisambiguationError("Mercury", ["planet", "element"]), RuntimeError("upstream")])
async def test_summary_errors_return_none(sdk, searcher, error):
    sdk.summary.side_effect = error
    assert await searcher.get_summary("missing") is None


async def test_concurrent_languages_are_set_when_sdk_work_runs(sdk, searcher, monkeypatch):
    # Hold dispatch until both calls are queued. With set_lang before dispatch,
    # both would see the second language instead of their own language.
    ready = asyncio.Event()
    pending = 0
    state = {}
    sdk.set_lang.side_effect = lambda lang: state.update(lang=lang)
    sdk.suggest.side_effect = lambda query: f"{state['lang']}:{query}"
    actual_executor = searcher._async_executor

    async def coordinated_executor(func, *args, **kwargs):
        nonlocal pending
        pending += 1
        if pending == 2:
            ready.set()
        await ready.wait()
        return await actual_executor(func, *args, **kwargs)

    monkeypatch.setattr(searcher, "_async_executor", coordinated_executor)
    result = await asyncio.wait_for(asyncio.gather(searcher.get_suggestion("hello", lang="en"), searcher.get_suggestion("bonjour", lang="fr")), 2)
    assert result == ["en:hello", "fr:bonjour"]


async def test_executor_runs_off_event_loop_and_supports_keyword_arguments(searcher):
    import threading

    caller_thread = threading.get_ident()

    def work(value, *, increment):
        assert threading.get_ident() != caller_thread
        return value + increment

    assert await searcher._async_executor(work, 4, increment=3) == 7


def test_destructor_handles_partial_initialization():
    instance = module.WikipediaSearcher.__new__(module.WikipediaSearcher)
    instance.__del__()
