"""Provider and Redis contracts without network or credentials."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from lumis.tools.search import search_engine_client as module

import pytest


@pytest.fixture
def client():
    instance = module.SearchEngineClient(tavily_client=SimpleNamespace(search=Mock()))
    yield instance
    instance.executor.shutdown(wait=True)


@pytest.fixture
def redis():
    return SimpleNamespace(get=AsyncMock(return_value=None), set=AsyncMock(), delete=AsyncMock(), scan_iter=Mock())


async def test_google_consumes_lazy_results_in_worker_thread(client, monkeypatch):
    caller_thread = threading.get_ident()

    def google(query, num_results, advanced):
        # A generator's body executes only while iterating.
        assert threading.get_ident() != caller_thread
        assert (query, num_results, advanced) == ("q", 2, True)
        yield SimpleNamespace(title="Title", url="https://example.test", description=None)
        yield SimpleNamespace()

    monkeypatch.setattr(module, "google_search", google)
    assert await client.search_google("q", 2) == [
        {"title": "Title", "url": "https://example.test", "description": "", "raw_content": None},
        {"title": "", "url": "", "description": "", "raw_content": None},
    ]


async def test_google_generator_error_propagates(client, monkeypatch):
    def broken(*args, **kwargs):
        yield SimpleNamespace()
        raise RuntimeError("provider failed")

    monkeypatch.setattr(module, "google_search", broken)
    with pytest.raises(RuntimeError, match="provider failed"):
        await client.search_google("q")


async def test_tavily_maps_content_and_preserves_absent_raw_content(client):
    client.tavily.search.return_value = {
        "results": [
            {"title": "Title", "url": "url", "content": "snippet", "raw_content": "full"},
            {"description": "legacy snippet"},
            {},
        ]
    }
    assert await client.search_tavily("q") == [
        {"title": "Title", "url": "url", "description": "snippet", "raw_content": "full"},
        {"title": "", "url": "", "description": "legacy snippet", "raw_content": None},
        {"title": "", "url": "", "description": "", "raw_content": None},
    ]


async def test_tavily_forwards_search_controls_and_runs_off_loop(client):
    caller_thread = threading.get_ident()

    def search(**kwargs):
        assert threading.get_ident() != caller_thread
        return {}

    client.tavily.search.side_effect = search
    assert (
        await client.search_tavily(
            "q",
            search_depth="advanced",
            topic="news",
            days=7,
            max_results=2,
            include_domains=["allowed.test"],
            exclude_domains=["blocked.test"],
            include_answer=True,
            include_raw_content=False,
            include_images=True,
            country="canada",
        )
        == []
    )
    client.tavily.search.assert_called_once_with(
        query="q",
        search_depth="advanced",
        topic="news",
        days=7,
        max_results=2,
        include_domains=["allowed.test"],
        exclude_domains=["blocked.test"],
        include_answer=True,
        include_raw_content=False,
        include_images=True,
        country="canada",
    )


async def test_news_awaits_tavily_and_forwards_options(client, monkeypatch):
    tavily = AsyncMock(return_value=[{"title": "News"}])
    monkeypatch.setattr(client, "search_tavily", tavily)
    result = await client.search_news("q", topic="news", days=10, max_results=2, include_domains=["example.test"])
    assert result == [{"title": "News"}]
    tavily.assert_awaited_once_with(
        query="q",
        search_depth="basic",
        topic="news",
        days=10,
        max_results=2,
        include_domains=["example.test"],
        exclude_domains=[],
        include_answer=False,
        include_raw_content=True,
        include_images=False,
    )


async def test_optional_sdk_not_required_to_construct_or_use_google(monkeypatch):
    monkeypatch.setattr(module, "TavilyClient", None)
    monkeypatch.setattr(module, "google_search", Mock(return_value=[SimpleNamespace(title="Google")]))
    instance = module.SearchEngineClient()
    try:
        result = await instance.search("q", search_engine="google")
        assert result[0]["title"] == "Google"
        with pytest.raises(ImportError, match=r"pip install lumis-ai\[search\]"):
            await instance.search_tavily("q")
    finally:
        await instance.shutdown()


async def test_default_tavily_client_is_created_lazily_once(monkeypatch):
    sdk_client = SimpleNamespace(search=Mock(return_value={"results": []}))
    constructor = Mock(return_value=sdk_client)
    monkeypatch.setattr(module, "TavilyClient", constructor)
    instance = module.SearchEngineClient()
    try:
        constructor.assert_not_called()
        await instance.search_tavily("first")
        await instance.search_tavily("second")
        constructor.assert_called_once_with()
    finally:
        await instance.shutdown()


@pytest.mark.parametrize("error", [RuntimeError("provider failed"), module.UsageLimitExceededError("quota")])
async def test_tavily_errors_propagate(client, error):
    client.tavily.search.side_effect = error
    with pytest.raises(type(error), match=str(error)):
        await client.search_tavily("q")


@pytest.mark.parametrize("google_result", [[], RuntimeError("Google failed")])
async def test_google_failure_or_empty_results_falls_back_to_tavily(client, monkeypatch, google_result):
    google = AsyncMock()
    if isinstance(google_result, Exception):
        google.side_effect = google_result
    else:
        google.return_value = google_result
    tavily = AsyncMock(return_value=[{"title": "Fallback"}])
    monkeypatch.setattr(client, "search_google", google)
    monkeypatch.setattr(client, "search_tavily", tavily)
    assert await client.search("q", search_engine="google", topic="news", max_results=3) == [{"title": "Fallback"}]
    tavily.assert_awaited_once_with("q", max_results=3, topic="news")


async def test_failed_fallback_propagates_final_error(client, monkeypatch):
    monkeypatch.setattr(client, "search_google", AsyncMock(side_effect=RuntimeError("google")))
    monkeypatch.setattr(client, "search_tavily", AsyncMock(side_effect=ValueError("fallback")))
    with pytest.raises(ValueError, match="fallback"):
        await client.search("q", search_engine="google")


async def test_tavily_failure_is_not_retried(client, monkeypatch):
    tavily = AsyncMock(side_effect=RuntimeError("failed"))
    monkeypatch.setattr(client, "search_tavily", tavily)
    with pytest.raises(RuntimeError, match="failed"):
        await client.search("q")
    tavily.assert_awaited_once()


async def test_cache_hit_avoids_provider_and_write(client, redis, monkeypatch):
    client.redis = redis
    redis.get.return_value = json.dumps([{"title": "Cached"}]).encode()
    provider = AsyncMock()
    monkeypatch.setattr(client, "search_tavily", provider)
    assert await client.search("q") == [{"title": "Cached"}]
    provider.assert_not_awaited()
    redis.set.assert_not_awaited()


@pytest.mark.parametrize("cached", [None, "broken json"])
async def test_cache_miss_or_invalid_json_fetches_and_writes_with_ttl(client, redis, monkeypatch, cached):
    client.redis = redis
    client.cache_ttl = 123
    redis.get.return_value = cached
    monkeypatch.setattr(client, "search_tavily", AsyncMock(return_value=[{"title": "Fresh"}]))
    assert await client.search("q") == [{"title": "Fresh"}]
    key = redis.get.call_args.args[0]
    redis.set.assert_awaited_once_with(key, json.dumps([{"title": "Fresh"}]), ex=123)


async def test_cache_write_failure_does_not_discard_results(client, redis, monkeypatch):
    client.redis = redis
    redis.set.side_effect = RuntimeError("Redis unavailable")
    monkeypatch.setattr(client, "search_tavily", AsyncMock(return_value=[]))
    assert await client.search("q") == []


async def test_cache_distinguishes_result_limit_and_normalizes_query(client, redis, monkeypatch):
    client.redis = redis
    monkeypatch.setattr(client, "search_tavily", AsyncMock(return_value=[]))
    await client.search("  Query  ", max_results=1)
    await client.search("query", max_results=10)
    await client.search("query", max_results=1)
    keys = [call.args[0] for call in redis.get.call_args_list]
    assert keys[0] != keys[1]
    assert keys[0] == keys[2]
    assert client._generate_cache_key("query", "google", "general") != keys[1]
    assert client._generate_cache_key("query", "tavily", "news") != keys[1]


def scan_keys(redis, keys):
    async def scan(**kwargs):
        assert kwargs == {"match": "search:*"}
        for key in keys:
            yield key

    redis.scan_iter.side_effect = scan


async def test_targeted_clear_removes_all_limits_and_legacy_key_only(client, redis):
    client.redis = redis
    prefix = "search:tavily:general:q[*]"
    selected = [prefix, f"{prefix}:limit:1", f"{prefix}:limit:10".encode()]
    scan_keys(redis, selected + ["search:tavily:general:other:limit:1", f"{prefix}:extra:limit:1"])
    await client.clear_cache(" Q[*] ", "tavily", "general")
    redis.delete.assert_awaited_once_with(*selected)


@pytest.mark.parametrize("keys", [[], [b"search:one", b"search:two"]])
async def test_clear_all_handles_empty_cache(client, redis, keys):
    client.redis = redis
    scan_keys(redis, keys)
    await client.clear_cache()
    if keys:
        redis.delete.assert_awaited_once_with(*keys)
    else:
        redis.delete.assert_not_awaited()


async def test_cache_clear_without_redis_is_noop(client):
    await client.clear_cache()


async def test_clear_scan_failure_is_handled(client, redis):
    client.redis = redis
    redis.scan_iter.side_effect = RuntimeError("scan failed")
    await client.clear_cache()
    redis.delete.assert_not_awaited()


def test_optional_sdk_import_uses_tavily_package(monkeypatch):
    import runpy
    import sys

    constructor = Mock()

    class QuotaError(Exception):
        pass

    monkeypatch.setitem(sys.modules, "tavily", SimpleNamespace(TavilyClient=constructor))
    monkeypatch.setitem(sys.modules, "tavily.errors", SimpleNamespace(UsageLimitExceededError=QuotaError))
    loaded = runpy.run_path(module.__file__)
    assert loaded["TavilyClient"] is constructor
    assert loaded["UsageLimitExceededError"] is QuotaError
