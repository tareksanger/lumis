"""Gemini adapter contracts, using SDK response models and in-memory clients."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import lumis.llm.gemini_llm as module
from lumis.llm.gemini_llm import _get_final_url, Gemini

from google.genai.errors import ClientError, ServerError
from google.genai.types import GenerateContentConfig, GenerateContentResponse, GenerateImagesResponse
import httpx
import pytest
from tenacity import wait_none


@pytest.fixture
def client():
    methods = {name: AsyncMock() for name in ("generate_content", "generate_content_stream", "generate_images", "generate_videos")}
    return SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(**methods)))


@pytest.fixture
def gemini(client):
    return Gemini(client=client)


def text_response(text="Answer", **kwargs):
    return GenerateContentResponse(candidates=[{"content": {"parts": [{"text": text}]}}], **kwargs)


async def test_injected_client_and_content_request(gemini, client):
    response = text_response()
    client.aio.models.generate_content.return_value = response
    config = {"temperature": 0.2}
    assert await gemini.generate_content(model="chosen-model", contents="Question", config=config, use_search=False) is response
    assert gemini.client is client
    client.aio.models.generate_content.assert_awaited_once_with(
        model="chosen-model",
        contents=["Question"],
        config=config,
    )


def test_constructs_default_client(monkeypatch, client):
    factory = Mock(return_value=client)
    monkeypatch.setattr(module.genai, "Client", factory)
    monkeypatch.setattr(module, "API_KEY", "unit-test-placeholder")
    assert Gemini().client is client
    factory.assert_called_once_with(api_key="unit-test-placeholder")


@pytest.mark.parametrize("config", [None, {"temperature": 0.2}, GenerateContentConfig(temperature=0.2)])
async def test_search_configuration_preserves_options(gemini, client, config):
    client.aio.models.generate_content.return_value = text_response()
    await gemini.generate_content(contents=["Question"], config=config)
    sent = client.aio.models.generate_content.await_args.kwargs["config"]
    assert any(tool.google_search is not None for tool in sent.tools)
    assert sent.response_modalities == ["TEXT"]
    if config is not None:
        assert sent.temperature == 0.2
        if isinstance(config, dict):
            assert config == {"temperature": 0.2}
        else:
            assert config.tools is None


def test_token_counts_accumulate_numeric_usage_and_ignore_details(gemini):
    response = text_response(
        usage_metadata={
            "prompt_token_count": 4,
            "candidates_token_count": 2,
            "total_token_count": 6,
            "prompt_tokens_details": [{"modality": "TEXT", "token_count": 4}],
            "cache_tokens_details": [],
            "candidates_tokens_details": [],
        }
    )
    assert gemini._count_tokens(response) is response
    gemini._count_tokens(response)
    assert gemini.token_count == {"prompt_token_count": 8, "candidates_token_count": 4, "total_token_count": 12}


@pytest.mark.parametrize("response", [GenerateContentResponse(), GenerateImagesResponse()])
def test_token_count_no_usage_or_unsupported_response(gemini, response):
    assert gemini._count_tokens(response) is response
    assert gemini.token_count == {}


@pytest.mark.parametrize("method", ["generate_content", "generate_images", "generate_videos"])
async def test_transient_failures_retry_without_sleep(gemini, client, method):
    response = text_response()
    provider = getattr(client.aio.models, method)
    provider.side_effect = [RuntimeError("Temporary transport error"), response]
    call = getattr(gemini, method).retry_with(wait=wait_none())
    kwargs = {"contents": "Question", "use_search": False} if method == "generate_content" else {"prompt": "Scene"}
    assert await call(gemini, **kwargs) is response
    assert provider.await_count == 2


@pytest.mark.parametrize("method", ["generate_content", "generate_images", "generate_videos"])
async def test_server_error_is_propagated_without_retries(gemini, client, method):
    provider = getattr(client.aio.models, method)
    error = ServerError(503, {"error": {"message": "Unavailable"}})
    provider.side_effect = error
    kwargs = {"contents": "Question"} if method == "generate_content" else {"prompt": "Scene"}
    with pytest.raises(ServerError) as caught:
        await getattr(gemini, method)(**kwargs)
    assert caught.value is error
    assert provider.await_count == 1


@pytest.mark.parametrize("method", ["generate_images", "generate_videos"])
async def test_media_request_forwarding_and_middlewares(gemini, client, method):
    response = GenerateImagesResponse()
    getattr(client.aio.models, method).return_value = response
    transformed = object()
    middleware = Mock(return_value=transformed, __name__="transform")
    gemini.add_middleware(middleware)
    kwargs = dict(prompt="Scene", model="chosen-model", config={"number_of_images": 1} if method == "generate_images" else {"duration_seconds": 5})
    if method == "generate_videos":
        kwargs["image"] = {"image_bytes": b"fake-image", "mime_type": "image/png"}
    assert await getattr(gemini, method)(**kwargs) is transformed
    getattr(client.aio.models, method).assert_awaited_once_with(**kwargs)
    middleware.assert_called_once_with(response)


async def test_stream_iterates_sdk_async_iterator_and_counts_chunks(gemini, client):
    chunks = [text_response("A", usage_metadata={"total_token_count": 1}), text_response("B", usage_metadata={"total_token_count": 2})]

    async def stream():
        for chunk in chunks:
            yield chunk

    client.aio.models.generate_content_stream.return_value = stream()
    output = [chunk async for chunk in gemini.generate_content_stream(contents="Question", use_search=True)]
    assert output == chunks
    assert gemini.token_count["total_token_count"] == 3
    assert client.aio.models.generate_content_stream.await_args.kwargs["config"].tools[0].google_search is not None


@pytest.mark.parametrize("error_type", [ClientError, RuntimeError])
async def test_stream_reraises_client_and_transport_errors(gemini, client, error_type):
    error = ClientError(400, {"error": {"message": "Invalid request"}}) if error_type is ClientError else RuntimeError("Transport failed")
    client.aio.models.generate_content_stream.side_effect = error
    with pytest.raises(error_type):
        _ = [item async for item in gemini.generate_content_stream(contents="Question")]


async def test_stream_server_error_ends_stream(gemini, client):
    client.aio.models.generate_content_stream.side_effect = ServerError(503, {"error": {"message": "Unavailable"}})
    assert [item async for item in gemini.generate_content_stream(contents="Question")] == []


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308, 200])
async def test_final_url_redirect_and_nonredirect(monkeypatch, status):
    response = httpx.Response(status, headers={"location": "https://destination.example/path"}, request=httpx.Request("GET", "https://source.example"))
    transport = AsyncMock()
    transport.get.return_value = response
    context = AsyncMock()
    context.__aenter__.return_value = transport
    monkeypatch.setattr(module.httpx, "AsyncClient", Mock(return_value=context))
    expected = "https://destination.example/path" if status != 200 else "https://source.example"
    assert await _get_final_url("https://source.example") == expected
    transport.get.assert_awaited_once_with("https://source.example", follow_redirects=False)


async def test_final_url_transport_failure_returns_original(monkeypatch):
    context = AsyncMock()
    context.__aenter__.side_effect = RuntimeError("Offline")
    monkeypatch.setattr(module.httpx, "AsyncClient", Mock(return_value=context))
    assert await _get_final_url("https://source.example") == "https://source.example"


async def test_extracts_answer_without_sources():
    result = await Gemini.extract_response_sources_and_answer(text_response("A finding"))
    assert result.answer == "A finding"
    assert result.sources == []


async def test_grounding_indices_survive_deduplication_and_nonweb_chunks(monkeypatch):
    resolve = AsyncMock(side_effect=lambda url: url + "/resolved")
    monkeypatch.setattr(module, "_get_final_url", resolve)
    response = GenerateContentResponse(
        candidates=[
            {
                "content": {"parts": [{"text": "Answer"}]},
                "grounding_metadata": {
                    "grounding_chunks": [
                        {"web": {"uri": "https://a.example", "title": "A"}},
                        {"web": {"uri": "https://a.example", "title": "Duplicate A"}},
                        {},
                        {"web": {"uri": "https://b.example"}},
                    ],
                    "grounding_supports": [
                        {"segment": {"text": "First"}, "grounding_chunk_indices": [1, 3], "confidence_scores": [0.6, 0.9]},
                        {"segment": {"text": "Second"}, "grounding_chunk_indices": [0], "confidence_scores": [0.8]},
                        {"segment": {"text": "Ignored"}, "grounding_chunk_indices": [-1, 2, 99], "confidence_scores": [0.9, 0.9, 0.9]},
                    ],
                },
            }
        ]
    )
    result = await Gemini.extract_response_sources_and_answer(response)
    assert result.answer == "Answer"
    assert [source.url for source in result.sources] == ["https://a.example/resolved", "https://b.example/resolved"]
    assert [source.title for source in result.sources] == ["A", "Unknown Source"]
    assert [source.confidence for source in result.sources] == [0.8, 0.9]
    assert [[segment.text for segment in source.segments] for source in result.sources] == [["First", "Second"], ["First"]]
    assert resolve.await_count == 2


async def test_extraction_failure_returns_empty_result(monkeypatch):
    monkeypatch.setattr(module, "_get_final_url", AsyncMock(side_effect=RuntimeError("Cannot resolve")))
    response = GenerateContentResponse(candidates=[{"grounding_metadata": {"grounding_chunks": [{"web": {"uri": "https://a.example"}}]}}])
    result = await Gemini.extract_response_sources_and_answer(response)
    assert result.answer == ""
    assert result.sources == []


@pytest.mark.parametrize("method", ["generate_content", "generate_images", "generate_videos"])
async def test_retry_exhaustion_reraises_original_exception(gemini, client, method):
    error = RuntimeError("Persistent transport failure")
    provider = getattr(client.aio.models, method)
    provider.side_effect = error
    call = getattr(gemini, method).retry_with(wait=wait_none())
    kwargs = {"contents": "Question", "use_search": False} if method == "generate_content" else {"prompt": "Scene"}
    with pytest.raises(RuntimeError) as caught:
        await call(gemini, **kwargs)
    assert caught.value is error
    assert provider.await_count == 5
