"""Chat, parsed-chat, and image adapter contracts with injected SDK clients."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import lumis.llm.openai_llm as module
from lumis.llm.openai_llm import OpenAILLM

import httpx
from openai import omit
from openai.types.chat import ChatCompletion, ParsedChatCompletion
from openai.types.images_response import ImagesResponse
from pydantic import BaseModel
import pytest
from tenacity import wait_none


class Answer(BaseModel):
    value: int


def completion(structured=False, count=1):
    cls = ParsedChatCompletion[Answer] if structured else ChatCompletion
    choices = []
    for index in range(count):
        message = {"role": "assistant", "content": f"Answer {index}"}
        if structured:
            message["parsed"] = Answer(value=index)
        choices.append({"index": index, "finish_reason": "stop", "message": message})
    return cls(id="chat-test", created=0, model="model-test", object="chat.completion", choices=choices)


@pytest.fixture
def client():
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=completion()),
                parse=AsyncMock(return_value=completion(structured=True)),
            )
        ),
        images=SimpleNamespace(generate=AsyncMock()),
    )


@pytest.fixture
def llm(client):
    return OpenAILLM(client=client)


def method_pair(llm, client, structured):
    if structured:
        return llm.structured_completion, client.chat.completions.parse
    return llm.completion, client.chat.completions.create


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        (None, omit, "chat-default"),
        (None, "high", "reasoning-default"),
        ("explicit-model", "high", "explicit-model"),
    ],
)
async def test_selects_model_from_reasoning_effort(llm, client, monkeypatch, structured, model, effort, expected):
    monkeypatch.setattr(module, "CHAT_MODEL", "chat-default")
    monkeypatch.setattr(module, "REASONING_MODEL", "reasoning-default")
    method, provider = method_pair(llm, client, structured)
    await method(model=model, reasoning_effort=effort, messages=[])
    assert provider.await_args.kwargs["model"] == expected
    assert provider.await_args.kwargs["reasoning_effort"] is effort


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("n", [omit, None, 1, 2])
async def test_returns_first_message_or_requested_multiple_messages(llm, client, structured, n):
    method, provider = method_pair(llm, client, structured)
    result = completion(structured, count=2)
    provider.return_value = result
    kwargs = {"response_format": Answer} if structured else {}
    output = await method(n=n, messages=[{"role": "user", "content": "Question"}], **kwargs)
    if n == 2:
        assert output == [choice.message for choice in result.choices]
    else:
        assert output is result.choices[0].message
        if structured:
            assert output.parsed == Answer(value=0)


@pytest.mark.parametrize("structured", [False, True])
async def test_forwards_all_chat_request_parameters(llm, client, structured):
    method, provider = method_pair(llm, client, structured)
    options = dict(
        model="chosen-model",
        messages=[{"role": "user", "content": "Question"}],
        n=1,
        frequency_penalty=0.1,
        logit_bias={"123": 2},
        logprobs=True,
        max_completion_tokens=25,
        max_tokens=20,
        metadata={"test": "yes"},
        parallel_tool_calls=False,
        presence_penalty=0.2,
        reasoning_effort="low",
        seed=42,
        service_tier="default",
        stop=["END"],
        store=False,
        temperature=0.3,
        tool_choice="none",
        tools=[],
        top_logprobs=2,
        top_p=0.8,
        user="test",
        extra_headers={"X-Test": "yes"},
        extra_query={"test": "yes"},
        extra_body={"extra": True},
        timeout=3.0,
    )
    if structured:
        options.update(response_format=Answer, web_search_options={"search_context_size": "low"})
    await method(**options)
    provider.assert_awaited_once_with(**options)


@pytest.mark.parametrize("structured", [False, True])
async def test_applies_middleware_before_extracting_message(llm, client, structured):
    method, _ = method_pair(llm, client, structured)
    replacement = completion(structured)
    replacement.choices[0].message.content = "Transformed"

    async def transform(response):
        return replacement

    llm.add_middleware(transform)
    assert await method(messages=[]) is replacement.choices[0].message


@pytest.mark.parametrize("structured", [False, True])
async def test_empty_choices_raise_instead_of_returning_invalid_message(llm, client, structured):
    method, provider = method_pair(llm, client, structured)
    provider.return_value = completion(structured, count=0)
    with pytest.raises(ValueError, match="No choices returned"):
        await method.retry_with(wait=wait_none())(llm, messages=[])


@pytest.mark.parametrize("structured", [False, True])
async def test_transient_transport_failure_retries_without_sleep(llm, client, structured):
    method, provider = method_pair(llm, client, structured)
    result = completion(structured)
    provider.side_effect = [httpx.ConnectError("Temporary failure"), result]
    assert await method.retry_with(wait=wait_none())(llm, messages=[]) is result.choices[0].message
    assert provider.await_count == 2


async def test_structured_refusal_is_returned_without_inventing_parsed_data(llm, client):
    result = completion(structured=True)
    result.choices[0].message.parsed = None
    result.choices[0].message.refusal = "Cannot fulfill"
    client.chat.completions.parse.return_value = result
    output = await llm.structured_completion(response_format=Answer, messages=[])
    assert output.parsed is None
    assert output.refusal == "Cannot fulfill"


async def test_chat_tool_calls_are_returned_for_caller_handling(llm, client):
    result = completion()
    result.choices[0].message.tool_calls = [{"id": "tool-test", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]
    client.chat.completions.create.return_value = result
    output = await llm.completion(messages=[])
    assert output is result.choices[0].message
    assert output.tool_calls == result.choices[0].message.tool_calls
    assert client.chat.completions.create.await_count == 1


@pytest.mark.parametrize(
    ("format", "data"),
    [
        ("url", {"url": "https://images.example/test.png"}),
        ("b64_json", {"b64_json": "dGVzdA=="}),
    ],
)
async def test_image_request_returns_sdk_response_without_decoding(llm, client, format, data):
    result = ImagesResponse(created=0, data=[data])
    client.images.generate.return_value = result
    options = dict(
        prompt="A mountain",
        model="image-model",
        n=1,
        quality="hd",
        response_format=format,
        size="1024x1024",
        style="natural",
        user="test",
        extra_headers={"X-Test": "yes"},
        extra_query={"test": "yes"},
        extra_body={"extra": True},
        timeout=2.0,
    )
    assert await llm.generate_image(**options) is result
    client.images.generate.assert_awaited_once_with(**options)


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout("Timed out"),
        httpx.ConnectError("Offline"),
        json.JSONDecodeError("Invalid JSON", "{", 0),
        RuntimeError("Provider failure"),
    ],
)
async def test_image_failures_exhaust_retries_and_preserve_exception(llm, client, error):
    client.images.generate.side_effect = error
    with pytest.raises(type(error)) as caught:
        await llm.generate_image.retry_with(wait=wait_none())(llm, prompt="A mountain")
    assert caught.value is error
    assert client.images.generate.await_count == 5


async def test_image_transient_error_recovers(llm, client):
    result = ImagesResponse(created=0, data=[])
    client.images.generate.side_effect = [httpx.ConnectError("Offline"), result]
    assert await llm.generate_image.retry_with(wait=wait_none())(llm, prompt="A mountain") is result
    assert client.images.generate.await_count == 2
