"""Responses API requests and bounded local tool execution without network calls."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from lumis.llm.openai_llm import OpenAILLM

import httpx
from openai import not_given, omit
from openai.types.responses import Response, ResponseCustomToolCall, ResponseFunctionToolCall
import pytest
from tenacity import wait_none


def response(*items, response_id="resp-test"):
    return Response(
        id=response_id,
        created_at=0,
        model="model-test",
        object="response",
        output=list(items),
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def call(name="double", arguments='{"value": 3}', call_id="call-test"):
    return ResponseFunctionToolCall(name=name, arguments=arguments, call_id=call_id, type="function_call")


@pytest.fixture
def client():
    return SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=response())))


@pytest.fixture
def llm(client):
    return OpenAILLM(client=client)


async def test_response_forwards_optional_parameters_and_returns_middleware_result(llm, client):
    options = dict(
        input=[{"role": "user", "content": "Question"}],
        model="chosen-model",
        background=False,
        conversation="conv-test",
        include=["file_search_call.results"],
        instructions="Be concise",
        max_output_tokens=20,
        max_tool_calls=2,
        metadata={"purpose": "test"},
        parallel_tool_calls=False,
        previous_response_id="resp-before",
        prompt={"id": "prompt-test"},
        prompt_cache_key="cache-test",
        prompt_cache_retention="24h",
        reasoning={"effort": "low"},
        safety_identifier="user-test",
        service_tier="default",
        store=False,
        temperature=0.2,
        text={"format": {"type": "text"}},
        tool_choice="none",
        tools=[],
        top_logprobs=2,
        top_p=0.5,
        truncation="auto",
        user="test",
        extra_headers={"X-Test": "yes"},
        extra_query={"test": "yes"},
        extra_body={"custom": True},
        timeout=2.0,
    )
    transformed = response(response_id="transformed")
    middleware = Mock(return_value=transformed, __name__="transform")
    llm.add_middleware(middleware)
    assert await llm.response(**options) is transformed
    client.responses.create.assert_awaited_once_with(**options)
    middleware.assert_called_once_with(client.responses.create.return_value)


async def test_default_optional_values_preserve_sdk_omission_sentinels(llm, client):
    await llm.response(input="Question")
    sent = client.responses.create.await_args.kwargs
    assert sent["input"] == "Question"
    assert sent["tools"] is omit
    assert sent["instructions"] is omit
    assert sent["timeout"] is not_given


async def test_response_retries_transport_errors_without_sleep(llm, client):
    final = response()
    client.responses.create.side_effect = [httpx.ConnectError("Offline"), final]
    assert await llm.response.retry_with(wait=wait_none())(llm, input="Question") is final
    assert client.responses.create.await_count == 2


async def test_response_reraises_nonretryable_error_once(llm, client):
    client.responses.create.side_effect = ValueError("Invalid model configuration")
    with pytest.raises(ValueError, match="Invalid model configuration"):
        await llm.response(input="Question")
    assert client.responses.create.await_count == 1


async def test_generate_returns_immediately_when_no_tools_are_available(llm, client):
    final = response(call())
    client.responses.create.return_value = final
    assert await llm.generate(input="Question") is final
    assert client.responses.create.await_count == 1
    assert client.responses.create.await_args.kwargs["input"] == [{"role": "user", "content": "Question"}]


@pytest.mark.parametrize("iterable_tools", [False, True])
async def test_generate_executes_multiple_rounds_without_consuming_tool_iterable(llm, client, iterable_tools):
    seen = []

    def double(value: int) -> int:
        """Double a supplied value."""
        seen.append(value)
        return value * 2

    first, second, final = response(call()), response(call(arguments='{"value": 4}', call_id="call-two")), response()
    client.responses.create.side_effect = [first, second, final]
    tools = iter([double]) if iterable_tools else [double]
    initial = [{"role": "user", "content": "Question"}]
    assert await llm.generate(input=initial, tools=tools) is final
    assert seen == [3, 4]
    assert initial == [{"role": "user", "content": "Question"}]
    requests = [entry.kwargs for entry in client.responses.create.await_args_list]
    assert [request["tools"][0]["name"] for request in requests] == ["double"] * 3
    assert requests[1]["input"] == initial + [first.output[0], {"type": "function_call_output", "call_id": "call-test", "output": "6"}]
    assert requests[2]["input"][-1] == {"type": "function_call_output", "call_id": "call-two", "output": "8"}


@pytest.mark.parametrize("limit", [0, 1, 2])
async def test_generate_stops_executing_tools_at_iteration_limit(llm, client, limit):
    seen = []

    def double(value: int) -> int:
        """Double a supplied value."""
        seen.append(value)
        return value * 2

    final = response(call())
    client.responses.create.return_value = final
    assert await llm.generate(input="Question", tools=[double], max_iterations=limit) is final
    assert seen == [3] * limit
    assert client.responses.create.await_count == limit + 1


async def test_generate_preserves_reasoning_items_needed_for_tool_continuation(llm, client):
    def double(value: int) -> int:
        """Double a supplied value."""
        return value * 2

    first = response({"id": "reasoning-test", "type": "reasoning", "summary": []}, call())
    client.responses.create.side_effect = [first, response()]
    await llm.generate(input="Question", tools=[double])
    continuation = client.responses.create.await_args_list[1].kwargs["input"]
    assert continuation[1:3] == first.output


async def test_handle_no_tool_calls_returns_none(llm):
    assert await llm.handle_response_tool_call(response(), {}) is None


async def test_function_tool_results_keep_call_ids_and_support_async_functions(llm):
    async_tool = AsyncMock(return_value="Async result")
    sync_tool = Mock(return_value={"answer": 6})
    output = await llm.handle_response_tool_call(
        response(call("sync"), call("async", '{"value": 4}', "call-two")),
        {"sync": sync_tool, "async": async_tool},
    )
    assert output == [
        {"type": "function_call_output", "call_id": "call-test", "output": "{'answer': 6}"},
        {"type": "function_call_output", "call_id": "call-two", "output": "Async result"},
    ]
    sync_tool.assert_called_once_with(value=3)
    async_tool.assert_awaited_once_with(value=4)


async def test_invalid_json_returns_tool_error_without_invoking_function(llm):
    tool = Mock()
    output = await llm.handle_response_tool_call(response(call(arguments="{")), {"double": tool})
    assert output == [{"type": "function_call_output", "call_id": "call-test", "output": "Invalid arguments for double"}]
    tool.assert_not_called()


async def test_tool_failure_does_not_prevent_later_tool_execution(llm):
    broken = Mock(side_effect=RuntimeError("Tool unavailable"))
    working = Mock(return_value="OK")
    output = await llm.handle_response_tool_call(response(call("broken"), call("working", call_id="call-two")), {"broken": broken, "working": working})
    assert [item["output"] for item in output] == ["Tool unavailable", "OK"]
    working.assert_called_once_with(value=3)


async def test_unknown_tool_returns_error_output(llm):
    output = await llm.handle_response_tool_call(response(call()), {})
    assert output[0]["output"] == "No such tool function: double"


@pytest.mark.parametrize(
    ("raw", "arguments"),
    [
        ('{"value": 3}', {"value": 3}),
        ("[1, 2]", {"input": [1, 2]}),
        ('"word"', {"input": "word"}),
        ("plain input", {"input": "plain input"}),
        ("", {}),
    ],
)
async def test_custom_tools_decode_input_and_use_custom_output_type(llm, raw, arguments):
    custom = ResponseCustomToolCall(call_id="custom-test", input=raw, name="custom", type="custom_tool_call")
    tool = Mock(return_value="Result")
    output = await llm.handle_response_tool_call(response(custom), {"custom": tool})
    tool.assert_called_once_with(**arguments)
    assert output == [{"type": "custom_tool_call_output", "call_id": "custom-test", "output": "Result"}]
