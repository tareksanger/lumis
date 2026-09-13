from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from lumis.llm.ollama_llm import DEFAULT_MODEL, OllamaLLM

from ollama import ChatResponse, GenerateResponse, Message
from pydantic import BaseModel, ValidationError
import pytest
from tenacity import wait_none


@pytest.fixture
def llm(monkeypatch):
    monkeypatch.setattr(OllamaLLM.response.retry, "wait", wait_none())
    monkeypatch.setattr(OllamaLLM.completion.retry, "wait", wait_none())
    return OllamaLLM(client=SimpleNamespace(generate=AsyncMock(), chat=AsyncMock()))


async def test_generate_forwards_options_and_counts_tokens(llm):
    result = GenerateResponse(model="local", response="answer", prompt_eval_count=2, eval_count=3)
    llm.client.generate.return_value = result
    response = await llm.response(
        "question", model="local", suffix="suffix", system="system", template="template", context=[1], think="low", logprobs=True, top_logprobs=2, raw=True, images=["image"], keep_alive="5m"
    )
    assert response is result
    request = llm.client.generate.call_args.kwargs
    assert request["context"] == [1] and request["images"] == ["image"]
    assert request["model"] == "local" and request["prompt"] == "question"
    assert request["think"] == "low" and request["raw"] is True
    assert request["suffix"] == "suffix" and request["system"] == "system"
    assert request["template"] == "template" and request["keep_alive"] == "5m"
    assert llm.token_count["total_tokens"] == 5


async def test_chat_forwards_options_without_mutating_input(llm):
    messages = [Message(role="user", content="hi")]
    result = ChatResponse(message=Message(role="assistant", content="answer"))
    llm.client.chat.return_value = result
    assert await llm.completion(messages, model="local", tools=[], think=True, logprobs=True, top_logprobs=2, format="json", options={"temperature": 0}, keep_alive=5) is result.message
    request = llm.client.chat.call_args.kwargs
    assert request["messages"] == messages and request["messages"] is not messages
    assert request["format"] == "json" and request["options"] == {"temperature": 0}
    assert request["stream"] is False and request["think"] is True


async def test_structured_response_uses_schema_and_optional_parameters(llm):
    class Output(BaseModel):
        answer: int

    llm.client.generate.return_value = GenerateResponse(response='{"answer": 42}')
    output = await llm.structured_response("q", Output, context=[1], system="s", images=[b"image"], raw=True)
    assert output.answer == 42
    request = llm.client.generate.call_args.kwargs
    assert request["format"] == Output.model_json_schema()
    assert request["model"] == DEFAULT_MODEL and request["context"] == [1]
    assert request["system"] == "s" and request["images"] == [b"image"]
    llm.client.generate.return_value = GenerateResponse(response='{"answer": "invalid"}')
    with pytest.raises(ValidationError):
        await llm.structured_response("q", Output)


async def test_response_retry_recovers_then_exhausts(llm):
    result = GenerateResponse(response="ok")
    llm.client.generate.side_effect = [RuntimeError("temporary"), result]
    assert await llm.response("q") is result
    assert llm.client.generate.await_count == 2
    llm.client.generate.reset_mock(side_effect=True)
    llm.client.generate.return_value = GenerateResponse(response="")
    with pytest.raises(ValueError, match="No response text"):
        await llm.response("q")
    assert llm.client.generate.await_count == 5


def test_malformed_usage_is_logged_not_raised(llm):
    response = Mock()
    response.get.side_effect = TypeError("bad data")
    assert llm._count_tokens(response) is response
    assert llm.token_count["total_tokens"] == 0
