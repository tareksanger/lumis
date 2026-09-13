"""LLM pipeline nodes using injected clients; no provider calls are made."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from lumis.core.utils.types import BaseSchema
from lumis.memory.simple_memory import SimpleMemory
from lumis.pipeline.graph import TERMINATE
from lumis.pipeline.nodes.llm_chat_node import LLMChatNode
from lumis.pipeline.nodes.llm_structured_node import LLMStructuredNode

import pytest


class Answer(BaseSchema):
    text: str

    def to_context_str(self, *args, **kwargs):
        return f"Answer: {self.text}"


def make_node(kind, client, **kwargs):
    if kind == "chat":
        return LLMChatNode(client, **kwargs)
    return LLMStructuredNode(client, Answer, "answer", **kwargs)


@pytest.fixture
def client():
    return SimpleNamespace(
        completion=AsyncMock(return_value=None),
        structured_completion=AsyncMock(return_value=None),
    )


@pytest.mark.parametrize("kind", ["chat", "structured"])
@pytest.mark.parametrize("state", [{}, {"memory": None}, {"memory": ["invalid"]}])
async def test_missing_or_invalid_memory_is_replaced(kind, state, client):
    result = await make_node(kind, client)(state)
    assert isinstance(result["memory"], SimpleMemory)
    assert await result["memory"].get() == []
    assert result["memory"] is not state.get("memory")


@pytest.mark.parametrize("kind", ["chat", "structured"])
async def test_passes_history_and_system_instruction_without_persisting_instruction(kind, client):
    history = [{"role": "user", "content": "Question"}]
    memory = SimpleMemory(messages=history.copy())
    node = make_node(kind, client, system_prompt="Be concise")
    state = {"memory": memory, "untouched": 3}
    result = await node(state)
    expected = history + [{"role": "system", "content": "Be concise"}]
    if kind == "chat":
        client.completion.assert_awaited_once_with(messages=expected)
    else:
        client.structured_completion.assert_awaited_once_with(response_format=Answer, messages=expected)
    assert result == {"memory": memory}
    assert await memory.get() == history
    assert state == {"memory": memory, "untouched": 3}


@pytest.mark.parametrize("kind", ["chat", "structured"])
async def test_no_system_instruction_sends_only_history(kind, client):
    memory = SimpleMemory(messages=[{"role": "user", "content": "Question"}])
    await make_node(kind, client)({"memory": memory})
    method = client.completion if kind == "chat" else client.structured_completion
    assert method.await_args.kwargs["messages"] == [{"role": "user", "content": "Question"}]


@pytest.mark.parametrize("kind", ["chat", "structured"])
async def test_client_failure_terminates_without_writing_memory(kind, client):
    method = client.completion if kind == "chat" else client.structured_completion
    method.side_effect = RuntimeError("Provider unavailable")
    memory = SimpleMemory(messages=[{"role": "user", "content": "Question"}])
    assert await make_node(kind, client)({"memory": memory}) is TERMINATE
    assert await memory.get() == [{"role": "user", "content": "Question"}]


@pytest.mark.parametrize("kind", ["chat", "structured"])
async def test_memory_write_failure_terminates(kind, client, monkeypatch):
    client.completion.return_value = SimpleNamespace(content="Answer")
    client.structured_completion.return_value = SimpleNamespace(parsed=Answer(text="Answer"))
    memory = SimpleMemory()
    monkeypatch.setattr(memory, "add", AsyncMock(side_effect=RuntimeError("Storage unavailable")))
    assert await make_node(kind, client)({"memory": memory}) is TERMINATE


class TestLLMChatNode:
    @pytest.mark.parametrize("content", ["An answer", ""])
    async def test_stores_plain_response_in_existing_memory(self, client, content):
        client.completion.return_value = SimpleNamespace(content=content)
        memory = SimpleMemory()
        result = await LLMChatNode(client)({"memory": memory})
        assert result == {"memory": memory}
        assert await memory.get() == [{"role": "assistant", "content": content}]

    async def test_concatenates_supported_content_parts_in_order(self, client):
        client.completion.return_value = SimpleNamespace(
            content=[
                "Plain ",
                SimpleNamespace(text="text "),
                SimpleNamespace(text=SimpleNamespace(value="nested")),
                SimpleNamespace(image_url="https://example.org/image.png"),
                SimpleNamespace(text=42),
                SimpleNamespace(text=SimpleNamespace(value=42)),
            ]
        )
        result = await LLMChatNode(client)({})
        assert await result["memory"].get() == [{"role": "assistant", "content": "Plain text nested"}]

    @pytest.mark.parametrize("content", [None, [], [SimpleNamespace(text=None)], [SimpleNamespace(image_url="image")]])
    async def test_missing_text_does_not_add_message(self, client, content):
        client.completion.return_value = SimpleNamespace(content=content)
        result = await LLMChatNode(client)({})
        assert await result["memory"].get() == []

    async def test_verbose_logs_assistant_text(self, client):
        client.completion.return_value = SimpleNamespace(content="Useful answer")
        node = LLMChatNode(client, verbose=True)
        node.logger = Mock()
        result = await node({})
        assert result is not TERMINATE
        assert "Useful answer" in node.logger.info.call_args.args[0]


class TestLLMStructuredNode:
    async def test_returns_parsed_model_under_requested_key_and_stores_context(self, client):
        answer = Answer(text="Useful answer")
        client.structured_completion.return_value = SimpleNamespace(parsed=answer)
        memory = SimpleMemory()
        node = LLMStructuredNode(client, Answer, "custom_result")
        result = await node({"memory": memory})
        assert result["custom_result"] is answer
        assert result["memory"] is memory
        assert await memory.get() == [{"role": "assistant", "content": "Answer: Useful answer"}]

    async def test_can_return_parsed_model_without_writing_memory(self, client):
        answer = Answer(text="Useful answer")
        client.structured_completion.return_value = SimpleNamespace(parsed=answer)
        node = LLMStructuredNode(client, Answer, "answer", add_result_to_memory=False)
        result = await node({})
        assert result["answer"] is answer
        assert await result["memory"].get() == []

    async def test_missing_parsed_result_omits_state_key(self, client):
        client.structured_completion.return_value = SimpleNamespace(parsed=None)
        result = await LLMStructuredNode(client, Answer, "answer")({"answer": "old"})
        assert set(result) == {"memory"}
        assert await result["memory"].get() == []

    async def test_context_conversion_failure_terminates(self, client):
        client.structured_completion.return_value = SimpleNamespace(
            parsed=SimpleNamespace(to_context_str=Mock(side_effect=ValueError("Cannot format"))),
        )
        assert await LLMStructuredNode(client, Answer, "answer")({}) is TERMINATE

    async def test_verbose_logs_context_even_if_memory_write_disabled(self, client):
        client.structured_completion.return_value = SimpleNamespace(parsed=Answer(text="Useful answer"))
        node = LLMStructuredNode(client, Answer, "answer", verbose=True, add_result_to_memory=False)
        node.logger = Mock()
        result = await node({})
        node.logger.info.assert_called_once_with("Assistant: Answer: Useful answer")
        assert await result["memory"].get() == []
