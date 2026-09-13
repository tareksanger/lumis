"""Base agent contracts tested with local memory and mocked model calls."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from lumis.agents.base.base_agent import BaseAgent
from lumis.agents.base.core_agent import CoreAgent

import pytest


class ConcreteBaseAgent(BaseAgent):
    async def run(self, *args, **kwargs):
        return "base result"


class ConcreteCoreAgent(CoreAgent):
    async def run(self, *args, **kwargs):
        return "core result"


def double(value: int) -> int:
    """Double the given integer."""
    return value * 2


@pytest.fixture
def llm():
    return SimpleNamespace(
        token_count=42,
        completion=AsyncMock(),
        _has_tool_calls=Mock(return_value=True),
        handle_chat_completion_tool_call=AsyncMock(return_value=[]),
    )


@pytest.fixture
def memory():
    return SimpleNamespace(length=0, add=AsyncMock())


@pytest.mark.parametrize("agent_class", [BaseAgent, CoreAgent])
def test_base_classes_require_run_implementation(agent_class):
    with pytest.raises(TypeError, match="abstract"):
        agent_class()


@pytest.mark.parametrize("agent_class", [ConcreteBaseAgent, ConcreteCoreAgent])
def test_injected_model_and_token_count_are_exposed(agent_class, llm):
    agent = agent_class(llm=llm, verbose=True)
    assert agent.llm is llm
    assert agent.verbose is True
    assert len(agent.agent_id) == 5
    assert agent.token_count == 42
    llm.token_count = 99
    assert agent.token_count == 99


@pytest.mark.parametrize(
    "agent_class,module",
    [
        (ConcreteBaseAgent, "lumis.agents.base.base_agent"),
        (ConcreteCoreAgent, "lumis.agents.base.core_agent"),
    ],
)
def test_missing_model_uses_default_constructor_without_network(agent_class, module, llm, monkeypatch):
    constructor = Mock(return_value=llm)
    monkeypatch.setattr(f"{module}.OpenAILLM", constructor)
    agent = agent_class()
    constructor.assert_called_once_with()
    assert agent.llm is llm


@pytest.mark.parametrize("agent_class", [ConcreteBaseAgent, ConcreteCoreAgent])
def test_constructor_preserves_custom_logger(agent_class, llm):
    logger = logging.getLogger("test.custom.agent")
    agent = agent_class(llm=llm, logger=logger)
    assert agent.logger is logger


async def test_default_memory_is_isolated_between_agents(llm):
    first, second = ConcreteBaseAgent(llm=llm), ConcreteBaseAgent(llm=llm)
    await first.add_message({"role": "user", "content": "private history"})
    assert await second.memory.get() == []
    assert first.memory is not second.memory


def test_injected_memory_and_tool_list_are_preserved_without_aliasing(llm, memory):
    tools = [double]
    agent = ConcreteBaseAgent(llm=llm, memory=memory, tools=tools)
    tools.clear()
    assert agent.memory is memory
    assert agent.tools == [double]
    definition = agent.tool_definitions[0]["function"]
    assert definition["name"] == "double"
    assert definition["parameters"]["properties"]["value"]["type"] == "integer"
    assert definition["parameters"]["required"] == ["value"]


@pytest.mark.parametrize(
    "agent_class,module",
    [
        (ConcreteBaseAgent, "lumis.agents.base.base_agent"),
        (ConcreteCoreAgent, "lumis.agents.base.core_agent"),
    ],
)
async def test_reset_changes_id_before_awaiting_extension_hook(agent_class, module, llm, monkeypatch):
    monkeypatch.setattr(f"{module}.get_random_string", Mock(side_effect=["first", "next1"]))
    agent = agent_class(llm=llm)

    async def reset_hook():
        assert agent.agent_id == "next1"

    agent._reset = AsyncMock(side_effect=reset_hook)
    assert agent.agent_id == "first"
    await agent.reset()
    agent._reset.assert_awaited_once_with()


@pytest.mark.parametrize("agent_class", [ConcreteBaseAgent, ConcreteCoreAgent])
async def test_default_reset_hook_completes_and_run_is_subclass_defined(agent_class, llm):
    agent = agent_class(llm=llm)
    await agent.reset()
    assert await agent.run() in ("base result", "core result")


@pytest.mark.parametrize("agent_class", [ConcreteBaseAgent, ConcreteCoreAgent])
async def test_events_are_instance_local_and_support_async_listeners(agent_class, llm):
    first, second = agent_class(llm=llm), agent_class(llm=llm)
    observed = []

    async def handler(value):
        observed.append(value)

    first.on("result", handler)
    await second.emit("result", "ignored")
    await first.emit("result", "answer")
    assert observed == ["answer"]


@pytest.mark.parametrize("role", ["user", "assistant", "tool"])
@pytest.mark.parametrize("verbose", [False, True])
async def test_add_message_forwards_each_role_once(llm, memory, role, verbose):
    agent = ConcreteBaseAgent(llm=llm, memory=memory, verbose=verbose)
    message = {"role": role, "content": "content"}
    await agent.add_message(message)
    memory.add.assert_awaited_once_with(message)


async def test_call_tool_forwards_options_and_records_assistant_then_each_result(llm, memory):
    agent = ConcreteBaseAgent(llm=llm, memory=memory, tools=[double])
    assistant_message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "double", "arguments": '{"value":2}'}}]}
    response = Mock()
    response.model_dump.return_value = assistant_message
    llm.completion.return_value = response
    llm.handle_chat_completion_tool_call.return_value = [{"result": 4, "call_id": "call-1"}, {"result": {"ok": True}, "call_id": "call-2"}]
    history = [{"role": "user", "content": "double two"}]
    await agent.call_tool(model="test-model", messages=history, temperature=0.2, max_completion_tokens=64, parallel_tool_calls=False, extra_body={"custom": True}, timeout=10)
    options = llm.completion.await_args.kwargs
    assert options["model"] == "test-model"
    assert options["messages"] is history
    assert options["temperature"] == 0.2
    assert options["max_completion_tokens"] == 64
    assert options["parallel_tool_calls"] is False
    assert options["extra_body"] == {"custom": True}
    assert options["timeout"] == 10
    assert options["tool_choice"] == "required"
    assert options["tools"] == agent.tool_definitions
    llm.handle_chat_completion_tool_call.assert_awaited_once_with(message=response, tool_map={"double": double})
    assert [call.args[0] for call in memory.add.await_args_list] == [
        assistant_message,
        {"role": "tool", "content": "4", "tool_call_id": "call-1"},
        {"role": "tool", "content": "{'ok': True}", "tool_call_id": "call-2"},
    ]


async def test_call_tool_without_calls_does_not_write_memory(llm, memory):
    llm._has_tool_calls.return_value = False
    agent = ConcreteBaseAgent(llm=llm, memory=memory)
    await agent.call_tool()
    memory.add.assert_not_awaited()
    llm.handle_chat_completion_tool_call.assert_not_awaited()


@pytest.mark.parametrize("results", [None, []])
async def test_call_tool_with_no_execution_results_only_records_assistant(llm, memory, results):
    response = Mock()
    response.model_dump.return_value = {"role": "assistant", "content": None}
    llm.completion.return_value = response
    llm.handle_chat_completion_tool_call.return_value = results
    await ConcreteBaseAgent(llm=llm, memory=memory).call_tool()
    memory.add.assert_awaited_once_with({"role": "assistant", "content": None})


async def test_model_failure_is_logged_and_does_not_write_memory(llm, memory, caplog):
    llm.completion.side_effect = RuntimeError("model unavailable")
    await ConcreteBaseAgent(llm=llm, memory=memory).call_tool()
    assert "model unavailable" in caplog.text
    memory.add.assert_not_awaited()


async def test_tool_execution_failure_is_logged_after_assistant_is_saved(llm, memory, caplog):
    response = Mock()
    response.model_dump.return_value = {"role": "assistant", "content": None}
    llm.completion.return_value = response
    llm.handle_chat_completion_tool_call.side_effect = RuntimeError("tool unavailable")
    await ConcreteBaseAgent(llm=llm, memory=memory).call_tool()
    assert "tool unavailable" in caplog.text
    memory.add.assert_awaited_once()


def test_tool_description_display_includes_name_and_description(llm):
    agent = ConcreteBaseAgent(llm=llm, tools=[double])
    description = agent._get_tool_definition_prompt()
    assert "double" in description
    assert "Double the given integer." in description
    assert ConcreteBaseAgent(llm=llm)._get_tool_definition_prompt() == ""
