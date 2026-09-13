"""Offline ReAct lifecycle tests using deterministic thought responses."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from lumis.agents.react_agent import ReactAgent, ReActThought
from lumis.memory import SimpleMemory

import pytest


def thought(action="reason", observations="Observed"):
    return ReActThought(action=action, thought="Consider the task", observations=observations)


def response(action="reason"):
    return SimpleNamespace(parsed=thought(action))


@pytest.fixture
def llm():
    return SimpleNamespace(structured_completion=AsyncMock(return_value=response("finish")))


@pytest.fixture
def agent(llm):
    return ReactAgent(llm=llm, memory=SimpleMemory())


def record_events(agent, *names):
    events = []
    for name in names:

        async def record(*args, _name=name):
            events.append((_name, args))

        agent.on(name, record)
    return events


async def test_agents_have_independent_default_memories(llm):
    first, second = ReactAgent(llm=llm), ReactAgent(llm=llm)
    await first.memory.add({"role": "user", "content": "first only"})
    assert await second.memory.get() == []


async def test_initialize_prepends_instructions_once_before_existing_history(agent, llm):
    user = {"role": "user", "content": "Question"}
    existing = {"role": "assistant", "content": "History"}
    await agent.memory.add(existing)
    events = record_events(agent, "initialize")
    first = await agent.initialize([user])
    history = await agent.memory.get()
    assert [message["role"] for message in history] == ["system", "system", "user", "assistant"]
    assert history[-2:] == [user, existing]
    assert first.action == "finish"
    assert llm.structured_completion.await_args.kwargs["response_format"] is ReActThought
    assert llm.structured_completion.await_args.kwargs["messages"] == history
    await agent.initialize([{"role": "user", "content": "not duplicated"}])
    assert await agent.memory.get() == history
    assert events == [("initialize", (agent,))]


@pytest.mark.parametrize("failure", [False, True])
async def test_initialize_returns_none_on_missing_response_or_exception(agent, llm, failure):
    if failure:
        llm.structured_completion.side_effect = RuntimeError("unavailable")
    else:
        llm.structured_completion.return_value = None
    assert await agent.initialize() is None


async def test_none_step_does_not_call_model_or_mutate_memory(agent, llm):
    assert await agent.step(None) is None
    llm.structured_completion.assert_not_awaited()
    assert agent.memory.length == 0


@pytest.mark.parametrize("observations", ["Observed", ""])
async def test_reason_step_records_thought_and_observation_before_next_request(agent, llm, observations):
    current = thought(observations=observations)
    events = record_events(agent, "step", "after_step")
    result = await agent.step(current)
    history = await agent.memory.get()
    assert result.action == "finish"
    assert history[0]["role"] == "assistant"
    assert "Action: reason" in history[0]["content"]
    assert current.thought in history[0]["content"]
    assert ("Observation:" in history[0]["content"]) == bool(observations)
    assert llm.structured_completion.await_args.kwargs["messages"] == history
    assert events == [("step", (agent, current)), ("after_step", (agent,))]


async def test_action_step_awaits_action_before_requesting_next_thought(agent):
    agent.act = AsyncMock()
    result = await agent.step(thought("act"))
    agent.act.assert_awaited_once_with()
    assert result.action == "finish"


async def test_finish_step_stops_without_requesting_another_thought(agent, llm):
    assert await agent.step(thought("finish")) is None
    llm.structured_completion.assert_not_awaited()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("block", [False, True])
async def test_finish_condition_supports_sync_and_async_callbacks(agent, llm, asynchronous, block):
    value = "Complete the missing check" if block else None
    callback = AsyncMock(return_value=value) if asynchronous else Mock(return_value=value)
    agent.finish_condition_callback = callback
    result = await agent.step(thought("finish"))
    callback.assert_called_once_with()
    if asynchronous:
        callback.assert_awaited_once()
    if block:
        assert result.action == "finish"
        assert (await agent.memory.get())[-1] == {"role": "system", "content": value}
        llm.structured_completion.assert_awaited_once()
    else:
        assert result is None
        llm.structured_completion.assert_not_awaited()


async def test_step_handles_missing_next_thought(agent, llm):
    llm.structured_completion.return_value = None
    assert await agent.step(thought()) is None


@pytest.mark.parametrize("callable_object", [False, True])
async def test_finish_condition_awaits_callback_result(agent, callable_object):
    async def pending_condition():
        return "Finish the check"

    class Condition:
        async def __call__(self):
            return await pending_condition()

    agent.finish_condition_callback = Condition() if callable_object else lambda: pending_condition()
    assert (await agent.step(thought("finish"))).action == "finish"
    assert (await agent.memory.get())[-1] == {"role": "system", "content": "Finish the check"}


async def test_step_errors_emit_error_and_stop(agent, llm):
    llm.structured_completion.side_effect = RuntimeError("bad response")
    events = record_events(agent, "error")
    assert await agent.step(thought()) is None
    assert events == [("error", (agent,))]


async def test_act_forwards_memory_and_parallel_setting_and_emits_success(agent):
    history = {"role": "user", "content": "act"}
    await agent.memory.add(history)
    agent.parallel_tool_calls = False
    agent.call_tool = AsyncMock()
    events = record_events(agent, "before_act", "after_act", "act_error")
    assert await agent.act() is None
    agent.call_tool.assert_awaited_once_with(messages=[history], parallel_tool_calls=False)
    assert events == [("before_act", (agent,)), ("after_act", (agent,))]


async def test_act_error_includes_agent_context_and_does_not_emit_success(agent):
    agent.call_tool = AsyncMock(side_effect=RuntimeError("tool failed"))
    events = record_events(agent, "before_act", "after_act", "act_error")
    assert await agent.act() is None
    assert events == [("before_act", (agent,)), ("act_error", (agent,))]


async def test_run_processes_reason_action_finish_cycle(agent, llm):
    llm.structured_completion.side_effect = [response("reason"), response("act"), response("finish")]
    agent.call_tool = AsyncMock()
    events = record_events(agent, "initialize", "before_act", "after_act", "complete", "max_steps_reached")
    assert await agent.run(max_steps=10) is None
    assert agent.step_count == 3
    assert agent.thought is None
    agent.call_tool.assert_awaited_once()
    assert [name for name, _ in events] == ["initialize", "before_act", "after_act", "complete"]


async def test_run_stops_at_maximum_steps(agent, llm):
    llm.structured_completion.return_value = response("reason")
    events = record_events(agent, "max_steps_reached", "complete")
    await agent.run(max_steps=2)
    assert agent.step_count == 2
    assert llm.structured_completion.await_count == 3
    assert [name for name, _ in events] == ["max_steps_reached", "complete"]


async def test_finish_on_last_allowed_step_is_not_reported_as_limit_failure(agent):
    events = record_events(agent, "max_steps_reached", "complete")
    await agent.run(max_steps=1)
    assert agent.step_count == 1
    assert [name for name, _ in events] == ["complete"]


async def test_zero_step_limit_does_not_execute_thoughts(agent):
    agent.step = AsyncMock()
    events = record_events(agent, "max_steps_reached", "complete")
    await agent.run(max_steps=0)
    agent.step.assert_not_awaited()
    assert agent.step_count == 0
    assert [name for name, _ in events] == ["max_steps_reached", "complete"]


async def test_run_unexpected_failure_emits_error_and_retries_five_times(agent):
    agent.initialize = AsyncMock(side_effect=RuntimeError("initialization crashed"))
    events = record_events(agent, "error", "complete")
    with pytest.raises(RuntimeError, match="initialization crashed"):
        await agent.run(max_steps=1)
    assert agent.initialize.await_count == 5
    assert [name for name, _ in events] == ["error"] * 5


async def test_reset_clears_lifecycle_state_and_allows_fresh_initialization(agent, monkeypatch):
    await agent.initialize()
    agent.step_count = 7
    agent.thought = thought()
    monkeypatch.setattr("lumis.agents.base.base_agent.get_random_string", lambda length: "fresh")
    events = record_events(agent, "reset")
    await agent.reset()
    assert await agent.memory.get() == []
    assert agent.step_count == 0
    assert agent.thought is None
    assert agent.has_initialized is False
    assert agent.agent_id == "fresh"
    assert events == [("reset", (agent,))]
    await agent.initialize()
    assert agent.has_initialized is True
    assert agent.memory.length == 2
