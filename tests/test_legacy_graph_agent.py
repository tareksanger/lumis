import importlib
from unittest.mock import Mock

from lumis.agents.base.graph_based_agent import GraphBasedAgent

import pytest


class CounterAgent(GraphBasedAgent):
    def construct_graph(self):
        self.graph.add_node("increment", lambda state: {"value": state["value"] + 1}, "start")

    async def setup(self, value=0):
        self.graph.set_initial_state({"value": value})


async def test_graph_agent_lifecycle():
    agent = CounterAgent(llm=Mock())
    with pytest.raises(AssertionError, match="not initialized"):
        await agent.step()
    await agent.run(value=4)
    assert agent.get_state()["value"] == 5
    await agent.step()
    assert agent.get_state()["value"] == 5
    assert "<html>" in agent.visualize().lower()


async def test_graph_agent_manual_step():
    agent = CounterAgent(llm=Mock())
    await agent.initialize(value=10)
    await agent.step()
    assert agent.get_state()["value"] == 11


@pytest.mark.parametrize(
    "legacy,target,names",
    [
        ("lumis.kit", "lumis.pipeline.graph", ["Graph", "Edge"]),
        ("lumis.kit.nodes", "lumis.pipeline.nodes", ["LLMChatNode", "LLMStructuredNode"]),
    ],
)
def test_deprecated_dynamic_exports_warn_and_preserve_identity(legacy, target, names):
    old = importlib.import_module(legacy)
    new = importlib.import_module(target)
    for name in names:
        with pytest.warns(DeprecationWarning):
            assert getattr(old, name) is getattr(new, name)
    with pytest.raises(AttributeError):
        getattr(old, "missing_export")


@pytest.mark.parametrize("legacy,target,name", [("lumis.kit.graph", "lumis.pipeline.graph", "Graph"), ("lumis.kit.utils", "lumis.pipeline.utils", "dict_diff")])
def test_deprecated_module_exports(legacy, target, name):
    with pytest.warns(DeprecationWarning):
        module = importlib.reload(importlib.import_module(legacy))
    assert getattr(module, name) is getattr(importlib.import_module(target), name)
