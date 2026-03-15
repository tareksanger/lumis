from __future__ import annotations

import warnings

import pytest

from lumis.pipeline import Pipeline, Graph, TERMINATE
from lumis.pipeline.graph import StateProtocol


# ---------------------------------------------------------------------------
# Concrete Pipeline for testing
# ---------------------------------------------------------------------------


class CounterPipeline(Pipeline):
    def build(self):
        self.graph.add_node("increment", self._increment, "start")
        self.graph.add_node("double", self._double)
        self.graph.chain("increment", "double")

    async def setup(self, initial_value: int = 0):
        self.graph.set_initial_state({"value": initial_value})

    def _increment(self, state):
        return {"value": state["value"] + 1}

    def _double(self, state):
        return {"value": state["value"] * 2}


# ---------------------------------------------------------------------------
# Pipeline lifecycle
# ---------------------------------------------------------------------------


class TestPipelineLifecycle:
    async def test_run_executes_build_setup_traverse(self):
        p = CounterPipeline()
        await p.run(initial_value=5)
        # 5 + 1 = 6, then 6 * 2 = 12
        assert p.state["value"] == 12

    async def test_run_with_default_args(self):
        p = CounterPipeline()
        await p.run()
        # 0 + 1 = 1, then 1 * 2 = 2
        assert p.state["value"] == 2

    async def test_step_without_init_raises(self):
        p = CounterPipeline()
        with pytest.raises(RuntimeError, match="not initialized"):
            await p.step()

    async def test_step_after_run_works(self):
        """After run completes, step should not raise (though graph may be exhausted)."""
        p = CounterPipeline()
        await p.run(initial_value=0)
        # Graph is exhausted, step should not raise but also not do anything
        await p.step()


# ---------------------------------------------------------------------------
# State and history access
# ---------------------------------------------------------------------------


class TestPipelineState:
    async def test_state_reflects_graph_state(self):
        p = CounterPipeline()
        await p.run(initial_value=10)
        assert p.state == p.graph.state

    async def test_history_empty_without_tracing(self):
        p = CounterPipeline()
        await p.run(initial_value=0)
        assert p.history == []

    async def test_history_populated_with_tracing(self):
        p = CounterPipeline(enable_tracing=True)
        await p.run(initial_value=0)
        assert len(p.history) == 2
        assert p.history[0]["node"] == "increment"
        assert p.history[1]["node"] == "double"


# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------


class TestPipelineVisualize:
    def test_visualize_returns_html(self):
        p = CounterPipeline()
        html = p.visualize()
        assert "<html>" in html.lower() or "<!doctype" in html.lower() or "<script" in html.lower()


# ---------------------------------------------------------------------------
# Graph property
# ---------------------------------------------------------------------------


class TestPipelineGraph:
    def test_graph_property_returns_graph_instance(self):
        p = CounterPipeline()
        assert isinstance(p.graph, Graph)


# ---------------------------------------------------------------------------
# TERMINATE sentinel
# ---------------------------------------------------------------------------


class TestTerminateSentinel:
    def test_terminate_is_singleton(self):
        from lumis.pipeline.graph import _Terminate
        assert TERMINATE is _Terminate()

    def test_terminate_equals_string(self):
        assert TERMINATE == "terminate"

    def test_terminate_repr(self):
        assert repr(TERMINATE) == "TERMINATE"


# ---------------------------------------------------------------------------
# Deprecation shims
# ---------------------------------------------------------------------------


class TestDeprecationShims:
    def test_kit_graph_import_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from lumis.kit import Graph as KitGraph  # noqa: F401
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1

    def test_agents_base_graph_based_agent_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from lumis.agents.base import GraphBasedAgent  # noqa: F401
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
