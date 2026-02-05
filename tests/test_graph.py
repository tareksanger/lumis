from __future__ import annotations

import pytest

from lumis.kit.graph import Graph


# ---------------------------------------------------------------------------
# Node management
# ---------------------------------------------------------------------------


class TestGraphNodes:
    def test_add_node(self):
        g = Graph()
        g.add_node("a", lambda s: s)
        assert "a" in g.nodes

    def test_duplicate_node_raises(self):
        g = Graph()
        g.add_node("a", lambda s: s)
        with pytest.raises(AssertionError, match="already exists"):
            g.add_node("a", lambda s: s)

    def test_starting_node_is_recorded(self):
        g = Graph()
        g.add_node("start", lambda s: s, starting_node="start")
        assert g.starting_node == "start"
        assert g.current_node == "start"

    def test_duplicate_starting_node_raises(self):
        g = Graph()
        g.add_node("a", lambda s: s, starting_node="start")
        with pytest.raises(AssertionError):
            g.add_node("b", lambda s: s, starting_node="start")


# ---------------------------------------------------------------------------
# Edge management
# ---------------------------------------------------------------------------


class TestGraphEdges:
    def test_add_edge(self):
        g = Graph()
        g.add_node("a", lambda s: s)
        g.add_node("b", lambda s: s)

        edge = g.add_edge("a", "b")

        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.condition is None

    def test_chain_creates_sequential_edges(self):
        g = Graph()
        for name in ("a", "b", "c"):
            g.add_node(name, lambda s: s)
        g.chain("a", "b", "c")

        assert g.edges["a"][0].target == "b"
        assert g.edges["b"][0].target == "c"
        assert "c" not in g.edges  # terminal – no outgoing edges

    def test_chain_requires_multiple_nodes(self):
        g = Graph()
        g.add_node("a", lambda s: s)
        with pytest.raises(AssertionError, match="more than one"):
            g.chain("a")

    def test_chain_missing_node_raises(self):
        g = Graph()
        g.add_node("a", lambda s: s)
        with pytest.raises(AssertionError, match="does not exists"):
            g.chain("a", "missing")

    def test_conditional_edges_all_must_have_conditions(self):
        g = Graph()
        for name in ("a", "b", "c"):
            g.add_node(name, lambda s: s)

        g.add_edge("a", "b", condition=lambda s: True)

        with pytest.raises(AssertionError, match="all edges must have conditions"):
            g.add_edge("a", "c")  # missing condition on second edge


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


class TestGraphState:
    def test_state_matches_initial(self):
        state = {"x": 1, "y": [2, 3]}
        assert Graph(initial_state=state).state == state

    def test_state_is_deep_copied(self):
        g = Graph(initial_state={"nested": {"value": 1}})
        copy = g.state
        copy["nested"]["value"] = 999
        assert g.state["nested"]["value"] == 1  # original unchanged

    def test_reset_restores_initial_state(self):
        g = Graph(initial_state={"count": 0})
        g.add_node("a", lambda s: {"count": 5}, starting_node="start")

        # Simulate post-traversal state
        g.step_count = 5
        g.history = [{"fake": True}]
        g.terminate = True

        g.reset()

        assert g.state == {"count": 0}
        assert g.step_count == 0
        assert g.history == []
        assert g.terminate is False

    def test_set_initial_state_blocked_while_running(self):
        g = Graph(initial_state={"v": 1})
        # Flip the private running flag (name-mangled from ___is_running)
        g._Graph___is_running = True

        g.set_initial_state({"v": 999})

        # initial_state attribute IS updated, but internal __state is not
        assert g.initial_state == {"v": 999}
        assert g.state == {"v": 1}


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------


class TestGraphTraversal:
    async def test_linear_chain_updates_state(self):
        g = Graph(initial_state={"value": 0})

        def add_one(s):
            return {"value": s["value"] + 1}

        def add_two(s):
            return {"value": s["value"] + 2}

        g.add_node("a", add_one, starting_node="start")
        g.add_node("b", add_two)
        g.chain("a", "b")

        await g.traverse()
        assert g.state["value"] == 3  # 0 + 1 + 2

    async def test_async_node_runs(self):
        g = Graph(initial_state={"value": 0})

        async def add_ten(s):
            return {"value": s["value"] + 10}

        g.add_node("a", add_ten, starting_node="start")
        await g.traverse()
        assert g.state["value"] == 10

    async def test_terminate_sentinel_stops_traversal(self):
        g = Graph(initial_state={"visited": []})

        def visit_a(s):
            return {"visited": s["visited"] + ["a"]}

        def stop(s):
            return "terminate"

        def visit_c(s):
            return {"visited": s["visited"] + ["c"]}

        g.add_node("a", visit_a, starting_node="start")
        g.add_node("b", stop)
        g.add_node("c", visit_c)
        g.chain("a", "b", "c")

        await g.traverse()

        assert g.terminate is True
        assert "c" not in g.state["visited"]

    async def test_conditional_edge_takes_high_branch(self):
        g = Graph(initial_state={"value": 10})

        g.add_node("check", lambda s: {}, starting_node="start")
        g.add_node("high", lambda s: {"result": "high"})
        g.add_node("low", lambda s: {"result": "low"})
        g.add_edge("check", "high", condition=lambda s: s["value"] >= 5)
        g.add_edge("check", "low", condition=lambda s: s["value"] < 5)

        await g.traverse()
        assert g.state["result"] == "high"

    async def test_conditional_edge_takes_low_branch(self):
        g = Graph(initial_state={"value": 2})

        g.add_node("check", lambda s: {}, starting_node="start")
        g.add_node("high", lambda s: {"result": "high"})
        g.add_node("low", lambda s: {"result": "low"})
        g.add_edge("check", "high", condition=lambda s: s["value"] >= 5)
        g.add_edge("check", "low", condition=lambda s: s["value"] < 5)

        await g.traverse()
        assert g.state["result"] == "low"

    async def test_step_count_increments_per_node(self):
        g = Graph(initial_state={})
        g.add_node("a", lambda s: {}, starting_node="start")
        g.add_node("b", lambda s: {})
        g.add_node("c", lambda s: {})
        g.chain("a", "b", "c")

        await g.traverse()
        assert g.step_count == 3

    async def test_no_starting_node_raises(self):
        g = Graph()
        with pytest.raises(AssertionError, match="No starting node"):
            await g.traverse()

    async def test_node_returning_none_does_not_change_state(self):
        g = Graph(initial_state={"value": 42})
        g.add_node("a", lambda s: None, starting_node="start")

        await g.traverse()
        assert g.state["value"] == 42


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class TestGraphEvents:
    async def test_step_event_fires_per_node(self):
        g = Graph(initial_state={})
        step_calls = []
        g.on("step", lambda graph: step_calls.append(1))

        g.add_node("a", lambda s: {}, starting_node="start")
        g.add_node("b", lambda s: {})
        g.chain("a", "b")

        await g.traverse()
        assert len(step_calls) == 2

    async def test_finish_event_fires_once(self):
        g = Graph(initial_state={})
        finished = []
        g.on("finish", lambda graph: finished.append(True))

        g.add_node("a", lambda s: {}, starting_node="start")
        await g.traverse()
        assert finished == [True]

    async def test_node_fail_event_fires_on_exception(self):
        g = Graph(initial_state={})
        failed = []
        g.on("node_fail", lambda node: failed.append(node))

        def boom(s):
            raise RuntimeError("kaboom")

        g.add_node("bad", boom, starting_node="start")

        with pytest.raises(RuntimeError, match="kaboom"):
            await g.traverse()

        assert failed == ["bad"]


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


class TestGraphTracing:
    async def test_tracing_records_history(self):
        g = Graph(initial_state={"count": 0}, enable_tracing=True)

        def increment(s):
            return {"count": s["count"] + 1}

        g.add_node("a", increment, starting_node="start")
        g.add_node("b", increment)
        g.chain("a", "b")

        await g.traverse()

        assert len(g.history) == 2
        assert g.history[0]["node"] == "a"
        assert g.history[0]["state_changes"] == {"count": 1}
        assert g.history[1]["node"] == "b"
        assert g.history[1]["state_changes"] == {"count": 2}

    async def test_tracing_disabled_by_default(self):
        g = Graph(initial_state={"x": 1})
        g.add_node("a", lambda s: {"x": 2}, starting_node="start")

        await g.traverse()
        assert g.history == []
