from __future__ import annotations

import collections.abc
import copy
from dataclasses import dataclass
import inspect
import logging
import warnings
from typing import Any, AsyncIterator, Awaitable, Callable, cast, Generic, Iterator, Literal, Optional, Protocol, TypedDict, TypeVar, Union

from lumis.core.common.logger_mixin import LoggerMixin
from lumis.core.event_emitter import EventEmitter
from lumis.core.utils.helpers import serialize

from .utils import dict_diff

from pyvis.network import Network


class _Terminate:
    """Singleton sentinel object for graph termination."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "TERMINATE"

    def __eq__(self, other):
        if isinstance(other, _Terminate):
            return True
        if other == "terminate":
            return True
        return NotImplemented

    def __hash__(self):
        return hash("TERMINATE")


TERMINATE = _Terminate()


class StateProtocol(TypedDict):
    """Protocol for state objects that behave like dictionaries."""

    ...


S = TypeVar("S", bound=StateProtocol)
# Output type should implement __concat__, as eg str, list, dict do
Output = TypeVar("Output", covariant=True)


class SyncNode(Protocol[S, Output]):
    def __call__(self, state: S) -> Output: ...


class AsyncNode(Protocol[S, Output]):
    async def __call__(self, state: S) -> Output: ...


class AwaitableNode(Protocol[S, Output]):
    def __call__(self, state: Iterator[S]) -> Awaitable[Output]: ...


class IteratorNode(Protocol[S, Output]):
    def __call__(self, state: Iterator[S]) -> Iterator[Output]: ...


class AsyncIteratorNode(Protocol[S, Output]):
    def __call__(self, state: AsyncIterator[S]) -> AsyncIterator[Output]: ...


NodeLike = Union[
    Callable[[S], Output],
    Callable[[S], Awaitable[Output]],
    Callable[[Iterator[S]], Iterator[Output]],
    Callable[[AsyncIterator[S]], AsyncIterator[Output]],
    type[SyncNode[S, Output]],
    type[AsyncNode[S, Output]],
    type[AwaitableNode[S, Output]],
    type[IteratorNode[S, Output]],
    type[AsyncIteratorNode[S, Output]],
]


NodeCallable = Union[
    Callable[[S], Output],
    Callable[[S], Awaitable[Output]],
    Callable[[Iterator[S]], Iterator[Output]],
    Callable[[AsyncIterator[S]], AsyncIterator[Output]],
    SyncNode[S, Output],
    AsyncNode[S, Output],
    AwaitableNode[S, Output],
    IteratorNode[S, Output],
    AsyncIteratorNode[S, Output],
]


# Backward-compatible aliases
RunnableCallableSync = SyncNode
RunnableCallableAsync = AsyncNode
RunnableCallableAwaitable = AwaitableNode
RunnableCallableIterator = IteratorNode
RunnableCallableAsyncIterator = AsyncIteratorNode
RunnableLike = NodeLike
Runnable = NodeCallable


@dataclass
class Edge(Generic[S]):
    source: str
    target: str
    condition: Optional[Callable[[S], bool]] = None


class Trace(TypedDict, Generic[S]):
    """
    A Trace object representing steps and states in a workflow.

    Args:
        step_id (int): The step identifier.
        from_node (Optional[str]): The starting node.
        node (str): The current node.
        to_node (Optional[str]): The destination node.
        result: (dict): The result returned from the state.
        state (S): The state of the graph after the step as completed with the trace.
    """

    step_id: int
    from_node: Optional[str]
    node: str
    to_node: Optional[str]
    state_changes: Optional[dict]
    state: Optional[S]


Events = Literal[
    "start",
    "step",
    "node_fail",
    "finish",
    "terminate",
]


class NodeConfig(TypedDict, Generic[S]):
    runnable: NodeLike[S, Any]
    init_kwargs: Optional[dict[str, Any]]


class Graph(Generic[S], EventEmitter[Events], LoggerMixin):
    """
    A directed graph structure to manage and chain together nodes and tasks.

    This class allows you to add nodes and edges, define conditions for transitions between nodes,
    and execute tasks in sequence or based on conditions. It maintains a shared state and logs each
    step in the process for traceability.

    Attributes:
        initial_state (S | None): The starting state for the graph.
        nodes (dict): A dictionary to hold nodes with their associated runnable tasks.
        edges (dict): A dictionary representing connections between nodes.
        step_count (int): Counter to track the number of steps executed.
        history (list[Trace]): List of Trace objects to keep a log of each step.

    Methods:
        add_node(): Adds a node to the graph.
        add_edge(): Adds a directional edge between nodes with an optional condition.
        chain(): Chains nodes in sequence based on provided order.
        traverse(): Executes the graph starting from the starting node.
        visualize_graph(): Generates an HTML representation of the graph.
    """

    # Keep class attribute for backward compatibility
    __TERMINATE__ = "terminate"

    def __init__(self, initial_state: S | None = None, logger: Optional[logging.Logger] = None, enable_tracing: bool = False):
        EventEmitter.__init__(self)
        LoggerMixin.__init__(self, logger=logger)

        """
        @Note: Make sure to update the reset method if necessary when adding properties to the graph
        """
        self.terminate = False
        self.step_count: int = 0
        self.history: list[Trace] = []
        self.___is_running = False
        self._trace = enable_tracing

        self.nodes: dict[str, NodeConfig[S]] = {}
        self.edges: dict[str, list[Edge]] = {}

        self.prev_node: str | None = None
        self.current_node: str | None = None
        self.starting_node: str | None = None

        self.set_initial_state(initial_state)

    @property
    def state(self) -> S:
        return cast(S, copy.deepcopy(self.__state))

    @property
    def is_running(self):
        return self.___is_running

    def set_initial_state(self, initial_state: dict[str, Any] | S | None = None):
        # This allows you to update the initial state of the graph. Useful when using reset

        self.initial_state = copy.deepcopy(initial_state or {})

        # If the graph has started or is currently running block this from modifying state
        if self.is_running:
            return

        self.__state = copy.deepcopy(initial_state or {})

    def reset(self):
        self.__state = copy.deepcopy(self.initial_state)
        self.history = []
        self.current_node = self.starting_node
        self.terminate = False
        self.step_count = 0

    def _get_state_for_node(self) -> dict:
        """Get state copy for node execution. Deep copy when tracing, shallow copy otherwise."""
        if self._trace:
            return copy.deepcopy(dict(self.__state))
        return dict(self.__state)

    def _get_state_for_condition(self) -> S:
        """Get state copy for condition evaluation. Always shallow copy (conditions should be read-only)."""
        return cast(S, dict(self.__state))

    def add_node(
        self,
        name: str,
        runnable: NodeLike[S, Any],
        starting_node: Literal["start"] | None = None,
        init_kwargs: dict | None = None,
    ):
        """Add a Node to the graph

        Args:
            name (str): The name of the node within the graph.
            runnable (NodeLike): Runnable action that accepts and augments the state.
            starting_node (Literal["start"] | None): Indicates whether this node is the starting point.
        """

        if name in self.nodes:
            raise ValueError("Node ('{}') already exists".format(name))

        if starting_node is not None:
            if self.starting_node is not None:
                raise ValueError("Node ('{}') is already set as the starting point.".format(self.starting_node))

        if starting_node:
            self.starting_node = name
            self.current_node = name

        self.nodes[name] = {"runnable": runnable, "init_kwargs": init_kwargs or {}}

    def _validate_edge_conditions(self, from_node: str, condition: Callable[[S], bool] | None):
        # Ensure that if a node has multiple edges, they all have conditions
        if len(self.edges[from_node]) > 0:
            has_condition = any(edge.condition is not None for edge in self.edges[from_node])
            if condition is None or not has_condition:
                raise ValueError(f"The node ('{from_node}') has multiple edges; all edges must have conditions.")

    def add_edge(self, from_node: str, to_node: str, condition: Callable[[S], bool] | None = None):
        if from_node not in self.edges:
            self.edges[from_node] = []
        self._validate_edge_conditions(from_node, condition)
        edge = Edge(from_node, to_node, condition)
        self.edges[from_node].append(edge)
        return edge

    def chain(self, *nodes: str):
        """Chains together nodes based on the order they are provided to the chain."""

        if len(nodes) <= 1:
            raise ValueError("There must be more than one node to chain.")
        for n in nodes:
            if n not in self.nodes:
                raise ValueError("A Node ('{}') does not exists please make sure to add it to the graph.".format(n))

        prev_node = nodes[0]
        for n in nodes[1:]:
            self.add_edge(prev_node, n)
            prev_node = n

    async def traverse(self):
        if self.starting_node is None:
            raise RuntimeError("No starting node was found.")

        if self.current_node is None:
            self.step_count = 0
            self.current_node = self.starting_node

        self.___is_running = True
        while self.current_node and not self.terminate:
            next_node = await self.step(self.current_node)
            if next_node is None:
                break
        self.___is_running = False
        await self.emit("finish", self)

    async def step(self, node: str):
        self.logger.debug(f"Running step with node ('{node}').")
        node_info = self.nodes.get(node)

        if node_info is None:
            raise RuntimeError("Runnable for node ('{}') not Found.".format(node))

        runnable = node_info["runnable"]
        init_kwargs = node_info.get("init_kwargs", {}) or {}

        # Get state for node execution
        before_state = self.state if self._trace else None
        node_state = self._get_state_for_node()

        if inspect.isclass(runnable):
            runnable_instance = runnable(**init_kwargs)
        else:
            runnable_instance = runnable

        try:
            result = self._call_runnable(runnable_instance, node_state)
            await self._process_result(result)
        except Exception as e:
            await self.emit("node_fail", node)
            raise e

        # Determine the next node; none if terminated
        next_node = self.__find_next_node(node) if not self.terminate else None

        if self._trace:
            self._record_trace(node, before_state, next_node)

        await self.emit("step", self)

        self.step_count += 1

        self.prev_node = self.current_node
        self.current_node = next_node
        return next_node

    def _record_trace(self, node: str, before_state: Any, next_node: Optional[str]):
        """Record a trace entry for the current step."""
        after_state = serialize(self.state)
        serialized_before = serialize(before_state)
        state_changes = dict_diff(serialized_before, after_state)

        trace = Trace(
            step_id=self.step_count,
            from_node=self.prev_node,
            node=node,
            to_node=next_node,
            state_changes=dict(state_changes),
            state=after_state,  # type: ignore
        )

        self.add_trace(trace)

    async def _call_runnable(self, runnable: NodeCallable[S, Any], state_data: dict) -> Any:
        """Call the runnable with the state data."""
        if inspect.iscoroutinefunction(runnable) or inspect.iscoroutinefunction(getattr(runnable, "__call__", None)) and not inspect.isawaitable(runnable):
            return await runnable(state_data)  # type: ignore
        else:
            return runnable(state_data)  # type: ignore

    async def _process_result(self, result: Any) -> None:  # noqa: C901
        """Process the result returned by the runnable."""
        if inspect.isawaitable(result) or inspect.iscoroutinefunction(result):
            result = await result

        if isinstance(result, collections.abc.AsyncIterator):
            async for item in result:
                self.handle_result(item)
                if self.terminate:
                    break
        elif isinstance(result, collections.abc.Iterator):
            for item in result:
                self.handle_result(item)
                if self.terminate:
                    break
        else:
            self.handle_result(result)

    def handle_result(self, result):
        """Handle the result from the runnable."""

        result_str = str(result)
        if len(result_str) > 100:
            result_str = result_str[:50] + "..." + result_str[-50:]
        self.logger.debug(f"Result from node ('{self.current_node}'): {result_str}")

        if result is TERMINATE or result == "terminate":
            if isinstance(result, str) and result == "terminate":
                warnings.warn(
                    'Returning the string "terminate" is deprecated. Use TERMINATE sentinel instead: from lumis.pipeline import TERMINATE',
                    DeprecationWarning,
                    stacklevel=4,
                )
            self.logger.debug("Graph terminated.")
            self.terminate = True
            return

        if isinstance(result, dict):
            self.logger.debug("Updating state with result.")
            new_state = copy.deepcopy(self.__state)
            new_state.update(result)
            self.__state = new_state

    def add_trace(self, trace: Trace):
        self.history.append(trace)

    def __find_next_node(self, node: str) -> Optional[str]:
        if node not in self.nodes:
            raise ValueError(f"Node ('{node}') is not a node in the graph.")

        edges = self.edges.get(node)
        if not edges:
            return None

        # Find the first edge where the condition is met
        for edge in edges:
            if edge.condition is None or edge.condition(self._get_state_for_condition()):
                return edge.target
        return None

    def visualize_graph(
        self,
    ) -> str:
        """Generate an HTML representation of the graph.

        Returns:
            str: An HTML string representation of the graph.
        """

        net = Network(notebook=True, directed=True, cdn_resources="in_line")
        # Add nodes to the network
        for node_name in self.nodes.keys():
            net.add_node(
                node_name,
                label=node_name,
                color="green" if self.starting_node == node_name else None,  # type: ignore
            )
        # Add edges to the network
        for _, edges in self.edges.items():
            for edge in edges:
                source = edge.source
                target = edge.target
                condition = edge.condition
                if condition:
                    condition_label = getattr(condition, "__name__", str(condition))
                    condition_label = None if condition_label == "<lambda>" else condition_label
                    net.add_edge(source, target, label=condition_label, arrows="to", dashes=True)
                else:
                    net.add_edge(source, target, arrows="to")
        net.repulsion()
        html_content = net.generate_html()
        return html_content
