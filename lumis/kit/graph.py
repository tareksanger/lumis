from __future__ import annotations

import collections.abc
import copy
from dataclasses import dataclass
import inspect
import logging
from typing import Any, AsyncIterator, Awaitable, Callable, cast, Generic, Iterator, Literal, Optional, Protocol, TypedDict, TypeVar, Union

from lumis.core.common.logger_mixin import LoggerMixin
from lumis.core.event_emitter import EventEmitter
from lumis.core.utils.helpers import serialize

from .utils import dict_diff

from pyvis.network import Network


class StateProtocol(TypedDict):
    """Protocol for state objects that behave like dictionaries."""

    ...


S = TypeVar("S", bound=StateProtocol)
# Output type should implement __concat__, as eg str, list, dict do
Output = TypeVar("Output", covariant=True)


class RunnableCallableSync(Protocol[S, Output]):
    def __call__(self, state: S) -> Output: ...


class RunnableCallableAsync(Protocol[S, Output]):
    async def __call__(self, state: S) -> Output: ...


class RunnableCallableAwaitable(Protocol[S, Output]):
    def __call__(self, state: Iterator[S]) -> Awaitable[Output]: ...


class RunnableCallableIterator(Protocol[S, Output]):
    def __call__(self, state: Iterator[S]) -> Iterator[Output]: ...


class RunnableCallableAsyncIterator(Protocol[S, Output]):
    def __call__(self, state: AsyncIterator[S]) -> AsyncIterator[Output]: ...


RunnableLike = Union[
    Callable[[S], Output],
    Callable[[S], Awaitable[Output]],
    Callable[[Iterator[S]], Iterator[Output]],
    Callable[[AsyncIterator[S]], AsyncIterator[Output]],
    type[RunnableCallableSync[S, Output]],
    type[RunnableCallableAsync[S, Output]],
    type[RunnableCallableAwaitable[S, Output]],
    type[RunnableCallableIterator[S, Output]],
    type[RunnableCallableAsyncIterator[S, Output]],
]


Runnable = Union[
    Callable[[S], Output],
    Callable[[S], Awaitable[Output]],
    Callable[[Iterator[S]], Iterator[Output]],
    Callable[[AsyncIterator[S]], AsyncIterator[Output]],
    RunnableCallableSync[S, Output],
    RunnableCallableAsync[S, Output],
    RunnableCallableAwaitable[S, Output],
    RunnableCallableIterator[S, Output],
    RunnableCallableAsyncIterator[S, Output],
]


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
    runnable: RunnableLike[S, Any]
    init_kwargs: Optional[dict[str, Any]]


class Graph(Generic[S], EventEmitter[Events], LoggerMixin):
    """
    A directed graph structure to manage and chain together nodes and tasks, similar to LangChain.

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

    def add_node(
        self,
        name: str,
        runnable: RunnableLike[S, Any],
        starting_node: Literal["start"] | None = None,
        init_kwargs: dict | None = None,
    ):
        """Add a Node to the graph

        Args:
            name (str): The name of the node within the graph.
            runnable (RunnableLike): Runnable action that accepts and augments the state.
            starting_node (Literal["start"] | None): Indicates whether this node is the starting point.
        """

        assert name not in self.nodes, "Node ('{}') already exists".format(name)

        if starting_node is not None:
            assert (self.starting_node is None) != (starting_node is None), "Node ('{}') is already set as the starting point.".format(self.starting_node)

        if starting_node:
            self.starting_node = name
            self.current_node = name

        self.nodes[name] = {"runnable": runnable, "init_kwargs": init_kwargs or {}}

    def _validate_edge_conditions(self, from_node: str, condition: Callable[[S], bool] | None):
        # Ensure that if a node has multiple edges, they all have conditions
        if len(self.edges[from_node]) > 0:
            has_condition = any(edge.condition is not None for edge in self.edges[from_node])
            assert condition is not None and has_condition, f"The node ('{from_node}') has multiple edges; all edges must have conditions."

    def add_edge(self, from_node: str, to_node: str, condition: Callable[[S], bool] | None = None):
        if from_node not in self.edges:
            self.edges[from_node] = []
        self._validate_edge_conditions(from_node, condition)
        edge = Edge(from_node, to_node, condition)
        self.edges[from_node].append(edge)
        return edge

    def chain(self, *nodes: str):
        """Chains together nodes based on the order they are provided to the chain."""

        assert len(nodes) > 1, "There must be more than one node to chain."
        for n in nodes:
            assert n in self.nodes, "A Node ('{}') does not exists please make sure to add it to the graph.".format(n)

        prev_node = nodes[0]
        for n in nodes[1:]:
            self.add_edge(prev_node, n)
            prev_node = n

    async def traverse(self):
        assert self.starting_node is not None, "No starting node was found."

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

        # Ensure runnable is not None - This should not happen as we ensure that runnables are not none during the addition phase
        assert node_info is not None, "Runnable for node ('{}') not Found.".format(node)

        runnable = node_info["runnable"]
        init_kwargs = node_info.get("init_kwargs", {}) or {}

        # Get a deep copy of the state before the node execution
        before_state = self.state

        if inspect.isclass(runnable):
            runnable_instance = runnable(**init_kwargs)
        else:
            runnable_instance = runnable

        try:
            # Call the runnable with the current state
            result = self._call_runnable(runnable_instance, dict(before_state))

            # Handle the result
            await self._process_result(result)
        except Exception as e:
            await self.emit("node_fail", node)
            raise e

        if self._trace:
            # Get a deep copy of the state after the node execution
            after_state = serialize(self.state)

            # Compute the differences between before_state and after_state
            before_state = serialize(before_state)
            state_changes = dict_diff(before_state, after_state)

        # Determine the next node none if a result was not a TERMINATE
        next_node = self.__find_next_node(node) if not self.terminate else None

        if self._trace:
            # Create a trace entry

            trace = Trace(
                step_id=self.step_count,  # fmt: ignore
                from_node=self.prev_node,
                node=node,
                to_node=next_node,
                state_changes=dict(state_changes),
                state=after_state,  # type: ignore
            )

            self.add_trace(trace)

        await self.emit("step", self)

        self.step_count += 1

        self.prev_node = self.current_node
        self.current_node = next_node
        return next_node

    async def _call_runnable(self, runnable: Runnable[S, Any], state_data: dict) -> Any:
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

        if result == self.__TERMINATE__:
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
        assert node in self.nodes, f"Node ('{node}') is not a node in the graph."

        edges = self.edges.get(node)
        if not edges:
            return None

        # Find the first edge where the condition is met
        for edge in edges:
            if edge.condition is None or edge.condition(copy.deepcopy(self.state)):
                return edge.target
        return None

    def visualize_graph(
        self,
    ) -> str:
        """_summary_

        Returns:
            str: An HTML string representation of the graph.
        """

        net = Network(notebook=True, directed=True, cdn_resources="in_line")
        # Add nodes to the network
        for node_name in self.nodes.keys():
            net.add_node(
                node_name,
                label=node_name,
                color="green" if self.starting_node == node_name else None,  # type: ignore - "green" is returning a Literal instead of a string
            )
        # Add edges to the network
        for _, edges in self.edges.items():
            for edge in edges:
                source = edge.source
                target = edge.target
                condition = edge.condition
                if condition:
                    # Try to get a meaningful label for the condition
                    condition_label = getattr(condition, "__name__", str(condition))
                    condition_label = None if condition_label == "<lambda>" else condition_label
                    net.add_edge(source, target, label=condition_label, arrows="to", dashes=True)
                else:
                    net.add_edge(source, target, arrows="to")
        net.repulsion()
        # Generate the HTML content without saving to a file
        html_content = net.generate_html()
        return html_content
