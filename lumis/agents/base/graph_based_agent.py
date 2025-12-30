from __future__ import annotations

from abc import abstractmethod
import logging
from typing import Generic, Optional, TypeVar

from lumis.agents.base.core_agent import CoreAgent
from lumis.kit.graph import Graph, StateProtocol
from lumis.llm.openai_llm import OpenAILLM

S = TypeVar("S", bound=StateProtocol)
E = TypeVar("E", bound=str | None)


class GraphBasedAgent(CoreAgent[E], Generic[S, E]):
    def __init__(self, llm: Optional[OpenAILLM] = None, logger: Optional[logging.Logger] = None, verbose: bool = False, *args, **kwargs) -> None:
        super().__init__(llm, verbose=verbose, logger=logger)

        self.__initialized = False
        self.graph = Graph[S]()

        self.construct_graph()

    @abstractmethod
    def construct_graph(self): ...

    async def initialize(self, *args, **kwargs):
        """
        Initializes the agent by setting up the initial state and preparing the agent for operation.

        This method marks the agent as initialized and then calls the `setup` method to perform
        any additional configuration or preparation. Override the `setup` method in subclasses
        to define custom initialization behavior.

        Args:
            *args: Positional arguments passed to the `setup` method for customization.
            **kwargs: Keyword arguments passed to the `setup` method for customization.

        Example:
            await agent.initialize(topic="Space Exploration")
        """
        self.__initialized = True
        await self.setup(*args, **kwargs)

    @abstractmethod
    async def setup(self, *args, **kwargs):
        """
        Abstract method to set up the agent. This method is called during initialization.

        Subclasses must implement this method to define the specific steps required to
        configure the agent. It can include tasks like loading data, preparing resources,
        or defining state variables.

        Args:
            *args: Positional arguments for setup configuration.
            **kwargs: Keyword arguments for setup configuration.

        Example (in subclass):
            async def setup(self, topic: str):
                self.topic = topic
                self.logger.info(f"Setting up the agent for topic: {self.topic}")
        """
        ...

    def visualize(self):
        """
        Visualize the agent's workflow graph. Useful for debugging or understanding the flow of steps.

        Returns:
            Any: A visualization object or representation from the underlying Graph class.
        """
        return self.graph.visualize_graph()

    async def run(self, *args, **kwargs):
        """
        Execute the full workflow to produce the article on the given topic.

        Steps:
        1. Initialize the agent with the topic.
        2. Run through each step of the graph, culminating in a complete article.

        Args:
            topic (str): The topic for which the article is being produced.
        """
        await self.initialize(*args, **kwargs)
        await self.graph.traverse()

    async def step(self):
        """
        Execute a single step/node of the workflow graph. This allows manual control of progression through
        the workflow for debugging or partial execution.

        Raises:
            AssertionError: If the agent is not yet initialized.
        """
        assert self.__initialized, "Agent is not initialized. Please call `.initialize(...)` first."
        if self.graph.current_node is None:
            self.logger.debug("No current node. Cannot perform step.")
            return

        await self.graph.step(self.graph.current_node)

    def get_state(self) -> S:
        """
        Get the current state of the agent.
        """
        return self.graph.state
