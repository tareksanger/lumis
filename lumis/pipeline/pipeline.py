from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Generic, Optional, TypeVar

from lumis.core.common.logger_mixin import LoggerMixin
from lumis.core.event_emitter import EventEmitter
from lumis.llm.base_llm import BaseLLM

from .graph import Graph, StateProtocol

from typing_extensions import TypeVar as TypeVarWithDefault

S = TypeVar("S", bound=StateProtocol)
E = TypeVar("E", bound=str)
LLMType = TypeVarWithDefault("LLMType", bound=BaseLLM, default=BaseLLM)


class Pipeline(EventEmitter[E], LoggerMixin, ABC, Generic[S, E, LLMType]):
    def __init__(
        self,
        llm: Optional[LLMType] = None,
        logger: Optional[logging.Logger] = None,
        verbose: bool = False,
        enable_tracing: bool = False,
    ) -> None:
        EventEmitter.__init__(self)
        LoggerMixin.__init__(self, logger=logger)
        self.llm = llm
        self.verbose = verbose
        self._graph = Graph[S](enable_tracing=enable_tracing)
        self._initialized = False
        self.build()

    def _require_llm(self) -> LLMType:
        """Return the configured provider, or fail clearly for LLM-dependent nodes."""
        if self.llm is None:
            raise RuntimeError("This pipeline requires an LLM; pass llm= when constructing it.")
        return self.llm

    @property
    def graph(self) -> Graph[S]:
        return self._graph

    @abstractmethod
    def build(self) -> None:
        """Define nodes and edges. Called once during __init__."""
        ...

    @abstractmethod
    async def setup(self, *args, **kwargs) -> None:
        """Initialize state before traversal. Called at the start of each run()."""
        ...

    async def run(self, *args, **kwargs):
        self._initialized = True
        await self.setup(*args, **kwargs)
        await self._graph.traverse()

    async def step(self):
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call run() or setup() first.")
        if self._graph.current_node:
            await self._graph.step(self._graph.current_node)

    @property
    def state(self) -> S:
        return self._graph.state

    @property
    def history(self) -> list:
        return self._graph.history

    def visualize(self) -> str:
        return self._graph.visualize_graph()
