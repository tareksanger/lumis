from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import os
from typing import Any, Optional, TypeVar

from lumis.core.common.logger_mixin import LoggerMixin
from lumis.core.event_emitter import EventEmitter
from lumis.core.utils.string import get_random_string
from lumis.llm.openai_llm import OpenAILLM

from openai.types import ChatModel

# For type variables
E = TypeVar("E", bound=str | None)

CHAT_MODEL: ChatModel = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")  # type: ignore


class CoreAgent(EventEmitter[E], LoggerMixin, ABC):
    def __init__(
        self,
        llm: Optional[OpenAILLM] = None,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        LoggerMixin.__init__(self, logger=logger)
        EventEmitter.__init__(self)

        if not llm:
            llm = OpenAILLM()
        self.llm = llm

        self._agent_id = get_random_string(5)

        self.verbose = verbose
        self.logger.debug(f"__init__ completed for {self.__class__.__name__} with ID: {self.agent_id}")

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def token_count(self):
        return self.llm.token_count

    @abstractmethod
    async def run(self, *args, **kwargs): ...

    async def reset(self):
        self._agent_id = get_random_string(5)
        await self._reset()

    async def _reset(self):
        return
