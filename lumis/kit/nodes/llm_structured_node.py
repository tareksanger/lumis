from __future__ import annotations

from typing import (
    Optional,
    Type,
    TypeVar,
)

from lumis.core.common.logger_mixin import LoggerMixin
from lumis.llm import OpenAILLM
from lumis.memory.base_memory import BaseMemory
from lumis.memory.simple_memory import SimpleMemory
from lumis.utils.types import BaseSchema

from ..graph import Graph, S

T = TypeVar("T", bound=BaseSchema)


class LLMStructuredNode(LoggerMixin):
    def __init__(self, llm: OpenAILLM, response_format: Type[T], state_key: str, system_prompt: Optional[str] = None, add_result_to_memory: bool = True, verbose: bool = False, *arg, **kwargs):
        LoggerMixin.__init__(self)
        self.llm = llm
        self.system_prompt = system_prompt
        self.response_format = response_format
        self.state_key = state_key
        self.add_result_to_memory = add_result_to_memory

        self.verbose = verbose

    def _get_memory(self, state: S):
        memory = state.get("memory", None)

        if not isinstance(memory, BaseMemory):
            self.logger.debug("Memory found invalid setting new memory object.")
            memory = SimpleMemory()
        else:
            length = memory.length
            self.logger.debug(f"Memory contains, {length} item(s).")

        return memory

    async def __call__(self, state: S):
        return await self.call_llm(state)

    async def call_llm(self, state: S):  # noqa: C901
        memory = self._get_memory(state)
        messages = await memory.get()

        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})

        content = None
        try:
            response = await self.llm.structured_completion(response_format=self.response_format, messages=messages)

            if response is not None and response.parsed is not None:
                content = response.parsed

                if self.add_result_to_memory:
                    await memory.add({"role": "assistant", "content": content.to_context_str()})

                if self.verbose:
                    self.logger.info(f"Assistant: {content.to_context_str()}")

        except Exception:
            # TODO: Handle all exceptions Properly
            # TEMP
            return Graph.__TERMINATE__

        updated_state = {"memory": memory}

        if content is not None:
            updated_state[self.state_key] = content  # type: ignore

        return updated_state
