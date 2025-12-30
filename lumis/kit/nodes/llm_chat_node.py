from __future__ import annotations

from typing import Optional

from lumis.core.common.logger_mixin import LoggerMixin
from lumis.llm import OpenAILLM
from lumis.memory.base_memory import BaseMemory
from lumis.memory.simple_memory import SimpleMemory

from ..graph import Graph, S

class LLMChatNode(LoggerMixin):
    def __init__(self, llm: OpenAILLM, system_prompt: Optional[str] = None, verbose: bool = False, *arg, **kwargs):
        LoggerMixin.__init__(self)
        self.llm = llm
        self.system_prompt = system_prompt
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

    async def call_llm(self, state: S):
        memory = self._get_memory(state)
        messages = await memory.get()

        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})

        try:
            response = await self.llm.completion(messages=messages)

            if response is not None and response.content is not None:
                content = response.content
                if isinstance(content, list):
                    content_parts = []
                    for part in content:
                        if isinstance(part, str):
                            content_parts.append(part)
                            continue

                        text = getattr(part, "text", None)
                        value = getattr(text, "value", None) if text is not None else None

                        if isinstance(text, str):
                            content_parts.append(text)
                        elif isinstance(value, str):
                            content_parts.append(value)

                    content = "".join(content_parts) if content_parts else None

                if content is not None:
                    await memory.add({"role": "assistant", "content": content})

                    if self.verbose:
                        self.logger.info(f"Assistant: {content}" + "\n" + ("-" * 100) + "\n")

        except Exception:
            return Graph.__TERMINATE__

        return {"memory": memory}
