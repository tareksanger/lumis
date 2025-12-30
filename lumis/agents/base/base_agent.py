from __future__ import annotations

from abc import ABC
from inspect import iscoroutinefunction
import json
import logging
import os
from typing import Any, Callable, Literal, Optional, TypeVar

from lumis.core.common.logger_mixin import LoggerMixin
from lumis.core.event_emitter import EventEmitter
from lumis.core.utils.string import get_random_string
from lumis.llm.openai_llm import OpenAILLM
from lumis.memory import BaseMemory
from lumis.memory.simple_memory import SimpleMemory

from .core_agent import CoreAgent

from langchain_core.utils.function_calling import convert_to_openai_tool
from openai import not_given, NotGiven, Omit, omit
from openai._types import Body, Headers, Query
from openai.types import ChatModel
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionReasoningEffort,
    ChatCompletionToolParam,
)

# For type variables
E = TypeVar("E", bound=str | None)

CHAT_MODEL: ChatModel = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")  # type: ignore


class BaseAgent(CoreAgent[E], ABC):
    def __init__(
        self,
        llm: Optional[OpenAILLM] = None,  # fmt: ignore
        memory: BaseMemory = SimpleMemory(),
        tools: list[Callable] = [],
        # Note: should we just use the events instead?
        # evaluator: Optional[Evaluation] = None,
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

        self._memory = memory
        self._process_tools(tools)

        self._agent_id = get_random_string(5)

        self.verbose = verbose
        self.logger.debug(f"__init__ completed for {self.__class__.__name__} with ID: {self.agent_id}")

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def memory(self) -> BaseMemory:
        return self._memory

    @property
    def tools(self) -> list[Callable]:
        return self.__tools

    @property
    def tool_definitions(self) -> list[ChatCompletionToolParam]:
        return self.__tool_definitions  # type: ignore

    @property
    def token_count(self):
        return self.llm.token_count

    async def call_tool(  # noqa: C901
        self,
        model: ChatModel = CHAT_MODEL,
        messages: list[ChatCompletionMessageParam] = [],
        n: Omit | Literal[1] = omit,
        frequency_penalty: float | Omit | None = omit,
        logit_bias: dict[str, int] | Omit | None = omit,
        logprobs: bool | Omit | None = omit,
        max_completion_tokens: int | Omit | None = omit,
        max_tokens: int | Omit | None = omit,
        metadata: dict[str, str] | Omit | None = omit,
        parallel_tool_calls: bool | Omit = omit,
        presence_penalty: float | Omit | None = omit,
        reasoning_effort: ChatCompletionReasoningEffort | Omit = omit,
        seed: int | Omit | None = omit,
        service_tier: Omit | Literal["auto", "default"] | None = omit,
        stop: str | list[str] | Omit | None = omit,
        store: bool | Omit | None = omit,
        temperature: float | Omit | None = omit,
        top_logprobs: int | Omit | None = omit,
        top_p: float | Omit | None = omit,
        user: str | Omit = omit,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | NotGiven | None = not_given,
    ):
        self.logger.debug("Starting call_tool")
        self.logger.debug(f"Agent<{self.agent_id}> calling tools: {self.tool_definitions}")

        try:
            tool_call_message = await self.llm.completion(
                model=model,
                messages=messages,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                logprobs=logprobs,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                metadata=metadata,
                parallel_tool_calls=parallel_tool_calls,
                presence_penalty=presence_penalty,
                seed=seed,
                reasoning_effort=reasoning_effort,
                n=n,
                service_tier=service_tier,
                stop=stop,
                store=store,
                temperature=temperature,
                tool_choice="required",
                tools=self.tool_definitions,
                top_logprobs=top_logprobs,
                top_p=top_p,
                user=user,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            )

            if self.llm._has_tool_calls(tool_call_message):
                # Add tool call to memory
                await self.memory.add(ChatCompletionAssistantMessageParam(**tool_call_message.model_dump()))

                calls = await self.llm.handle_chat_completion_tool_call(message=tool_call_message, tool_map=self.__tool_name_map)
                if calls:
                    for call in calls:
                        await self.memory.add(
                            {
                                "role": "tool",
                                "content": str(call["result"]),
                                "tool_call_id": call["call_id"],
                            }
                        )
            else:
                self.logger.debug("No tool call message received")
        except Exception as e:
            self.logger.error(f"Exception during call_tool: {e}")

    async def add_message(self, message: ChatCompletionMessageParam):
        length = self.memory.length
        self.logger.debug(f"Adding message to memory: {message}, memory contains {length} messages.")
        await self.memory.add(message)
        if self.verbose:
            role = message.get("role", None)
            if role == "tool":
                return

    def _process_tools(self, tools: list[Callable]):
        # Process initial tools (if any)
        self.__tools = list(tools)
        self.__tool_name_map: dict[str, Callable] = {tool.__name__: tool for tool in self.__tools}
        self.__tool_definitions = [convert_to_openai_tool(tool) for tool in self.__tools]

    def _get_tool_definition_prompt(self):
        tool_definitions_list = []
        for tool in self.tool_definitions:
            function_definition = tool["function"]
            tool_str = f"- {function_definition['name']}:\n\t{function_definition.get('description', 'No description provided')}\n\t"
            tool_definitions_list.append(tool_str)
        tool_definitions_str = "\n".join(tool_definitions_list)
        self.logger.debug(f"Generated tool definition prompt: {tool_definitions_str}")
        return tool_definitions_str
