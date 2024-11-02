from abc import ABC, abstractmethod
from inspect import iscoroutinefunction
import json
import logging
from typing import Any, Callable, Coroutine, Literal, Optional, TypeVar, Union

from lumis.common.event_emitter import EventEmitter
from lumis.common.logger_mixin import LoggerMixin
from lumis.llm.openai_llm import LLM
from lumis.memory import BaseMemory, SimpleMemory
from lumis.utils.coloured_logger import ColorPrinter
from lumis.utils.string import get_random_string

from langchain_core.utils.function_calling import convert_to_openai_tool
from openai import NOT_GIVEN, NotGiven
from openai._types import Body, Headers, Query
from openai.types import ChatModel
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessageParam, ChatCompletionToolParam

E = TypeVar("E", bound=str)
EventHandler = Union[Callable[..., Any], Callable[..., Coroutine[Any, Any, Any]]]


# class Evaluation(Protocol):
#     @abstractmethod
#     async def evaluate(self, agent: "BaseAgent", *args: Any, **kwargs: Any) -> None:
#         pass


class BaseAgent(EventEmitter[E], LoggerMixin, ABC):
    def __init__(
        self,
        llm: Optional[LLM] = None,  # fmt: ignore
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
            llm = LLM()

        self.llm = llm

        # self.evaluator = evaluator

        self._agent_id = get_random_string(5)
        self.__tools = tools
        self.__memory = memory

        # Set up tools
        self.__tool_name_map: dict[str, Callable] = {tool.__name__: tool for tool in self.tools}
        self.__tool_definitions = [convert_to_openai_tool(tool) for tool in self.tools]

        self.verbose = verbose

        self.logger.debug(f"__init__ completed for {self.__class__.__name__} with ID: {self.agent_id}")

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def memory(self) -> BaseMemory:
        return self.__memory

    @property
    def tools(self) -> list[Callable]:
        return self.__tools

    @property
    def tool_definitions(self) -> list[ChatCompletionToolParam]:
        """
        Converts the tools to OpenAI tool format.

        Returns:
            List[ChatCompletionToolParam]: The list of tools in OpenAI tool format.
        """
        return self.__tool_definitions  # type: ignore

    @property
    def token_count(self):
        return self.llm.token_count

    @abstractmethod
    async def run(self, *args, **kwargs): ...

    async def reset(self):
        """
        Resets the agents preparing it for a new run.

        By default it changes the `agent_id` and clears memory.

        Each agent must define its own

        """

        self._agent_id = get_random_string(5)
        self.memory.clear()
        await self._reset()

    @abstractmethod
    async def _reset(self): ...

    async def call_tool(
        self,
        model: ChatModel = "gpt-4o",
        messages: list[ChatCompletionMessageParam] = [],
        frequency_penalty: float | NotGiven | None = NOT_GIVEN,
        logit_bias: dict[str, int] | NotGiven | None = NOT_GIVEN,
        logprobs: bool | NotGiven | None = NOT_GIVEN,
        max_completion_tokens: int | NotGiven | None = NOT_GIVEN,
        max_tokens: int | NotGiven | None = NOT_GIVEN,
        metadata: dict[str, str] | NotGiven | None = NOT_GIVEN,
        n: int | NotGiven | None = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        presence_penalty: float | NotGiven | None = NOT_GIVEN,
        seed: int | NotGiven | None = NOT_GIVEN,
        service_tier: NotGiven | Literal["auto", "default"] | None = NOT_GIVEN,
        stop: str | list[str] | NotGiven | None = NOT_GIVEN,
        store: bool | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        top_logprobs: int | NotGiven | None = NOT_GIVEN,
        top_p: float | NotGiven | None = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | NotGiven | None = NOT_GIVEN,
    ):
        """
        Gets a tool call from the OpenAI API.

        Args:
            model (ChatModel): The model to use for tool calls.

        Returns:
            Optional[ChatCompletionMessageParam]: The message with tool calls or None if an error occurred.
        """
        self.logger.debug("Starting call_tool")
        self.logger.info(f"Agent<{self.agent_id}> calling tools: {self.tool_definitions}")

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
                n=n,
                parallel_tool_calls=parallel_tool_calls,
                presence_penalty=presence_penalty,
                seed=seed,
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
            if tool_call_message and hasattr(tool_call_message, "tool_calls") and tool_call_message.tool_calls:
                self.logger.debug(f"Received tool call message: {tool_call_message.model_dump()}")
                message = ChatCompletionAssistantMessageParam(**tool_call_message.model_dump())

                self.memory.add(message)
                tool_calls = tool_call_message.tool_calls

                self.logger.debug(f"Processing {len(tool_calls)} tool calls")
                for call in tool_calls:
                    if call:
                        function_name = call.function.name
                        self.logger.debug(f"Calling tool function: {function_name}")
                        tool_result_content = ""
                        try:
                            arguments = json.loads(call.function.arguments)
                            f = self.__tool_name_map.get(function_name)
                            if not f:
                                tool_result_content = f"No such tool function: {function_name}"
                            elif iscoroutinefunction(f):
                                tool_result_content = await f(**arguments)
                            else:
                                tool_result_content = f(**arguments)
                            self.logger.debug(f"Tool {function_name} executed successfully")
                        except json.JSONDecodeError as je:
                            self.log_exception(je, level=logging.ERROR)
                            tool_result_content = f"Invalid arguments for {function_name}"
                        except Exception as e:
                            self.log_exception(e, level=logging.ERROR)
                            tool_result_content = f"An error occurred while calling {function_name}"

                        self.logger.info(f"Tool Call complete with response: {str(tool_result_content)}")
                        self.memory.add(
                            {
                                "role": "tool",
                                "content": str(tool_result_content),
                                "tool_call_id": call.id,
                            }
                        )

            else:
                # self.add_message({"role": "system", "content": "An issue occurred while calling the tool, please try again."})
                self.logger.warning("No tool call message received")
        except Exception as e:
            self.log_exception(e, level=logging.ERROR)

    def add_message(self, message: ChatCompletionMessageParam):
        self.logger.debug(f"Adding message to memory: {message}, memory contains {self.memory.length} messages.")
        self.memory.add(message)
        if self.verbose:
            role = message.get("role", None)
            if role == "tool":
                return
            content = message.get("content", "unknown")
            name = message.get("name", None)
            tag = name if name is not None else self.agent_id if role not in ["system", "tool"] else role

            ColorPrinter.print(f"<{name or role}>\n{content}", tag)

    def _get_tool_definition_prompt(self):
        """
        Generates a prompt for the tool definitions.

        Returns:
            str: The tool definitions prompt.
        """
        tool_definitions = self.tool_definitions

        tool_definitions_list = []
        for tool in tool_definitions:
            function_definition = tool["function"]

            tool_str = f"- {function_definition['name']}:\n\t"  # name
            tool_str += f"{function_definition.get('description', 'No description provided')}\n\t"  # description

            # TODO: format and extract the args
            # if "parameters" in function_definition:
            #     parameters = function_definition["parameters"]
            #     tool_str += "Args:\n\t"  # parameters
            #     for parameter in parameters:
            #         print(parameter)

            # tool_str += f"{parameter}: {parameter.get('type', 'unknown')}\n\t"
            tool_definitions_list.append(tool_str)

        tool_definitions_str = "\n".join(tool_definitions_list)
        self.logger.debug(f"Generated tool definition prompt: {tool_definitions_str}")
        return tool_definitions_str
