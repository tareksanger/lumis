from __future__ import annotations

from inspect import iscoroutinefunction
import json
import logging
import os
from typing import (
    Any,
    Callable,
    cast,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    overload,
    Type,
    TypedDict,
    TypeVar,
    Union,
)

from lumis.llm.base_llm import BaseLLM

import httpx
from httpx import TimeoutException
from langchain_core.utils.function_calling import convert_to_openai_tool
import numpy as np
from openai import AsyncOpenAI, not_given, NotGiven, Omit, omit, OpenAIError, Timeout
from openai._types import Body, Headers, Query
from openai.types import ChatModel, ImageModel, Metadata, Reasoning, ResponsesModel
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionReasoningEffort,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    ParsedChatCompletion,
)
from openai.types.chat.chat_completion import ChoiceLogprobs
from openai.types.chat.chat_completion_token_logprob import TopLogprob
from openai.types.chat.completion_create_params import (
    WebSearchOptions,
)
from openai.types.chat.parsed_chat_completion import ParsedChatCompletionMessage
from openai.types.images_response import ImagesResponse
from openai.types.responses import (
    Response,
    ResponseCustomToolCall,
    ResponseFunctionToolCall,
    ResponseIncludable,
    ResponseInputParam,
    ResponsePromptParam,
    ResponseTextConfigParam,
    ToolParam,
)
from openai.types.responses.response_create_params import Conversation, ToolChoice
from openai.types.responses.response_input_param import FunctionCallOutput
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

# from openai.types.completion_usage import CompletionUsage
T = TypeVar("T", bound=BaseModel)

Completion = ChatCompletion | ParsedChatCompletion | ChatCompletionChunk
CompletionT = TypeVar("CompletionT")

CHAT_MODEL: ChatModel = cast(ChatModel, os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
REASONING_MODEL: ChatModel = cast(ChatModel, os.getenv("OPENAI_REASONING_MODEL", "o3-mini"))
IMAGE_MODEL: Union[str, ImageModel] = cast(Union[str, ImageModel], os.getenv("OPENAI_IMAGE_MODEL", "gpt-4o-mini"))


class ToolCallResult(TypedDict):
    result: Any
    call_id: str


ToolCallResults = list[ToolCallResult]


def _is_retryable_exception(exception: BaseException) -> bool:
    if isinstance(exception, OpenAIError):
        return bool(getattr(exception, "should_retry", False))
    return isinstance(exception, (TimeoutException, httpx.RequestError, json.JSONDecodeError))


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        client: Optional[AsyncOpenAI] = None,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        # Initialize BaseLLM first
        super().__init__(verbose=verbose, logger=logger)

        if client is None:
            client = AsyncOpenAI()

        self.__client = client

    @property
    def client(self) -> AsyncOpenAI:
        return self.__client

    def _count_tokens(self, completion: Completion | Response):
        """
        Counts the tokens in the completion.

        Args:
            completion (Completion): The completion to count the tokens of.

        Returns:
            Completion: The completion with the token count.
        """
        # with sentry_sdk.start_span(op="llm.token.count", name="OpenAI LLM") as span:
        try:
            if usage := completion.usage:
                # input_tokens = usage.prompt_tokens if isinstance(usage, CompletionUsage) else usage.input_tokens
                # output_tokens = usage.completion_tokens if isinstance(usage, CompletionUsage) else usage.output_tokens
                # record_token_usage(
                #     span,
                #     total_tokens=usage.total_tokens,
                #     input_tokens=input_tokens,
                #     output_tokens=output_tokens,
                # )
                usage_dict = usage.model_dump(exclude_none=True)
                usage_dict.pop("search_context_size", None)

                def _flatten(counts: Any, prefix: str = "") -> dict[str, int]:
                    """Flatten nested mappings into dot-notated keys with numeric values."""
                    flattened: dict[str, int] = {}
                    if isinstance(counts, Mapping):
                        for key, value in counts.items():
                            full_key = f"{prefix}.{key}" if prefix else key
                            flattened.update(_flatten(value, full_key))
                    elif isinstance(counts, (int, float)):
                        if prefix:
                            flattened[prefix] = int(counts)
                    return flattened

                flat_counts = _flatten(usage_dict)
                self._token_count.update(flat_counts)
                self.logger.debug(f"Updated token count: {self._token_count}")
        except Exception as e:
            self.log_exception(e, level=logging.ERROR)
        return completion

    @overload
    async def completion(
        self,
        model: Optional[ChatModel] = None,
        messages: list[ChatCompletionMessageParam] = [],
        n: Literal[1] | None | Omit = omit,
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
        stop: str | List[str] | Omit | None = omit,
        store: bool | Omit | None = omit,
        temperature: float | Omit | None = omit,
        tool_choice: ChatCompletionToolChoiceOptionParam | Omit = omit,
        tools: Iterable[ChatCompletionToolParam] | Omit = omit,
        top_logprobs: int | Omit | None = omit,
        top_p: float | Omit | None = omit,
        user: str | Omit = omit,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | NotGiven | None = not_given,
    ) -> ChatCompletionMessage: ...

    @overload
    async def completion(
        self,
        model: Optional[ChatModel] = None,
        messages: list[ChatCompletionMessageParam] = [],
        n: int = 2,
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
        stop: str | List[str] | Omit | None = omit,
        store: bool | Omit | None = omit,
        temperature: float | Omit | None = omit,
        tool_choice: ChatCompletionToolChoiceOptionParam | Omit = omit,
        tools: Iterable[ChatCompletionToolParam] | Omit = omit,
        top_logprobs: int | Omit | None = omit,
        top_p: float | Omit | None = omit,
        user: str | Omit = omit,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | NotGiven | None = not_given,
    ) -> List[ChatCompletionMessage]: ...

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        retry=retry_if_exception(_is_retryable_exception),
        reraise=True,
    )
    async def completion(  # noqa: C901
        self,
        model: Optional[ChatModel] = None,
        messages: list[ChatCompletionMessageParam] = [],
        n: int | Omit | None = omit,
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
        stop: str | List[str] | Omit | None = omit,
        store: bool | Omit | None = omit,
        temperature: float | Omit | None = omit,
        tool_choice: ChatCompletionToolChoiceOptionParam | Omit = omit,
        tools: Iterable[ChatCompletionToolParam] | Omit = omit,
        top_logprobs: int | Omit | None = omit,
        top_p: float | Omit | None = omit,
        user: str | Omit = omit,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | NotGiven | None = not_given,
    ) -> Union[ChatCompletionMessage, List[ChatCompletionMessage]]:
        """
        Gets a completion from the OpenAI API.

        Args:
            model (ChatModel): The model to use for completions.
            messages (list[ChatCompletionMessageParam]): The messages to send to the API.
            n (int | Omit): Number of completions to generate. If not provided, returns a single completion.
                               If provided, returns a list of completions.

        Returns:
            Union[ChatCompletionMessageParam, List[ChatCompletionMessageParam]]:
                If n is not provided or is 1, returns a single completion message.
                If n > 1, returns a list of completion messages.
        """
        self.logger.debug(f"Starting completion with model={model}, messages_count={len(messages)}, n={n}")

        # Use the reasoning model if reasoning effort is provided and model is not provided
        if model is None:
            model = CHAT_MODEL if isinstance(reasoning_effort, Omit) else REASONING_MODEL

        try:
            completion = await self.client.chat.completions.create(
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
                reasoning_effort=reasoning_effort,
                seed=seed,
                service_tier=service_tier,
                stop=stop,
                store=store,
                temperature=temperature,
                tool_choice=tool_choice,
                tools=tools,
                top_logprobs=top_logprobs,
                top_p=top_p,
                user=user,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            )
            self.logger.debug("Received completion from OpenAI API")
            completion = await self._apply_middlewares(completion)

            if not completion.choices:
                self.logger.warning("No choices returned in completion")
                raise ValueError("No choices returned in completion")

            # Return all choices if n > 1, otherwise return the first choice
            if isinstance(n, int) and n > 1:
                return [choice.message for choice in completion.choices]  # type: ignore
            return completion.choices[0].message

        except Exception as e:
            self.log_exception(e, level=logging.ERROR)
            raise e

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        retry=retry_if_exception(_is_retryable_exception),
        reraise=True,
    )
    async def response(
        self,
        input: str | ResponseInputParam,
        model: ResponsesModel = CHAT_MODEL,
        background: bool | Omit | None = omit,
        conversation: Conversation | Omit | None = omit,
        include: List[ResponseIncludable] | Omit | None = omit,
        instructions: str | Omit | None = omit,
        max_output_tokens: int | Omit | None = omit,
        max_tool_calls: int | Omit | None = omit,
        metadata: Metadata | Omit | None = omit,
        parallel_tool_calls: bool | Omit | None = omit,
        previous_response_id: str | Omit | None = omit,
        prompt: ResponsePromptParam | Omit | None = omit,
        prompt_cache_key: str | Omit = omit,
        prompt_cache_retention: Literal["in-memory", "24h"] | Omit | None = omit,
        reasoning: Reasoning | Omit | None = None,
        safety_identifier: str | Omit = omit,
        service_tier: Omit | Literal["auto", "default", "flex", "scale", "priority"] | None = omit,
        store: bool | Omit | None = omit,
        temperature: float | Omit | None = omit,
        text: ResponseTextConfigParam | Omit = omit,
        tool_choice: ToolChoice | Omit = omit,
        tools: Iterable[ToolParam] | Omit = omit,
        top_logprobs: int | Omit | None = omit,
        top_p: float | Omit | None = omit,
        truncation: Omit | Literal["auto", "disabled"] | None = omit,
        user: str | Omit = omit,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | Timeout | NotGiven | None = not_given,
    ):
        self.logger.debug(f"Starting response with model={model}, input_type={type(input)}")

        try:
            response = await self.client.responses.create(
                background=background,
                conversation=conversation,
                input=input,
                model=model,
                include=include,
                instructions=instructions,
                max_output_tokens=max_output_tokens,
                max_tool_calls=max_tool_calls,
                metadata=metadata,
                parallel_tool_calls=parallel_tool_calls,
                previous_response_id=previous_response_id,
                prompt=prompt,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
                reasoning=reasoning,  # type: ignore
                safety_identifier=safety_identifier,
                service_tier=service_tier,
                store=store,
                temperature=temperature,
                text=text,
                tool_choice=tool_choice,
                tools=tools,
                top_logprobs=top_logprobs,
                top_p=top_p,
                truncation=truncation,
                user=user,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            )

            self.logger.debug("Received response from OpenAI API")

            return await self._apply_middlewares(response)

        except Exception as e:
            self.log_exception(e, level=logging.ERROR)
            raise e

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        retry=retry_if_exception(_is_retryable_exception),
        reraise=True,
    )
    async def stream(  # noqa: C901
        self,
        model: Optional[ChatModel] = None,
        messages: list[ChatCompletionMessageParam] = [],
        frequency_penalty: float | Omit | None = omit,
        logit_bias: dict[str, int] | Omit | None = omit,
        logprobs: bool | Omit | None = omit,
        max_completion_tokens: int | Omit | None = omit,
        max_tokens: int | Omit | None = omit,
        metadata: dict[str, str] | Omit | None = omit,
        n: int | Omit | None = omit,
        parallel_tool_calls: bool | Omit = omit,
        presence_penalty: float | Omit | None = omit,
        reasoning_effort: ChatCompletionReasoningEffort | Omit = omit,
        seed: int | Omit | None = omit,
        stream_options: ChatCompletionStreamOptionsParam | Omit | None = omit,
        service_tier: Omit | Literal["auto", "default"] | None = omit,
        stop: str | List[str] | Omit | None = omit,
        store: bool | Omit | None = omit,
        temperature: float | Omit | None = omit,
        tool_choice: ChatCompletionToolChoiceOptionParam | Omit = omit,
        tools: Iterable[ChatCompletionToolParam] | Omit = omit,
        top_logprobs: int | Omit | None = omit,
        top_p: float | Omit | None = omit,
        user: str | Omit = omit,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | NotGiven | None = not_given,
    ):
        """
        Gets a streaming completion from the OpenAI API.

        Args:
            model (ChatModel): The model to use for completions.

        Returns:
            A stream of completion chunks with middlewares applied.
        """
        self.logger.debug(f"Starting streaming completion with model={model}, messages_count={len(messages)}")

        # Use the reasoning model if reasoning effort is provided and model is not provided
        if model is None:
            model = CHAT_MODEL if isinstance(reasoning_effort, Omit) else REASONING_MODEL

        try:
            # Ensure stream_options includes usage information
            if isinstance(stream_options, Omit):
                stream_options = {"include_usage": True}
            elif stream_options is None:
                stream_options = {"include_usage": True}
            elif isinstance(stream_options, dict) and "include_usage" not in stream_options:
                stream_options["include_usage"] = True

            # Create a stream manager

            async def stream_manager():
                accumulated_content = ""
                async with await self.client.chat.completions.create(
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
                    reasoning_effort=reasoning_effort,
                    seed=seed,
                    service_tier=service_tier,
                    stop=stop,
                    store=store,
                    stream=True,
                    stream_options=stream_options,
                    temperature=temperature,
                    tool_choice=tool_choice,
                    tools=tools,
                    top_logprobs=top_logprobs,
                    top_p=top_p,
                    user=user,
                    extra_headers=extra_headers,
                    extra_query=extra_query,
                    extra_body=extra_body,
                    timeout=timeout,
                ) as stream:
                    async for chunk in stream:
                        await self._apply_middlewares(chunk)
                        if isinstance(chunk, ChatCompletionChunk) and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                            accumulated_content += chunk.choices[0].delta.content
                            yield accumulated_content, "delta"
                    yield accumulated_content, "done"

            return stream_manager()
        except Exception as e:
            self.log_exception(e, level=logging.ERROR)
            raise e

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        retry=retry_if_exception(_is_retryable_exception),
        reraise=True,
    )
    async def structured_stream(  # noqa: C901
        self,
        response_format: Type[T],
        model: Optional[ChatModel] = None,
        messages: list[ChatCompletionMessageParam] = [],
        frequency_penalty: float | Omit | None = omit,
        logit_bias: dict[str, int] | Omit | None = omit,
        logprobs: bool | Omit | None = omit,
        max_completion_tokens: int | Omit | None = omit,
        max_tokens: int | Omit | None = omit,
        metadata: dict[str, str] | Omit | None = omit,
        n: int | Omit | None = omit,
        parallel_tool_calls: bool | Omit = omit,
        presence_penalty: float | Omit | None = omit,
        reasoning_effort: ChatCompletionReasoningEffort | Omit = omit,
        seed: int | Omit | None = omit,
        stream_options: ChatCompletionStreamOptionsParam | Omit | None = omit,
        service_tier: Omit | Literal["auto", "default"] | None = omit,
        stop: str | List[str] | Omit | None = omit,
        store: bool | Omit | None = omit,
        temperature: float | Omit | None = omit,
        tool_choice: ChatCompletionToolChoiceOptionParam | Omit = omit,
        tools: Iterable[ChatCompletionToolParam] | Omit = omit,
        top_logprobs: int | Omit | None = omit,
        top_p: float | Omit | None = omit,
        user: str | Omit = omit,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | NotGiven | None = not_given,
    ):
        """
        Gets a streaming structured completion from the OpenAI API.

        See examples usage:https://platform.openai.com/docs/guides/structured-outputs

        Args:
            response_format (Type[T]): The expected response format.
            model (ChatModel): The model to use for completions.
            messages (list[ChatCompletionMessageParam]): The messages to send to the API.
            stream_options (ChatCompletionStreamOptionsParam): Options for streaming.

        Returns:
            A stream of structured completion chunks with middlewares applied.
        """
        self.logger.debug(f"Starting stream_structured_completion with model={model}, response_format={response_format.__name__}, messages_count={len(messages)}")

        # Use the reasoning model if reasoning effort is provided and model is not provided
        if model is None:
            model = CHAT_MODEL if isinstance(reasoning_effort, Omit) else REASONING_MODEL

        try:
            # Ensure stream_options includes usage information
            if isinstance(stream_options, Omit) or stream_options is None:
                stream_options = {"include_usage": True}
            elif isinstance(stream_options, dict) and "include_usage" not in stream_options:
                stream_options["include_usage"] = True

            # Create an async generator that handles the stream context manager
            async def stream_generator():
                async with self.client.chat.completions.stream(
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
                    reasoning_effort=reasoning_effort,
                    seed=seed,
                    service_tier=service_tier,
                    stop=stop,
                    store=store,
                    stream_options=stream_options,
                    temperature=temperature,
                    tool_choice=tool_choice,
                    tools=tools,
                    top_logprobs=top_logprobs,
                    top_p=top_p,
                    user=user,
                    extra_headers=extra_headers,
                    extra_query=extra_query,
                    extra_body=extra_body,
                    timeout=timeout,
                    response_format=response_format,
                ) as stream:
                    # Process each chunk in the stream
                    async for event in stream:
                        # Apply middlewares to the accumulated completion if it's the final chunk
                        if event.type == "chunk":
                            await self._apply_middlewares(event.chunk)  # type: ignore

                        # Yield the original chunk
                        yield event

                    yield stream.get_final_completion()

            # Return the stream generator
            return stream_generator()

        except Exception as e:
            self.log_exception(e, level=logging.ERROR)
            raise e

    @overload
    async def structured_completion(
        self,
        response_format: Type[T],
        model: Optional[ChatModel] = None,
        messages: list[ChatCompletionMessageParam] = [],
        n: Literal[1] | Omit = omit,
        frequency_penalty: float | Omit | None = omit,
        logit_bias: dict[str, int] | Omit | None = omit,
        logprobs: bool | Omit | None = omit,
        web_search_options: WebSearchOptions | Omit = omit,
        max_completion_tokens: int | Omit | None = omit,
        max_tokens: int | Omit | None = omit,
        metadata: dict[str, str] | Omit | None = omit,
        parallel_tool_calls: bool | Omit = omit,
        presence_penalty: float | Omit | None = omit,
        reasoning_effort: ChatCompletionReasoningEffort | Omit = omit,
        seed: int | Omit | None = omit,
        service_tier: Omit | Literal["auto", "default"] | None = omit,
        stop: str | List[str] | Omit | None = omit,
        store: bool | Omit | None = omit,
        temperature: float | Omit | None = omit,
        tool_choice: ChatCompletionToolChoiceOptionParam | Omit = omit,
        tools: Iterable[ChatCompletionToolParam] | Omit = omit,
        top_logprobs: int | Omit | None = omit,
        top_p: float | Omit | None = omit,
        user: str | Omit = omit,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | NotGiven | None = not_given,
    ) -> ParsedChatCompletionMessage[T]: ...

    @overload
    async def structured_completion(
        self,
        response_format: Type[T],
        model: Optional[ChatModel] = None,
        messages: list[ChatCompletionMessageParam] = [],
        n: int = 2,
        frequency_penalty: float | Omit | None = omit,
        logit_bias: dict[str, int] | Omit | None = omit,
        logprobs: bool | Omit | None = omit,
        web_search_options: WebSearchOptions | Omit = omit,
        max_completion_tokens: int | Omit | None = omit,
        max_tokens: int | Omit | None = omit,
        metadata: dict[str, str] | Omit | None = omit,
        parallel_tool_calls: bool | Omit = omit,
        presence_penalty: float | Omit | None = omit,
        reasoning_effort: ChatCompletionReasoningEffort | Omit = omit,
        seed: int | Omit | None = omit,
        service_tier: Omit | Literal["auto", "default"] | None = omit,
        stop: str | List[str] | Omit | None = omit,
        store: bool | Omit | None = omit,
        temperature: float | Omit | None = omit,
        tool_choice: ChatCompletionToolChoiceOptionParam | Omit = omit,
        tools: Iterable[ChatCompletionToolParam] | Omit = omit,
        top_logprobs: int | Omit | None = omit,
        top_p: float | Omit | None = omit,
        user: str | Omit = omit,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | NotGiven | None = not_given,
    ) -> List[ParsedChatCompletionMessage[T]]: ...

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        reraise=True,
    )
    async def structured_completion(  # noqa: C901
        self,
        response_format: Type[T] | Omit = omit,
        model: Optional[ChatModel] = None,
        messages: list[ChatCompletionMessageParam] = [],
        n: int | Omit | None = omit,
        frequency_penalty: float | Omit | None = omit,
        logit_bias: dict[str, int] | Omit | None = omit,
        logprobs: bool | Omit | None = omit,
        web_search_options: WebSearchOptions | Omit = omit,
        max_completion_tokens: int | Omit | None = omit,
        max_tokens: int | Omit | None = omit,
        metadata: dict[str, str] | Omit | None = omit,
        parallel_tool_calls: bool | Omit = omit,
        presence_penalty: float | Omit | None = omit,
        reasoning_effort: ChatCompletionReasoningEffort | Omit = omit,
        seed: int | Omit | None = omit,
        service_tier: Omit | Literal["auto", "default"] | None = omit,
        stop: str | List[str] | Omit | None = omit,
        store: bool | Omit | None = omit,
        temperature: float | Omit | None = omit,
        tool_choice: ChatCompletionToolChoiceOptionParam | Omit = omit,
        tools: Iterable[ChatCompletionToolParam] | Omit = omit,
        top_logprobs: int | Omit | None = omit,
        top_p: float | Omit | None = omit,
        user: str | Omit = omit,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | NotGiven | None = not_given,
    ) -> Union[ParsedChatCompletionMessage[T], List[ParsedChatCompletionMessage[T]]]:
        """
        Gets a structured completion from the OpenAI API.

        Args:
            response_format (Type[T]): The expected response format.
            model (ChatModel): The model to use for completions.
            messages (list[ChatCompletionMessageParam]): The messages to send to the API.
            n (int | Omit): Number of completions to generate. If not provided, returns a single completion.
                               If provided, returns a list of completions.

        Returns:
            Union[ParsedChatCompletionMessage[T], List[ParsedChatCompletionMessage[T]]]:
                If n is not provided or is 1, returns a single parsed completion message.
                If n > 1, returns a list of parsed completion messages.
        """
        self.logger.debug(
            f"Starting structured_completion with model={model}, response_format={response_format.__name__ if not isinstance(response_format, Omit) else None}, messages_count={len(messages)}, n={n}"  # noqa: E501
        )

        # Use the reasoning model if reasoning effort is provided and model is not provided
        if model is None:
            model = CHAT_MODEL if isinstance(reasoning_effort, Omit) else REASONING_MODEL

        try:
            completion = await self.client.chat.completions.parse(
                response_format=response_format,
                model=model,
                messages=messages,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                logprobs=logprobs,
                max_completion_tokens=max_completion_tokens,
                web_search_options=web_search_options,
                max_tokens=max_tokens,
                metadata=metadata,
                n=n,
                parallel_tool_calls=parallel_tool_calls,
                presence_penalty=presence_penalty,
                reasoning_effort=reasoning_effort,
                seed=seed,
                service_tier=service_tier,
                stop=stop,
                store=store,
                temperature=temperature,
                tool_choice=tool_choice,
                tools=tools,
                top_logprobs=top_logprobs,
                top_p=top_p,
                user=user,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            )
            self.logger.debug("Received structured completion from OpenAI API")
            completion = await self._apply_middlewares(completion)

            if not completion.choices:
                self.logger.warning("No choices returned in structured completion")
                raise ValueError("No choices returned in structured completion")

            # Return all choices if n > 1, otherwise return the first choice
            if isinstance(n, int) and n > 1:
                return [choice.message for choice in completion.choices]
            return completion.choices[0].message

        except Exception as e:
            self.log_exception(e, level=logging.ERROR)
            raise e

    @retry(stop=stop_after_attempt(5), reraise=True)
    async def generate_image(
        self,
        prompt: str,
        model: Union[str, ImageModel, None] | Omit = IMAGE_MODEL,
        n: Optional[int] | Omit = omit,
        quality: Literal["standard", "hd"] | Omit = omit,
        response_format: Optional[Literal["url", "b64_json"]] | Omit = omit,
        size: (Optional[Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]] | Omit) = "256x256",
        style: Optional[Literal["vivid", "natural"]] | Omit = omit,
        user: str | Omit = omit,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = not_given,
    ) -> ImagesResponse:
        """
        Generates an image using the OpenAI DALL-E model.

        Args:
            prompt (str): The prompt to generate the image.
            model (str): The model to use for image generation.
            size (str): The size of the image. Defaults to "1024x1024".
            quality (str): The quality of the image. Defaults to "standard".
            n (int): The number of images to generate. Defaults to 1.

        Returns:
            List[str]: A list of URLs to the generated images.
        """
        self.logger.debug(f"Starting image generation with prompt='{prompt}', model='{model}', size='{size}', quality='{quality}', n={n}")
        try:
            response = await self.client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                n=n,
                response_format=response_format,
                style=style,
                user=user,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            )
            self.logger.debug("Received response from image generation API")
            return response
        except TimeoutException as te:
            self.log_exception(te, level=logging.ERROR)
            raise te
        except httpx.RequestError as re:
            self.log_exception(re, level=logging.ERROR)
            raise re
        except json.JSONDecodeError as je:
            self.log_exception(je, level=logging.ERROR)
            raise je
        except Exception as e:
            self.log_exception(e, level=logging.ERROR)
            raise e

    # TODO: Revisit this is too Agentic with no hooks, state or controls (Great for testing though)
    async def generate(
        self,
        input: str | ResponseInputParam,
        model: ResponsesModel = CHAT_MODEL,
        background: bool | Omit | None = omit,
        conversation: Conversation | Omit | None = omit,
        include: List[ResponseIncludable] | Omit | None = omit,
        instructions: str | Omit | None = omit,
        max_output_tokens: int | Omit | None = omit,
        max_tool_calls: int | Omit | None = omit,
        metadata: Metadata | Omit | None = omit,
        parallel_tool_calls: bool | Omit | None = omit,
        previous_response_id: str | Omit | None = omit,
        prompt: ResponsePromptParam | Omit | None = omit,
        prompt_cache_key: str | Omit = omit,
        prompt_cache_retention: Literal["in-memory", "24h"] | Omit | None = omit,
        reasoning: Reasoning | Omit | None = None,
        safety_identifier: str | Omit = omit,
        service_tier: Omit | Literal["auto", "default", "flex", "scale", "priority"] | None = omit,
        store: bool | Omit | None = omit,
        temperature: float | Omit | None = omit,
        text: ResponseTextConfigParam | Omit = omit,
        tool_choice: ToolChoice | Omit = omit,
        tools: Iterable[Callable] | None = None,
        top_logprobs: int | Omit | None = omit,
        top_p: float | Omit | None = omit,
        truncation: Omit | Literal["auto", "disabled"] | None = omit,
        user: str | Omit = omit,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | Timeout | NotGiven | None = not_given,
        max_iterations: int = 8,
    ):
        iteration = 0

        if isinstance(input, str):
            current_input: ResponseInputParam = [{"role": "user", "content": input}]
        else:
            current_input = input

        while True:
            self.logger.debug(f"generate iteration={iteration} starting; tools_enabled={bool(tools)}")

            tool_definitions = omit
            if tools:
                tool_map, tool_definitions = self._process_tools(list(tools))
                print(tool_map, tool_definitions)

            response = await self.response(
                input=current_input,
                model=model,
                background=background,
                conversation=conversation,
                include=include,
                instructions=instructions,
                max_output_tokens=max_output_tokens,
                max_tool_calls=max_tool_calls,
                metadata=metadata,
                parallel_tool_calls=parallel_tool_calls,
                previous_response_id=previous_response_id,
                prompt=prompt,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
                reasoning=reasoning,
                safety_identifier=safety_identifier,
                service_tier=service_tier,
                store=store,
                temperature=temperature,
                text=text,
                tool_choice=tool_choice,
                tools=tool_definitions,
                top_logprobs=top_logprobs,
                top_p=top_p,
                truncation=truncation,
                user=user,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
            )

            tool_calls = self._get_response_tool_calls(response)
            if not (tools and tool_calls):
                self.logger.debug(f"generate iteration={iteration} completed with no tool calls; returning response")
                return response

            if iteration >= max_iterations:
                self.logger.warning("Max tool-call iterations reached; returning last response")
                return response

            tool_call_names = [getattr(call, "name", "unknown") for call in tool_calls]
            self.logger.debug(f"generate iteration={iteration} processing {len(tool_calls)} tool calls: {tool_call_names}")

            current_input = current_input + tool_calls  # type: ignore
            calls = await self.handle_response_tool_call(response, tool_map)
            if calls:
                current_input = current_input + calls
                self.logger.debug(f"generate iteration={iteration} appended tool call outputs; continuing")

            iteration += 1

    async def handle_response_tool_call(
        self,
        response: Response,
        tool_map: dict[str, Callable[..., Any]],
    ) -> list[FunctionCallOutput] | None:
        tool_calls = self._get_response_tool_calls(response)
        if not tool_calls:
            return None

        self.logger.debug(f"Received response tool calls: {[call.model_dump() for call in tool_calls]}")
        calls: list[FunctionCallOutput] = []
        for call in tool_calls:
            if isinstance(call, ResponseFunctionToolCall):
                arguments = self._parse_tool_arguments(call.arguments, call.name)
                name = call.name
            else:  # ResponseCustomToolCall
                arguments = self._parse_custom_tool_arguments(call)
                name = call.name

            if arguments is None:
                calls.append(FunctionCallOutput(output=f"Invalid arguments for {name}", call_id=call.call_id, type="function_call_output"))
                continue

            tool_result_content = await self._invoke_tool(name, arguments, tool_map)
            self.logger.debug(f"Tool call complete with response: {str(tool_result_content)}")
            calls.append(FunctionCallOutput(output=str(tool_result_content), call_id=call.call_id, type="function_call_output"))

        return calls

    async def handle_chat_completion_tool_call(
        self,
        message: ChatCompletionMessage,
        tool_map: dict[str, Callable[..., object]],
    ) -> ToolCallResults | None:
        if not self._has_tool_calls(message):
            return None

        self.logger.debug(f"Received tool call message: {message.model_dump()}")
        tool_calls = message.tool_calls or []
        self.logger.debug(f"Processing {len(tool_calls)} tool calls")

        calls: ToolCallResults = []
        for call in tool_calls:
            payload = self._get_tool_payload(call)
            if payload is None:
                continue

            function_name, raw_arguments = payload
            arguments = self._parse_tool_arguments(raw_arguments, function_name)
            if arguments is None:
                calls.append({"result": f"Invalid arguments for {function_name}", "call_id": call.id})
                continue

            tool_result_content = await self._invoke_tool(function_name, arguments, tool_map)
            self.logger.debug(f"Tool call complete with response: {str(tool_result_content)}")
            calls.append({"result": tool_result_content, "call_id": call.id})

        return calls

    def _get_response_tool_calls(
        self,
        response: Response,
    ) -> List[ResponseFunctionToolCall | ResponseCustomToolCall]:
        output_items = getattr(response, "output", None) or []
        return [item for item in output_items if isinstance(item, (ResponseFunctionToolCall, ResponseCustomToolCall))]

    def _has_tool_calls(self, message: ChatCompletionMessage) -> bool:
        return bool(message and hasattr(message, "tool_calls") and message.tool_calls)

    def _get_tool_payload(
        self,
        call: ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall,
    ) -> tuple[str, str | Mapping[str, object] | None] | None:
        tool_payload = getattr(call, "function", None) or getattr(call, "custom", None)
        if not tool_payload:
            self.logger.error(f"Unsupported tool call payload: {call}")
            return None

        function_name = getattr(tool_payload, "name", None)
        if not function_name:
            self.logger.error(f"Tool call missing function name: {call}")
            return None

        raw_arguments = getattr(tool_payload, "arguments", {}) or {}
        return function_name, raw_arguments

    def _parse_tool_arguments(
        self,
        raw_arguments: str | Mapping[str, object] | None,
        function_name: str,
    ) -> dict[str, object] | None:
        try:
            if isinstance(raw_arguments, str):
                return json.loads(raw_arguments) if raw_arguments else {}
            if raw_arguments is None:
                return {}
            if isinstance(raw_arguments, Mapping):
                return dict(raw_arguments)
            self.logger.error(f"Unexpected argument type for {function_name}: {type(raw_arguments)}")
            return {}
        except json.JSONDecodeError as je:
            self.logger.error(f"JSON decode error for {function_name}: {je}")
            return None

    def _parse_custom_tool_arguments(self, call: ResponseCustomToolCall) -> dict[str, object]:
        if not call.input:
            return {}
        try:
            parsed = json.loads(call.input)
            if isinstance(parsed, Mapping):
                return dict(parsed)
            return {"input": parsed}
        except json.JSONDecodeError:
            self.logger.debug(f"Using raw input for custom tool {call.name}")
            return {"input": call.input}

    async def _invoke_tool(
        self,
        function_name: str,
        arguments: dict[str, object],
        tool_map: dict[str, Callable[..., Any]],
    ) -> Any:
        tool_function = tool_map.get(function_name)
        if not tool_function:
            return f"No such tool function: {function_name}"

        self.logger.debug(f"Calling tool function: {function_name}")
        try:
            if iscoroutinefunction(tool_function):
                result = await tool_function(**arguments)
            else:
                result = tool_function(**arguments)
            self.logger.debug(f"Tool {function_name} executed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Error calling tool {function_name}: {e}")
            error_message = str(e).strip() or e.__class__.__name__
            return error_message

    def confidence_score(self, top_logprob: TopLogprob) -> float:
        """
        Calculates the confidence score for a single token in the completion.

        Args:
            logprobs (ChoiceLogprobs): The log probabilities for the completion.

        Returns:
            float: The confidence score.
        """
        _, _, confidence_score = self.token_abstraction(top_logprob=top_logprob)
        return confidence_score

    def token_abstraction(self, top_logprob: TopLogprob):
        return (
            top_logprob.token,
            top_logprob.logprob,
            np.round(np.exp(top_logprob.logprob) * 100, 2),
        )

    def perplexity(self, choice_logprobs: ChoiceLogprobs):
        """
        Calculates how perplexed the llm is based on the logprobs.

        When looking to assess the model's confidence in a result, it can be useful to calculate perplexity,
        which is a measure of the uncertainty

        Args:
            logprobs (ChoiceLogprobs): The log probabilities for the completion.

        Returns:
            float: The perplexity score.
        """

        if not choice_logprobs.content:
            return 0.0
        logprobs = [lp.logprob for lp in choice_logprobs.content if lp.logprob is not None]
        if not logprobs:
            return 0.0

        avg_logprob = sum(logprobs) / len(logprobs)
        perplexity_score: float = float(np.exp(-avg_logprob))
        return max(perplexity_score, 1.0)

    def _convert_function_to_tool_param(self, function: Callable) -> ToolParam:
        convert = convert_to_openai_tool(function)
        result = convert.get("function", {})
        result["type"] = convert.get("type", "function")
        return result

    def _process_tools(self, tools: list[Callable]):
        tool_map: dict[str, Callable] = {}
        tool_definitions: list[ToolParam] = []

        for tool in tools:
            if not callable(tool):
                self.logger.error(f"Tool {tool} is not callable; skipping")
                continue

            tool_map[tool.__name__] = tool
            tool_definitions.append(self._convert_function_to_tool_param(tool))

        tool_names = [td.get("name") if isinstance(td, Mapping) else None for td in tool_definitions]
        self.logger.debug(f"Prepared tool definitions: {tool_names}")

        return tool_map, tool_definitions
