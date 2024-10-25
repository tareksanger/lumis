from abc import ABC
from collections import Counter
from inspect import iscoroutinefunction
import json
import logging
from typing import Callable, Generic, Iterable, List, Literal, Optional, Type, TypeVar

from lumis.common.logger_mixin import LoggerMixin

import httpx
from httpx import TimeoutException
from openai import AsyncOpenAI, NOT_GIVEN, NotGiven
from openai._types import Body, Headers, Query
from openai.types import ChatModel
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    ParsedChatCompletion,
    ParsedChatCompletionMessage,
)
from openai.types.chat.chat_completion import ChoiceLogprobs
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt

T = TypeVar("T", bound=BaseModel)

Completion = ChatCompletion | ParsedChatCompletion
CompletionT = TypeVar("CompletionT", bound=Completion)


class Response(BaseModel):
    content: str = Field(..., description="The response content")
    confidence: float = Field(..., description="The confidence of the response")


class StructuredResponse(BaseModel, Generic[T]):
    content: Type[T]
    confidence: float = Field(..., description="The confidence of the response")


class LLM(LoggerMixin, ABC):
    def __init__(
        self,
        client: Optional[AsyncOpenAI] = None,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        # Initialize LoggerMixin first
        super().__init__(logger=logger)

        if client is None:
            client = AsyncOpenAI()

        self.__client = client

        self.__token_count = Counter()
        self.__verbose = verbose

        self.__middlewares: List[Callable] = []
        self.__initialize_default_middlewares()
        self.logger.debug(f"Initialized {self.__class__.__name__}")

    @property
    def client(self) -> AsyncOpenAI:
        return self.__client

    @property
    def verbose(self) -> bool:
        return self.__verbose

    @property
    def token_count(self) -> Counter:
        return self.__token_count

    def __initialize_default_middlewares(self):
        """
        Initializes the default middlewares.
        """
        self.__middlewares.append(self.__count_tokens)
        self.logger.debug("Initialized default middlewares")

    def add_middleware(self, middleware: Callable):
        """
        Adds a custom middleware to the agent.

        Args:
            middleware (Callable): The middleware to add.
        """
        self.__middlewares.append(middleware)
        self.logger.debug(f"Added middleware: {middleware.__name__}")

    async def __apply_middlewares(self, completion: CompletionT) -> CompletionT:
        """
        Applies the middlewares to the completion.

        Args:
            completion (CompletionT): The completion to apply the middlewares to.

        Returns:
            CompletionT: The completion with the middlewares applied.
        """
        self.logger.debug("Applying middlewares")
        for middleware in self.__middlewares:
            try:
                self.logger.debug(f"Applying middleware: {middleware.__name__}")
                if iscoroutinefunction(middleware):
                    completion = await middleware(completion)
                else:
                    completion = middleware(completion)
                self.logger.debug(f"Middleware {middleware.__name__} applied successfully")
            except Exception as e:
                self.log_exception(e, level=logging.ERROR)
        return completion

    def __count_tokens(self, completion: Completion):
        """
        Counts the tokens in the completion.

        Args:
            completion (Completion): The completion to count the tokens of.

        Returns:
            Completion: The completion with the token count.
        """
        try:
            if completion.usage:
                usage_dict = completion.usage.model_dump(exclude_none=True)
                completion_details = usage_dict.pop("completion_tokens_details", None)
                prompt_details = usage_dict.pop("prompt_tokens_details", None)

                self.__token_count.update(usage_dict)
                if completion_details:
                    self.__token_count.update(completion_details)
                if prompt_details:
                    self.__token_count.update(prompt_details)
                self.logger.debug(f"Updated token count: {self.__token_count}")
        except Exception as e:
            self.log_exception(e, level=logging.ERROR)
        return completion

    @retry(stop=stop_after_attempt(5), reraise=True)
    async def completion(
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
        stop: str | List[str] | NotGiven | None = NOT_GIVEN,
        store: bool | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_logprobs: int | NotGiven | None = NOT_GIVEN,
        top_p: float | NotGiven | None = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | NotGiven | None = NOT_GIVEN,
    ) -> Optional[ChatCompletionMessage]:
        """
        Gets a completion from the OpenAI API.

        Args:
            model (ChatModel): The model to use for completions.

        Returns:
            Optional[ChatCompletionMessage]: The completion message or None if an error occurred.
        """
        self.logger.debug(f"Starting completion with model={model}, messages_count={len(messages)}")
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
            completion = await self.__apply_middlewares(completion)
            if completion.choices:
                self.logger.debug("Returning first choice from completion")
                return completion.choices[0].message
            else:
                self.logger.warning("No choices returned in completion")
                return None
        except TimeoutException as te:
            self.log_exception(te, level=logging.ERROR)
            return None
        except httpx.RequestError as re:
            self.log_exception(re, level=logging.ERROR)
            return None
        except json.JSONDecodeError as je:
            self.log_exception(je, level=logging.ERROR)
            return None
        except Exception as e:
            self.log_exception(e, level=logging.ERROR)
            return None

    @retry(stop=stop_after_attempt(5), reraise=True)
    async def structured_completion(
        self,
        response_format: Type[T],
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
        stop: str | List[str] | NotGiven | None = NOT_GIVEN,
        store: bool | NotGiven | None = NOT_GIVEN,
        temperature: float | NotGiven | None = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_logprobs: int | NotGiven | None = NOT_GIVEN,
        top_p: float | NotGiven | None = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | NotGiven | None = NOT_GIVEN,
    ) -> Optional[ParsedChatCompletionMessage[T]]:
        """
        Gets a structured completion from the OpenAI API.

        Args:
            response_format (Type[T]): The expected response format.
            model (ChatModel): The model to use for completions.

        Returns:
            Optional[ParsedChatCompletionMessage[T]]: The parsed completion message or None if an error occurred.
        """
        self.logger.debug(f"Starting structured_completion with model={model}, response_format={response_format.__name__}, messages_count={len(messages)}")
        try:
            completion = await self.client.beta.chat.completions.parse(
                response_format=response_format,
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
            completion = await self.__apply_middlewares(completion)
            if completion.choices:
                self.logger.debug("Returning first choice from structured completion")
                return completion.choices[0].message
            else:
                self.logger.warning("No choices returned in structured completion")
                return None
        except TimeoutException as te:
            self.log_exception(te, level=logging.ERROR)
            return None
        except httpx.RequestError as re:
            self.log_exception(re, level=logging.ERROR)
            return None
        except json.JSONDecodeError as je:
            self.log_exception(je, level=logging.ERROR)
            return None
        except Exception as e:
            self.log_exception(e, level=logging.ERROR)
            return None

    def get_confidence_score(self, logprobs: ChoiceLogprobs):
        """
        Calculates the confidence score for a completion.

        Args:
            logprobs (ChoiceLogprobs): The log probabilities for the completion.

        Returns:
            float: The confidence score.
        """
        # Placeholder for confidence score calculation
        self.logger.debug("Calculating confidence score")
        raise NotImplementedError("Confidence score calculation is not implemented")
