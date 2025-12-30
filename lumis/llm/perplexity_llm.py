from __future__ import annotations

import json
import logging
import os
from typing import cast, Generic, Iterable, List, Literal, Optional, Type, TypeVar

import httpx
from httpx import TimeoutException
from lumis.llm.openai_llm import OpenAILLM
from openai import AsyncClient, NOT_GIVEN, NotGiven
from openai._types import Body, Headers, Query
from openai.types import ChatModel
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    ParsedChatCompletion,
)
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

T = TypeVar("T", bound=BaseModel)

Completion = ChatCompletion | ParsedChatCompletion
CompletionT = TypeVar("CompletionT", bound=Completion)


CHAT_MODEL: ChatModel = cast(ChatModel, os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
PERPLEXITY_API_KEY: str = cast(str, os.getenv("PERPLEXITYAI_API_KEY"))
PERPLEXITY_BASE_URL: str = cast(str, os.getenv("PERPLEXITYAI_BASE_URL"))


class Response(BaseModel):
    content: str = Field(..., description="The response content")
    confidence: float = Field(..., description="The confidence of the response")


class StructuredResponse(BaseModel, Generic[T]):
    content: Type[T]
    confidence: float = Field(..., description="The confidence of the response")


class ResearchChatCompletionMessage(ChatCompletionMessage):
    citations: Optional[list[str]]


class PerplexityLLM(OpenAILLM):
    def __init__(
        self,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
        # @depreciated
        client: Optional[AsyncClient] = None,
    ):
        client = AsyncClient(api_key=PERPLEXITY_API_KEY, base_url=PERPLEXITY_BASE_URL)
        super().__init__(client, verbose=verbose, logger=logger)

        if client is not None:
            self.logger.debug("Passing client to PerplexityLLM does nothing and is depreciated")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=20), reraise=True)
    async def completion(  # noqa: C901
        self,
        model: ChatModel = CHAT_MODEL,
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
    ):
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
            completion = await self._apply_middlewares(completion)

            if completion.choices:
                first_choice_message = completion.choices[0].message

                # Check if the citations property exists and is not null/undefined
                citations = getattr(completion, "citations", None)

                # Recreate the message as a ChatCompletionMessage object with citations
                message_with_citations = ResearchChatCompletionMessage(
                    content=first_choice_message.content,
                    refusal=first_choice_message.refusal,
                    role=first_choice_message.role,
                    audio=first_choice_message.audio,
                    function_call=first_choice_message.function_call,
                    tool_calls=first_choice_message.tool_calls,
                    citations=citations if citations else None,
                )

                self.logger.debug("Returning first choice from completion with citations")
                return message_with_citations
            else:
                self.logger.warning("No choices returned in completion")
                raise ValueError("No choices returned in completion")

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
