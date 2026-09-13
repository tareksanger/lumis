"""Static-only public OpenAI adapter contracts, checked by Pyright, never executed."""

from typing import assert_type

from lumis.llm.openai_llm import OpenAILLM

import httpx
from openai import Omit, omit, Timeout
from openai.types import Reasoning
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
from openai.types.chat.parsed_chat_completion import ParsedChatCompletionMessage
from pydantic import BaseModel


class Decision(BaseModel):
    next_step: str


async def check_completion_contract(
    llm: OpenAILLM,
    messages: list[ChatCompletionMessageParam],
    count: int,
    optional_count: int | None | Omit,
) -> None:
    assert_type(await llm.completion(messages=messages), ChatCompletionMessage)
    assert_type(await llm.completion(messages=messages, n=1), ChatCompletionMessage)
    assert_type(await llm.completion(messages=messages, n=None), ChatCompletionMessage)
    assert_type(await llm.completion(messages=messages, n=omit), ChatCompletionMessage)
    assert_type(await llm.completion(messages=messages, n=2), list[ChatCompletionMessage])
    assert_type(await llm.completion(messages=messages, n=10), list[ChatCompletionMessage])
    assert_type(await llm.completion(messages=messages, n=count), ChatCompletionMessage | list[ChatCompletionMessage])
    assert_type(await llm.completion(messages=messages, n=optional_count), ChatCompletionMessage | list[ChatCompletionMessage])
    assert_type(await llm.completion(None, messages, 1), ChatCompletionMessage)


async def check_structured_contract(
    llm: OpenAILLM,
    messages: list[ChatCompletionMessageParam],
    count: int,
    optional_count: int | None | Omit,
) -> None:
    result = await llm.structured_completion(response_format=Decision, messages=messages)
    assert_type(result, ParsedChatCompletionMessage[Decision])
    assert_type(result.parsed, Decision | None)
    if result.parsed is not None:
        assert_type(result.parsed.next_step, str)
    assert_type(await llm.structured_completion(Decision, messages=messages, n=1), ParsedChatCompletionMessage[Decision])
    assert_type(await llm.structured_completion(Decision, messages=messages, n=None), ParsedChatCompletionMessage[Decision])
    assert_type(await llm.structured_completion(Decision, messages=messages, n=omit), ParsedChatCompletionMessage[Decision])
    assert_type(await llm.structured_completion(Decision, messages=messages, n=2), list[ParsedChatCompletionMessage[Decision]])
    assert_type(await llm.structured_completion(Decision, messages=messages, n=10), list[ParsedChatCompletionMessage[Decision]])
    assert_type(await llm.structured_completion(Decision, messages=messages, n=count), ParsedChatCompletionMessage[Decision] | list[ParsedChatCompletionMessage[Decision]])
    assert_type(await llm.structured_completion(Decision, messages=messages, n=optional_count), ParsedChatCompletionMessage[Decision] | list[ParsedChatCompletionMessage[Decision]])
    assert_type(await llm.structured_completion(Decision, None, messages, 1), ParsedChatCompletionMessage[Decision])
    assert_type(await llm.structured_completion(messages=messages), ParsedChatCompletionMessage[None])
    assert_type(await llm.structured_completion(messages=messages, n=2), list[ParsedChatCompletionMessage[None]])
    assert_type(await llm.structured_completion(messages=messages, n=count), ParsedChatCompletionMessage[None] | list[ParsedChatCompletionMessage[None]])


async def check_sdk_compatibility(llm: OpenAILLM) -> None:
    await llm.response(input="Question", reasoning={"effort": "low"})
    await llm.response(input="Question", reasoning=Reasoning(effort="low"))
    await llm.response(input="Question", prompt_cache_retention="in_memory")
    await llm.response(input="Question", prompt_cache_retention="in-memory")
    await llm.generate_image(prompt="Scene", timeout=httpx.Timeout(2))
    await llm.generate_image(prompt="Scene", timeout=Timeout(2))
