"""Runtime middleware contract, including awaitable-returning callable objects."""

import asyncio
from collections import Counter
from collections.abc import Awaitable
from dataclasses import dataclass, replace
from unittest.mock import Mock
from typing import Generic, TypeVar

from google.genai.types import GenerateContentResponse, GenerateContentResponseUsageMetadata
from ollama import GenerateResponse

from lumis.llm.base_llm import BaseLLM
from lumis.llm.gemini_llm import Gemini
from lumis.llm.ollama_llm import OllamaLLM


T = TypeVar("T")


@dataclass
class Response:
    value: int


async def test_typed_sync_async_and_callable_middlewares_preserve_order():
    llm = BaseLLM()

    def sync(response: T) -> T:
        return replace(response, value=response.value + 1) if isinstance(response, Response) else response

    async def async_function(response: T) -> T:
        return replace(response, value=response.value * 2) if isinstance(response, Response) else response

    class AsyncObject:
        async def __call__(self, response: T) -> T:
            return replace(response, value=response.value + 3) if isinstance(response, Response) else response

    class SyncObject:
        def __call__(self, response: T) -> T:
            return replace(response, value=response.value * 4) if isinstance(response, Response) else response

    llm.add_middleware(sync)
    llm.add_middleware(async_function)
    llm.add_middleware(AsyncObject())
    llm.add_middleware(SyncObject())
    assert await llm._apply_middlewares(Response(2)) == Response(36)
    unrelated = object()
    assert await llm._apply_middlewares(unrelated) is unrelated

    @dataclass
    class DetailedResponse(Response):
        detail: str

    transformed = await llm._apply_middlewares(DetailedResponse(2, "keep"))
    assert transformed == DetailedResponse(36, "keep")


async def test_regular_function_returning_future_is_awaited():
    llm = BaseLLM()

    def future(response: T) -> Awaitable[T]:
        result: asyncio.Future[T] = asyncio.get_running_loop().create_future()
        result.set_result(replace(response, value=response.value + 1) if isinstance(response, Response) else response)
        return result

    llm.add_middleware(future)
    assert await llm._apply_middlewares(Response(2)) == Response(3)


async def test_custom_awaitable_from_regular_callable_object_is_awaited():
    class Deferred(Generic[T]):
        def __init__(self, response: T):
            self.response = response

        def __await__(self):
            async def resolve() -> T:
                return replace(self.response, value=self.response.value + 1) if isinstance(self.response, Response) else self.response

            return resolve().__await__()

    class Middleware:
        def __call__(self, response: T) -> Awaitable[T]:
            return Deferred(response)

    llm = BaseLLM()
    llm.add_middleware(Middleware())
    assert await llm._apply_middlewares(Response(2)) == Response(3)


async def test_awaited_middleware_failure_preserves_previous_response(caplog):
    llm = BaseLLM()

    async def fail(response: T) -> T:
        raise ValueError("middleware failed")

    def increment(response: T) -> T:
        return replace(response, value=response.value + 1) if isinstance(response, Response) else response

    llm.add_middleware(fail)
    llm.add_middleware(increment)
    assert await llm._apply_middlewares(Response(2)) == Response(3)
    assert "middleware failed" in caplog.text


async def test_gemini_response_type_and_token_counter_are_preserved():
    gemini = Gemini(client=Mock())
    response = GenerateContentResponse(usage_metadata=GenerateContentResponseUsageMetadata(prompt_token_count=2, total_token_count=5))
    assert await gemini._apply_middlewares(response) is response
    assert gemini.token_count == Counter({"prompt_token_count": 2, "total_token_count": 5})
    unrelated = Response(1)
    assert gemini._count_tokens(unrelated) is unrelated


async def test_ollama_response_type_and_token_counter_are_preserved():
    ollama = OllamaLLM(client=Mock())
    response = GenerateResponse(response="result", prompt_eval_count=2, eval_count=3)
    assert await ollama._apply_middlewares(response) is response
    assert ollama.token_count == Counter(prompt_tokens=2, completion_tokens=3, total_tokens=5)
    unrelated = Response(1)
    assert ollama._count_tokens(unrelated) is unrelated
