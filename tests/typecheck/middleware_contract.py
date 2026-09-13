# pyright: reportUnnecessaryTypeIgnoreComment=true
"""Static middleware examples checked by Pyright; never calls provider APIs."""

from collections import Counter
from collections.abc import Awaitable
from typing import TypeVar, assert_type

from lumis.llm.base_llm import BaseLLM
from lumis.llm.gemini_llm import Gemini
from lumis.llm.ollama_llm import OllamaLLM

from google.genai.types import GenerateContentResponse, GenerateImagesResponse
from ollama import ChatResponse, GenerateResponse


T = TypeVar("T")


def sync_middleware(response: T) -> T:
    return response


async def async_middleware(response: T) -> T:
    return response


class SyncMiddleware:
    def __call__(self, response: T) -> T:
        return response


class AsyncMiddleware:
    async def __call__(self, response: T) -> T:
        return response


def awaitable_middleware(response: T) -> Awaitable[T]:
    return async_middleware(response)


class AwaitableMiddleware:
    def __call__(self, response: T) -> Awaitable[T]:
        return async_middleware(response)


async def check_contract(base: BaseLLM, gemini: Gemini, ollama: OllamaLLM, content: GenerateContentResponse, images: GenerateImagesResponse, chat: ChatResponse, generated: GenerateResponse) -> None:
    base.add_middleware(sync_middleware)
    base.add_middleware(async_middleware)
    base.add_middleware(SyncMiddleware())
    base.add_middleware(AsyncMiddleware())
    base.add_middleware(awaitable_middleware)
    base.add_middleware(AwaitableMiddleware())
    assert_type(await base._apply_middlewares(content), GenerateContentResponse)
    assert_type(await gemini._apply_middlewares(content), GenerateContentResponse)
    assert_type(await gemini._apply_middlewares(images), GenerateImagesResponse)
    assert_type(await ollama._apply_middlewares(chat), ChatResponse)
    assert_type(await ollama._apply_middlewares(generated), GenerateResponse)
    assert_type(gemini._count_tokens(content), GenerateContentResponse)
    assert_type(gemini._count_tokens(images), GenerateImagesResponse)
    assert_type(ollama._count_tokens(chat), ChatResponse)
    assert_type(ollama._count_tokens(generated), GenerateResponse)
    assert_type(base.token_count, Counter[str])
    assert_type(gemini.token_count, Counter[str])
    assert_type(ollama.token_count, Counter[str])
    assert_type(await gemini.generate_content(use_search=False), GenerateContentResponse)
    assert_type(await gemini.generate_images("prompt"), GenerateImagesResponse)
    assert_type(await ollama.response("prompt"), GenerateResponse)


def response_specific(response: GenerateContentResponse) -> GenerateContentResponse:
    return response


def type_changing(response: object) -> str:
    return str(response)


def rejected_registrations(base: BaseLLM) -> None:
    # Each suppression is required: removing the generic protocol would make
    # these invalid registrations pass and Pyright flag the unused suppression.
    base.add_middleware(response_specific)  # pyright: ignore[reportArgumentType]
    base.add_middleware(type_changing)  # pyright: ignore[reportArgumentType]
