from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
import logging
import os
from typing import Any, Callable, Literal, Optional, Type, TypeVar

from lumis.llm.base_llm import BaseLLM

from ollama import AsyncClient, ChatResponse, GenerateResponse, Image, Message, Options, Tool
from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaValue
from tenacity import retry, stop_after_attempt, wait_exponential

T = TypeVar("T", bound=BaseModel)


DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


Completion = ChatResponse | GenerateResponse


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        client: Optional[AsyncClient] = None,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.__client = client or AsyncClient()

    @classmethod
    def from_client(
        cls,
        client: AsyncClient,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> "OllamaLLM":
        return cls(client=client, verbose=verbose, logger=logger)

    @classmethod
    def from_host(
        cls,
        base_url: str,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> "OllamaLLM":
        return cls(client=AsyncClient(host=base_url), verbose=verbose, logger=logger)

    @property
    def client(self) -> AsyncClient:
        return self.__client

    def _count_tokens(self, response: Completion) -> Completion:
        try:
            prompt_tokens = response.get("prompt_eval_count")
            completion_tokens = response.get("eval_count")
            counts: dict[str, int] = {}

            if isinstance(prompt_tokens, int):
                counts["prompt_tokens"] = prompt_tokens
            if isinstance(completion_tokens, int):
                counts["completion_tokens"] = completion_tokens
            if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                counts["total_tokens"] = prompt_tokens + completion_tokens

            if counts:
                self._token_count.update(counts)
                self.logger.debug(f"Updated Ollama token counts: {counts}")
        except Exception as e:
            self.log_exception(e, level=logging.ERROR)
        return response

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        reraise=True,
    )
    async def completion(
        self,
        messages: Optional[Sequence[Message]] = None,
        model: Optional[str] = None,
        *,
        tools: Optional[Sequence[Tool | Mapping[str, Any] | Callable[..., Any]]] = None,
        think: bool | Literal["low", "medium", "high"] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        format: JsonSchemaValue | Literal["", "json"] | None = None,
        options: Options | Mapping[str, Any] | None = None,
        keep_alive: float | str | None = None,
    ) -> Message:
        message_payload = list(messages or [])
        target_model = model or DEFAULT_MODEL
        self.logger.debug(f"Starting Ollama completion with model={target_model}, messages_count={len(message_payload)}")

        try:
            completion: ChatResponse = await self.client.chat(
                model=target_model,
                messages=message_payload,
                tools=tools,
                stream=False,
                think=think,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                format=format,
                options=options,
                keep_alive=keep_alive,
            )
            completion = await self._apply_middlewares(completion)
            if not completion.message:
                raise ValueError("No message returned in completion")
            return completion.message
        except Exception as e:
            self.log_exception(e, level=logging.ERROR)
            raise

    # @retry(
    #     stop=stop_after_attempt(5),
    #     wait=wait_exponential(multiplier=1, min=4, max=20),
    #     reraise=True,
    # )
    async def structured_response(
        self,
        prompt: str,
        format: Type[T],
        model: Optional[str] = None,
        *,
        suffix: Optional[str] = None,
        system: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[Sequence[int]] = None,
        think: bool | Literal["low", "medium", "high"] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        raw: bool = False,
        images: Optional[Sequence[str | bytes | Image]] = None,
        options: Options | Mapping[str, Any] | None = None,
        keep_alive: float | str | None = None,
    ) -> T:
        if not model:
            model = DEFAULT_MODEL

        try:
            response = await self.client.generate(
                model=model,
                prompt=prompt,
                suffix=suffix or "",
                system=system or "",
                template=template or "",
                context=context,
                stream=False,
                think=think,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                raw=raw,
                format=format.model_json_schema(),
                images=images,
                options=options,
                keep_alive=keep_alive,
            )

            response = await self._apply_middlewares(response)
            if not getattr(response, "response", None):
                raise ValueError("No response text returned from Ollama")

            return format.model_validate_json(response.response)

        except Exception as e:
            self.log_exception(e, level=logging.ERROR)
            raise

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        reraise=True,
    )
    async def response(
        self,
        prompt: str,
        model: Optional[str] = None,
        *,
        suffix: Optional[str] = None,
        system: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[Sequence[int]] = None,
        think: bool | Literal["low", "medium", "high"] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        raw: bool = False,
        images: Optional[Sequence[str | bytes | Image]] = None,
        options: Options | None = None,
        keep_alive: float | str | None = None,
    ) -> GenerateResponse:
        if not model:
            model = DEFAULT_MODEL

        self.logger.debug(f"Starting Ollama response with model={model}")

        try:
            generate_response = await self.client.generate(
                model=model,
                prompt=prompt,
                suffix=suffix or "",
                system=system or "",
                template=template or "",
                context=context,
                stream=False,
                think=think,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                raw=raw,
                images=images,
                options=options,
                keep_alive=keep_alive,
            )
            generate_response = await self._apply_middlewares(generate_response)
            if not getattr(generate_response, "response", None):
                raise ValueError("No response text returned from Ollama")
            return generate_response
        except Exception as e:
            self.log_exception(e, level=logging.ERROR)
            raise

    # @retry(
    #     stop=stop_after_attempt(5),
    #     wait=wait_exponential(multiplier=1, min=4, max=20),
    #     reraise=True,
    # )
    # async def stream_response(
    #     self,
    #     prompt: str,
    #     model: Optional[str] = None,
    #     *,
    #     suffix: Optional[str] = None,
    #     system: Optional[str] = None,
    #     template: Optional[str] = None,
    #     context: Optional[Sequence[int]] = None,
    #     think: bool | Literal["low", "medium", "high"] | None = None,
    #     logprobs: bool | None = None,
    #     top_logprobs: int | None = None,
    #     raw: bool = False,
    #     format: JsonSchemaValue | Literal["", "json"] | None = None,
    #     images: Optional[Sequence[str | bytes | Image]] = None,
    #     options: Options | None = None,
    #     keep_alive: float | str | None = None,
    # ) -> AsyncIterator[GenerateResponse]:
    #     if not model:
    #         model = DEFAULT_MODEL
    #     self.logger.debug(f"Starting streaming Ollama response with model={model}")

    #     async def stream_manager() -> AsyncIterator[tuple[str, Literal["delta", "done"]]]:
    #         async for chunk in await self.client.generate(
    #             model=model,
    #             prompt=prompt,
    #             suffix=suffix or "",
    #             system=system or "",
    #             template=template or "",
    #             context=context,
    #             stream=True,
    #             think=think,
    #             logprobs=logprobs,
    #             top_logprobs=top_logprobs,
    #             raw=raw,
    #             format=format,
    #             images=images,
    #             options=options,
    #             keep_alive=keep_alive,
    #         ):
    #             if chunk.get("done"):
    #                 await self._apply_middlewares(chunk)
    #             yield chunk

    #     return stream_manager()

    # TODO: Fix streaming
    # @retry(
    #     stop=stop_after_attempt(5),
    #     wait=wait_exponential(multiplier=1, min=4, max=20),
    #     reraise=True,
    # )
    # async def structured_stream(
    #     self,
    #     prompt: str,
    #     format: Type[T],
    #     model: Optional[str] = None,
    #     *,
    #     suffix: Optional[str] = None,
    #     system: Optional[str] = None,
    #     template: Optional[str] = None,
    #     context: Optional[Sequence[int]] = None,
    #     think: bool | Literal["low", "medium", "high"] | None = None,
    #     logprobs: bool | None = None,
    #     top_logprobs: int | None = None,
    #     raw: bool = False,
    #     images: Optional[Sequence[str | bytes | Image]] = None,
    #     options: Options | Mapping[str, Any] | None = None,
    #     keep_alive: float | str | None = None,
    # ) -> AsyncIterator[T]:
    #     target_model = model or self.__model
    #     self.logger.debug(f"Starting streaming structured Ollama response with model={target_model}")

    #     async def stream_manager() -> AsyncIterator[T]:
    #         async for chunk in await self.client.generate(
    #             model=target_model,
    #             prompt=prompt,
    #             suffix=suffix,
    #             system=system,
    #             template=template,
    #             context=context,
    #             stream=True,
    #             think=think,
    #             logprobs=logprobs,
    #             top_logprobs=top_logprobs,
    #             raw=raw,
    #             format=format.model_json_schema(),
    #             images=images,
    #             options=options,
    #             keep_alive=keep_alive,
    #         ):
    #             if chunk.get("done"):
    #                 await self._apply_middlewares(chunk)

    #             # self.logger.info(f"Streaming structured chunk: {chunk}")
    #             yield chunk.response

    #     return stream_manager()
