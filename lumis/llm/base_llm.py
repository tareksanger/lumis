from __future__ import annotations

from abc import ABC
from collections import Counter
from inspect import isawaitable
import logging
from typing import Awaitable, Optional, Protocol, TypeVar

from lumis.core.common.logger_mixin import LoggerMixin

T = TypeVar("T")


class ResponseMiddleware(Protocol):
    """A callback preserving every input response type, synchronously or asynchronously."""

    def __call__(self, response: T, /) -> T | Awaitable[T]: ...


class BaseLLM(LoggerMixin, ABC):
    def __init__(self, verbose: bool = False, logger: Optional[logging.Logger] = None):
        super().__init__(logger=logger)

        self._token_count: Counter[str] = Counter()
        self._verbose = verbose
        self._middlewares: list[ResponseMiddleware] = []
        self._initialize_default_middlewares()
        self.logger.debug(f"Initialized {self.__class__.__name__}")

    @property
    def token_count(self) -> Counter[str]:
        return self._token_count

    @property
    def verbose(self) -> bool:
        return self._verbose

    def _initialize_default_middlewares(self) -> None:
        """
        Initializes the default middlewares.
        """
        self._middlewares.append(self._count_tokens)
        self.logger.debug("Initialized default middlewares")

    def add_middleware(self, middleware: ResponseMiddleware) -> None:
        """
        Add a callback that preserves the concrete type of every response.

        Typed callbacks must be generic in their response type. Guard access
        to provider-specific fields and return unsupported response types
        unchanged; a single instance can process several SDK response types.
        The callback may return its response directly or through an awaitable.
        """
        self._middlewares.append(middleware)
        self.logger.debug(f"Added middleware: {getattr(middleware, '__name__', type(middleware).__name__)}")

    async def _apply_middlewares(self, response: T) -> T:
        """
        Applies the middlewares to the response.

        Args:
            response (T): The response to apply the middlewares to.

        Returns:
            T: The response with the middlewares applied.
        """
        self.logger.debug(f"Applying middlewares from {self.__class__.__name__}")
        for middleware in self._middlewares:
            try:
                result = middleware(response)
                if isawaitable(result):
                    response = await result
                else:
                    response = result
            except Exception as e:
                self.log_exception(e, level=logging.ERROR)
        self.logger.debug(f"Finished applying middlewares from {self.__class__.__name__}")
        return response

    def _count_tokens(self, response: T) -> T:
        """
        Base implementation for counting tokens.
        Should be overridden by subclasses with actual implementation.
        """
        return response
