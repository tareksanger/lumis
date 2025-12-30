from __future__ import annotations

from abc import ABC
from collections import Counter
from inspect import iscoroutinefunction
import logging
from typing import Callable, Optional, TypeVar

from lumis.core.common.logger_mixin import LoggerMixin

T = TypeVar("T")


class BaseLLM(LoggerMixin, ABC):
    def __init__(self, verbose: bool = False, logger: Optional[logging.Logger] = None):
        super().__init__(logger=logger)

        self._token_count = Counter()
        self._verbose = verbose
        self._middlewares: list[Callable] = []
        self._initialize_default_middlewares()
        self.logger.debug(f"Initialized {self.__class__.__name__}")

    @property
    def token_count(self) -> Counter:
        return self._token_count

    @property
    def verbose(self) -> bool:
        return self._verbose

    def _initialize_default_middlewares(self):
        """
        Initializes the default middlewares.
        """
        self._middlewares.append(self._count_tokens)
        self.logger.debug("Initialized default middlewares")

    def add_middleware(self, middleware: Callable[[T], T]):
        """
        Adds a custom middleware to the agent.

        Args:
            middleware (Callable): The middleware to add.
        """
        self._middlewares.append(middleware)
        self.logger.debug(f"Added middleware: {middleware.__name__}")

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
                if iscoroutinefunction(middleware):
                    response = await middleware(response)
                else:
                    response = middleware(response)
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
