from __future__ import annotations

"""
Wikipedia search operations with integrated logging and caching.

This module provides asynchronous methods for searching, retrieving pages,
and other Wikipedia API operations with built-in caching and rate limiting.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from decimal import Decimal
import functools
import logging
from typing import Any, Callable, List, Optional, Tuple, Union

from lumis.core.common.logger_mixin import LoggerMixin
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError

WikiPage = wikipedia.WikipediaPage


class WikipediaSearcher(LoggerMixin):
    """
    A class to handle Wikipedia search operations using the 'wikipedia' package with integrated logging.
    Provides asynchronous methods for searching, retrieving pages, and other Wikipedia API operations.
    """

    def __init__(
        self,
        user_agent: str = "",
        logger: Optional[logging.Logger] = None,
        rate_limit: bool = True,
        rate_limit_wait_ms: int = 50,
        max_workers: Optional[int] = None,
    ):
        """Initialize the WikipediaSearcher with custom settings."""
        super().__init__(logger=logger)
        self.logger.debug("WikipediaSearcher initialized.")

        self.user_agent = user_agent
        wikipedia.set_user_agent(self.user_agent)
        self.logger.debug(f"User-Agent set to: {self.user_agent}")

        wikipedia.set_rate_limiting(rate_limit, timedelta(milliseconds=rate_limit_wait_ms))
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def __del__(self):
        """Cleanup resources."""
        if self._executor:
            self._executor.shutdown(wait=False)

    async def _async_executor(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a synchronous function asynchronously in the thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, functools.partial(func, **kwargs), *args)

    @functools.lru_cache(maxsize=1000, typed=True)
    def _get_page_sync(self, title: str, lang: str = "en") -> Optional[WikiPage]:
        """
        Synchronous method to get a Wikipedia page with caching.

        The lru_cache decorator will cache up to 1000 most recently used pages.
        Different language versions of the same title are cached separately due to typed=True.
        """
        try:
            wikipedia.set_lang(lang)
            return wikipedia.page(title)
        except DisambiguationError as e:
            self.logger.warning(f"DisambiguationError for title '{title}': {e.options}")
        except PageError:
            self.logger.warning(f"PageError: The page '{title}' does not exist")
        except Exception as e:
            self.log_exception(e)
        return None

    async def _get_page(self, title: str, lang: str = "en") -> Optional[WikiPage]:
        """Get a page from cache or API with error handling."""
        return await self._async_executor(self._get_page_sync, title, lang=lang)

    async def search(self, query: str, num_results: int = 5, lang: str = "en") -> List[WikiPage]:
        """Search Wikipedia and retrieve page details concurrently."""
        wikipedia.set_lang(lang)
        self.logger.debug(f"Searching for: '{query}' in {lang}")

        try:
            search_results = await self._async_executor(wikipedia.search, query, num_results)
            if not search_results:
                self.logger.warning(f"No results found for: {query}")
                return []

            tasks = [self._get_page(title, lang) for title in search_results]
            results = await asyncio.gather(*tasks)
            return [r for r in results if r is not None]
        except Exception as e:
            self.log_exception(e)
            return []

    async def get_page_details(self, title: str, lang: str = "en") -> Optional[WikiPage]:
        """Get detailed information about a specific Wikipedia page."""
        return await self._get_page(title, lang)

    async def get_random_pages(self, num_pages: int = 1, lang: str = "en") -> List[WikiPage]:
        """Get random Wikipedia pages."""
        wikipedia.set_lang(lang)
        self.logger.debug(f"Fetching {num_pages} random pages")

        try:
            titles = await self._async_executor(wikipedia.random, num_pages)
            titles = [titles] if isinstance(titles, str) else titles
            tasks = [self._get_page(title, lang) for title in titles]
            results = await asyncio.gather(*tasks)
            return [r for r in results if r is not None]
        except Exception as e:
            self.log_exception(e)
            return []

    async def get_suggestion(self, query: str, lang: str = "en") -> Optional[str]:
        """Get a search suggestion for the query."""
        wikipedia.set_lang(lang)
        try:
            return await self._async_executor(wikipedia.suggest, query)
        except Exception as e:
            self.log_exception(e)
            return None

    async def geosearch(
        self,
        latitude: Union[float, Decimal],
        longitude: Union[float, Decimal],
        title: Optional[str] = None,
        radius: int = 1000,
        num_results: int = 10,
        lang: str = "en",
    ) -> List[WikiPage]:
        """Search for Wikipedia pages near specified coordinates."""
        wikipedia.set_lang(lang)
        self.logger.debug(f"Geosearch at ({latitude}, {longitude}), radius: {radius}m")

        try:
            titles = await self._async_executor(wikipedia.geosearch, latitude=latitude, longitude=longitude, title=title, results=num_results, radius=radius)
            if not titles:
                return []

            tasks = [self._get_page(title, lang) for title in titles]
            results = await asyncio.gather(*tasks)
            return [r for r in results if r is not None]
        except Exception as e:
            self.log_exception(e)
            return []

    async def search_with_suggestion(self, query: str, num_results: int = 5, lang: str = "en") -> Tuple[List[WikiPage], Optional[str]]:
        """Search with query suggestions."""
        wikipedia.set_lang(lang)
        self.logger.debug(f"Search with suggestion: '{query}'")

        try:
            results, suggestion = await self._async_executor(lambda: wikipedia.search(query, results=num_results, suggestion=True))
            if not results:
                return [], suggestion

            tasks = [self._get_page(title, lang) for title in results]
            pages = await asyncio.gather(*tasks)
            return [p for p in pages if p is not None], suggestion
        except Exception as e:
            self.log_exception(e)
            return [], None

    @functools.lru_cache(maxsize=1000, typed=True)
    def _get_summary_sync(self, title: str, sentences: int = 0, chars: int = 0, auto_suggest: bool = True, lang: str = "en") -> Optional[str]:
        """Synchronous method to get a page summary with caching."""
        try:
            wikipedia.set_lang(lang)
            return wikipedia.summary(title, sentences=sentences, chars=chars, auto_suggest=auto_suggest)
        except (PageError, DisambiguationError) as e:
            self.logger.warning(f"Error getting summary for '{title}': {str(e)}")
            return None
        except Exception as e:
            self.log_exception(e)
            return None

    async def get_summary(self, title: str, sentences: int = 0, chars: int = 0, auto_suggest: bool = True, lang: str = "en") -> Optional[str]:
        """Get a plain text summary of a Wikipedia page."""
        return await self._async_executor(self._get_summary_sync, title, sentences=sentences, chars=chars, auto_suggest=auto_suggest, lang=lang)
