from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from typing import Literal, Optional, Sequence

from googlesearch import search as google_search
import redis.asyncio as aioredis
from tavily import TavilyClient, UsageLimitExceededError
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class SearchResult(TypedDict):
    """
    Represents the search result.

    Attributes:
        title (str): The title of the search result.
        url (str): The URL of the search result.
        description (str): The snippet or description of the search result.
        raw_content (Optional[str]): Raw content if available.
    """

    title: str
    url: str
    description: str
    raw_content: Optional[str]


SearchEngine = Literal["google", "tavily"]


class SearchEngineClient:
    def __init__(
        self,
        redis_client: Optional[aioredis.Redis] = None,
        cache_ttl: int = 3600,  # Time-to-live for cache entries in seconds
        tavily_client: Optional[TavilyClient] = None,
        max_workers: int = 10,
    ):
        """
        Initializes the SearchEngineClient.

        Args:
            redis_client (Optional[aioredis.Redis], optional): External Redis client for caching.
            cache_ttl (int, optional): Cache expiration time in seconds.
            tavily_client (Optional[TavilyClient], optional): Optional TavilyClient instance.
            max_workers (int, optional): Maximum number of worker threads for the executor.
        """
        self.tavily = tavily_client if tavily_client is not None else TavilyClient()
        self.redis = redis_client
        self.cache_ttl = cache_ttl
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def search(  # noqa: C901
        self,
        query: str,
        topic: Literal["general", "news"] = "general",
        max_results: int = 10,
        search_engine: SearchEngine = "tavily",
    ) -> list[SearchResult]:
        """
        Performs a search using the specified search engine. If the selected search engine
        fails, it falls back to Tavily.

        Args:
            query (str): The search query.
            topic (Literal["general", "news"], optional): The topic for Tavily search.
            max_results (int, optional): The maximum number of search results.
            search_engine (SearchEngine, optional): The search engine to use.

        Returns:
            list[SearchResult]: The search results.
        """
        cache_key = self._generate_cache_key(query, search_engine, topic)
        if self.redis:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for key: {cache_key}")
                try:
                    search_results = json.loads(cached_data)
                    return search_results
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode cache data for key: {cache_key}")

        try:
            if search_engine == "google":
                results = await self.search_google(query, max_results)
                if not results:
                    raise ValueError("Google search returned None. Falling back to Tavily.")
            # elif search_engine == "duckduckgo":
            #     results = await self.search_duckduckgo(query, max_results)
            elif search_engine == "tavily":
                results = await self.search_tavily(query, max_results=max_results, topic=topic)
            else:
                logger.error(f"Unsupported search engine: {search_engine}")
                raise ValueError(f"Unsupported search engine: {search_engine}")
        except Exception as e:
            logger.exception(f"Error using {search_engine}. Falling back to Tavily. Error: {e}")
            if search_engine != "tavily":
                try:
                    results = await self.search_tavily(query, max_results=max_results, topic=topic)
                except Exception as fallback_e:
                    logger.exception(f"Fallback to Tavily also failed. Error: {fallback_e}")
                    raise fallback_e
            else:
                raise e

        if self.redis:
            try:
                serialized = json.dumps(results)
                await self.redis.set(cache_key, serialized, ex=self.cache_ttl)
                logger.debug(f"Cached results for key: {cache_key} with TTL: {self.cache_ttl}s")
            except Exception as cache_e:
                logger.warning(f"Failed to cache results for key: {cache_key}. Error: {cache_e}")

        return results

    def _generate_cache_key(self, query: str, search_engine: SearchEngine, topic: str) -> str:
        """
        Generates a unique cache key based on the search parameters.

        Args:
            query (str): The search query.
            search_engine (SearchEngine): The search engine used.
            topic (str): The topic for Tavily search.

        Returns:
            str: The generated cache key.
        """
        # Normalize the query to ensure consistent cache keys
        normalized_query = query.strip().lower()
        return f"search:{search_engine}:{topic}:{normalized_query}"

    async def search_google(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """
        Asynchronously searches Google for the query and returns the search results.

        Args:
            query (str): The search query.
            max_results (int, optional): The maximum number of search results to return.

        Returns:
            list[SearchResult]: The search results.
        """
        loop = asyncio.get_event_loop()
        try:
            logger.debug(f"Initiating Google search for query: {query}")
            results = await loop.run_in_executor(
                self.executor,
                lambda: google_search(query, num_results=max_results, advanced=True),
            )
            search_results = [
                SearchResult(
                    title=getattr(result, "title", ""),
                    url=getattr(result, "url", ""),
                    description=getattr(result, "description", "") or "",
                    raw_content=None,
                )
                for result in results
            ]
            logger.debug(f"Google search returned {len(search_results)} results for query: {query}")
            return search_results
        except Exception as e:
            logger.exception(f"Google search failed for query: {query}")
            raise e

    async def search_duckduckgo(self, query: str, max_results: int = 10):
        pass

    async def search_news(
        self,
        query: str,
        search_depth: Literal["basic", "advanced"] = "basic",
        topic: Literal["general", "news"] = "general",
        days: int = 3,
        max_results: int = 5,
        include_domains: Sequence[str] = [],
        exclude_domains: Sequence[str] = [],
        include_answer: bool = False,
        include_raw_content: bool = True,
        include_images: bool = False,
    ):
        return self.search_tavily(
            query=query,
            search_depth=search_depth,
            topic=topic,
            days=days,
            max_results=max_results,
            include_domains=include_domains or [],
            exclude_domains=exclude_domains or [],
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_images=include_images,
        )

    async def search_tavily(
        self,
        query: str,
        search_depth: Literal["basic", "advanced"] = "basic",
        topic: Literal["general", "news"] = "general",
        days: int = 3,
        max_results: int = 5,
        include_domains: Sequence[str] = [],
        exclude_domains: Sequence[str] = [],
        include_answer: bool = False,
        include_raw_content: bool = True,
        include_images: bool = False,
        **kwargs,
    ) -> list[SearchResult]:
        """
        Asynchronously searches Tavily for the query and returns the search results.

        Args:
            query (str): The search query.
            max_results (int, optional): The maximum number of search results to return.
            topic (Literal["general", "news"], optional): The topic for the search.

        Returns:
            list[SearchResult]: The search results.
        """

        loop = asyncio.get_event_loop()
        try:
            logger.debug(f"Initiating Tavily search for query: {query} with topic: {topic}")
            results = await loop.run_in_executor(
                self.executor,
                lambda: self.tavily.search(
                    query=query,
                    # search_depth=search_depth,
                    topic=topic,
                    # days=days,
                    max_results=max_results,
                    # include_domains=include_domains,
                    # exclude_domains=exclude_domains,
                    # include_answer=include_answer,
                    # include_raw_content=include_raw_content,
                    # include_images=include_images,
                ),
            )
            search_results = [
                SearchResult(title=str(result.get("title", "")), url=str(result.get("url", "")), description=str(result.get("description", "")), raw_content=str(result.get("raw_content")))
                for result in results.get("results", [])
            ]
            logger.debug(f"Tavily search returned {len(search_results)} results for query: {query}")
            return search_results
        except UsageLimitExceededError as e:
            logger.critical(f"Tavily usage limit exceeded. Error: {e}")
            raise e
        except Exception as e:
            logger.exception(f"Tavily search failed for query: {query}")
            raise e

    async def clear_cache(  # noqa: C901
        self,
        query: Optional[str] = None,
        search_engine: Optional[SearchEngine] = None,
        topic: Optional[str] = None,
    ):
        """
        Clears specific cache entries or the entire cache.

        Args:
            query (Optional[str], optional): The search query to clear from cache.
            search_engine (Optional[SearchEngine], optional): The search engine to clear from cache.
            topic (Optional[str], optional): The topic to clear from cache.
        """
        if not self.redis:
            logger.warning("Redis is not provided. Cannot clear cache.")
            return

        if query and search_engine and topic:
            cache_key = self._generate_cache_key(query, search_engine, topic)
            await self.redis.delete(cache_key)
            logger.debug(f"Cleared cache for key: {cache_key}")
        else:
            # Clear all keys matching the search cache pattern
            pattern = "search:*"
            keys = []
            try:
                async for key in self.redis.scan_iter(match=pattern):
                    keys.append(key)
                if keys:
                    await self.redis.delete(*keys)
                    logger.debug(f"Cleared {len(keys)} cache entries.")
                else:
                    logger.debug("No cache entries to clear.")
            except Exception as e:
                logger.exception(f"Failed to clear cache. Error: {e}")

    async def shutdown(self):
        """
        Shuts down the thread pool executor.

        Note:
            Since the Redis client is managed externally, it is not closed here.
        """
        self.executor.shutdown(wait=True)
        logger.debug("Thread pool executor shut down.")
