from __future__ import annotations

import logging
from typing import Literal, Optional

from ..scraper import WebScrapper
from .search_engine_client import SearchEngine, SearchEngineClient

from lumis.core.document import Chunk, Document
from lumis.embedding import BaseEmbeddingModel
from lumis.nlp import SemanticParser, VectorSimilarityRetriever
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class SearchContent(TypedDict):
    """
    Represents the cached content for a URL.

    Attributes:
        document (Document): The fetched and processed document.
        chunks (list[Chunk]): The chunks parsed from that document.
    """

    document: Document
    chunks: list[Chunk]


class VectorSearchRetrievalEngine:
    """
    A retrieval engine that performs semantic search over documents and their associated text chunks.
    Uses a cache to store previously processed documents to avoid redundant scraping/parsing.

    The retrieval process:
    1. Use the Tavily client to search for related URLs given a query.
    2. Check the cache for previously parsed documents.
        - If cached, retrieve chunks directly.
        - If not cached, scrape the URL and parse the document into chunks, then cache it.
    3. Use vector similarity to select the top k most relevant chunks for the given query.
    """

    def __init__(
        self,
        embedding: BaseEmbeddingModel,
        max_results: int = 5,
        k: int = 5,
        search_engine_client: Optional[SearchEngineClient] = None,
        web_scrapper: Optional[WebScrapper] = None,
        semantic_parser: Optional[SemanticParser] = None,
        similarity_retriever: Optional[VectorSimilarityRetriever] = None,
        # depreciated
        max_cache_size: int = 100,
        use_cache: bool = False,
    ):
        """
        Initialize the VectorSearchRetrievalEngine.

        Args:
            embedding (BaseEmbeddingModel): Embedding model used for semantic analysis.
            max_results (int): Default maximum number of Tavily results to fetch.
            k (int): Default number of top relevant chunks to return.
            max_cache_size (int): Maximum number of entries in the cache. Older entries are evicted.
        """
        self.search_engine_client = search_engine_client if search_engine_client is not None else SearchEngineClient()
        self.scraper = web_scrapper if web_scrapper is not None else WebScrapper(gb_client=gb_client)

        self.parser = semantic_parser if semantic_parser is not None else SemanticParser(embedding_model=embedding, breakpoint_percentile_threshold=75)
        self.retriever = similarity_retriever if similarity_retriever is not None else VectorSimilarityRetriever(embedding_model=embedding)

        self.default_max_results = max_results
        self.default_k = k

    async def search(
        self,
        query: str,
        topic: Literal["general", "news"] = "general",
        search_engine: SearchEngine = "google",
        max_results: Optional[int] = None,
        k: Optional[int] = None,
    ) -> list[Chunk]:
        """
        Perform a semantic search on the provided query and return the top `k` most relevant chunks.

        Args:
            query (str): The user query.
            topic (Literal["general", "news"]): Topic category, defaults to "general".
            max_results (int, optional): Override the default max results.
            k (int, optional): Override the default number of relevant chunks to return.

        Returns:
            list[Chunk]: The top `k` most relevant chunks.
        """

        max_results = max_results if max_results is not None else self.default_max_results
        k = k if k is not None else self.default_k

        logger.debug(f"Starting search for query='{query}', topic='{topic}', max_results={max_results}, k={k}")

        # Fetch search results with error handling
        try:
            search_results = await self.search_engine_client.search(query, topic=topic, max_results=max_results)
        except Exception as e:
            logger.error(f"Failed to retrieve search results from search engine: {e}")
            return []

        if len(search_results) == 0:
            logger.debug("No search results found.")
            return []

        data = [self.__extract_url_and_metadata(r) for r in search_results]
        urls, metadatas = zip(*data)
        urls, metadatas = list(urls), list(metadatas)

        logger.debug(f"Found {len(urls)} URLs from search engine.")

        # Gather chunks from cache or by scraping new content
        all_chunks = await self._collect_chunks(urls, metadatas)

        if not all_chunks:
            logger.debug("No chunks collected, returning empty list.")
            return []

        # Use vector similarity to find top k chunks
        logger.debug(f"Extracting top {k} relevant chunks from {len(all_chunks)} collected chunks.")
        relevant_k_chunks = await self.retriever.aextract_relevant_chunks(query, all_chunks, k)
        if not relevant_k_chunks:
            logger.debug("No relevant chunks found based on vector similarity.")
            return []

        _, selected_chunks = zip(*relevant_k_chunks)
        return list(selected_chunks)

    def __extract_url_and_metadata(self, result: dict) -> tuple[str, dict]:
        url = result["url"]
        return (url, result)

    async def _collect_chunks(self, urls: list[str], metadatas: list[dict]) -> list[Chunk]:
        """
        Retrieve and parse chunks for the given URLs. Leverage the cache to avoid reprocessing.

        Args:
            urls (list[str]): URLs to process.
            metadatas (list[dict]): Metadata for each URL.

        Returns:
            list[Chunk]: All combined chunks from cached and newly processed documents.
        """
        chunks = []
        docs = await self.scraper.batch_fetch_content(urls, metadatas)

        if docs:
            logger.debug(f"Fetching and parsing {len(docs)} new documents.")
        else:
            logger.debug("No new documents to parse; all were from cache.")

        for doc in docs:
            url = (doc.metadata or {}).get("url", "")
            if not url.strip():
                logger.debug("Skipping document with no valid URL in metadata.")
                continue

            try:
                # Parse the document into chunks
                new_chunks = await self.parser.aparse_document(doc)
                # Cache the new chunks
                # if self.use_cache:
                #     self.cache.set_content(url, {"document": doc, "chunks": chunks})
                chunks.extend(new_chunks)
            except Exception as e:
                logger.error(f"Error parsing document for URL='{url}': {e}")

        return chunks

    async def _split_new_chunks_from_old(self, urls: list[str], metadatas: list[dict]) -> tuple[list[Document], list[Chunk]]:
        cached_chunks: list[Chunk] = []
        new_urls: list[str] = []
        new_metadata: list[dict] = []

        for i, url in enumerate(urls):
            cached = self.cache.get_content(url)
            if cached is not None:
                logger.debug(f"Cache hit for URL='{url}'.")
                cached_chunks.extend(cached.get("chunks", []))
            else:
                logger.debug(f"Cache miss for URL='{url}', will need to fetch and parse.")
                new_urls.append(url)
                new_metadata.append(metadatas[i])

        # Fetch and scrape new URLs with error handling
        try:
            if len(new_urls) > 0:
                docs = await self.scraper.batch_fetch_content(new_urls, new_metadata)
            else:
                docs = []
        except Exception as e:
            logger.error(f"Failed to fetch content for URLs {new_urls}: {e}")
            docs = []

        return docs, cached_chunks
