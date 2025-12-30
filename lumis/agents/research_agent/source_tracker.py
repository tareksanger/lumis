from __future__ import annotations

"""
Source tracking functionality for the ResearchAgent.

This module provides classes for tracking and managing sources and metadata
from various tool calls during agent interactions.
"""

from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Awaitable, Callable, Literal, Optional, Union

from agents import Agent, AgentHooks, RunContextWrapper, TContext, Tool

from .source_models import SourceType
from .types import ResearchAgentResponse

from lumis.embedding import OpenAIEmbeddingModel
import numpy as np
from pydantic import ValidationError

logger = logging.getLogger(__name__)

# Type for callback that can be either sync or async
CallbackT = Callable[[RunContextWrapper[TContext], Agent[TContext], Any], Union[None, Awaitable[None]]]


@dataclass
class SourceMatch:
    source: SourceType
    confidence: float
    match_type: Literal["explicit", "semantic"]


@dataclass
class ResearchSourceTracker:
    """Base context class for tracking research sources used during agent interactions.

    This class can be inherited from to add source tracking capabilities to any context.
    """

    sources: dict[str, SourceType] = field(default_factory=dict)  # url -> BaseSource
    insights: list[ResearchAgentResponse] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    query: str = ""

    # Use a default embedding model if none is provided (Need to be careful about this when we change our embedding model as ConceptAgentContext also has an embedding model)
    embedding_model: OpenAIEmbeddingModel = field(default_factory=OpenAIEmbeddingModel)

    def add_source(self, source: SourceType) -> None:
        """
        Add a source to the tracker.

        Args:
            source (BaseSource): The source to add
        """
        # Use URL as key if available, otherwise use title
        key = source.url or source.title
        logger.debug(f"Adding source: {key} with title: {source.title}")
        self.sources[key] = source

    def get_sources(self) -> list[dict[str, Any]]:
        """Get all tracked sources with their details."""
        sources = [source.model_dump() for source in self.sources.values()]
        logger.debug(f"Retrieved {len(sources)} sources")
        return sources

    def get_source_by_url(self, url: str) -> Optional[SourceType]:
        """Get a source by its URL."""
        return self.sources.get(url)

    def clear(self) -> None:
        """Clear all tracked sources."""
        logger.debug("Clearing source tracker")
        self.sources.clear()
        self.query = ""

    async def find_relevant_sources(self, content: str, threshold: float = 0.6) -> list[SourceMatch]:  # noqa: C901
        """Find sources relevant to the given content using both explicit and semantic matching."""
        matches: list[SourceMatch] = []

        if not self.sources:
            return matches

        # 1. Check for explicit URL matches
        for source in self.sources.values():
            if source.url and source.url in content:
                matches.append(SourceMatch(source=source, confidence=1.0, match_type="explicit"))

        # 2. Semantic search for remaining sources
        # Get sources that weren't matched by URL
        remaining_sources = [source for source in self.sources.values() if not (source.url and source.url in content)]
        if not remaining_sources:
            return matches

        # Prepare texts for batch embedding
        source_texts = [f"{source.title} {source.summary}" for source in remaining_sources]

        # Batch embed content and all source texts at once
        content_embedding = await self.embedding_model.aembed(content)
        source_embeddings = await self.embedding_model.aembed(source_texts)

        # Compute all similarities at once using numpy operations
        # Normalize embeddings for cosine similarity
        content_norm = content_embedding / np.linalg.norm(content_embedding)
        source_norms = source_embeddings / np.linalg.norm(source_embeddings, axis=1, keepdims=True)

        # Compute cosine similarities (dot product of normalized vectors)
        similarities = np.dot(source_norms, content_norm)

        # Add matches that exceed threshold
        for source, similarity in zip(remaining_sources, similarities):
            if similarity >= threshold:
                matches.append(
                    SourceMatch(
                        source=source,
                        confidence=float(similarity),
                        match_type="semantic",
                    )
                )

        # Sort by confidence
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches


class ResearchAgentHooks(AgentHooks[TContext]):
    """Hooks for tracking research agent behavior and sources."""

    def __init__(self, on_end_callback: Optional[CallbackT] = None):
        self.on_end_callback = on_end_callback

    async def on_start(self, ctx: RunContextWrapper[TContext], agent: Agent[TContext]) -> None:
        """Called before the agent starts processing a query."""
        query = str(ctx)
        logger.info(f"Starting new query: {query}")

        # Check if context is a ResearchSourceTracker
        if isinstance(ctx.context, ResearchSourceTracker):
            ctx.context.clear()
            ctx.context.query = query
            ctx.context.timestamp = datetime.now()

    async def on_end(
        self,
        ctx: RunContextWrapper[TContext],
        agent: Agent[TContext],
        output: Any,
    ) -> None:
        """Called when the agent produces a final output."""
        import inspect

        # Save insights to the context for later use
        if isinstance(output, ResearchAgentResponse) and isinstance(ctx.context, ResearchSourceTracker):
            ctx.context.insights.append(output)

        # Call the on_end_callback if it exists
        logger.info(f"On end callback: {self.on_end_callback}")
        if self.on_end_callback is not None:
            logger.info(f"Calling on_end_callback: {self.on_end_callback}")
            result = self.on_end_callback(ctx, agent, output)
            if inspect.isawaitable(result):
                await result

    def _parse_arxiv_dates(self, item: dict[str, Any]) -> tuple[Optional[datetime], Optional[datetime]]:
        """Parse published and updated dates from arXiv item."""
        published_date = None
        updated_date = None

        if "published_date" in item:
            try:
                published_date = datetime.fromisoformat(item["published_date"])
            except (ValueError, TypeError):
                pass

        if "updated_date" in item:
            try:
                updated_date = datetime.fromisoformat(item["updated_date"])
            except (ValueError, TypeError):
                pass

        return published_date, updated_date

    def _build_arxiv_metadata(
        self,
        item: dict[str, Any],
        published_date: Optional[datetime],
        updated_date: Optional[datetime],
    ) -> dict[str, Any]:
        """Build metadata dictionary for arXiv source."""
        metadata = {}

        # Add basic fields if present
        fields_to_copy = [
            "authors",
            "arxiv_id",
            "categories",
            "primary_category",
            "comment",
            "journal_ref",
            "doi",
            "content",
            "abstract",
        ]
        for field_name in fields_to_copy:
            if field_name in item:
                metadata[field_name] = item.get(field_name, [] if field_name in ["authors", "categories"] else "")

        # Add parsed dates
        if published_date:
            metadata["published_date"] = published_date
        if updated_date:
            metadata["updated_date"] = updated_date

        # Merge any additional metadata
        metadata.update(item.get("metadata", {}))

        return metadata

    def _handle_arxiv_source(self, item: dict[str, Any], tool_name: str) -> Optional[SourceType]:
        """Handle arXiv paper source data."""
        try:
            # Parse dates
            published_date, updated_date = self._parse_arxiv_dates(item)

            # Build metadata
            metadata = self._build_arxiv_metadata(item, published_date, updated_date)

            return SourceType(
                url=item.get("url"),
                title=item.get("title", ""),
                summary=item.get("abstract", ""),
                keywords=item.get("categories", []),
                credibility=0.9,
                metadata=metadata,
                source_type="arxiv",
            )
        except (KeyError, ValidationError) as e:
            logger.error(f"Error processing arXiv source: {e}")
            return None

    def _handle_wiki_source(self, item: dict[str, Any], tool_name: str) -> Optional[SourceType]:
        """Handle Wikipedia article source data."""
        try:
            metadata = {}
            if "content" in item:
                metadata["content"] = item.get("content", None)
            if "categories" in item:
                metadata["categories"] = item.get("categories", [])

            return SourceType(
                url=item.get("url"),
                title=item.get("title", ""),
                summary=item.get("summary", ""),
                keywords=item.get("categories", []),
                credibility=0.7,
                metadata=metadata,
                source_type="wiki",
            )
        except (KeyError, ValidationError) as e:
            logger.error(f"Error processing Wikipedia source: {e}")
            return None

    def _handle_openai_web_source(self, item: dict[str, Any], tool_name: str) -> Optional[SourceType]:
        """Handle web search source data."""
        try:
            metadata = {
                "content": item.get("raw_content", None),
                "description": item.get("description", ""),
            }

            return SourceType(
                url=item.get("url"),
                title=item["title"],
                summary=item.get("description", ""),
                keywords=[],  # Web sources don't typically have keywords
                credibility=0.6,
                metadata=metadata,
                source_type="web",
            )
        except (KeyError, ValidationError) as e:
            logger.error(f"Error processing web source: {e}")
            return None

    def _handle_gemini_source(self, item: dict[str, Any], tool_name: str) -> Optional[SourceType]:  # noqa: C901
        """Handle Gemini search source data."""
        try:
            metadata = {}

            if "content" in item:
                metadata["content"] = item.get("content", None)
            if "summary" in item:
                metadata["summary"] = item.get("summary", "")
            if "segments" in item:
                metadata["segments"] = item.get("segments", [])

            metadata.update(item.get("metadata", {}))

            if "confidence" in item:
                metadata["confidence"] = item.get("confidence", 0.0)

            return SourceType(
                url=item.get("url"),
                title=item["title"],
                summary=item.get("summary", ""),
                keywords=[],  # Gemini sources don't typically have keywords
                credibility=0.6,  # Use calculated confidence score
                metadata=metadata,
                source_type="gemini",
            )
        except (KeyError, ValidationError) as e:
            logger.error(f"Error processing Gemini source: {e}")
            return None

    def _process_source(self, ctx: RunContextWrapper[TContext], item: dict[str, Any], tool_name: str) -> None:  # noqa: C901
        """Process a source based on its type and add it to the tracker."""
        if not isinstance(ctx.context, ResearchSourceTracker):
            return

        source: Optional[SourceType] = None

        # Determine source type from tool name
        if "arxiv" in tool_name:
            source = self._handle_arxiv_source(item, tool_name)
        elif "wiki" in tool_name:
            source = self._handle_wiki_source(item, tool_name)
        elif "web" in tool_name or "gemini" in tool_name:
            source = self._handle_gemini_source(item, tool_name)
        elif "openai" in tool_name:
            # OpenAI web search
            source = self._handle_openai_web_source(item, tool_name)
        else:
            logger.warning(f"Unknown tool name: {tool_name}")

        if source:
            ctx.context.add_source(source)

    async def on_tool_end(  # noqa: C901
        self,
        ctx: RunContextWrapper[TContext],
        agent: Agent[TContext],
        tool: Tool,
        result: Any,
    ) -> None:
        """Called after each tool call to track sources."""
        logger.info(f"Processing results from tool: {tool.name}")
        logger.info(f"Result: {result}")

        try:
            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        self._process_source(ctx, item, tool.name)
            elif isinstance(result, dict):
                # Handle Gemini response format
                if "answer" in result and "sources" in result:
                    # Process each source from the Gemini response
                    for source in result["sources"]:
                        if isinstance(source, dict):
                            # Add the answer to each source for context
                            source["answer"] = result["answer"]
                            self._process_source(ctx, source, tool.name)

                # Handle other response formats
                elif "url" in result or "title" in result:
                    self._process_source(ctx, result, tool.name)
                elif "results" in result:
                    for item in result["results"]:
                        if isinstance(item, dict):
                            self._process_source(ctx, item, tool.name)
        except Exception as e:
            logger.error(f"Error processing tool result: {e}")
            # Don't re-raise the exception to avoid breaking the agent's flow
