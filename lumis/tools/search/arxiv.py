from __future__ import annotations

import asyncio
from dataclasses import dataclass
from io import BytesIO
import logging
from typing import Any, Dict, Optional
import xml.etree.ElementTree as ET

import arxiv
from arxiv import Client, SortCriterion, SortOrder
import httpx
from lumis.core.document import Document
import pymupdf

logger = logging.getLogger(__name__)


@dataclass
class ArxivResult:
    title: str
    authors: list[str]
    abstract: str
    pdf_url: Optional[str]
    arxiv_id: str
    published_date: str
    categories: list[str]
    comment: Optional[str]
    journal_ref: Optional[str]
    doi: Optional[str]
    primary_category: str
    updated_date: str
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ArxivSearcher:
    def __init__(self, max_results: int = 10, max_concurrent_pdfs: int = 5) -> None:
        self.max_results: int = max_results
        self.client = Client()
        self._http_client = httpx.AsyncClient()
        self._result_cache: dict[str, ArxivResult] = {}  # Cache for results by arxiv_id
        self._pdf_semaphore = asyncio.Semaphore(max_concurrent_pdfs)

    async def search(  # noqa: C901
        self,
        query: str,
        max_results: Optional[int] = None,
        sort_by: SortCriterion = SortCriterion.SubmittedDate,
        sort_order: SortOrder = SortOrder.Descending,
        id_list: Optional[list[str]] = None,
        read_pdfs: bool = False,
    ) -> list[ArxivResult]:
        """
        Search arXiv using the official arxiv library.

        Args:
            query: Search query string
            max_results: Optional override for max results
            sort_by: Sort criterion (Relevance, LastUpdatedDate, SubmittedDate)
            sort_order: Sort order (Ascending, Descending)
            id_list: Optional list of specific arXiv IDs to search for
            read_pdfs: Whether to download and read PDF content for each result

        Returns:
            list of ArxivResult objects
        """
        # Create search object with all available options
        search = arxiv.Search(query=query, max_results=max_results or self.max_results, sort_by=sort_by, sort_order=sort_order, id_list=id_list or [])

        results: list[ArxivResult] = []
        pdf_tasks = []

        # Execute search using the client
        for result in self.client.results(search):
            arxiv_id = result.get_short_id()

            # Check if we already have this result in cache
            if arxiv_id in self._result_cache:
                cached_result = self._result_cache[arxiv_id]
                # If we need PDF content and don't have it yet, update the cached result
                if read_pdfs and not cached_result.content and result.pdf_url:
                    pdf_tasks.append(self._update_cached_result(cached_result, result.pdf_url, arxiv_id))
                results.append(cached_result)
                continue

            # Convert arxiv.Result to ArxivResult with all available fields
            arxiv_result = ArxivResult(
                title=result.title,
                authors=[author.name for author in result.authors],
                abstract=result.summary,
                pdf_url=result.pdf_url,
                arxiv_id=arxiv_id,
                published_date=result.published.strftime("%Y-%m-%d"),
                categories=result.categories,
                comment=result.comment,
                journal_ref=result.journal_ref,
                doi=result.doi,
                primary_category=result.primary_category,
                updated_date=result.updated.strftime("%Y-%m-%d"),
            )

            # If PDF reading is requested and we have a PDF URL, add to tasks
            if read_pdfs and result.pdf_url:
                pdf_tasks.append(self._read_and_cache_pdf(arxiv_result, result.pdf_url))

            # Cache the result
            self._result_cache[arxiv_id] = arxiv_result
            results.append(arxiv_result)

        # Wait for all PDF reading tasks to complete
        if pdf_tasks:
            await asyncio.gather(*pdf_tasks)

        return results

    async def _update_cached_result(self, cached_result: ArxivResult, pdf_url: str, arxiv_id: str):
        """Update cached result with PDF content."""
        try:
            logger.debug(f"Updating cached result with PDF content for {arxiv_id}")
            pdf_content = await self._read_pdf(pdf_url)
            if pdf_content:
                cached_result.content = pdf_content.content
                cached_result.metadata = pdf_content.metadata
                logger.debug(f"Successfully updated cached result with PDF content ({len(pdf_content.content)} chars)")
        except Exception as e:
            logger.error(f"Failed to update cached result with PDF content for {arxiv_id}: {e}")

    async def _read_and_cache_pdf(self, arxiv_result: ArxivResult, pdf_url: str):
        """Read PDF and update result with content."""
        try:
            logger.debug(f"Attempting to read PDF from {pdf_url}")
            pdf_content = await self._read_pdf(pdf_url)
            if pdf_content:
                logger.debug(f"Successfully read PDF content ({len(pdf_content.content)} chars)")
                arxiv_result.content = pdf_content.content
                arxiv_result.metadata = pdf_content.metadata
            else:
                logger.warning(f"No content extracted from PDF at {pdf_url}")
        except Exception as e:
            logger.error(f"Failed to read PDF for {pdf_url}: {e}")

    async def _read_pdf(self, pdf_url: str) -> Optional[Document]:  # noqa: C901
        """Download and parse a PDF from arXiv."""
        try:
            # Download PDF
            logger.debug(f"Downloading PDF from {pdf_url}")
            response = await self._http_client.get(pdf_url)
            response.raise_for_status()
            logger.debug(f"Successfully downloaded PDF ({len(response.content)} bytes)")

            # Parse PDF content
            buffer = BytesIO(response.content)
            with pymupdf.open(stream=buffer, filetype="pdf") as doc:
                logger.debug(f"Opened PDF with {len(doc)} pages")
                # Extract metadata
                pdf_meta = doc.metadata or {}
                title = pdf_meta.get("title", "").strip()
                description = pdf_meta.get("subject", "").strip()

                # Try to get description from XMP metadata
                if not description and hasattr(doc, "xmp_metadata") and doc.xmp_metadata:  # type: ignore
                    try:
                        root = ET.fromstring(doc.xmp_metadata)  # type: ignore
                        ns = {"dc": "http://purl.org/dc/elements/1.1/", "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"}
                        desc_elements = root.findall(".//dc:description//rdf:Alt//rdf:li", ns)
                        if desc_elements and desc_elements[0].text:
                            description = desc_elements[0].text.strip()
                    except Exception as e:
                        logger.debug(f"Failed to parse XMP metadata: {e}")

                # Extract text content
                content = ""
                for page in doc:
                    page_text = page.get_text("text")  # type: ignore
                    content += page_text + chr(12)
                    logger.debug(f"Extracted {len(page_text)} chars from page {page.number + 1}")  # type: ignore

                if not description and content.strip():
                    description = content.strip()[:300] + "..."

                if content.strip():
                    metadata = {}
                    if title:
                        metadata["title"] = title
                    if description:
                        metadata["description"] = description

                    logger.debug(f"Successfully extracted content ({len(content)} chars) and metadata")
                    return Document(content=content, metadata=metadata)
                else:
                    logger.warning("No text content extracted from PDF")

        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            raise

        return None

    async def close(self):
        """Close the HTTP client."""
        await self._http_client.aclose()

    async def get_document(self, arxiv_id: str, read_pdf: bool = True) -> Optional[ArxivResult]:
        """
        Fetch and read a specific arXiv document by its ID.

        Args:
            arxiv_id: The arXiv ID of the document (e.g., '2303.08774')
            read_pdf: Whether to download and read the PDF content

        Returns:
            Optional[ArxivResult]: The document details and optionally its PDF content, or None if not found
        """
        try:
            # First check if we have it in cache
            if arxiv_id in self._result_cache:
                cached_result = self._result_cache[arxiv_id]
                # If we need PDF content and don't have it yet, update the cached result
                if read_pdf and not cached_result.content and cached_result.pdf_url:
                    await self._update_cached_result(cached_result, cached_result.pdf_url, arxiv_id)
                return cached_result

            # If not in cache, search for it
            results = await self.search(query=f"id:{arxiv_id}", max_results=1, id_list=[arxiv_id], read_pdfs=read_pdf)

            if results:
                return results[0]
            else:
                logger.warning(f"No document found with ID: {arxiv_id}")
                return None

        except Exception as e:
            logger.error(f"Error fetching document {arxiv_id}: {e}")
            return None
