from __future__ import annotations

import asyncio
from io import BytesIO
import logging
from typing import Callable, Dict, List, Optional
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
from dateutil.parser import parse
import httpx
from lumis.core.document import Document
import lxml.html
from minify_html import minify
import pymupdf
from readability import Document as RDocument

logger = logging.getLogger("WebScrapper")


class WebScrapper:
    HEADERS = {
        "accept": "application/json, text/plain, */*",
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) QtWebEngine/5.15.2 Chrome/87.0.4280.144 Safari/537.36",
        "content-type": "application/json",
        "accept-language": "en-US,en;q=0.9",
    }

    def __init__(
        self,

        http_client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,
        max_concurrency: int = 20,  # limit the number of concurrent fetches
        default_headers: Optional[dict] = {},
    ):
        """
        Initialize the WebScrapper with optional dependency injection.

        Args:
            http_client_factory (Optional[Callable[[], httpx.AsyncClient]]): A factory function that returns
                an httpx.AsyncClient instance. Useful for testing. If not provided, a default client is used.
        """
        self.http_client_factory = http_client_factory if http_client_factory is not None else lambda: httpx.AsyncClient(timeout=httpx.Timeout(10.0))
        self.semaphore = asyncio.Semaphore(max_concurrency)

        self.default_headers = self.HEADERS
        self.default_headers.update(default_headers)

    async def batch_fetch_content(self, urls: List[str], metadatas: Optional[List[Dict]] = None) -> List[Document]:
        """
        Fetches content for multiple URLs concurrently with a concurrency limit.
        """
        logger.info(f"Fetching content for {len(urls)} URLs")
        if metadatas is not None and len(metadatas) != len(urls):
            logger.error("The number of URLs must match the number of metadata objects.")
            raise ValueError("The number of URLs must match the number of metadata objects.")

        if metadatas is None:
            metadatas = [{} for _ in urls]

        # Create tasks with semaphore to limit concurrency
        tasks = [asyncio.create_task(self._fetch_with_semaphore(url, meta)) for url, meta in zip(urls, metadatas)]

        logger.info(f"Created {len(tasks)} tasks")

        # gather all tasks, fail fast if one task raises an exception:
        # using return_exceptions=False means the first exception will immediately propagate.
        # If you want all tasks to complete and collect exceptions, set return_exceptions=True and handle them afterward.
        results = await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"Finished fetching content for {len(urls)} URLs")

        documents: List[Document] = []
        for url, result in zip(urls, results):
            if isinstance(result, list):
                documents.extend(result)
            else:
                logger.error(f"Failed to fetch content for {url}: {result}")

        return documents

    async def _fetch_with_semaphore(self, url: str, metadata: dict) -> List[Document]:
        async with self.semaphore:
            return await self.fetch_content(url, metadata)

    async def fetch_content(self, url: str, metadata: Optional[dict] = None, params: Optional[dict] = None) -> list[Document]:  # noqa: C901
        url = self._url_with_protocol(url)

        documents: list[Document] = []
        try:
            async with self.http_client_factory() as client:
                response = await client.get(
                    url=url,
                    headers=self.default_headers,
                    params=params,
                    follow_redirects=True,
                )

                content_type = response.headers.get("content-type", "")

                # In the case of a redirect update the url
                url = str(response.url)

                # Add url to metadata
                if metadata is None:
                    metadata = {}

                metadata["url"] = url

                if response.status_code in [200, 201, 202, 203] and content_type:
                    if "application/pdf" in content_type:
                        logger.debug(f"Parsing PDF content from {url}")
                        document = await self.parse_pdf(response.content, metadata)
                    elif "text/html" in content_type:
                        logger.debug(f"Parsing HTML content from {url}")

                        if not response.text.strip():
                            raise ValueError("No HTML body content found in the response.")
                        document = await self.parse_html(url, response.text, metadata)

                        # If content is too short, try Gotenberg
                        if len(document.content) < 100:
                            logger.debug(f"Content from {url} is too short. Content={document.content}. Trying Gotenberg.")
                            raise ValueError("Short content trigger")

                    else:
                        logger.debug(f"Unsupported content type '{content_type}' for {url}. Using Gotenberg.")

                else:
                    logger.debug(f"Non-OK status code or no content-type for {url}. Using Gotenberg.")


        except Exception as e:
            logger.debug(f"Error fetching content from {url} (Using Gotenberg): {str(e)}")

        if document is not None:
            logger.debug(f"Adding document for {url}")
            documents.append(document)
        else:
            logger.debug(f"No document generated for {url}")

        return documents

    def cleanup_html(self, url: str, html_content: str):  # noqa: C901
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()

        text_lower = soup.get_text(separator=" ").lower()
        if "captcha" in text_lower or "cloudflare" in text_lower:
            return None, None, None, None

        # Extract title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else "No Title Found"

        # Extract metadata
        metadata = {
            "publication_date": None,
            "author": None,
            "description": None,
            "logo": None,
        }

        meta_tags = soup.find_all("meta")
        for meta in meta_tags:
            if meta.get("name") in [
                "date",
                "publication_date",
                "article:published_time",
            ]:
                metadata["publication_date"] = meta.get("content")
            elif meta.get("name") in ["author"]:
                metadata["author"] = meta.get("content")
            elif meta.get("name") in ["description", "og:description"]:
                metadata["description"] = meta.get("content")
            elif meta.get("property") in ["og:image", "twitter:image"]:
                metadata["logo"] = meta.get("content")  # Prioritize Open Graph or Twitter image

        # Extract favicon/logo from <link> tags
        if not metadata["logo"]:
            icon_link = soup.find("link", rel=["icon", "shortcut icon", "apple-touch-icon"])
            if icon_link and icon_link.get("href"):
                metadata["logo"] = urljoin(url, icon_link["href"])  # Resolve relative URLs

        # Attempt to parse the publication date if available
        if metadata["publication_date"]:
            try:
                metadata["publication_date"] = parse(metadata["publication_date"]).isoformat()
            except ValueError:
                metadata["publication_date"] = None

        # Extract links
        link_tags = soup.find_all("a", href=True)
        link_urls = self._process_links(url=url, links=[a["href"] for a in link_tags])

        # Extract main content
        cleaned_html = str(soup)
        doc = RDocument(cleaned_html)
        summary_html = doc.summary()
        summary_title = doc.short_title() or title

        summary_html_min = minify(summary_html)
        summary_tree = lxml.html.fromstring(summary_html_min)
        body_text = summary_tree.text_content().strip()

        if not body_text:
            raise ValueError("No readable content found.")

        return summary_title, body_text, link_urls, metadata

    async def parse_html(self, url: str, content: str, metadata: Optional[dict] = None) -> Document:  # noqa: C901
        # Extract title, body, links, and metadata (which now includes publication_date, author, description, and logo)
        title, body, links, metadata_extracted = self.cleanup_html(url, content)

        if metadata is None:
            metadata = {}

        if title is not None:
            metadata["title"] = title
        if body is not None:
            metadata["content_length"] = len(body)  # Store content length for reference
        if links is not None:
            metadata["links"] = list(set(links))

        # Merge extracted metadata
        if metadata_extracted:
            metadata.update({k: v for k, v in metadata_extracted.items() if v})  # Only keep non-empty values

        return Document(content=body or "", metadata=metadata)

    async def parse_pdf(self, b: bytes, metadata: Optional[dict] = None) -> Optional[Document]:  # noqa: C901
        try:
            buffer = BytesIO(b)
            with pymupdf.open(stream=buffer, filetype="pdf") as doc:
                # 1. Extract basic metadata from the document info dictionary.
                pdf_meta = doc.metadata or {}
                title = pdf_meta.get("title", "").strip()
                description = pdf_meta.get("subject", "").strip()

                # 2. If description is empty, try extracting it from the XMP metadata.
                if not description and hasattr(doc, "xmp_metadata") and doc.xmp_metadata:
                    try:
                        # Parse the XML in the XMP metadata.
                        root = ET.fromstring(doc.xmp_metadata)
                        # Define the namespaces used in the XMP metadata.
                        ns = {"dc": "http://purl.org/dc/elements/1.1/", "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"}
                        # Look for the description element. Typically it is nested under:
                        # <dc:description><rdf:Alt><rdf:li>Actual description</rdf:li></rdf:Alt></dc:description>
                        desc_elements = root.findall(".//dc:description//rdf:Alt//rdf:li", ns)
                        if desc_elements and desc_elements[0].text:
                            description = desc_elements[0].text.strip()
                    except Exception as e:
                        logger.debug(f"Failed to parse XMP metadata for description: {e}")

                # 3. As a fallback, if no description is found, you could extract the first paragraph
                #    or the first 300 characters from the PDF content.
                content = chr(12).join(page.get_text() for page in doc)
                if not description and content.strip():
                    # For example, take the first 300 characters as a fallback description.
                    description = content.strip()[:300] + "..."

            # Only return a Document if some content was extracted.
            if content.strip():
                if metadata is None:
                    metadata = {}
                if title:
                    metadata["title"] = title
                if description:
                    metadata["description"] = self._clean_string(description)

                return Document(content=content, metadata=metadata)
        except Exception as e:
            logger.debug(f"Failed to parse PDF: {e}")
            raise e

        return None

    # async def from_gotenberg(self, url: str, metadata: Optional[dict]) -> Optional[Document]:
    #     # Same logic as before, but uses injected gb_client
    #     try:
    #         response = await self.gb_client.afrom_url(url)
    #         return await self.parse_pdf(response.content, metadata)
    #     except Exception as e:
    #         logger.debug(f"Gotenberg failed for {url}: {e}")
    #         return None

    def _url_with_protocol(self, url: str):
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url
        return url

    def _parse_domain(self, url: str):
        # If no scheme is present, prepend one:
        url = self._url_with_protocol(url)

        parsed = urlparse(url)
        domain = parsed.netloc
        return domain

    def _process_links(self, url: str, links: list[str]):
        urls = set()

        # If no scheme is present, prepend one:
        url = self._url_with_protocol(url)

        domain = self._parse_domain(url)

        for href in links:
            if href == "" or href is None:
                # href empty tag
                continue

            # join the URL if it's relative (not absolute link)
            href = urljoin(url, href)
            parsed_href = urlparse(href)

            # Filter out unwanted schemes like mailto, javascript, etc.
            if parsed_href.scheme not in ["http", "https"]:
                continue

            # remove URL GET parameters, URL fragments, etc.
            href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
            if domain not in href:
                # not an internal link
                continue
            urls.add(href)

        return list(urls)

    def _clean_string(self, text: str) -> str:
        filter_chars = "".join([chr(i) for i in range(1, 32)])
        translation_table = str.maketrans(filter_chars, " " * len(filter_chars))
        return text.translate(translation_table)
