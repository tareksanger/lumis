from __future__ import annotations


from .scraper import WebScrapper
from .search.wiki import WikiPage, WikipediaSearcher

__all__ = ["WebScrapper", "WikipediaSearcher", "WikiPage"]
