from __future__ import annotations

from typing import Any, Dict, List

from asgiref.sync import sync_to_async
from pytrends.request import TrendReq


class PyTrends:
    """
    Wrapper around pytrends for Google Trends data.
    """

    def __init__(self, tz: int = 0, retries: int = 3, backoff: float = 0.1):
        self.pytrends = TrendReq(retries=retries, backoff_factor=backoff, tz=tz)  # type: ignore

    @sync_to_async
    def interest_over_time(self, keywords: List[str], timeframe: str = "today 3-m", geo: str = "", cat: int = 0, gprop: str = "") -> Any:
        """
        Get interest over time for given keywords.
        :param keywords: list of search terms
        :param timeframe: date range string (e.g. 'today 3-m')
        :param geo: geographical area code (e.g. 'US')
        :return: pandas DataFrame
        """
        self.pytrends.build_payload(keywords, cat=cat, timeframe=timeframe, geo=geo, gprop=gprop)
        return self.pytrends.interest_over_time()

    @sync_to_async
    def related_queries(self, keyword: str) -> Dict[str, Any]:
        """
        Fetch related queries for a single keyword.
        :param keyword: search term
        :return: dict with "top" and "rising" queries
        """
        self.pytrends.build_payload([keyword])
        return self.pytrends.related_queries().get(keyword, {})

    @sync_to_async
    def interest_by_region(self, keywords: List[str], timeframe: str = "today 3-m", geo: str = "", resolution: str = "CITY") -> Any:
        """
        Get geographic interest by region.
        :param resolution: REGION, CITY, or DMA
        """
        self.pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
        return self.pytrends.interest_by_region(resolution=resolution)
