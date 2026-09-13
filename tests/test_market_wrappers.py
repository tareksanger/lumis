import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

from lumis.tools.search.pytrends import PyTrends
from lumis.tools.search.yahoofinance import YahooFinance

import pandas as pd
import pytest


@pytest.fixture
def trends(monkeypatch):
    client = Mock()
    module = ModuleType("pytrends.request")
    module.TrendReq = Mock(return_value=client)
    monkeypatch.setitem(sys.modules, "pytrends", ModuleType("pytrends"))
    monkeypatch.setitem(sys.modules, "pytrends.request", module)
    wrapper = PyTrends(tz=60, retries=2, backoff=0.5)
    module.TrendReq.assert_called_once_with(tz=60, retries=2, backoff_factor=0.5)
    return wrapper, client


async def test_trends_time_series_and_region(trends):
    wrapper, client = trends
    result = pd.DataFrame({"topic": [2]})
    client.interest_over_time.return_value = result
    assert await wrapper.interest_over_time(["topic"], timeframe="today 1-m", geo="CA", cat=1, gprop="news") is result
    client.build_payload.assert_called_with(["topic"], cat=1, timeframe="today 1-m", geo="CA", gprop="news")
    client.interest_by_region.return_value = result
    assert await wrapper.interest_by_region(["topic"], geo="US", resolution="COUNTRY") is result
    client.interest_by_region.assert_called_with(resolution="COUNTRY")


async def test_related_queries_missing_keyword(trends):
    wrapper, client = trends
    client.related_queries.return_value = {"topic": {"top": [1]}}
    assert await wrapper.related_queries("topic") == {"top": [1]}
    assert await wrapper.related_queries("missing") == {}


@pytest.fixture
def finance(monkeypatch):
    module = ModuleType("yfinance")
    module.Ticker = Mock()
    module.Tickers = Mock()
    monkeypatch.setitem(sys.modules, "yfinance", module)
    return YahooFinance(), module


async def test_quote_mapping_and_optional_defaults(finance):
    wrapper, module = finance
    module.Ticker.return_value.info = {"shortName": "Example", "currentPrice": 10, "volume": 12, "trailingPE": 5}
    quote = await wrapper.get_quote("EX")
    assert (quote.symbol, quote.short_name, quote.current_price, quote.volume, quote.pe_ratio) == ("EX", "Example", 10, 12, 5)
    assert quote.previous_close == 0
    assert quote.market_cap is None


async def test_history_converts_rows_and_forwards_range(finance):
    wrapper, module = finance
    frame = pd.DataFrame([{"Open": 1, "High": 3, "Low": 1, "Close": 2, "Volume": 10, "Adj Close": 1.9}], index=pd.to_datetime(["2025-01-01"]))
    module.Ticker.return_value.history.return_value = frame
    result = await wrapper.get_history("EX", period="1y", interval="1wk", start="2025-01-01", end="2025-02-01")
    assert len(result) == 1 and result[0].adj_close == 1.9 and result[0].volume == 10
    module.Ticker.return_value.history.assert_called_once_with(period="1y", interval="1wk", start="2025-01-01", end="2025-02-01")
    module.Ticker.return_value.history.return_value = frame.iloc[:0]
    assert await wrapper.get_history("EX") == []


async def test_ticker_search_falls_back_to_symbol(finance):
    wrapper, module = finance
    module.Tickers.return_value.tickers = {"A": SimpleNamespace(info={"shortName": "Alpha"}), "B": SimpleNamespace(info={})}
    assert await wrapper.search_ticker("A B") == [{"symbol": "A", "name": "Alpha"}, {"symbol": "B", "name": "B"}]


async def test_recommendations_and_holders(finance):
    wrapper, module = finance
    ticker = module.Ticker.return_value
    ticker.recommendations = None
    ticker.major_holders = None
    assert await wrapper.get_recommendations("EX") == []
    assert await wrapper.get_major_holders("EX") == {}
    ticker.recommendations = pd.DataFrame([{"buy": 2, "hold": 1}])
    ticker.major_holders = pd.DataFrame({"value": [10]})
    assert await wrapper.get_recommendations("EX") == [{"buy": 2, "hold": 1}]
    assert await wrapper.get_major_holders("EX") == {"value": {0: 10}}


@pytest.mark.parametrize("method", ["get_quote", "get_history", "search_ticker", "get_recommendations", "get_major_holders"])
async def test_finance_provider_errors_propagate(finance, method):
    wrapper, module = finance
    module.Ticker.side_effect = RuntimeError("offline")
    module.Tickers.side_effect = RuntimeError("offline")
    with pytest.raises(RuntimeError, match="offline"):
        await getattr(wrapper, method)("EX")


@pytest.mark.parametrize("package,constructor", [("yfinance", YahooFinance), ("pytrends.request", PyTrends)])
def test_optional_dependency_has_install_hint(monkeypatch, package, constructor):
    monkeypatch.setitem(sys.modules, package, None)
    with pytest.raises(ImportError, match=r"lumis-ai\[search\]"):
        constructor()
