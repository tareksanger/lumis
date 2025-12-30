from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, cast, Optional

from asgiref.sync import sync_to_async
import pandas as pd
from pydantic import BaseModel, Field
import yfinance as yf

logger = logging.getLogger(__name__)


class QuoteData(BaseModel):
    """Schema for quote data returned by Yahoo Finance."""

    symbol: str = Field(..., description="Stock ticker symbol")
    short_name: Optional[str] = Field(None, description="Short company name")
    long_name: Optional[str] = Field(None, description="Full company name")
    current_price: float = Field(..., description="Current stock price")
    previous_close: float = Field(..., description="Previous closing price")
    open: float = Field(..., description="Opening price")
    day_high: float = Field(..., description="Day's highest price")
    day_low: float = Field(..., description="Day's lowest price")
    volume: int = Field(..., description="Trading volume")
    market_cap: Optional[int] = Field(None, description="Market capitalization")
    fifty_two_week_high: Optional[float] = Field(None, description="52-week high")
    fifty_two_week_low: Optional[float] = Field(None, description="52-week low")
    dividend_yield: Optional[float] = Field(None, description="Dividend yield")
    pe_ratio: Optional[float] = Field(None, description="Price to earnings ratio")


class HistoricalData(BaseModel):
    """Schema for historical market data."""

    date: datetime = Field(..., description="Date of the data point")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="Highest price")
    low: float = Field(..., description="Lowest price")
    close: float = Field(..., description="Closing price")
    volume: int = Field(..., description="Trading volume")
    adj_close: float = Field(..., description="Adjusted closing price")


class YahooFinance:
    """
    Wrapper around yfinance for stock/company data.

    This class provides async methods to interact with Yahoo Finance API
    for retrieving stock quotes, historical data, and company information.
    All methods are wrapped with sync_to_async to ensure compatibility
    with async codebases.
    """

    @sync_to_async
    def get_quote(self, ticker: str) -> QuoteData:
        """
        Get current quote and basic info for a ticker symbol.

        Args:
            ticker: Stock ticker symbol (e.g. 'AAPL')

        Returns:
            QuoteData object containing quote information

        Raises:
            ValueError: If ticker symbol is invalid
            Exception: For other API errors
        """
        try:
            logger.debug(f"Fetching quote data for ticker: {ticker}")
            t = yf.Ticker(ticker)
            info = t.info

            return QuoteData(
                symbol=ticker,
                short_name=info.get("shortName"),
                long_name=info.get("longName"),
                current_price=info.get("currentPrice", 0.0),
                previous_close=info.get("previousClose", 0.0),
                open=info.get("open", 0.0),
                day_high=info.get("dayHigh", 0.0),
                day_low=info.get("dayLow", 0.0),
                volume=info.get("volume", 0),
                market_cap=info.get("marketCap"),
                fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
                fifty_two_week_low=info.get("fiftyTwoWeekLow"),
                dividend_yield=info.get("dividendYield"),
                pe_ratio=info.get("trailingPE"),
            )
        except Exception as e:
            logger.error(f"Error fetching quote for {ticker}: {str(e)}")
            raise

    @sync_to_async
    def get_history(self, ticker: str, period: str = "1mo", interval: str = "1d", start: Optional[str] = None, end: Optional[str] = None) -> list[HistoricalData]:
        """
        Get historical market data.

        Args:
            ticker: Stock ticker symbol
            period: Time period (e.g. '1mo', '1y', 'max')
            interval: Data interval (e.g. '1d', '1h', '1wk')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)

        Returns:
            List of HistoricalData objects

        Raises:
            ValueError: If parameters are invalid
            Exception: For other API errors
        """
        try:
            logger.debug(f"Fetching historical data for {ticker} with period={period}, interval={interval}")
            t = yf.Ticker(ticker)
            hist = t.history(period=period, interval=interval, start=start, end=end)

            return [
                HistoricalData(
                    date=datetime.fromtimestamp(pd.Timestamp(str(index)).timestamp()),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                    adj_close=float(row["Adj Close"]),
                )
                for index, row in hist.iterrows()
            ]
        except Exception as e:
            logger.error(f"Error fetching history for {ticker}: {str(e)}")
            raise

    @sync_to_async
    def search_ticker(self, query: str) -> list[dict[str, Any]]:
        """
        Search for ticker symbols by keyword.

        Args:
            query: Search term

        Returns:
            List of dictionaries containing symbol and name

        Raises:
            Exception: For API errors
        """
        try:
            logger.debug(f"Searching for tickers matching: {query}")
            # Using yf.Tickers() as get_tickers_by_search is deprecated
            tickers = yf.Tickers(query)
            return [{"symbol": symbol, "name": ticker.info.get("shortName", symbol)} for symbol, ticker in tickers.tickers.items()]
        except Exception as e:
            logger.error(f"Error searching tickers for {query}: {str(e)}")
            raise

    @sync_to_async
    def get_recommendations(self, ticker: str) -> list[dict[str, Any]]:
        """
        Get analyst recommendations for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of recommendation data

        Raises:
            Exception: For API errors
        """
        try:
            logger.debug(f"Fetching recommendations for {ticker}")
            t = yf.Ticker(ticker)
            if t.recommendations is None:
                return []
            df = cast(pd.DataFrame, t.recommendations)
            return [dict(zip(df.columns, row)) for row in df.values]
        except Exception as e:
            logger.error(f"Error fetching recommendations for {ticker}: {str(e)}")
            raise

    @sync_to_async
    def get_major_holders(self, ticker: str) -> dict[str, Any]:
        """
        Get major holders of a stock.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing major holders data

        Raises:
            Exception: For API errors
        """
        try:
            logger.debug(f"Fetching major holders for {ticker}")
            t = yf.Ticker(ticker)
            if t.major_holders is None:
                return {}
            df = cast(pd.DataFrame, t.major_holders)
            return {str(k): v for k, v in df.to_dict().items()}
        except Exception as e:
            logger.error(f"Error fetching major holders for {ticker}: {str(e)}")
            raise
