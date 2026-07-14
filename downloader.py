"""Historical data acquisition.

Three sources, in priority order chosen by the user:
1. yfinance download (optional dependency; graceful error if offline/missing).
2. CSV upload (must contain a date column and a price column).
3. Manual parameters (no data needed).
"""

from __future__ import annotations

import io
import logging

import numpy as np
import pandas as pd

from simulator.validation import ValidationError, validate_price_series

logger = logging.getLogger("mcsim.data")


class DataSourceError(RuntimeError):
    """Raised when historical data cannot be obtained."""


def download_prices(ticker: str, period: str = "5y") -> pd.Series:
    """Download adjusted close prices via yfinance.

    Raises :class:`DataSourceError` with a readable message if yfinance is not
    installed, the ticker is invalid, or the network is unavailable.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataSourceError(
            "yfinance is not installed. Run `pip install yfinance`, or use "
            "CSV upload / manual parameters instead."
        ) from exc

    ticker = (ticker or "").strip().upper()
    if not ticker or any(c in ticker for c in " ;,'\""):
        raise DataSourceError(f"'{ticker}' is not a valid ticker symbol.")

    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    except Exception as exc:  # network / API errors
        raise DataSourceError(f"Download failed for {ticker}: {exc}") from exc

    if df is None or df.empty:
        raise DataSourceError(
            f"No data returned for '{ticker}'. Check the symbol and your "
            "internet connection."
        )
    close = df["Close"]
    if isinstance(close, pd.DataFrame):        # multi-index single ticker
        close = close.iloc[:, 0]
    close = close.dropna()
    validate_price_series(close.to_numpy())
    logger.info("Downloaded %d prices for %s", close.size, ticker)
    return close.rename(ticker)


def load_csv(file_or_buffer, price_column: str | None = None) -> pd.Series:
    """Load a price series from a CSV file or buffer.

    The date column is auto-detected (first parseable column); the price
    column is ``price_column`` if given, else the first numeric column named
    like close/adj close/price, else the first numeric column.
    """
    if isinstance(file_or_buffer, (str, bytes)):
        file_or_buffer = io.StringIO(
            file_or_buffer.decode() if isinstance(file_or_buffer, bytes)
            else file_or_buffer
        )
    df = pd.read_csv(file_or_buffer)
    if df.empty:
        raise DataSourceError("CSV is empty.")

    # Date index
    date_col = None
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        except (TypeError, ValueError):
            continue
        if parsed.notna().mean() > 0.9:
            date_col = col
            df[col] = parsed
            break
    if date_col is not None:
        df = df.set_index(date_col).sort_index()

    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        raise DataSourceError("CSV contains no numeric price column.")

    if price_column and price_column in numeric.columns:
        series = numeric[price_column]
    else:
        preferred = [c for c in numeric.columns
                     if str(c).lower() in ("adj close", "adj_close", "close", "price")]
        series = numeric[preferred[0]] if preferred else numeric.iloc[:, 0]

    series = series.dropna()
    try:
        validate_price_series(series.to_numpy())
    except ValidationError as exc:
        raise DataSourceError(f"CSV price column invalid: {exc}") from exc
    return series
