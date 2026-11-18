import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import PRICE_CACHE_DIR, ensure_dirs  # noqa: E402

# Kept for backwards compatibility with older scripts that import CACHE_DIR.
CACHE_DIR = str(PRICE_CACHE_DIR)


class PriceDataError(RuntimeError):
    """Raised when prices are neither cached nor downloadable."""


def _cache_path(ticker: str, period: str):
    return PRICE_CACHE_DIR / f"{ticker}_{period}.csv"


def _read_cache(path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception:
        return None
    if "Close" not in df.columns:
        return None
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df[df["Close"].notna()]
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def get_price_data(ticker: str, period: str = "3y", cache_only: bool = False) -> pd.DataFrame:
    """
    Fetch historical daily prices for a ticker, using a CSV cache.

    The cache is read first, so once a ticker has been downloaded the project runs
    without internet. Set cache_only=True to never touch
    the network.
    """
    ensure_dirs()
    cache_path = _cache_path(ticker, period)

    if cache_path.exists():
        df = _read_cache(cache_path)
        if df is not None:
            return df
        try:
            os.remove(cache_path)  # corrupted cache file
        except OSError:
            pass

    if cache_only:
        raise PriceDataError(
            f"No cached prices for {ticker} ({period}). Run once with internet to download them."
        )

    try:
        import yfinance as yf

        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    except Exception as e:  # network errors, rate limits
        raise PriceDataError(f"Could not download {ticker}: {e}") from e

    if df is None or df.empty:
        raise PriceDataError(
            f"No price data returned for {ticker}. Check the symbol or your internet connection."
        )

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_csv(cache_path)
    return df


def list_cached_tickers(period: str = "3y") -> list[str]:
    """Tickers whose prices are already cached (no download needed)."""
    if not PRICE_CACHE_DIR.exists():
        return []
    suffix = f"_{period}.csv"
    return sorted(p.name[: -len(suffix)] for p in PRICE_CACHE_DIR.glob(f"*{suffix}"))


def split_data(df: pd.DataFrame, train_ratio: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronologically split a price DataFrame into train and test sets.

    The train set (research window) is what the Hypothesis/Backtest/Critique/Refine
    loop sees and iterates on. The test set (validation window) is held out and
    NEVER seen during the loop — it is used only for final out-of-sample scoring.
    """
    n = len(df)
    split_idx = int(n * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


if __name__ == "__main__":
    df = get_price_data("TCS.NS")
    print(f"Full data: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    print("Close Dtype:", df["Close"].dtype)

    train, test = split_data(df)
    print(f"\nTrain split: {len(train)} rows, {train.index[0].date()} to {train.index[-1].date()}")
    print(f"Test split:  {len(test)} rows,  {test.index[0].date()} to {test.index[-1].date()}")
