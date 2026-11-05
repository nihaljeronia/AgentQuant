import yfinance as yf
import pandas as pd
import os

CACHE_DIR = "data_layer/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_price_data(ticker: str, period: str = "3y") -> pd.DataFrame:
    """
    Fetch historical price data for a given ticker with clean CSV caching.
    Flattens multi-index headers from yfinance and enforces float data types.
    """
    cache_path = f"{CACHE_DIR}/{ticker}_{period}.csv"
    
    # If cached file exists, check if header has multi-index artifact
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0)
            # Check if read was corrupted by multi-index lines
            if "Close" in df.columns and pd.api.types.is_numeric_dtype(df["Close"]):
                df.index = pd.to_datetime(df.index)
                return df
        except Exception:
            pass
        # Remove corrupted cache file if present
        try:
            os.remove(cache_path)
        except OSError:
            pass

    df = yf.download(ticker, period=period, auto_adjust=True)
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_csv(cache_path)
    return df


def split_data(df: pd.DataFrame, train_ratio: float = 0.70) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronologically split a price DataFrame into train and test sets.
    
    The train set (research window) is what the Hypothesis/Backtest/Critique/Refine
    loop sees and iterates on. The test set (validation window) is held out and
    NEVER seen during the loop — it is used only for final out-of-sample scoring.
    
    Args:
        df: Full historical price DataFrame (datetime-indexed, sorted ascending).
        train_ratio: Fraction of rows to use for training (default 0.70 = 70%).
    
    Returns:
        (train_df, test_df) tuple.
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
