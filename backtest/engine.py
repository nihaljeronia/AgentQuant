import pandas as pd
import numpy as np
import re
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schema import sanitize_strategy, DEFAULT_PARAMS

# Minimum number of trades required for metrics to be considered statistically meaningful.
MIN_TRADES = 20

def get_close_series(df: pd.DataFrame) -> pd.Series:
    """Extract a clean 1D Series for Close prices, handling multi-index or DataFrame structures."""
    if "Close" in df.columns:
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            return close.iloc[:, 0]
        return close
    elif isinstance(df.columns, pd.MultiIndex):
        return df.xs("Close", axis=1, level=0).iloc[:, 0]
    return df.iloc[:, 0]

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    period = max(2, int(period))
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_bollinger(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Calculate Bollinger Bands (middle, upper, lower)."""
    period = max(2, int(period))
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return sma, upper, lower

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD (macd_line, signal_line, histogram)."""
    fast = max(2, int(fast))
    slow = max(fast + 1, int(slow))
    signal = max(2, int(signal))
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# --- Parameterized Signal Interpreters ---

def interpret_rsi(df: pd.DataFrame, params: dict) -> pd.Series:
    close = get_close_series(df)
    period = int(params.get("rsi_period", 14))
    thresh = float(params.get("rsi_entry_threshold", 30))
    rsi = calculate_rsi(close, period)
    return rsi <= thresh

def interpret_ma_crossover(df: pd.DataFrame, params: dict) -> pd.Series:
    close = get_close_series(df)
    fast_p = int(params.get("ma_fast", 20))
    slow_p = int(params.get("ma_slow", 50))
    if fast_p >= slow_p:
        fast_p = max(5, slow_p - 10)
    sma_fast = close.rolling(fast_p).mean()
    sma_slow = close.rolling(slow_p).mean()
    signal = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))
    return signal.fillna(False)

def interpret_price_dip(df: pd.DataFrame, params: dict) -> pd.Series:
    close = get_close_series(df)
    dip_pct = float(params.get("dip_pct", 0.05))
    daily_return = close.pct_change()
    return daily_return <= -dip_pct

def interpret_bollinger(df: pd.DataFrame, params: dict) -> pd.Series:
    close = get_close_series(df)
    period = int(params.get("bollinger_period", 20))
    std_dev = float(params.get("bollinger_std", 2.0))
    _, _, lower_band = calculate_bollinger(close, period, std_dev)
    return close <= lower_band

def interpret_macd(df: pd.DataFrame, params: dict) -> pd.Series:
    close = get_close_series(df)
    fast_p = int(params.get("macd_fast", 12))
    slow_p = int(params.get("macd_slow", 26))
    signal_p = int(params.get("macd_signal", 9))
    macd_line, signal_line, _ = calculate_macd(close, fast_p, slow_p, signal_p)
    signal = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    return signal.fillna(False)

SIGNAL_INTERPRETERS = {
    "rsi": interpret_rsi,
    "ma_crossover": interpret_ma_crossover,
    "price_dip": interpret_price_dip,
    "bollinger": interpret_bollinger,
    "macd": interpret_macd,
}

def simulate_trades(df: pd.DataFrame, entry_signal: pd.Series, params: dict) -> dict:
    """
    Universal trade simulation loop with parameter-driven hold period,
    stop loss, and take profit exit rules.
    """
    close_col = get_close_series(df)
    hold_days = max(1, int(params.get("hold_days", 5)))
    stop_loss_pct = params.get("stop_loss_pct")
    take_profit_pct = params.get("take_profit_pct")

    trades = []
    n = len(df)
    i = 0

    while i < n - 1:
        if bool(entry_signal.iloc[i]):
            entry_price = float(close_col.iloc[i])
            if entry_price <= 0:
                i += 1
                continue

            exit_price = None
            exit_idx = i
            max_exit = min(i + hold_days, n - 1)

            # Check day-by-day for stop loss / take profit triggers before hold_days expires
            for t in range(i + 1, max_exit + 1):
                curr_price = float(close_col.iloc[t])
                ret = (curr_price - entry_price) / entry_price

                if stop_loss_pct is not None and ret <= -abs(float(stop_loss_pct)):
                    exit_price = curr_price
                    exit_idx = t
                    break
                if take_profit_pct is not None and ret >= abs(float(take_profit_pct)):
                    exit_price = curr_price
                    exit_idx = t
                    break

            if exit_price is None:
                exit_idx = max_exit
                exit_price = float(close_col.iloc[exit_idx])

            if exit_idx > i:
                trade_return = (exit_price - entry_price) / entry_price
                trades.append(trade_return)
                i = exit_idx  # Skip to trade exit day to prevent overlapping trades
            else:
                i += 1
        else:
            i += 1

    return compute_metrics(trades)

def simple_drop_and_hold(df: pd.DataFrame, drop_pct=0.05, hold_days=5):
    """Baseline Rule: buy after single-day drop >= drop_pct, sell after hold_days."""
    strategy = {
        "signal_type": "price_dip",
        "params": {"dip_pct": drop_pct, "hold_days": hold_days}
    }
    return evaluate_strategy(df, strategy)

def evaluate_strategy(df: pd.DataFrame, hypothesis: dict) -> dict:
    """
    Dynamically interprets a parameterized strategy JSON and backtests it.
    """
    strategy = sanitize_strategy(hypothesis)
    signal_type = strategy["signal_type"]
    params = strategy["params"]

    interpreter = SIGNAL_INTERPRETERS.get(signal_type, interpret_price_dip)
    entry_signal = interpreter(df, params)

    if strategy.get("combine_with") and isinstance(strategy["combine_with"], dict):
        comb = strategy["combine_with"]
        c_type = comb.get("signal_type")
        c_params = comb.get("params", {})
        if c_type in SIGNAL_INTERPRETERS:
            sec_signal = SIGNAL_INTERPRETERS[c_type](df, c_params)
            entry_signal = entry_signal & sec_signal

    return simulate_trades(df, entry_signal, params)

def compute_metrics(trades: list[float]) -> dict:
    """
    Computes summary backtest metrics: number of trades, win rate, max drawdown, and Sharpe ratio.
    Flags strategies with fewer than MIN_TRADES as insufficient_sample.
    """
    if not trades:
        return {
            "num_trades": 0, "win_rate": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
            "insufficient_sample": True,
        }

    trades_arr = np.array(trades)
    win_rate = float((trades_arr > 0).mean())

    equity = np.cumprod(1 + trades_arr)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min())

    std_dev = trades_arr.std()
    sharpe = float((trades_arr.mean() / std_dev) * np.sqrt(len(trades_arr))) if std_dev > 0 else 0.0

    insufficient = len(trades_arr) < MIN_TRADES

    return {
        "num_trades": len(trades_arr),
        "win_rate": round(win_rate, 3),
        "max_drawdown": round(max_drawdown, 3),
        "sharpe": round(sharpe, 3),
        "insufficient_sample": insufficient,
    }

if __name__ == "__main__":
    from data_layer.fetch_prices import get_price_data, split_data
    df = get_price_data("TCS.NS")
    
    train, test = split_data(df)
    print(f"Train: {len(train)} rows | Test: {len(test)} rows")
    
    metrics_train = simple_drop_and_hold(train)
    print("Baseline Drop & Hold Train Metrics:", metrics_train)
    
    # Test all 5 signal interpreters
    for s_type in ["rsi", "ma_crossover", "price_dip", "bollinger", "macd"]:
        sample_strat = {
            "rule_name": f"Test {s_type.upper()}",
            "signal_type": s_type,
            "params": DEFAULT_PARAMS[s_type]
        }
        res = evaluate_strategy(train, sample_strat)
        print(f"Signal {s_type:12s} Train Metrics:", res)
