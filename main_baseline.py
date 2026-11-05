from data_layer.fetch_prices import get_price_data
from backtest.engine import evaluate_strategy
from agents.hypothesis_agent import propose_hypothesis

def run_baseline(ticker: str = "TCS.NS", df=None):
    """
    Run a single-shot baseline strategy proposal and backtest.
    
    Args:
        ticker: Stock ticker symbol.
        df: Optional pre-split DataFrame. If None, uses full data from get_price_data().
    """
    print(f"--- Running Baseline Single-Shot Pipeline for {ticker} ---")
    if df is None:
        df = get_price_data(ticker)
    
    hypothesis = propose_hypothesis(ticker)
    print("Proposed strategy:", hypothesis)

    result = evaluate_strategy(df, hypothesis)
    print("Backtest baseline result:", result)
    return hypothesis, result

if __name__ == "__main__":
    run_baseline("TCS.NS")
