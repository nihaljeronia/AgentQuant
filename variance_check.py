"""
Task 6 — Run-to-Run Variance Check

Runs the full research loop multiple times on the same ticker with the same
train/test split but fresh LLM calls each time, to measure whether the
measured loop-vs-baseline improvement exceeds normal run-to-run noise.
"""
import pandas as pd
import numpy as np
from data_layer.fetch_prices import get_price_data, split_data
from backtest.engine import evaluate_strategy
from main_baseline import run_baseline
from graph import run_research_loop

# Tickers to test variance on (subset of the full evaluation set)
VARIANCE_TICKERS = ["TCS.NS", "RELIANCE.NS", "HDFCBANK.NS"]
RUNS_PER_TICKER = 3
TRAIN_RATIO = 0.70


def run_variance_check():
    print("=" * 60)
    print("  AGENTQUANT — RUN-TO-RUN VARIANCE CHECK (Task 6)")
    print("=" * 60)
    
    all_rows = []
    
    for ticker in VARIANCE_TICKERS:
        print(f"\n{'=' * 50}")
        print(f"Ticker: {ticker} — {RUNS_PER_TICKER} runs")
        print(f"{'=' * 50}")
        
        df_full = get_price_data(ticker)
        train_df, test_df = split_data(df_full, train_ratio=TRAIN_RATIO)
        
        # Single baseline run (deterministic on data, stochastic on LLM proposal)
        b_hypo, b_research = run_baseline(ticker, df=train_df)
        b_valid = evaluate_strategy(test_df, b_hypo)
        
        for run_idx in range(RUNS_PER_TICKER):
            print(f"\n  --- Run {run_idx + 1}/{RUNS_PER_TICKER} ---")
            l_result = run_research_loop(ticker, df_train=train_df)
            l_research = l_result.get("metrics", {})
            l_valid = evaluate_strategy(test_df, l_result.get("hypothesis", {}))
            
            all_rows.append({
                "ticker": ticker,
                "run": run_idx + 1,
                "baseline_valid_sharpe": b_valid.get("sharpe"),
                "loop_valid_sharpe": l_valid.get("sharpe"),
                "loop_valid_win": l_valid.get("win_rate"),
                "loop_valid_dd": l_valid.get("max_drawdown"),
                "loop_valid_trades": l_valid.get("num_trades"),
                "loop_iterations": l_result.get("iteration"),
                "loop_decision": l_result.get("decision"),
                "final_rule": l_result.get("hypothesis", {}).get("rule_name", "N/A"),
            })
    
    df = pd.DataFrame(all_rows)
    
    # --- Per-ticker variance summary ---
    print("\n" + "=" * 60)
    print("VARIANCE REPORT")
    print("=" * 60)
    
    for ticker in VARIANCE_TICKERS:
        subset = df[df["ticker"] == ticker]
        sharpes = subset["loop_valid_sharpe"].values
        baseline = subset["baseline_valid_sharpe"].iloc[0]
        
        print(f"\n  {ticker}:")
        print(f"    Baseline validation Sharpe:   {baseline:.3f}")
        print(f"    Loop validation Sharpes:      {list(sharpes)}")
        print(f"    Mean:  {np.mean(sharpes):.3f}")
        print(f"    Std:   {np.std(sharpes):.3f}")
        print(f"    Min:   {np.min(sharpes):.3f}")
        print(f"    Max:   {np.max(sharpes):.3f}")
        print(f"    Spread (max-min):             {np.max(sharpes) - np.min(sharpes):.3f}")
        gap = np.mean(sharpes) - baseline
        print(f"    Mean loop - baseline gap:     {gap:.3f}")
        if np.std(sharpes) > 0:
            signal_to_noise = abs(gap) / np.std(sharpes)
            print(f"    Signal-to-noise ratio:        {signal_to_noise:.2f}")
        else:
            print(f"    Signal-to-noise ratio:        N/A (zero variance)")
    
    # Save
    df.to_csv("variance_report_v2.csv", index=False)
    print(f"\nSaved to variance_report_v2.csv")

if __name__ == "__main__":
    run_variance_check()
