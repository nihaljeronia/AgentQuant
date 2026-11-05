import pandas as pd
import json
from data_layer.fetch_prices import get_price_data, split_data
from backtest.engine import evaluate_strategy
from main_baseline import run_baseline
from graph import run_research_loop

# --- Expanded Ticker Set ---
TICKERS = [
    # IT
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS",
    # Energy / Industrial
    "RELIANCE.NS", "ONGC.NS", "NTPC.NS",
    # Banking / Finance
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
    # Consumer / Pharma
    "HINDUNILVR.NS", "SUNPHARMA.NS",
    # Auto / Metals
    "TATAMOTORS.NS", "TATASTEEL.NS", "MARUTI.NS",
]

TRAIN_RATIO = 0.70  # 70% research window, 30% validation window

# Overfitting flag threshold: if validation Sharpe drops by more than 50% of research Sharpe
OVERFIT_SHARPE_DROP = 0.5

# Materiality threshold (Phase 9 Task 1): Sharpe delta must be >= 0.1 over baseline to count as genuine win
MIN_MEANINGFUL_DELTA = 0.1


def run_evaluation():
    """
    Phase 9 evaluation: Train/Test split, research vs validation scoring,
    internal generalization check logging, degenerate strategy exclusion,
    and aggregate statistics exported to evaluation_summary_v4.csv.
    """
    print("=" * 60)
    print("  AGENTQUANT PHASE 9 — RESULT INTEGRITY & ACCEPT-GATE SUITE")
    print("=" * 60)
    
    results = []
    
    for ticker in TICKERS:
        print(f"\n{'=' * 50}")
        print(f"[Evaluation] Processing Ticker: {ticker}")
        print(f"{'=' * 50}")
        
        # --- Fetch full data and split chronologically ---
        try:
            df_full = get_price_data(ticker)
        except Exception as e:
            print(f"[ERROR] Failed to fetch data for {ticker}: {e}")
            continue
        
        if len(df_full) < 100:
            print(f"[SKIP] {ticker}: insufficient data ({len(df_full)} rows)")
            continue
            
        train_df, test_df = split_data(df_full, train_ratio=TRAIN_RATIO)
        print(f"  Train: {len(train_df)} rows ({train_df.index[0].date()} to {train_df.index[-1].date()})")
        print(f"  Test:  {len(test_df)} rows ({test_df.index[0].date()} to {test_df.index[-1].date()})")
        
        # --- Baseline: propose on train, score on both ---
        print(f"\n  [Baseline] Single-shot proposal...")
        b_hypo, b_research_metrics = run_baseline(ticker, df=train_df)
        b_validation_metrics = evaluate_strategy(test_df, b_hypo)
        print(f"  [Baseline] Research metrics: {b_research_metrics}")
        print(f"  [Baseline] Validation metrics: {b_validation_metrics}")
        
        # --- Loop: iterate on train only, then score final strategy on test ---
        print(f"\n  [Loop] Starting iterative research loop on TRAIN data only...")
        l_result = run_research_loop(ticker, df_train=train_df)
        l_research_metrics = l_result.get("metrics", {})
        l_validation_metrics = evaluate_strategy(test_df, l_result.get("hypothesis", {}))
        print(f"  [Loop] Research metrics: {l_research_metrics}")
        print(f"  [Loop] Validation metrics: {l_validation_metrics}")
        
        # --- Overfitting detection ---
        r_sharpe = l_research_metrics.get("sharpe", 0)
        v_sharpe = l_validation_metrics.get("sharpe", 0)
        overfit_flag = False
        if r_sharpe > 0 and v_sharpe < r_sharpe * OVERFIT_SHARPE_DROP:
            overfit_flag = True
        
        # --- Degenerate strategy detection (Phase 9 Task 1) ---
        # Flagged if insufficient_sample on EITHER window or total trades == 0
        r_trades = l_research_metrics.get("num_trades", 0)
        v_trades = l_validation_metrics.get("num_trades", 0)
        r_insuff = l_research_metrics.get("insufficient_sample", True)
        v_insuff = l_validation_metrics.get("insufficient_sample", True)
        
        degenerate_flag = bool(r_insuff or v_insuff or r_trades == 0 or v_trades == 0)

        results.append({
            "ticker": ticker,
            # Baseline - research window
            "b_research_win": b_research_metrics.get("win_rate"),
            "b_research_sharpe": b_research_metrics.get("sharpe"),
            "b_research_dd": b_research_metrics.get("max_drawdown"),
            "b_research_trades": b_research_metrics.get("num_trades"),
            # Baseline - validation window
            "b_valid_win": b_validation_metrics.get("win_rate"),
            "b_valid_sharpe": b_validation_metrics.get("sharpe"),
            "b_valid_dd": b_validation_metrics.get("max_drawdown"),
            "b_valid_trades": b_validation_metrics.get("num_trades"),
            # Loop - research window
            "l_research_win": l_research_metrics.get("win_rate"),
            "l_research_sharpe": l_research_metrics.get("sharpe"),
            "l_research_dd": l_research_metrics.get("max_drawdown"),
            "l_research_trades": l_research_metrics.get("num_trades"),
            # Loop - validation window
            "l_valid_win": l_validation_metrics.get("win_rate"),
            "l_valid_sharpe": l_validation_metrics.get("sharpe"),
            "l_valid_dd": l_validation_metrics.get("max_drawdown"),
            "l_valid_trades": l_validation_metrics.get("num_trades"),
            # Loop metadata
            "loop_iterations": l_result.get("iteration"),
            "loop_decision": l_result.get("decision"),
            "overfit_flag": overfit_flag,
            "degenerate_flag": degenerate_flag,
        })
    
    if not results:
        print("\n[ERROR] No results collected.")
        return
    
    df_eval = pd.DataFrame(results)
    
    # Sharpe delta: loop_validation - baseline_validation
    df_eval["sharpe_delta_valid"] = df_eval["l_valid_sharpe"] - df_eval["b_valid_sharpe"]
    df_eval["raw_loop_beats_baseline"] = df_eval["sharpe_delta_valid"] > 0
    df_eval["genuine_win"] = (df_eval["sharpe_delta_valid"] >= MIN_MEANINGFUL_DELTA) & (~df_eval["degenerate_flag"])
    
    # --- Print per-ticker results ---
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS — Per Ticker (Research vs Validation)")
    print("=" * 80)
    display_cols = [
        "ticker",
        "b_valid_sharpe", "l_valid_sharpe", "sharpe_delta_valid",
        "loop_iterations", "loop_decision", "overfit_flag", "degenerate_flag", "genuine_win"
    ]
    print(df_eval[display_cols].to_string(index=False))
    
    # --- Aggregate statistics ---
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS (Phase 9 Corrected)")
    print("=" * 80)
    
    mean_delta = df_eval["sharpe_delta_valid"].mean()
    std_delta = df_eval["sharpe_delta_valid"].std()
    raw_beat_pct = df_eval["raw_loop_beats_baseline"].mean() * 100
    genuine_beat_pct = df_eval["genuine_win"].mean() * 100
    n_degenerate = df_eval["degenerate_flag"].sum()
    n_overfit = df_eval["overfit_flag"].sum()
    
    print(f"  Tickers evaluated:                      {len(df_eval)}")
    print(f"  Mean Sharpe delta (loop - baseline):     {mean_delta:.3f}")
    print(f"  Std dev of Sharpe delta:                 {std_delta:.3f}")
    print(f"  Raw % tickers loop beat baseline (>0):   {raw_beat_pct:.1f}%")
    print(f"  Genuine % win (delta >= 0.1 & non-deg):  {genuine_beat_pct:.1f}%")
    print(f"  Strategies flagged as degenerate:       {n_degenerate}")
    print(f"  Strategies flagged as overfit:          {n_overfit}")
    print(f"  Loops that accepted early:              {(df_eval['loop_decision'] == 'accept').sum()}")
    print(f"  Loops that hit max iterations:          {(df_eval['loop_decision'] != 'accept').sum()}")
    
    # --- Save ---
    df_eval.to_csv("evaluation_summary_v4.csv", index=False)
    print(f"\nSaved full results to evaluation_summary_v4.csv")

if __name__ == "__main__":
    run_evaluation()
