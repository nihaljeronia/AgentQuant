"""
AgentQuant Strategy Schema Definition

Defines the contract between LLM agents (Hypothesis & Refinement) and the Backtest Engine.
"""

ALLOWED_SIGNAL_TYPES = ["rsi", "ma_crossover", "price_dip", "bollinger", "macd"]

# Standard parameter defaults by signal type
DEFAULT_PARAMS = {
    "rsi": {
        "rsi_period": 14,
        "rsi_entry_threshold": 30,
        "rsi_exit_threshold": 70,
        "hold_days": 5,
    },
    "ma_crossover": {
        "ma_fast": 20,
        "ma_slow": 50,
        "hold_days": 10,
    },
    "price_dip": {
        "dip_pct": 0.05,
        "hold_days": 5,
    },
    "bollinger": {
        "bollinger_period": 20,
        "bollinger_std": 2.0,
        "hold_days": 5,
    },
    "macd": {
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "hold_days": 5,
    }
}

SCHEMA_PROMPT_TEMPLATE = """
Strategy Proposals MUST be formatted strictly as JSON matching this schema:

{
  "rule_name": "<Short descriptive name>",
  "signal_type": "<rsi | ma_crossover | price_dip | bollinger | macd>",
  "params": {
    // Include ONLY parameters relevant to signal_type:
    // For "rsi": "rsi_period" (e.g. 14), "rsi_entry_threshold" (e.g. 30), "rsi_exit_threshold" (e.g. 70), "hold_days" (e.g. 5), "stop_loss_pct" (optional, e.g. 0.03), "take_profit_pct" (optional, e.g. 0.08)
    // For "ma_crossover": "ma_fast" (e.g. 20), "ma_slow" (e.g. 50), "hold_days" (e.g. 10), "stop_loss_pct" (optional), "take_profit_pct" (optional)
    // For "price_dip": "dip_pct" (e.g. 0.05 for 5%), "hold_days" (e.g. 5), "stop_loss_pct" (optional), "take_profit_pct" (optional)
    // For "bollinger": "bollinger_period" (e.g. 20), "bollinger_std" (e.g. 2.0), "hold_days" (e.g. 5), "stop_loss_pct" (optional), "take_profit_pct" (optional)
    // For "macd": "macd_fast" (e.g. 12), "macd_slow" (e.g. 26), "macd_signal" (e.g. 9), "hold_days" (e.g. 5), "stop_loss_pct" (optional), "take_profit_pct" (optional)
  },
  "reasoning": "<Explanation of strategy logic and why parameters were chosen>"
}
"""

def sanitize_strategy(strategy: dict) -> dict:
    """Ensure strategy dictionary has required structure and reasonable fallback values."""
    if not isinstance(strategy, dict):
        strategy = {}
    
    signal_type = str(strategy.get("signal_type", "price_dip")).lower()
    if signal_type not in ALLOWED_SIGNAL_TYPES:
        signal_type = "price_dip"
        
    params = strategy.get("params", {})
    if not isinstance(params, dict):
        params = {}
        
    defaults = DEFAULT_PARAMS.get(signal_type, DEFAULT_PARAMS["price_dip"]).copy()
    
    # Merge existing params over defaults
    merged_params = {}
    for k, v in defaults.items():
        merged_params[k] = params.get(k, v)
        
    # Copy optional exit parameters if present
    for opt in ["stop_loss_pct", "take_profit_pct"]:
        if opt in params and params[opt] is not None:
            try:
                merged_params[opt] = float(params[opt])
            except (ValueError, TypeError):
                pass

    return {
        "rule_name": strategy.get("rule_name", f"{signal_type.upper()} Strategy"),
        "signal_type": signal_type,
        "params": merged_params,
        "reasoning": strategy.get("reasoning", ""),
        "combine_with": strategy.get("combine_with", None)
    }
