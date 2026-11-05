from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import json

load_dotenv()

# --- Acceptance Thresholds ---
# These are the minimum bars a strategy must clear for the Critique Agent to accept it.
# Chosen as reasonable starting points; easy to tune by changing these constants.
# If the loop still never accepts after tuning, that itself is a finding worth reporting.
ACCEPT_MIN_TRADES = 20       # Minimum trade count for statistical significance
ACCEPT_MIN_SHARPE = 0.5      # Minimum Sharpe ratio for risk-adjusted quality
ACCEPT_MAX_DRAWDOWN = -0.25  # Maximum allowable drawdown (must be >= this, i.e. not worse)
ACCEPT_MIN_WIN_RATE = 0.40   # Minimum win rate (40%)

# Critique agent uses full gpt-4o for high reasoning quality on evaluation
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    model_kwargs={"response_format": {"type": "json_object"}},
)

PROMPT = """You are reviewing a backtested trading strategy result.

Strategy: {hypothesis}
Backtest metrics: {metrics}

You MUST apply these concrete acceptance thresholds — accept ONLY if ALL are met:
1. num_trades >= {min_trades} (statistical significance — fewer trades means the metrics are unreliable)
2. sharpe >= {min_sharpe} (risk-adjusted return quality)
3. max_drawdown >= {max_drawdown} (drawdown must not be worse than this; note max_drawdown is negative)
4. win_rate >= {min_win_rate} (minimum win rate)

If the strategy has "insufficient_sample": true in its metrics, you MUST return "refine" regardless of other metrics.

If ALL thresholds are met, return "accept". If ANY threshold is NOT met, return "refine" and explain which specific thresholds failed and by how much.

Respond ONLY with JSON:
{{"decision": "accept" or "refine", "reasoning": "..."}}
"""

def critique(hypothesis: dict, metrics: dict) -> dict:
    msg = PROMPT.format(
        hypothesis=json.dumps(hypothesis),
        metrics=json.dumps(metrics),
        min_trades=ACCEPT_MIN_TRADES,
        min_sharpe=ACCEPT_MIN_SHARPE,
        max_drawdown=ACCEPT_MAX_DRAWDOWN,
        min_win_rate=ACCEPT_MIN_WIN_RATE,
    )
    response = llm.invoke([HumanMessage(content=msg)])
    text = str(response.content).strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(text)

if __name__ == "__main__":
    # Test case that should be accepted (meets all thresholds)
    c1 = critique(
        {"rule_name": "RSI Dip Strategy"},
        {"num_trades": 35, "win_rate": 0.55, "sharpe": 0.8, "max_drawdown": -0.12, "insufficient_sample": False}
    )
    print("Should ACCEPT:", c1)
    
    # Test case that should be rejected (insufficient sample)
    c2 = critique(
        {"rule_name": "MA Crossover"},
        {"num_trades": 7, "win_rate": 0.857, "sharpe": 1.838, "max_drawdown": -0.03, "insufficient_sample": True}
    )
    print("Should REFINE (insufficient sample):", c2)
