import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import HYPOTHESIS_MODEL  # noqa: E402
from schema import SCHEMA_PROMPT_TEMPLATE, sanitize_strategy  # noqa: E402
from agents.llm import invoke_json  # noqa: E402

PROMPT = """You are a quantitative strategy researcher. Propose ONE parameterized trading rule
for {ticker} based on a technical pattern (e.g. price drop, moving average crossover, RSI, Bollinger Bands, MACD).

{schema_template}

Ensure you choose ONE valid signal_type ("rsi", "ma_crossover", "price_dip", "bollinger", "macd") and populate its relevant params.
"""


def build_prompt(ticker: str, prior_context: str = "") -> str:
    prompt_text = PROMPT.format(ticker=ticker, schema_template=SCHEMA_PROMPT_TEMPLATE)
    if prior_context:
        prompt_text += f"\n\nPrior attempts and lessons learned to avoid repeating:\n{prior_context}"
    return prompt_text


def propose_hypothesis(ticker: str, prior_context: str = "") -> dict:
    raw_data = invoke_json(HYPOTHESIS_MODEL, 0.7, build_prompt(ticker, prior_context))
    return sanitize_strategy(raw_data)


if __name__ == "__main__":
    h = propose_hypothesis("TCS.NS")
    print("Hypothesis Proposal:", json.dumps(h, indent=2))
