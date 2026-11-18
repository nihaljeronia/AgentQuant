import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import REFINEMENT_MODEL  # noqa: E402
from schema import SCHEMA_PROMPT_TEMPLATE, sanitize_strategy  # noqa: E402
from agents.llm import invoke_json  # noqa: E402

# Refinement agent uses gpt-4o-mini (REFINEMENT_MODEL) for cost efficiency during iteration.
# The client is created lazily in agents/llm.py.

PROMPT = """Your previous trading strategy proposal was judged as needing refinement.

Previous strategy JSON:
{hypothesis_json}

Critique feedback:
{critique_reasoning}

Propose a REVISED strategy proposal that directly addresses the critique. You may adjust parameters (e.g. thresholds, periods, hold days, stop loss / take profit) or switch to a better signal type.

{schema_template}
"""

def refine(hypothesis: dict, critique_reasoning: str) -> dict:
    msg = PROMPT.format(
        hypothesis_json=json.dumps(hypothesis, indent=2),
        critique_reasoning=critique_reasoning,
        schema_template=SCHEMA_PROMPT_TEMPLATE
    )
    raw_data = invoke_json(REFINEMENT_MODEL, 0.7, msg)
    return sanitize_strategy(raw_data)

if __name__ == "__main__":
    sample_hypo = {
        "rule_name": "RSI Oversold 30",
        "signal_type": "rsi",
        "params": {"rsi_period": 14, "rsi_entry_threshold": 30, "hold_days": 5},
        "reasoning": "Standard RSI 30 oversold buy"
    }
    r = refine(sample_hypo, "Sample size too small (only 12 trades). Try relaxing threshold to 35 or increasing period.")
    print("Refinement Result:", json.dumps(r, indent=2))
