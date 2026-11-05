from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from schema import SCHEMA_PROMPT_TEMPLATE, sanitize_strategy

load_dotenv()

# Strategy proposal agent using ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    model_kwargs={"response_format": {"type": "json_object"}},
)

PROMPT = """You are a quantitative strategy researcher. Propose ONE parameterized trading rule
for {ticker} based on a technical pattern (e.g. price drop, moving average crossover, RSI, Bollinger Bands, MACD).

{schema_template}

Ensure you choose ONE valid signal_type ("rsi", "ma_crossover", "price_dip", "bollinger", "macd") and populate its relevant params.
"""

def propose_hypothesis(ticker: str, prior_context: str = "") -> dict:
    prompt_text = PROMPT.format(ticker=ticker, schema_template=SCHEMA_PROMPT_TEMPLATE)
    if prior_context:
        prompt_text += f"\n\nPrior attempts and lessons learned to avoid repeating:\n{prior_context}"
    
    response = llm.invoke([HumanMessage(content=prompt_text)])
    text = str(response.content).strip().removeprefix("```json").removesuffix("```").strip()
    try:
        raw_data = json.loads(text)
    except json.JSONDecodeError:
        raw_data = {}
        
    return sanitize_strategy(raw_data)

if __name__ == "__main__":
    h = propose_hypothesis("TCS.NS")
    print("Hypothesis Proposal:", json.dumps(h, indent=2))
