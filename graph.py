from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
import json

from data_layer.fetch_prices import get_price_data
from backtest.engine import evaluate_strategy
from agents.hypothesis_agent import propose_hypothesis
from agents.critique_agent import critique
from agents.refinement_agent import refine
from memory.log_store import init_db, log_attempt
from memory.vector_store import store_hypothesis, find_similar

MAX_ITERATIONS = 8

# Module-level variable to hold a pre-loaded DataFrame for the research loop.
# This prevents the loop from calling get_price_data() (which returns the full dataset)
# and instead forces it to use only the train split passed by the caller.
_df_override: Optional[dict] = None  # {ticker: DataFrame}

class ResearchState(TypedDict):
    ticker: str
    hypothesis: dict
    metrics: dict
    decision: str
    reasoning: str
    iteration: int
    history: list

def node_hypothesize(state: ResearchState):
    if state["iteration"] == 0:
        # Retrieve past similar strategy ideas from ChromaDB memory
        similar = find_similar(state["ticker"], n_results=3)
        prior_context = ""
        if similar and isinstance(similar, dict) and "documents" in similar:
            flat_docs = [doc for sublist in similar["documents"] if sublist for doc in sublist]
            if flat_docs:
                prior_context = "\n".join(flat_docs)
        
        h = propose_hypothesis(state["ticker"], prior_context=prior_context)
    else:
        h = state["hypothesis"]  # updated by refine node
    
    # --- Per-iteration verbose logging ---
    print(f"  [Iter {state['iteration']}] Hypothesis: {h.get('rule_name', 'N/A')} "
          f"({h.get('signal_type', 'N/A')}: {h.get('params', {})})")
    return {"hypothesis": h}

def node_backtest(state: ResearchState):
    # Use the override DataFrame (train split) if provided, otherwise full data
    if _df_override and state["ticker"] in _df_override:
        df = _df_override[state["ticker"]]
    else:
        df = get_price_data(state["ticker"])
    
    # Dynamic strategy backtest evaluation
    metrics = evaluate_strategy(df, state["hypothesis"])
    
    # --- Per-iteration verbose logging (Task 3) ---
    print(f"  [Iter {state['iteration']}] Backtest: trades={metrics.get('num_trades')}, "
          f"win={metrics.get('win_rate')}, sharpe={metrics.get('sharpe')}, "
          f"dd={metrics.get('max_drawdown')}, insuff={metrics.get('insufficient_sample')}")
    return {"metrics": metrics}

def node_critique(state: ResearchState):
    result = critique(state["hypothesis"], state["metrics"])
    
    # --- Internal Generalization Check (Phase 9 Task 2) ---
    if result["decision"] == "accept":
        if _df_override and state["ticker"] in _df_override:
            df_research = _df_override[state["ticker"]]
        else:
            df_research = get_price_data(state["ticker"])
            
        n_research = len(df_research)
        split_idx = int(n_research * 0.80)
        df_inner_train = df_research.iloc[:split_idx]
        df_inner_check = df_research.iloc[split_idx:]
        
        m_inner_train = evaluate_strategy(df_inner_train, state["hypothesis"])
        m_inner_check = evaluate_strategy(df_inner_check, state["hypothesis"])
        
        sharpe_train = m_inner_train.get("sharpe", 0)
        sharpe_check = m_inner_check.get("sharpe", 0)
        
        gen_failed = False
        gen_reason = ""
        
        if sharpe_check < 0:
            gen_failed = True
            gen_reason = f"Inner-check Sharpe is negative ({sharpe_check:.3f})."
        elif sharpe_train > 0 and sharpe_check < 0.5 * sharpe_train:
            gen_failed = True
            gen_reason = f"Inner-check Sharpe ({sharpe_check:.3f}) degraded by >50% from inner-train ({sharpe_train:.3f})."
        elif m_inner_check.get("num_trades", 0) < 3:
            gen_failed = True
            gen_reason = f"Inner-check sample too small ({m_inner_check.get('num_trades', 0)} trades)."

        if gen_failed:
            print(f"  [Iter {state['iteration']}] Generalization Gate REJECT: {gen_reason}")
            result["decision"] = "refine"
            result["reasoning"] += f"\n[Internal Generalization Gate Rejected] Overfit in research window: {gen_reason} Forcing refinement."

    # --- Per-iteration verbose logging ---
    print(f"  [Iter {state['iteration']}] Critique: decision={result['decision']}")
    
    history_entry = {
        "iteration": state["iteration"],
        "hypothesis": state["hypothesis"],
        "metrics": state["metrics"],
        "decision": result["decision"],
        "reasoning": result["reasoning"],
    }
    
    # Persistent SQLite log recording
    try:
        conn = init_db()
        log_attempt(conn, state["ticker"], history_entry)
        conn.close()
    except Exception as e:
        print(f"[Warning] Failed to log attempt to SQLite: {e}")

    # Persistent ChromaDB vector embedding storage
    try:
        hypo_id = f"{state['ticker']}_iter_{state['iteration']}"
        hypo_text = (f"Rule: {state['hypothesis'].get('rule_name')}. "
                     f"Type: {state['hypothesis'].get('signal_type')}. "
                     f"Params: {json.dumps(state['hypothesis'].get('params'))}. "
                     f"Reasoning: {state['hypothesis'].get('reasoning')}")
        store_hypothesis(hypo_id, hypo_text, {"ticker": state["ticker"], "decision": result["decision"]})
    except Exception as e:
        print(f"[Warning] Failed to log attempt to ChromaDB: {e}")

    return {
        "decision": result["decision"],
        "reasoning": result["reasoning"],
        "history": state["history"] + [history_entry],
        "iteration": state["iteration"] + 1,
    }

def node_refine(state: ResearchState):
    new_hypothesis = refine(state["hypothesis"], state["reasoning"])
    # --- Per-iteration verbose logging ---
    print(f"  [Iter {state['iteration']}] Refined -> {new_hypothesis.get('rule_name', 'N/A')} "
          f"({new_hypothesis.get('signal_type', 'N/A')}: {new_hypothesis.get('params', {})})")
    return {"hypothesis": new_hypothesis}

def route_after_critique(state: ResearchState):
    if state["decision"] == "accept" or state["iteration"] >= MAX_ITERATIONS:
        return "end"
    return "refine"

# Construct State Graph
graph = StateGraph(ResearchState)
graph.add_node("hypothesize", node_hypothesize)
graph.add_node("backtest", node_backtest)
graph.add_node("critique", node_critique)
graph.add_node("refine", node_refine)

graph.set_entry_point("hypothesize")
graph.add_edge("hypothesize", "backtest")
graph.add_edge("backtest", "critique")
graph.add_conditional_edges("critique", route_after_critique, {"refine": "refine", "end": END})
graph.add_edge("refine", "backtest")

app = graph.compile()

def run_research_loop(ticker: str = "TCS.NS", df_train=None):
    """
    Run the full research loop for a given ticker.
    
    Args:
        ticker: Stock ticker symbol.
        df_train: Optional pre-split train DataFrame. If provided, the loop uses ONLY
                  this data for backtesting — the validation window is never seen.
                  If None, uses the full dataset from get_price_data().
    """
    global _df_override
    
    if df_train is not None:
        _df_override = {ticker: df_train}
    else:
        _df_override = None
    
    print(f"=== Starting AgentQuant Research Loop for {ticker} ===")
    if df_train is not None:
        print(f"    (Using train split: {len(df_train)} rows, "
              f"{df_train.index[0].date()} to {df_train.index[-1].date()})")
    
    initial_state = {
        "ticker": ticker,
        "hypothesis": {},
        "metrics": {},
        "decision": "",
        "reasoning": "",
        "iteration": 0,
        "history": [],
    }
    result = app.invoke(initial_state)
    
    # Clean up override
    _df_override = None
    
    print(f"\nFinal decision after {result['iteration']} iteration(s):")
    print("Hypothesis:", result["hypothesis"])
    print("Metrics:", result["metrics"])
    print("Decision:", result["decision"])
    return result

if __name__ == "__main__":
    run_research_loop("TCS.NS")
