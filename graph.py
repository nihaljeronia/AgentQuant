from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Callable
import argparse
import json
import uuid

from data_layer.fetch_prices import get_price_data
from backtest.engine import evaluate_strategy
from agents.hypothesis_agent import propose_hypothesis
from agents.critique_agent import critique
from agents.refinement_agent import refine
from memory.log_store import init_db, log_attempt, start_run, finish_run, prior_attempts_summary
from memory.vector_store import store_hypothesis, find_similar

MAX_ITERATIONS = 8

# Per-run pre-loaded DataFrames for the research loop, keyed by run_id.
# This prevents the loop from calling get_price_data() (which returns the full dataset)
# and instead forces it to use only the train split passed by the caller.
_RUN_DATA: dict = {}       # {run_id: DataFrame}
_RUN_CALLBACKS: dict = {}  # {run_id: callable(event, payload)} for live dashboards


def _research_df(state):
    df = _RUN_DATA.get(state.get("run_id"))
    if df is not None:
        return df
    return get_price_data(state["ticker"])


def _emit(state, event: str, payload: dict):
    cb = _RUN_CALLBACKS.get(state.get("run_id"))
    if cb is not None:
        try:
            cb(event, {"iteration": state.get("iteration"), **payload})
        except Exception as e:
            print(f"[Warning] progress callback failed: {e}")


class ResearchState(TypedDict):
    run_id: str
    ticker: str
    hypothesis: dict
    metrics: dict
    decision: str
    reasoning: str
    iteration: int
    history: list

def node_hypothesize(state: ResearchState):
    if state["iteration"] == 0:
        prior_parts = []
        # Structured memory: best and worst earlier attempts on this ticker, with their metrics
        try:
            conn = init_db()
            summary = prior_attempts_summary(conn, state["ticker"], exclude_run=state.get("run_id"))
            conn.close()
            if summary:
                prior_parts.append(summary)
        except Exception as e:
            print(f"[Warning] Could not read SQLite memory: {e}")

        # Semantic memory: similar strategy ideas for this ticker from ChromaDB
        try:
            similar = find_similar(f"Trading strategy for {state['ticker']}", n_results=3,
                                   ticker=state["ticker"], exclude_run=state.get("run_id"))
            if similar and isinstance(similar, dict) and "documents" in similar:
                flat_docs = [doc for sublist in similar["documents"] if sublist for doc in sublist]
                prior_parts.extend(flat_docs)
        except Exception as e:
            print(f"[Warning] Could not read ChromaDB memory: {e}")

        prior_context = "\n".join(prior_parts)
        h = propose_hypothesis(state["ticker"], prior_context=prior_context)
    else:
        h = state["hypothesis"]  # updated by refine node
    
    # --- Per-iteration verbose logging ---
    print(f"  [Iter {state['iteration']}] Hypothesis: {h.get('rule_name', 'N/A')} "
          f"({h.get('signal_type', 'N/A')}: {h.get('params', {})})")
    if state["iteration"] == 0:
        _emit(state, "hypothesis", {"hypothesis": h})
    return {"hypothesis": h}

def node_backtest(state: ResearchState):
    # Use the train split for this run if provided, otherwise full data
    df = _research_df(state)
    
    # Dynamic strategy backtest evaluation
    metrics = evaluate_strategy(df, state["hypothesis"])
    
    # --- Per-iteration verbose logging (Task 3) ---
    print(f"  [Iter {state['iteration']}] Backtest: trades={metrics.get('num_trades')}, "
          f"win={metrics.get('win_rate')}, sharpe={metrics.get('sharpe')}, "
          f"dd={metrics.get('max_drawdown')}, insuff={metrics.get('insufficient_sample')}")
    _emit(state, "backtest", {"metrics": metrics})
    return {"metrics": metrics}

def node_critique(state: ResearchState):
    result = critique(state["hypothesis"], state["metrics"])
    
    # --- Internal Generalization Check (Phase 9 Task 2) ---
    if result["decision"] == "accept":
        df_research = _research_df(state)

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
        log_attempt(conn, state["ticker"], history_entry, run_id=state.get("run_id"))
        conn.close()
    except Exception as e:
        print(f"[Warning] Failed to log attempt to SQLite: {e}")

    # Persistent ChromaDB vector embedding storage
    try:
        hypo_id = f"{state.get('run_id', 'run')}_iter_{state['iteration']}"
        m = state["metrics"]
        hypo_text = (f"Rule: {state['hypothesis'].get('rule_name')}. "
                     f"Type: {state['hypothesis'].get('signal_type')}. "
                     f"Params: {json.dumps(state['hypothesis'].get('params'))}. "
                     f"Result: trades={m.get('num_trades')}, sharpe={m.get('sharpe')}, "
                     f"win={m.get('win_rate')}, dd={m.get('max_drawdown')}, decision={result['decision']}. "
                     f"Reasoning: {state['hypothesis'].get('reasoning')}")
        store_hypothesis(hypo_id, hypo_text, {"ticker": state["ticker"], "decision": result["decision"],
                                              "run_id": state.get("run_id", "")})
    except Exception as e:
        print(f"[Warning] Failed to log attempt to ChromaDB: {e}")

    _emit(state, "critique", {"decision": result["decision"], "reasoning": result["reasoning"]})

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
    _emit(state, "refine", {"hypothesis": new_hypothesis})
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

def run_research_loop(ticker: str = "TCS.NS", df_train=None,
                      on_event: Optional[Callable[[str, dict], None]] = None,
                      run_id: Optional[str] = None):
    """
    Run the full research loop for a given ticker.

    Args:
        ticker: Stock ticker symbol.
        df_train: Optional pre-split train DataFrame. If provided, the loop uses ONLY
                  this data for backtesting — the validation window is never seen.
                  If None, uses the full dataset from get_price_data().
        on_event: Optional callback(event, payload) for live progress
                  (events: hypothesis, backtest, critique, refine).
        run_id: Optional id for this run; generated if not given.
    """
    run_id = run_id or f"{ticker.replace('.', '_')}_{uuid.uuid4().hex[:8]}"
    if df_train is not None:
        _RUN_DATA[run_id] = df_train
    if on_event is not None:
        _RUN_CALLBACKS[run_id] = on_event

    print(f"=== Starting AgentQuant Research Loop for {ticker} (run {run_id}) ===")
    if df_train is not None:
        print(f"    (Using train split: {len(df_train)} rows, "
              f"{df_train.index[0].date()} to {df_train.index[-1].date()})")

    try:
        conn = init_db()
        start_run(conn, run_id, ticker,
                  df_train.index[0].date() if df_train is not None else None,
                  df_train.index[-1].date() if df_train is not None else None)
        conn.close()
    except Exception as e:
        print(f"[Warning] Failed to register run in SQLite: {e}")

    initial_state = {
        "run_id": run_id,
        "ticker": ticker,
        "hypothesis": {},
        "metrics": {},
        "decision": "",
        "reasoning": "",
        "iteration": 0,
        "history": [],
    }
    try:
        result = app.invoke(initial_state, {"recursion_limit": 4 * MAX_ITERATIONS + 10})
    finally:
        # Clean up per-run state
        _RUN_DATA.pop(run_id, None)
        _RUN_CALLBACKS.pop(run_id, None)

    try:
        conn = init_db()
        finish_run(conn, run_id, result)
        conn.close()
    except Exception as e:
        print(f"[Warning] Failed to finalize run in SQLite: {e}")

    print(f"\nFinal decision after {result['iteration']} iteration(s):")
    print("Hypothesis:", result["hypothesis"])
    print("Metrics:", result["metrics"])
    print("Decision:", result["decision"])
    return result


if __name__ == "__main__":
    from data_layer.fetch_prices import split_data

    parser = argparse.ArgumentParser(description="Run the AgentQuant research loop for one ticker.")
    parser.add_argument("--ticker", default="TCS.NS")
    parser.add_argument("--full-data", action="store_true",
                        help="Research on the full history (default: train split only, validation held out)")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    args = parser.parse_args()

    df_all = get_price_data(args.ticker)
    if args.full_data:
        run_research_loop(args.ticker)
    else:
        train, test = split_data(df_all, args.train_ratio)
        res = run_research_loop(args.ticker, df_train=train)
        val = evaluate_strategy(test, res["hypothesis"])
        print("Validation (held-out) metrics:", val)
        try:
            from memory.log_store import save_validation
            conn = init_db()
            save_validation(conn, res["run_id"], val)
            conn.close()
        except Exception as e:
            print(f"[Warning] Failed to save validation metrics: {e}")
