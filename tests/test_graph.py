"""Runs the real LangGraph loop end to end with the OpenAI call replaced by a stub."""
import json

import agents.critique_agent as critique_agent
import agents.hypothesis_agent as hypothesis_agent
import agents.refinement_agent as refinement_agent
import graph
from data_layer.fetch_prices import split_data
from memory.log_store import init_db


def test_loop_runs_and_logs(prices, monkeypatch):
    proposals = iter([
        {"rule_name": "RSI", "signal_type": "rsi", "params": {"rsi_entry_threshold": 30, "hold_days": 5}},
        {"rule_name": "Dip", "signal_type": "price_dip", "params": {"dip_pct": 0.01, "hold_days": 3}},
    ] * 10)
    monkeypatch.setattr(hypothesis_agent, "invoke_json", lambda *a, **k: next(proposals))
    monkeypatch.setattr(refinement_agent, "invoke_json", lambda *a, **k: next(proposals))
    monkeypatch.setattr(critique_agent, "invoke_json",
                        lambda *a, **k: {"decision": "refine", "reasoning": "needs work"})

    events = []
    train, _ = split_data(prices)
    res = graph.run_research_loop("TEST.NS", df_train=train, on_event=lambda e, p: events.append(e))

    assert res["iteration"] == graph.MAX_ITERATIONS
    assert res["decision"] == "refine"
    assert events.count("backtest") == graph.MAX_ITERATIONS
    conn = init_db()
    n = conn.execute("SELECT COUNT(*) FROM attempts WHERE run_id=?", (res["run_id"],)).fetchone()[0]
    run = conn.execute("SELECT final_decision, iterations FROM runs WHERE run_id=?", (res["run_id"],)).fetchone()
    conn.close()
    assert n == graph.MAX_ITERATIONS
    assert run == ("refine", graph.MAX_ITERATIONS)


def test_critique_fallback_on_bad_reply(monkeypatch):
    from agents.llm import LLMError

    def boom(*a, **k):
        raise LLMError("bad json")

    monkeypatch.setattr(critique_agent, "invoke_json", boom)
    out = critique_agent.critique({}, {"num_trades": 30})
    assert out["decision"] == "refine"
