"""Renders the Research runs page from a seeded database (no API key or network needed)."""
import json
import os

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from config import PRICE_CACHE_DIR, ensure_dirs
from memory.log_store import init_db

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")
STRATEGY = {"rule_name": "RSI", "signal_type": "rsi", "params": {"rsi_entry_threshold": 30, "hold_days": 5}}


def seed_run(run_id, started_at, decision, loop_sharpe, baseline_sharpe):
    metrics = {"sharpe": 0.4, "num_trades": 7, "win_rate": 0.5, "max_drawdown": -0.05, "insufficient_sample": True}
    validation = {**metrics, "sharpe": loop_sharpe}
    baseline = {"hypothesis": STRATEGY, "research": metrics, "validation": {**metrics, "sharpe": baseline_sharpe}}
    conn = init_db()
    conn.execute(
        "INSERT INTO runs (run_id, ticker, started_at, train_end, iterations, final_decision, final_hypothesis, "
        "final_metrics, validation_metrics, baseline) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (run_id, "APP.NS", started_at, "2024-06-28", 3, decision, json.dumps(STRATEGY), json.dumps(metrics),
         json.dumps(validation), json.dumps(baseline)),
    )
    for i in range(3):
        conn.execute(
            "INSERT INTO attempts (ticker, iteration, hypothesis, metrics, decision, reasoning, run_id) "
            "VALUES (?,?,?,?,?,?,?)",
            ("APP.NS", i, json.dumps(STRATEGY), json.dumps(metrics), decision if i == 2 else "refine", "why", run_id),
        )
    conn.commit()
    conn.close()


@pytest.mark.parametrize("run_id, started_at, decision, loop, base, badge, note", [
    ("app_better", "2099-01-01T00:00:00", "refine", 0.5, -0.5, "LOOP BETTER", "not met within 3 iterations"),
    ("app_worse", "2099-01-02T00:00:00", "accept", -0.5, 0.5, "LOOP WORSE", "criteria: met"),
])
def test_run_page_headline_and_badge(prices, run_id, started_at, decision, loop, base, badge, note):
    ensure_dirs()
    prices.to_csv(PRICE_CACHE_DIR / "APP.NS_3y.csv")
    seed_run(run_id, started_at, decision, loop, base)
    st.cache_data.clear()

    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.radio[0].set_value("Research runs").run()

    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Final strategy after 2 refinements" in text
    assert badge in text
    assert "NOT ACCEPTED" not in text
    assert any(note in c.value for c in at.caption)
