from backtest.engine import evaluate_strategy, backtest_details, compute_metrics
from schema import DEFAULT_PARAMS


def test_details_match_metrics_for_every_signal(prices):
    for s_type, params in DEFAULT_PARAMS.items():
        strat = {"signal_type": s_type, "params": dict(params, stop_loss_pct=0.03, take_profit_pct=0.06)}
        m = evaluate_strategy(prices, strat)
        d = backtest_details(prices, strat)
        assert d["metrics"] == m
        assert len(d["trades"]) == m["num_trades"]


def test_trades_do_not_overlap_and_move_forward(prices):
    d = backtest_details(prices, {"signal_type": "rsi", "params": {"rsi_entry_threshold": 45, "hold_days": 5}})
    t = d["trades"]
    assert (t["exit_date"] > t["entry_date"]).all()
    assert (t["entry_date"].iloc[1:].values >= t["exit_date"].iloc[:-1].values).all()


def test_compute_metrics_empty_and_flag():
    assert compute_metrics([])["insufficient_sample"] is True
    m = compute_metrics([0.01] * 10 + [-0.005] * 15)
    assert m["num_trades"] == 25 and m["insufficient_sample"] is False
