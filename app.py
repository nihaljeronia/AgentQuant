"""
AgentQuant dashboard.

    streamlit run app.py

Pages:
  Overview      what the system does and what is in the research database
  Run research  run the LangGraph loop live on a ticker (OpenAI API) and see the result
  Research runs replay any recorded run: iterations, critique, trades and equity curves
  Evaluation    results of evaluate.py / variance_check.py (CSV files)
"""
import glob
import json
import os
import sqlite3
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import DB_PATH, DEFAULT_TICKERS, RESULTS_DIR, has_openai_key, HYPOTHESIS_MODEL, CRITIQUE_MODEL, REFINEMENT_MODEL  # noqa: E402
from backtest.engine import backtest_details, evaluate_strategy  # noqa: E402
from data_layer.fetch_prices import get_price_data, list_cached_tickers, split_data, PriceDataError  # noqa: E402
from agents.critique_agent import (  # noqa: E402
    ACCEPT_MIN_TRADES, ACCEPT_MIN_SHARPE, ACCEPT_MAX_DRAWDOWN, ACCEPT_MIN_WIN_RATE,
)

st.set_page_config(page_title="AgentQuant", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

ACCENT = "#4F46E5"
GOOD = "#16A34A"
BAD = "#DC2626"
MUTED = "#94A3B8"

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1300px;}
      .aq-hero h1 {margin-bottom: 0.2rem;}
      .aq-sub {color: #64748B; font-size: 1.05rem; margin-bottom: 1.2rem;}
      .aq-card {border: 1px solid rgba(148,163,184,0.35); border-radius: 12px; padding: 14px 16px; height: 100%;}
      .aq-card h4 {margin: 0 0 6px 0;}
      .aq-card p {margin: 0; color: #64748B; font-size: 0.92rem;}
      .aq-pill {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600;}
      .aq-accept {background: rgba(22,163,74,0.15); color: #16A34A;}
      .aq-refine {background: rgba(234,179,8,0.18); color: #B45309;}
    </style>
    """,
    unsafe_allow_html=True,
)

SHARPE_HELP = ("Per-trade score: mean / std of trade returns × √(number of trades). "
               "Not annualised; used for relative comparison.")


# ---------------------------------------------------------------- data access
def db_conn():
    if not os.path.exists(DB_PATH):
        return None
    from memory.log_store import init_db  # upgrades old databases (adds run_id)
    return init_db()


@st.cache_data(ttl=5)
def load_runs() -> pd.DataFrame:
    conn = db_conn()
    if conn is None:
        return pd.DataFrame()
    runs = pd.read_sql_query("SELECT * FROM runs ORDER BY started_at DESC", conn)
    # Older attempts recorded before run ids existed: one pseudo-run per ticker
    legacy = pd.read_sql_query(
        "SELECT ticker, COUNT(*) AS n, MAX(id) AS last_id FROM attempts WHERE run_id IS NULL GROUP BY ticker", conn)
    conn.close()
    rows = [r for _, r in runs.iterrows()]
    out = pd.DataFrame(rows) if rows else pd.DataFrame(columns=runs.columns)
    if not legacy.empty:
        leg = pd.DataFrame({
            "run_id": "legacy:" + legacy["ticker"],
            "ticker": legacy["ticker"],
            "started_at": None,
            "iterations": legacy["n"],
            "final_decision": None,
        })
        out = pd.concat([out, leg], ignore_index=True)
    return out


@st.cache_data(ttl=5)
def load_attempts(run_id: str) -> pd.DataFrame:
    conn = db_conn()
    if conn is None:
        return pd.DataFrame()
    if run_id.startswith("legacy:"):
        df = pd.read_sql_query("SELECT * FROM attempts WHERE run_id IS NULL AND ticker=? ORDER BY id",
                               conn, params=(run_id.split(":", 1)[1],))
    else:
        df = pd.read_sql_query("SELECT * FROM attempts WHERE run_id=? ORDER BY iteration, id", conn, params=(run_id,))
    conn.close()
    if not df.empty:
        df["hypothesis"] = df["hypothesis"].apply(json.loads)
        df["metrics"] = df["metrics"].apply(json.loads)
    return df


@st.cache_data(show_spinner=False)
def load_prices(ticker: str) -> pd.DataFrame:
    return get_price_data(ticker)


def db_stats():
    conn = db_conn()
    if conn is None:
        return 0, 0, 0, 0
    a = conn.execute("SELECT COUNT(*), COUNT(DISTINCT ticker) FROM attempts").fetchone()
    r = conn.execute("SELECT COUNT(*), SUM(final_decision='accept') FROM runs").fetchone()
    conn.close()
    return r[0] or 0, a[0] or 0, a[1] or 0, r[1] or 0


# ---------------------------------------------------------------- small UI helpers
def decision_pill(decision: str | None) -> str:
    if decision == "accept":
        return '<span class="aq-pill aq-accept">ACCEPTED</span>'
    if decision:
        return '<span class="aq-pill aq-refine">NOT ACCEPTED (max iterations)</span>'
    return '<span class="aq-pill aq-refine">IN PROGRESS / UNKNOWN</span>'


def fmt_pct(x):
    return "–" if x is None else f"{x * 100:.1f}%"


def metric_row(m: dict, label: str):
    st.markdown(f"**{label}**")
    c = st.columns(4)
    c[0].metric("Sharpe", m.get("sharpe", "–"), help=SHARPE_HELP)
    c[1].metric("Trades", m.get("num_trades", "–"),
                help=f"At least {ACCEPT_MIN_TRADES} trades needed for a reliable result")
    c[2].metric("Win rate", fmt_pct(m.get("win_rate")))
    c[3].metric("Max drawdown", fmt_pct(m.get("max_drawdown")))
    if m.get("insufficient_sample"):
        st.caption(f"⚠️ Fewer than {ACCEPT_MIN_TRADES} trades: treat these numbers as unreliable.")


def strategy_card(h: dict):
    st.markdown(f"**{h.get('rule_name', 'Strategy')}**  ·  `{h.get('signal_type')}`")
    params = h.get("params", {})
    st.markdown(" ".join(f"`{k} = {v}`" for k, v in params.items()))
    if h.get("reasoning"):
        st.caption(h["reasoning"])


def split_for_run(df: pd.DataFrame, run: dict):
    """Rebuild the exact train / validation split used by a run."""
    train_end = run.get("train_end")
    if train_end and isinstance(train_end, str):
        end = pd.Timestamp(train_end)
        train = df[df.index <= end]
        test = df[df.index > end]
        if len(train) and len(test):
            return train, test
    return split_data(df)


# ---------------------------------------------------------------- charts
def price_chart(ticker, train, test, d_train, d_test):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train["Close"], name="Research window", line=dict(color=ACCENT, width=1.6)))
    fig.add_trace(go.Scatter(x=test.index, y=test["Close"], name="Validation window (held out)",
                             line=dict(color="#0EA5E9", width=1.6)))
    for d, tag in ((d_train, "research"), (d_test, "validation")):
        t = d["trades"]
        if t.empty:
            continue
        fig.add_trace(go.Scatter(x=t["entry_date"], y=t["entry_price"], mode="markers", name=f"Buy ({tag})",
                                 marker=dict(symbol="triangle-up", size=9, color=GOOD),
                                 hovertemplate="Buy %{x|%d %b %Y}<br>₹%{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=t["exit_date"], y=t["exit_price"], mode="markers", name=f"Sell ({tag})",
                                 marker=dict(symbol="triangle-down", size=9, color=BAD),
                                 customdata=(t["return"] * 100).round(2),
                                 hovertemplate="Sell %{x|%d %b %Y}<br>₹%{y:.2f}<br>Return %{customdata}%<extra></extra>"))
    if len(test):
        fig.add_vrect(x0=test.index[0], x1=test.index[-1], fillcolor="#0EA5E9", opacity=0.07, line_width=0,
                      annotation_text="held-out validation", annotation_position="top left")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.15),
                      yaxis_title=f"{ticker} close (₹)", hovermode="closest")
    return fig


def equity_chart(d_train, d_test):
    fig = go.Figure()
    for d, name, color in ((d_train, "Research window", ACCENT), (d_test, "Validation window", "#0EA5E9")):
        eq = d["equity"]
        if len(eq):
            fig.add_trace(go.Scatter(x=eq.index, y=(eq - 1) * 100, name=name, mode="lines+markers",
                                     line=dict(color=color, width=2), marker=dict(size=4)))
    fig.add_hline(y=0, line_color=MUTED, line_dash="dot")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Cumulative return (%)",
                      legend=dict(orientation="h", y=-0.2))
    return fig


def iteration_chart(attempts: pd.DataFrame):
    it = attempts["iteration"].tolist()
    sharpe = [m.get("sharpe", 0) for m in attempts["metrics"]]
    trades = [m.get("num_trades", 0) for m in attempts["metrics"]]
    colors = [GOOD if d == "accept" else ACCENT for d in attempts["decision"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=it, y=trades, name="Trades", marker_color="rgba(148,163,184,0.45)", yaxis="y2"))
    fig.add_trace(go.Scatter(x=it, y=sharpe, name="Sharpe (per-trade)", mode="lines+markers",
                             line=dict(color=ACCENT, width=2.5), marker=dict(size=10, color=colors)))
    fig.add_hline(y=ACCEPT_MIN_SHARPE, line_dash="dash", line_color=GOOD,
                  annotation_text=f"accept ≥ {ACCEPT_MIN_SHARPE}", annotation_position="top left")
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(title="Iteration", dtick=1),
        yaxis=dict(title="Sharpe (per-trade)"),
        yaxis2=dict(title="Trades", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


# ---------------------------------------------------------------- run detail
def render_run(run: dict):
    run_id = run["run_id"]
    ticker = run["ticker"]
    attempts = load_attempts(run_id)
    if attempts.empty:
        st.info("No iterations recorded for this run yet.")
        return

    final = attempts.iloc[-1]
    final_h = final["hypothesis"]
    decision = run.get("final_decision") or final["decision"]

    st.markdown(f"### {ticker} {decision_pill(decision)}", unsafe_allow_html=True)
    st.caption(f"Run `{run_id}` · {len(attempts)} iteration(s)"
               + (f" · started {str(run['started_at'])[:16].replace('T', ' ')}" if run.get("started_at") else ""))

    try:
        prices = load_prices(ticker)
    except PriceDataError as e:
        st.error(str(e))
        prices = None

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Final strategy")
        strategy_card(final_h)
    d_train = d_test = None
    if prices is not None:
        train, test = split_for_run(prices, run)
        d_train = backtest_details(train, final_h)
        d_test = backtest_details(test, final_h)
        with right:
            metric_row(d_train["metrics"], f"Research window · {train.index[0]:%b %Y} – {train.index[-1]:%b %Y}")
            metric_row(d_test["metrics"], f"Held-out validation · {test.index[0]:%b %Y} – {test.index[-1]:%b %Y}")

    tabs = st.tabs(["📈 Price & trades", "💹 Equity curve", "🔁 Iterations", "🧠 Agent trail", "📋 Trades"])
    with tabs[0]:
        if d_train is not None:
            st.plotly_chart(price_chart(ticker, train, test, d_train, d_test), use_container_width=True)
            st.caption("The agents only ever saw the research window. The shaded validation window is scored once, after the loop ends.")
    with tabs[1]:
        if d_train is not None:
            st.plotly_chart(equity_chart(d_train, d_test), use_container_width=True)
            st.caption("Compounded return of the final strategy's trades (no transaction costs).")
    with tabs[2]:
        st.plotly_chart(iteration_chart(attempts), use_container_width=True)
        rows = []
        for _, a in attempts.iterrows():
            m, h = a["metrics"], a["hypothesis"]
            rows.append({"Iteration": a["iteration"], "Signal": h.get("signal_type"),
                         "Params": json.dumps(h.get("params", {})), "Trades": m.get("num_trades"),
                         "Sharpe": m.get("sharpe"), "Win rate": m.get("win_rate"),
                         "Max DD": m.get("max_drawdown"), "Decision": a["decision"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with tabs[3]:
        for _, a in attempts.iterrows():
            label = "✅ accept" if a["decision"] == "accept" else "🔄 refine"
            h, m = a["hypothesis"], a["metrics"]
            with st.expander(f"Iteration {a['iteration']} · {h.get('signal_type')} · Sharpe {m.get('sharpe')} · "
                             f"{m.get('num_trades')} trades · {label}", expanded=(a["iteration"] == attempts["iteration"].max())):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Hypothesis / Refinement agent proposed**")
                    strategy_card(h)
                with c2:
                    st.markdown("**Critique agent said**")
                    st.write(a["reasoning"])
    with tabs[4]:
        if d_train is not None:
            frames = []
            for d, tag in ((d_train, "research"), (d_test, "validation")):
                t = d["trades"].copy()
                if not t.empty:
                    t.insert(0, "window", tag)
                    frames.append(t)
            if frames:
                t = pd.concat(frames)
                t["return"] = (t["return"] * 100).round(2)
                t = t.rename(columns={"return": "return %"})
                st.dataframe(t, use_container_width=True, hide_index=True)
            else:
                st.info("This strategy made no trades.")


# ---------------------------------------------------------------- pages
def page_overview():
    st.markdown('<div class="aq-hero"><h1>📈 AgentQuant</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="aq-sub">Can an LLM improve a trading strategy step by step better than a single prompt? '
                'A multi-agent research loop on NSE stocks, with deterministic backtesting and a held-out validation window.</div>',
                unsafe_allow_html=True)

    runs, attempts, tickers, accepted = db_stats()
    c = st.columns(4)
    c[0].metric("Research runs", runs)
    c[1].metric("Strategies tested", attempts)
    c[2].metric("Tickers", tickers)
    c[3].metric("Runs accepted", accepted)

    st.markdown("#### How the loop works")
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR; bgcolor="transparent";
          node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#4F46E5", fillcolor="#EEF2FF"];
          edge [color="#64748B", fontname="Helvetica", fontsize=9];
          mem [label="Memory\\nSQLite + ChromaDB", fillcolor="#F1F5F9", color="#94A3B8"];
          H [label="Hypothesis Agent\\n(%s)"];
          B [label="Backtest Engine\\n(deterministic, no LLM)", fillcolor="#ECFDF5", color="#16A34A"];
          C [label="Critique Agent\\n(%s)"];
          R [label="Refinement Agent\\n(%s)"];
          G [label="Generalization gate\\n(80/20 inside research window)", fillcolor="#FFFBEB", color="#D97706"];
          V [label="Validation\\n(held-out window)", fillcolor="#F0F9FF", color="#0EA5E9"];
          mem -> H [label="past attempts"];
          H -> B -> C;
          C -> G [label="accept"];
          G -> V [label="passes"];
          G -> R [label="overfit"];
          C -> R [label="refine"];
          R -> B;
          C -> V [label="max 8 iterations", style=dashed];
        }
        """ % (HYPOTHESIS_MODEL, CRITIQUE_MODEL, REFINEMENT_MODEL),
        use_container_width=True,
    )

    cols = st.columns(4)
    cards = [
        ("1 · Propose", f"{HYPOTHESIS_MODEL} proposes a rule as strict JSON: RSI, MA crossover, price dip, Bollinger or MACD, with parameters and exits."),
        ("2 · Backtest", "Pure Python engine scores it on the research window only: trades, win rate, drawdown, Sharpe."),
        ("3 · Critique", f"{CRITIQUE_MODEL} checks it against fixed thresholds (≥{ACCEPT_MIN_TRADES} trades, Sharpe ≥ {ACCEPT_MIN_SHARPE}, DD ≥ {int(ACCEPT_MAX_DRAWDOWN*100)}%, win ≥ {int(ACCEPT_MIN_WIN_RATE*100)}%) and explains what to fix."),
        ("4 · Refine", f"{REFINEMENT_MODEL} revises the rule from the critique. Repeat up to 8 times, then score once on held-out data."),
    ]
    for col, (title, body) in zip(cols, cards):
        col.markdown(f'<div class="aq-card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)

    with st.expander("Design choices and known limitations"):
        st.markdown(
            "- **The LLM never writes scoring code.** Strategies are JSON validated by `schema.py`; the backtester is deterministic, so every number is reproducible.\n"
            "- **No look-ahead into validation.** The loop only receives the research (train) split; validation is scored after the loop ends.\n"
            "- **Sharpe here is per-trade** (mean/std × √trades), closer to a t-statistic; it is not annualised.\n"
            "- **No transaction costs** are modelled; entries fill at the signal day's close.\n"
            "- **Small samples:** many strategies make fewer than 20 trades in a ~1-year validation window.\n"
            "- **Unequal comparison:** the loop gets up to 8 attempts, the single-shot baseline gets one."
        )


def page_run():
    st.header("🚀 Run the research loop")
    if not has_openai_key():
        st.error("OPENAI_API_KEY is not set. Copy `.env.example` to `.env`, add your key, and restart the app.")
        return

    cached = list_cached_tickers()
    options = list(dict.fromkeys(cached + DEFAULT_TICKERS))
    c1, c2, c3 = st.columns([2, 1, 1])
    ticker = c1.selectbox("Ticker (NSE)", options, index=options.index("TCS.NS") if "TCS.NS" in options else 0,
                          help="Tickers already downloaded are listed first and load without internet.")
    custom = c1.text_input("…or type another Yahoo Finance symbol", placeholder="e.g. ITC.NS").strip().upper()
    if custom:
        ticker = custom
    train_ratio = c2.slider("Research window", 0.5, 0.9, 0.7, 0.05, help="Share of history the agents may see")
    with_baseline = c3.checkbox("Also run single-shot baseline", value=True)
    st.caption(f"Uses the OpenAI API: {HYPOTHESIS_MODEL} (hypothesis, refinement) and {CRITIQUE_MODEL} (critique). "
               "A run takes about 1–3 minutes.")

    if st.button("Run research loop", type="primary"):
        try:
            prices = load_prices(ticker)
        except PriceDataError as e:
            st.error(str(e))
            return
        train, test = split_data(prices, train_ratio)

        from graph import run_research_loop
        from memory.log_store import init_db, save_validation

        status = st.status(f"Researching {ticker} on {train.index[0]:%b %Y} – {train.index[-1]:%b %Y}…", expanded=True)

        def on_event(event, p):
            i = p.get("iteration")
            if event == "hypothesis":
                h = p["hypothesis"]
                status.write(f"💡 **Hypothesis**: {h.get('rule_name')} · `{h.get('signal_type')}` {h.get('params')}")
            elif event == "backtest":
                m = p["metrics"]
                status.write(f"🧪 **Backtest (iter {i})**: {m.get('num_trades')} trades · Sharpe {m.get('sharpe')} · "
                             f"win {fmt_pct(m.get('win_rate'))} · DD {fmt_pct(m.get('max_drawdown'))}")
            elif event == "critique":
                icon = "✅" if p["decision"] == "accept" else "🔄"
                status.write(f"{icon} **Critique (iter {i})**: {p['decision']}. {p['reasoning'][:220]}"
                             + ("…" if len(p["reasoning"]) > 220 else ""))
            elif event == "refine":
                h = p["hypothesis"]
                status.write(f"🛠️ **Refined**: {h.get('rule_name')} · `{h.get('signal_type')}` {h.get('params')}")

        try:
            result = run_research_loop(ticker, df_train=train, on_event=on_event)
        except Exception as e:
            status.update(label="Run failed", state="error")
            st.error(f"The research loop failed: {e}")
            return

        val = evaluate_strategy(test, result["hypothesis"])
        try:
            conn = init_db()
            save_validation(conn, result["run_id"], val)
            conn.close()
        except Exception:
            pass

        baseline = None
        if with_baseline:
            status.write("🎯 Running single-shot baseline (one proposal, no feedback)…")
            try:
                from agents.hypothesis_agent import propose_hypothesis
                b_h = propose_hypothesis(ticker)
                baseline = (b_h, evaluate_strategy(train, b_h), evaluate_strategy(test, b_h))
            except Exception as e:
                status.write(f"Baseline failed: {e}")

        status.update(label=f"Finished: {result['decision']} after {result['iteration']} iteration(s)",
                      state="complete", expanded=False)
        load_runs.clear()
        load_attempts.clear()

        if baseline:
            st.markdown("#### Loop vs single-shot baseline (held-out validation)")
            b_h, b_train, b_test = baseline
            comp = pd.DataFrame([
                {"": "Iterative loop", "Strategy": result["hypothesis"].get("rule_name"),
                 "Research Sharpe": result["metrics"].get("sharpe"), "Validation Sharpe": val.get("sharpe"),
                 "Validation trades": val.get("num_trades"), "Validation win": val.get("win_rate")},
                {"": "Single-shot baseline", "Strategy": b_h.get("rule_name"),
                 "Research Sharpe": b_train.get("sharpe"), "Validation Sharpe": b_test.get("sharpe"),
                 "Validation trades": b_test.get("num_trades"), "Validation win": b_test.get("win_rate")},
            ])
            st.dataframe(comp, use_container_width=True, hide_index=True)

        run = {"run_id": result["run_id"], "ticker": ticker, "final_decision": result["decision"],
               "train_end": str(train.index[-1].date()), "started_at": None}
        render_run(run)


def page_runs():
    st.header("🔎 Research runs")
    runs = load_runs()
    if runs.empty:
        st.info("No runs recorded yet. Use **Run research** or `python graph.py --ticker TCS.NS`.")
        return

    tickers = ["All"] + sorted(runs["ticker"].dropna().unique().tolist())
    c1, c2 = st.columns([1, 3])
    t = c1.selectbox("Ticker", tickers)
    view = runs if t == "All" else runs[runs["ticker"] == t]

    def label(r):
        if str(r["run_id"]).startswith("legacy:"):
            return f"{r['ticker']} · earlier runs (before run ids) · {int(r['iterations'])} attempts"
        dec = r.get("final_decision") or "unfinished"
        when = (r.get("started_at") or "")[:16].replace("T", " ")
        return f"{r['ticker']} · {when} · {dec} · {int(r['iterations']) if pd.notna(r.get('iterations')) else '?'} iters"

    records = view.to_dict("records")
    choice = c2.selectbox("Run", range(len(records)), format_func=lambda i: label(records[i]))
    render_run(records[choice])


def page_evaluation():
    st.header("📊 Evaluation results")
    files = sorted(glob.glob(os.path.join(str(RESULTS_DIR), "evaluation_summary*.csv")), reverse=True)
    if not files:
        st.info("No evaluation results found. Run `python evaluate.py` (all 15 tickers, ~30–60 minutes) "
                "and the CSV will appear here.")
        return
    f = st.selectbox("Results file", files, format_func=os.path.basename)
    df = pd.read_csv(f)

    if "sharpe_delta_valid" in df.columns:
        n = len(df)
        c = st.columns(5)
        c[0].metric("Tickers", n)
        c[1].metric("Mean Sharpe Δ", f"{df['sharpe_delta_valid'].mean():+.3f}",
                    help="Loop minus baseline, on the held-out validation window")
        c[2].metric("Std of Δ", f"{df['sharpe_delta_valid'].std():.3f}")
        if "raw_loop_beats_baseline" in df:
            c[3].metric("Loop beats baseline (raw)", f"{int(df['raw_loop_beats_baseline'].sum())}/{n}")
        if "genuine_win" in df:
            c[4].metric("Genuine wins", f"{int(df['genuine_win'].sum())}/{n}",
                        help="Δ ≥ 0.1 and not degenerate (enough trades in both windows)")

        d = df.sort_values("sharpe_delta_valid")
        colors = [GOOD if g else (MUTED if deg else ACCENT)
                  for g, deg in zip(d.get("genuine_win", [False] * n), d.get("degenerate_flag", [False] * n))]
        fig = go.Figure(go.Bar(x=d["sharpe_delta_valid"], y=d["ticker"], orientation="h", marker_color=colors))
        fig.add_vline(x=0, line_color=MUTED)
        fig.update_layout(height=40 + 28 * n, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Validation Sharpe: loop − baseline")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Green = genuine win · grey = degenerate (too few trades to trust) · purple = other.")

        if {"b_valid_sharpe", "l_valid_sharpe"} <= set(df.columns):
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=df["ticker"], y=df["b_valid_sharpe"], name="Baseline", marker_color=MUTED))
            fig2.add_trace(go.Bar(x=df["ticker"], y=df["l_valid_sharpe"], name="Loop", marker_color=ACCENT))
            fig2.update_layout(barmode="group", height=360, margin=dict(l=10, r=10, t=10, b=10),
                               yaxis_title="Validation Sharpe (per-trade)", legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(df, use_container_width=True, hide_index=True)

    var_files = sorted(glob.glob(os.path.join(str(RESULTS_DIR), "variance_report*.csv")), reverse=True)
    if var_files:
        st.markdown("#### Run-to-run variance")
        vf = st.selectbox("Variance file", var_files, format_func=os.path.basename)
        st.dataframe(pd.read_csv(vf), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------- main
with st.sidebar:
    st.markdown("## 📈 AgentQuant")
    page = st.radio("Navigate", ["Overview", "Run research", "Research runs", "Evaluation"], label_visibility="collapsed")
    st.divider()
    if has_openai_key():
        st.success("OpenAI key loaded", icon="🔑")
    else:
        st.warning("No OpenAI key in .env: live runs disabled", icon="🔑")
    st.caption("Research prototype · historical backtests only · not financial advice")

{"Overview": page_overview, "Run research": page_run, "Research runs": page_runs, "Evaluation": page_evaluation}[page]()
