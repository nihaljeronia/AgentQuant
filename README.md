# AgentQuant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/LangGraph-Agent%20Loop-6A5AE0" alt="LangGraph" />
  <img src="https://img.shields.io/badge/OpenAI-GPT%204o%2F4o-mini-412991" alt="OpenAI" />
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-FFA726" alt="Status" />
</p>

AgentQuant is an experimental agentic trading research project that tests whether a multi-step AI workflow can improve stock strategy discovery compared with a single-shot model baseline.

## Overview

The system combines:

- real NSE stock price data
- deterministic backtesting logic
- structured strategy schemas
- LLM-based hypothesis generation
- critique and refinement loops
- memory persistence across attempts

The core idea is simple: generate a strategy, backtest it, critique it, revise it, and repeat until the loop converges on a stronger idea.

## Why this project matters

Most strategy work is slow, manual, and highly dependent on human intuition. AgentQuant explores a different pattern:

- propose a rule
- evaluate it on historical data
- get model feedback on quality and risk
- refine the idea with explicit critique
- validate against a held-out window

This makes it a useful project for learning agentic AI design, research loops, and systematic strategy evaluation.

## Architecture

```text
Hypothesis Agent --> Backtest Engine --> Critique Agent --> Refinement Agent
      ^                                              |
      |______________________________________________|
```

The loop is implemented in `graph.py` and uses `LangGraph` to orchestrate the workflow. Strategy output is structured through `schema.py`, while historical data and memory are managed in `data_layer/` and `memory/`.

## Key features

- Multi-agent research loop for strategy generation and refinement
- Rule-based stock backtesting with trade simulation
- Guardrails against obvious overfitting and weak sampling
- Persistent SQLite and ChromaDB memory
- Streamlit dashboard for viewing the research trail
- Chronological train/test split for more disciplined evaluation

## Project structure

```text
AgentQuant/
├── app.py                  # Streamlit dashboard (overview, live run, research runs, evaluation)
├── config.py               # paths, model names, ticker list
├── graph.py                # LangGraph research loop (CLI: python graph.py --ticker TCS.NS)
├── schema.py               # strategy JSON contract
├── evaluate.py             # 15-ticker loop vs baseline evaluation
├── variance_check.py       # run-to-run variance check
├── main_baseline.py        # single-shot baseline
├── agents/                 # hypothesis, critique, refinement agents + shared OpenAI client
├── backtest/engine.py      # deterministic backtester
├── data_layer/             # yfinance download + CSV cache
├── memory/                 # SQLite run log + ChromaDB similarity memory
├── report/dashboard.py     # old entry point, opens app.py
└── tests/                  # pytest suite (no API key needed)
```

## Quick start (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then put your OPENAI_API_KEY in .env
streamlit run app.py
```

The dashboard has four pages:

- **Overview**: the pipeline and what is in the research database
- **Run research**: pick a ticker and watch the agents propose, backtest, critique and refine live, then see the held-out validation score next to a single-shot baseline
- **Research runs**: replay any recorded run with the price chart and buy/sell markers, equity curve, per-iteration metrics and the full critique trail
- **Evaluation**: charts for `evaluation_summary*.csv` and `variance_report*.csv`

Command line:

```powershell
python graph.py --ticker TCS.NS      # one research run (train split only, prints validation score)
python main_baseline.py              # single-shot baseline
python evaluate.py                   # full loop-vs-baseline evaluation
python variance_check.py             # run-to-run variance
python -m pytest -q                  # tests (no API key needed)
```

Prices are cached in `data_layer/cache/` after the first download, so later runs don't need Yahoo Finance.

## Research and validation approach

The project is designed around a simple but important principle: do not evaluate on the same data used to generate the strategy.

It uses a chronological split:

- train/research window: used for strategy generation and refinement
- validation/test window: held out and evaluated only after the loop finishes

This helps reduce the chance that a strategy is only looking good because it has memorized the research window.

## Limitations

These are known and deliberate to state up front:

- **"Sharpe" is a per-trade score, not an annualised Sharpe ratio.** It is computed as mean / std of trade returns multiplied by √(number of trades), so it behaves more like a t-statistic and rises with trade count. It is used only for relative comparison between strategies on the same window.
- **No transaction costs or slippage are modelled.** NSE round-trip costs (STT, exchange charges, brokerage) of roughly 0.2–0.3% would reduce returns, especially for short holding periods.
- **Entries fill at the signal day's close**, and stop-loss / take-profit are checked on closing prices only.
- **Small sample.** Most strategies produce fewer than 20 trades in the roughly one-year validation window, so many results are flagged as degenerate. With 14 tickers and a Sharpe-delta standard deviation larger than its mean, the aggregate improvement is not statistically significant.
- **Unequal comparison.** The iterative loop gets up to 8 attempts plus memory, while the baseline gets one stochastic proposal that is regenerated on every run. A best-of-N baseline is the planned fair control.
- **Validation window reuse.** The same held-out window was inspected across evaluation versions, so it is no longer a fully untouched holdout.
- **The Critique Agent applies fixed numeric thresholds**; its main value is the natural-language feedback that drives refinement.

## Future work

1. Annualised Sharpe alongside a separate significance statistic, transaction costs, and a buy-and-hold benchmark.
2. Cached multi-draw baseline with a best-of-N control.
3. Walk-forward evaluation over a longer history, plus a final untouched holdout run once.
4. Turn memory off (or isolate it) during evaluation so runs are fully independent.

## Notes

This repository is a research prototype and a learning project. It is not financial advice and should not be treated as a production trading system.

## Team

Developed as a B.Tech Major Project at ITM University Gwalior.

- Nihal Jeronia
- Devang Shrivastava
- Rohit Prajapati
- Satya Mishra
- Sharad Rathore

**Project guide:** Mr. Ashish Yadav, Department of CSA, SOCIS, ITM University Gwalior.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## GitHub

Repository:

- https://github.com/nihaljeronia/AgentQuant
