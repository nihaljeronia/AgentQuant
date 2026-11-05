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
├── agents/
│   ├── critique_agent.py
│   ├── hypothesis_agent.py
│   └── refinement_agent.py
├── backtest/
│   └── engine.py
├── data_layer/
│   ├── cache/
│   └── fetch_prices.py
├── memory/
│   ├── __init__.py
│   ├── log_store.py
│   └── vector_store.py
├── report/
│   └── dashboard.py
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── evaluate.py
├── graph.py
├── main_baseline.py
├── pyproject.toml
├── requirements.txt
├── schema.py
├── variance_check.py
└── .venv/
```

## Quick start

1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Configure your OpenAI key

```bash
copy .env.example .env
```

Update `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

4. Run the baseline

```bash
python main_baseline.py
```

5. Run the iterative agent loop

```bash
python graph.py
```

6. Launch the dashboard

```bash
streamlit run report/dashboard.py
```

## Research and validation approach

The project is designed around a simple but important principle: do not evaluate on the same data used to generate the strategy.

It uses a chronological split:

- train/research window: used for strategy generation and refinement
- validation/test window: held out and evaluated only after the loop finishes

This helps reduce the chance that a strategy is only looking good because it has memorized the research window.

## Notes

This repository is a research prototype and a learning project. It is not financial advice and should not be treated as a production trading system.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## GitHub

Repository:

- https://github.com/nihaljeronia/AgentQuant
