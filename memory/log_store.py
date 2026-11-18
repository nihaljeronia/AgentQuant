import json
import os
import sqlite3
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DB_PATH, ensure_dirs  # noqa: E402


def init_db(path=None):
    """Initialize SQLite database for structured execution logging."""
    ensure_dirs()
    conn = sqlite3.connect(str(path or DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT, iteration INTEGER,
            hypothesis TEXT, metrics TEXT, decision TEXT, reasoning TEXT
        )
    """)
    # Upgrade older databases in place: add run_id / created_at if missing.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(attempts)")}
    if "run_id" not in cols:
        conn.execute("ALTER TABLE attempts ADD COLUMN run_id TEXT")
    if "created_at" not in cols:
        conn.execute("ALTER TABLE attempts ADD COLUMN created_at TEXT")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            ticker TEXT,
            started_at TEXT,
            finished_at TEXT,
            train_start TEXT,
            train_end TEXT,
            iterations INTEGER,
            final_decision TEXT,
            final_hypothesis TEXT,
            final_metrics TEXT,
            validation_metrics TEXT,
            baseline TEXT
        )
    """)
    # Upgrade older databases in place: add baseline if missing.
    run_cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
    if "baseline" not in run_cols:
        conn.execute("ALTER TABLE runs ADD COLUMN baseline TEXT")
    conn.commit()
    return conn


def log_attempt(conn, ticker: str, entry: dict, run_id: str | None = None):
    """Insert a single research attempt into the database."""
    conn.execute(
        "INSERT INTO attempts (ticker, iteration, hypothesis, metrics, decision, reasoning, run_id, created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            ticker,
            entry["iteration"],
            json.dumps(entry["hypothesis"]),
            json.dumps(entry["metrics"]),
            entry["decision"],
            entry["reasoning"],
            run_id,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()


def start_run(conn, run_id: str, ticker: str, train_start=None, train_end=None):
    conn.execute(
        "INSERT OR REPLACE INTO runs (run_id, ticker, started_at, train_start, train_end) VALUES (?,?,?,?,?)",
        (run_id, ticker, datetime.now().isoformat(timespec="seconds"),
         str(train_start) if train_start is not None else None,
         str(train_end) if train_end is not None else None),
    )
    conn.commit()


def finish_run(conn, run_id: str, result: dict):
    conn.execute(
        "UPDATE runs SET finished_at=?, iterations=?, final_decision=?, final_hypothesis=?, final_metrics=? "
        "WHERE run_id=?",
        (
            datetime.now().isoformat(timespec="seconds"),
            result.get("iteration"),
            result.get("decision"),
            json.dumps(result.get("hypothesis", {})),
            json.dumps(result.get("metrics", {})),
            run_id,
        ),
    )
    conn.commit()


def save_validation(conn, run_id: str, metrics: dict):
    conn.execute("UPDATE runs SET validation_metrics=? WHERE run_id=?", (json.dumps(metrics), run_id))
    conn.commit()


def save_baseline(conn, run_id: str, baseline: dict):
    """Store the single-shot baseline (hypothesis, research and validation metrics) with its run."""
    conn.execute("UPDATE runs SET baseline=? WHERE run_id=?", (json.dumps(baseline), run_id))
    conn.commit()


def prior_attempts_summary(conn, ticker: str, exclude_run: str | None = None, limit: int = 5) -> str:
    """
    Text summary of the best and worst earlier attempts on this ticker (from other runs),
    including their metrics and decision, for the Hypothesis Agent's prompt.
    """
    rows = conn.execute(
        "SELECT hypothesis, metrics, decision FROM attempts WHERE ticker=? AND (run_id IS NULL OR run_id != ?)",
        (ticker, exclude_run or ""),
    ).fetchall()
    if not rows:
        return ""
    parsed = []
    for h, m, d in rows:
        try:
            hyp, met = json.loads(h), json.loads(m)
        except (TypeError, ValueError):
            continue
        parsed.append((met.get("sharpe") or 0.0, hyp, met, d))
    parsed.sort(key=lambda r: r[0], reverse=True)
    picks = parsed[: (limit + 1) // 2] + parsed[-(limit // 2):] if len(parsed) > limit else parsed
    seen, lines = set(), []
    for sharpe, hyp, met, d in picks:
        key = json.dumps([hyp.get("signal_type"), hyp.get("params")], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"- {hyp.get('signal_type')} {json.dumps(hyp.get('params', {}))}: "
            f"trades={met.get('num_trades')}, sharpe={met.get('sharpe')}, win={met.get('win_rate')}, "
            f"dd={met.get('max_drawdown')} -> {d}"
        )
    return "\n".join(lines)
