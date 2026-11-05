import sqlite3
import json
import os

DB_DIR = "memory"
os.makedirs(DB_DIR, exist_ok=True)

def init_db(path="memory/agentquant.db"):
    """Initialize SQLite database for structured execution logging."""
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT, iteration INTEGER,
            hypothesis TEXT, metrics TEXT, decision TEXT, reasoning TEXT
        )
    """)
    return conn

def log_attempt(conn, ticker: str, entry: dict):
    """Insert a single research attempt into the database."""
    conn.execute(
        "INSERT INTO attempts (ticker, iteration, hypothesis, metrics, decision, reasoning) VALUES (?,?,?,?,?,?)",
        (
            ticker,
            entry["iteration"],
            json.dumps(entry["hypothesis"]),
            json.dumps(entry["metrics"]),
            entry["decision"],
            entry["reasoning"]
        )
    )
    conn.commit()
