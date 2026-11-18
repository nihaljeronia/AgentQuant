"""
Central configuration for AgentQuant.

All paths are resolved relative to the repository root, so scripts work no matter
which folder they are launched from (useful on Windows / PowerShell).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

# Where generated data lives. Override with AGENTQUANT_DATA_DIR (used by tests).
DATA_DIR = Path(os.getenv("AGENTQUANT_DATA_DIR", str(ROOT_DIR))).resolve()

PRICE_CACHE_DIR = DATA_DIR / "data_layer" / "cache"
DB_PATH = DATA_DIR / "memory" / "agentquant.db"
CHROMA_DIR = DATA_DIR / "memory" / "chroma_db"
RESULTS_DIR = DATA_DIR  # evaluate.py / variance_check.py write their CSVs here

# Models (unchanged from the original design)
HYPOTHESIS_MODEL = os.getenv("AGENTQUANT_HYPOTHESIS_MODEL", "gpt-4o-mini")
REFINEMENT_MODEL = os.getenv("AGENTQUANT_REFINEMENT_MODEL", "gpt-4o-mini")
CRITIQUE_MODEL = os.getenv("AGENTQUANT_CRITIQUE_MODEL", "gpt-4o")

DEFAULT_TICKERS = [
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS",
    "RELIANCE.NS", "ONGC.NS", "NTPC.NS",
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
    "HINDUNILVR.NS", "SUNPHARMA.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "MARUTI.NS",
]


def ensure_dirs() -> None:
    PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def has_openai_key() -> bool:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(key) and not key.startswith("your_")
