import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Keep test databases out of the real project data.
os.environ.setdefault("AGENTQUANT_DATA_DIR", tempfile.mkdtemp(prefix="agentquant_test_"))


@pytest.fixture
def prices():
    """Synthetic daily prices (geometric random walk) used only for tests."""
    rng = np.random.default_rng(7)
    n = 750
    rets = rng.normal(0.0004, 0.015, n)
    close = 1000 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                         "Close": close, "Volume": 1_000_000}, index=idx)
