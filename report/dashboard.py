"""Kept for backwards compatibility: `streamlit run report/dashboard.py` opens the main app."""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app.py"), run_name="__main__")
