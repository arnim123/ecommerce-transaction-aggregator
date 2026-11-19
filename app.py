"""ASGI entry point for Vercel and for `python app.py`.

The pages live in streamlit_app.py. From this folder, the local dashboard is:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

# Vercel can read the project files but cannot write next to them.
# Matplotlib and Streamlit keep their caches in /tmp instead.
_cache = Path("/tmp/matplotlib")
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("STREAMLIT_SERVER_ENABLE_CORS", "false")
os.environ.setdefault("STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION", "false")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

import streamlit as st

_ROOT = Path(__file__).resolve().parent
app = st.App(str(_ROOT / "streamlit_app.py"))

if __name__ == "__main__":
    app.run()
