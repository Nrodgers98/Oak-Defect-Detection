"""Model dashboard: every run, its Optuna study, training curves and metrics.

    cd notebooks/dashboard && streamlit run Home.py --server.port 8502
"""

import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Oak Model Dashboard", page_icon=":material/forest:", layout="wide")

# Pages import ui/charts as top-level modules from this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui  # noqa: E402

pages = {
    "Dashboard": [
        st.Page(ui.PAGES["home"], title="Overview", icon=":material/space_dashboard:", default=True),
        st.Page(ui.PAGES["runs"], title="Runs", icon=":material/table_rows:"),
        st.Page(ui.PAGES["detail"], title="Run detail", icon=":material/insights:"),
    ],
    "Analysis": [
        st.Page(ui.PAGES["optuna"], title="Optuna studies", icon=":material/tune:"),
        st.Page(ui.PAGES["compare"], title="Compare runs", icon=":material/compare_arrows:"),
    ],
}

nav = st.navigation(pages)
ui.setup_page(nav.title)
nav.run()
