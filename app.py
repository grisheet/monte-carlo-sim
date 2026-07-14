"""Monte Carlo Stock Market Simulator — Streamlit entry point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from config import APP_NAME, APP_VERSION, setup_logging
from ui.controls import sidebar_controls
from ui.dashboard import (
    render_data_tab,
    render_export_tab,
    render_math_tab,
    render_portfolio_tab,
    render_results_tab,
    render_risk_tab,
    render_sensitivity_tab,
    render_statistics_tab,
    run_simulation,
)

logger = setup_logging()

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Monte Carlo Stock Market Simulator")
st.caption(f"v{APP_VERSION} — multi-model stochastic simulation, risk "
           "analytics, and scenario analysis. Educational software, not "
           "investment advice.")

cfg, ui = sidebar_controls()

tabs = st.tabs(["📊 Data", "🎲 Simulation", "📐 Statistics", "⚠️ Risk",
                "🎚 Sensitivity", "💼 Portfolio", "📤 Export", "📚 Math"])

with tabs[0]:
    render_data_tab(cfg)

with tabs[1]:
    if ui["run"]:
        result = run_simulation(cfg)
        if result is not None:
            st.session_state["result"] = result
    result = st.session_state.get("result")
    if result is not None:
        render_results_tab(result, ui["dark"], ui["max_lines"])
    else:
        st.info("Configure parameters in the sidebar and press **Run "
                "Simulation**. Tip: hover any parameter's ⓘ for its formula "
                "and meaning.")

result = st.session_state.get("result")
with tabs[2]:
    if result is not None:
        render_statistics_tab(result, ui["dark"])
    else:
        st.info("Run a simulation first.")
with tabs[3]:
    if result is not None:
        render_risk_tab(result)
    else:
        st.info("Run a simulation first.")
with tabs[4]:
    render_sensitivity_tab(cfg, ui["dark"])
with tabs[5]:
    render_portfolio_tab(ui["dark"])
with tabs[6]:
    if result is not None:
        render_export_tab(result, ui["dark"])
    else:
        st.info("Run a simulation first.")
with tabs[7]:
    render_math_tab()
