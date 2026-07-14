"""Chart rendering helpers for the Streamlit UI.

Wraps figures with consistent config (interactive hover, zoom, pan, PNG/SVG
camera export) and provides HTML download for any figure.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "toImageButtonOptions": {"format": "png", "scale": 2},
    "modeBarButtonsToAdd": ["toggleSpikelines"],
}


def render(fig: go.Figure, key: str, download: bool = True) -> None:
    """Render a Plotly figure with standard interactivity + HTML download."""
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=key)
    if download:
        html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
        st.download_button(
            "⬇ Download interactive HTML", html, file_name=f"{key}.html",
            mime="text/html", key=f"dl_{key}",
        )


def figure_bytes(fig: go.Figure, fmt: str = "png") -> bytes | None:
    """Static image bytes via kaleido if available (PNG/SVG/PDF)."""
    try:
        return fig.to_image(format=fmt, scale=2)
    except Exception:
        return None
