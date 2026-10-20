"""Shared look and feel: CSS, colours, Plotly styling and small components."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

BG, SURFACE, SURFACE_2, BORDER = "#0B1020", "#121933", "#18214A", "#232C4D"
TEXT, MUTED = "#E6E9F5", "#8A93B5"
ACCENT, ACCENT_2 = "#8B7CF6", "#5EC8D8"
GOOD, FAIR, POOR = "#6FD3A3", "#F5C26B", "#F2777A"
CLASS_COLORS = {"Poor": POOR, "Fair": FAIR, "Good": GOOD}

MOON = (f'<svg width="{{s}}" height="{{s}}" viewBox="0 0 24 24" fill="none">'
        f'<path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5a8.5 8.5 0 1 0 11 11Z" fill="{ACCENT}" fill-opacity=".92"/>'
        f'<circle cx="17.5" cy="5.5" r="1.1" fill="{ACCENT_2}"/><circle cx="20.5" cy="9" r=".7" fill="{ACCENT_2}"/></svg>')


def inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, label, textarea {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1120px; }}
[data-testid="stHeader"] {{ background: rgba(11,16,32,.85); backdrop-filter: blur(8px); border-bottom: 1px solid {BORDER}; }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display:none; }}
[data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"] {{ display:none; }}
footer {{ visibility:hidden; }}
.sw-brand {{ display:flex; align-items:center; gap:.55rem; }}
.sw-brand span {{ font-family:'Fraunces', Georgia, serif; font-size:1.5rem; font-weight:600; color:{TEXT}; }}
.sw-eyebrow {{ color:{ACCENT}; font-size:.76rem; font-weight:600; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.2rem; }}
.sw-title {{ font-family:'Fraunces', Georgia, serif; font-size:2rem; font-weight:600; color:{TEXT}; margin:0 0 .25rem; line-height:1.15; }}
.sw-sub {{ color:{MUTED}; font-size:.98rem; margin-bottom:1.4rem; max-width:720px; }}
.sw-card {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:16px; padding:1.05rem 1.2rem; height:100%; }}
.sw-kpi-label {{ color:{MUTED}; font-size:.74rem; font-weight:600; text-transform:uppercase; letter-spacing:.07em; }}
.sw-kpi-value {{ color:{TEXT}; font-size:1.7rem; font-weight:700; margin-top:.15rem; font-variant-numeric:tabular-nums; }}
.sw-kpi-value small {{ font-size:.95rem; color:{MUTED}; font-weight:500; }}
.sw-kpi-note {{ color:{MUTED}; font-size:.82rem; margin-top:.1rem; }}
.sw-pill {{ display:inline-block; padding:.25rem .75rem; border-radius:999px; font-weight:600; font-size:.84rem; }}
.sw-section {{ color:{TEXT}; font-weight:600; font-size:1.05rem; margin:1.5rem 0 .6rem; }}
.sw-muted {{ color:{MUTED}; font-size:.88rem; }}
.sw-insight {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:14px; padding:.9rem 1.05rem; margin-bottom:.6rem; }}
.sw-insight .h {{ display:flex; justify-content:space-between; align-items:baseline; gap:1rem; }}
.sw-insight .t {{ color:{TEXT}; font-weight:600; }}
.sw-insight .d {{ font-weight:700; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.sw-insight .b {{ color:{MUTED}; font-size:.88rem; margin-top:.25rem; }}
.sw-lever {{ display:flex; justify-content:space-between; align-items:baseline; padding:.55rem 0; border-bottom:1px solid {BORDER}; }}
.sw-lever:last-child {{ border-bottom:none; }}
.sw-lever .name {{ color:{TEXT}; font-weight:500; }}
.sw-lever .detail {{ color:{MUTED}; font-size:.82rem; }}
.sw-lever .delta {{ font-weight:700; font-variant-numeric:tabular-nums; }}
.sw-progress {{ height:8px; background:{BORDER}; border-radius:99px; overflow:hidden; margin:.55rem 0 .35rem; }}
.sw-progress > div {{ height:100%; background:linear-gradient(90deg, {ACCENT}, {ACCENT_2}); border-radius:99px; }}
.sw-note {{ background:rgba(245,194,107,.08); border:1px solid rgba(245,194,107,.32); color:#F5D9A8;
            padding:.6rem .85rem; border-radius:10px; font-size:.86rem; margin:.6rem 0; }}
.sw-hero h1 {{ font-family:'Fraunces', Georgia, serif; font-size:2.7rem; line-height:1.1; color:{TEXT}; margin:.6rem 0 .8rem; }}
.sw-hero p {{ color:{MUTED}; font-size:1.05rem; max-width:460px; }}
.sw-feat {{ display:flex; gap:.75rem; align-items:flex-start; margin:.9rem 0; color:{TEXT}; }}
.sw-feat .dot {{ width:30px; height:30px; border-radius:9px; background:{SURFACE_2}; border:1px solid {BORDER};
                 display:flex; align-items:center; justify-content:center; flex:none; color:{ACCENT_2}; font-weight:700; }}
.sw-feat .s {{ color:{MUTED}; font-size:.88rem; }}
div[data-testid="stForm"] {{ border:1px solid {BORDER}; border-radius:16px; background:{SURFACE}; padding:1.2rem 1.3rem; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius:16px; }}
.stTabs [data-baseweb="tab-list"] {{ gap:.3rem; }}
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{ border-radius:10px; font-weight:600; }}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{ background:{ACCENT}; border-color:{ACCENT}; }}
@media (max-width: 640px) {{ .sw-title {{ font-size:1.6rem; }} .sw-hero h1 {{ font-size:2rem; }} .block-container {{ padding-top:3.6rem; }} }}
</style>""", unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=13), margin=dict(l=8, r=8, t=36, b=8),
        showlegend=legend, title_font=dict(size=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", title=None),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font_color=TEXT),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    return fig


def chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def header(eyebrow: str, title: str, sub: str = "") -> None:
    st.markdown(f'<div class="sw-eyebrow">{eyebrow}</div><div class="sw-title">{title}</div>'
                + (f'<div class="sw-sub">{sub}</div>' if sub else '<div style="height:1rem"></div>'),
                unsafe_allow_html=True)


def kpi(col, label: str, value: str, note: str = "") -> None:
    col.markdown(f'<div class="sw-card"><div class="sw-kpi-label">{label}</div>'
                 f'<div class="sw-kpi-value">{value}</div><div class="sw-kpi-note">{note}</div></div>',
                 unsafe_allow_html=True)


def pill(label: str) -> str:
    c = CLASS_COLORS.get(label, ACCENT)
    return f'<span class="sw-pill" style="background:{c}22;color:{c};border:1px solid {c}55">{label}</span>'


def section(text: str, top: bool = False) -> None:
    st.markdown(f'<div class="sw-section"{" style=margin-top:0" if top else ""}>{text}</div>', unsafe_allow_html=True)


def quality_color(q: float) -> str:
    return POOR if q < 6.5 else (FAIR if q < 7.5 else GOOD)
