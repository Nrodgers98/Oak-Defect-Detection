"""Shared look-and-feel, data access and widgets for the model dashboard pages."""

from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = DASHBOARD_DIR.parent
REPO_ROOT = NOTEBOOKS_DIR.parent
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from functions import run_registry as reg  # noqa: E402

# ---------------------------------------------------------------------------
# Palette (dataviz reference palette, dark-surface steps). Data colors follow the
# entity, never its rank: each family owns one categorical slot.
# ---------------------------------------------------------------------------
SURFACE = "#141413"
SURFACE_2 = "#1d1d1b"
BORDER = "#2e2e2b"
TEXT = "#f4f3ee"
TEXT_2 = "#c3c2b7"
TEXT_MUTED = "#8a8980"
GRID = "#2a2a27"
ACCENT = "#E0A040"  # oak amber: UI accent only, never a data series

SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
FAMILY_COLORS = {
    "Segmentation": SERIES[0],
    "YOLO Seg": SERIES[1],
    "YOLO Pith": SERIES[2],
    "NHLA Grade": SERIES[6],
}
FAMILY_SYMBOLS = {"Segmentation": "circle", "YOLO Seg": "diamond", "YOLO Pith": "square", "NHLA Grade": "triangle-up"}
FAMILY_ICONS = {"Segmentation": ":material/texture:", "YOLO Seg": ":material/center_focus_strong:",
                "YOLO Pith": ":material/adjust:", "NHLA Grade": ":material/grade:"}

_template = go.layout.Template()
_template.layout = go.Layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color=TEXT_2, size=13),
    title=dict(font=dict(color=TEXT, size=15), x=0, xanchor="left", pad=dict(b=8)),
    colorway=SERIES,
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER, tickcolor=BORDER, showgrid=False),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER, tickcolor=BORDER),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT_2)),
    hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=BORDER, font=dict(color=TEXT, family="Inter, sans-serif")),
    margin=dict(l=8, r=8, t=48, b=8),
    colorscale=dict(sequential=[[0, "#1d2a3c"], [1, "#86b6ef"]]),
)
pio.templates["oak"] = _template
pio.templates.default = "plotly_dark+oak"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame, button, input {{ font-family: 'Inter', system-ui, sans-serif; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; font-weight: 650; }}
code {{ font-family: 'JetBrains Mono', monospace; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}
div[data-testid="stMetric"] {{
  background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 14px; padding: 14px 18px;
}}
div[data-testid="stMetricLabel"] p {{ color: {TEXT_MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; }}
div[data-testid="stMetricValue"] {{ font-weight: 650; }}
.oak-hero {{ display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; margin-bottom:.4rem; }}
.oak-hero h1 {{ margin:0; padding:0; font-size:2rem; }}
.oak-sub {{ color:{TEXT_MUTED}; margin-top:.2rem; font-size:.95rem; }}
.oak-badges {{ display:flex; flex-wrap:wrap; gap:.4rem; margin:.5rem 0 1rem; }}
.oak-badge {{
  display:inline-flex; align-items:center; gap:.35rem; padding:.18rem .62rem; border-radius:999px;
  font-size:.78rem; font-weight:500; border:1px solid {BORDER}; background:{SURFACE_2}; color:{TEXT_2};
}}
.oak-dot {{ width:.55rem; height:.55rem; border-radius:50%; display:inline-block; }}
.oak-card {{
  background:{SURFACE_2}; border:1px solid {BORDER}; border-radius:14px; padding:16px 18px; height:100%;
}}
.oak-card .lbl {{ color:{TEXT_MUTED}; font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; }}
.oak-card .val {{ color:{TEXT}; font-size:1.65rem; font-weight:650; margin:.15rem 0; }}
.oak-card .name {{ color:{TEXT_2}; font-size:.85rem; word-break:break-all; }}
.oak-empty {{
  border:1px dashed {BORDER}; border-radius:14px; padding:22px; color:{TEXT_MUTED}; text-align:center;
}}
.oak-path {{ font-family:'JetBrains Mono', monospace; font-size:.8rem; color:{TEXT_2}; word-break:break-all; }}
</style>
"""


def setup_page(title: str) -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    sidebar_footer()


def sidebar_footer() -> None:
    with st.sidebar:
        if st.button("Rescan disk", icon=":material/refresh:", width="stretch",
                     help="Re-read models/, checkpoints/, logs/ and data/metrics/"):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"Repo: `{REPO_ROOT.name}`")


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------

def _registry_version() -> float:
    """Changes whenever run_registry.py is edited, so cached Run objects/frames built by
    older code (missing new fields or columns) are never served to newer pages."""
    return _mtime(Path(reg.__file__))


@st.cache_data(show_spinner="Scanning runs…", ttl=600)
def _load_runs(version: float) -> list[reg.Run]:
    return reg.discover_runs(REPO_ROOT)


def load_runs() -> list[reg.Run]:
    return _load_runs(_registry_version())


@st.cache_data(show_spinner=False, ttl=600)
def _load_runs_frame(version: float) -> pd.DataFrame:
    return reg.runs_frame(_load_runs(version))


def load_runs_frame() -> pd.DataFrame:
    return _load_runs_frame(_registry_version())


@st.cache_data(show_spinner=False, ttl=600)
def _load_studies(version: float) -> list[reg.StudyRef]:
    return reg.discover_studies(REPO_ROOT)


def load_studies() -> list[reg.StudyRef]:
    return _load_studies(_registry_version())


def get_run(key: str) -> reg.Run | None:
    return next((r for r in load_runs() if r.key == key), None)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False, max_entries=256)
def _tb_scalars(path: str, mtime: float) -> pd.DataFrame:
    return reg.load_tb_scalars(Path(path))


def tb_scalars(path: Path) -> pd.DataFrame:
    newest = max((_mtime(p) for p in path.glob("events.out.tfevents*")), default=0.0)
    return _tb_scalars(str(path), newest)


@st.cache_data(show_spinner=False, max_entries=256)
def _ultra(path: str, mtime: float) -> pd.DataFrame:
    return reg.load_ultralytics_results(Path(path))


def ultra_results(path: Path) -> pd.DataFrame:
    return _ultra(str(path), _mtime(path))


@st.cache_data(show_spinner=False, max_entries=256)
def _history(path: str, mtime: float) -> pd.DataFrame:
    return reg.load_history(Path(path))


def history(path: Path) -> pd.DataFrame:
    return _history(str(path), _mtime(path))


@st.cache_data(show_spinner=False, max_entries=64)
def _trials(db: str, study: str, mtime: float) -> pd.DataFrame:
    return reg.load_trials_df(Path(db), study)


def trials_df(db: Path, study: str) -> pd.DataFrame:
    return _trials(str(db), study, _mtime(db))


@st.cache_resource(show_spinner=False, max_entries=16)
def _study(db: str, study: str, mtime: float):
    return reg.load_study(Path(db), study)


def study_obj(db: Path, study: str):
    return _study(str(db), study, _mtime(db))


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

def hero(title: str, subtitle: str = "") -> None:
    sub = f'<div class="oak-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(f'<div class="oak-hero"><div><h1>{html.escape(title)}</h1>{sub}</div></div>',
                unsafe_allow_html=True)


def badges(items: list[tuple[str, str | None]]) -> None:
    """Render pill badges; each item is (text, optional dot color)."""
    parts = []
    for text, color in items:
        dot = f'<span class="oak-dot" style="background:{color}"></span>' if color else ""
        parts.append(f'<span class="oak-badge">{dot}{html.escape(text)}</span>')
    st.markdown(f'<div class="oak-badges">{"".join(parts)}</div>', unsafe_allow_html=True)


def card(label: str, value: str, name: str = "", color: str | None = None) -> None:
    dot = f'<span class="oak-dot" style="background:{color}"></span> ' if color else ""
    st.markdown(
        f'<div class="oak-card"><div class="lbl">{dot}{html.escape(label)}</div>'
        f'<div class="val">{html.escape(value)}</div><div class="name">{html.escape(name)}</div></div>',
        unsafe_allow_html=True,
    )


def empty(message: str) -> None:
    st.markdown(f'<div class="oak-empty">{html.escape(message)}</div>', unsafe_allow_html=True)


def fmt_metric(value: float | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{digits}f}"


def rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def open_run(key: str) -> None:
    st.session_state["run_key"] = key
    st.switch_page(PAGES["detail"])


def open_study(db: Path, study_name: str) -> None:
    st.session_state["optuna_db"] = str(db)
    st.session_state["optuna_study"] = study_name
    st.switch_page(PAGES["optuna"])


PAGES = {
    "home": "pages/0_Overview.py",
    "runs": "pages/1_Runs.py",
    "detail": "pages/2_Run_Detail.py",
    "optuna": "pages/3_Optuna.py",
    "compare": "pages/4_Compare.py",
}


def class_count_label(n: int) -> str:
    return f"{n} class" if n == 1 else f"{n} classes"


def run_filters(df: pd.DataFrame, key: str = "f") -> pd.DataFrame:
    """Global sidebar filters shared by list pages. Selections persist across pages."""
    with st.sidebar:
        st.subheader("Filters")
        families = [f for f in FAMILY_COLORS if f in set(df["family"])]
        chosen = st.pills("Family", families, selection_mode="multi", default=families, key="flt_family")
        archs = sorted(df["arch"].dropna().unique())
        arch_sel = st.multiselect("Architecture", archs, key="flt_arch", placeholder="All")
        counts = sorted(int(n) for n in df["n_classes"].dropna().unique())
        count_opts = [class_count_label(n) for n in counts] + (["Unknown"] if df["n_classes"].isna().any() else [])
        count_sel = st.pills("Classes", count_opts, selection_mode="multi", key="flt_ncls",
                             help="Foreground classes the model predicts. Segmentation counts exclude "
                                  "Background, so mIoU over 4 outputs vs 5 outputs is not comparable.")
        sets = sorted(df.loc[df["class_names"].notna(), "classes"].unique())
        set_sel = st.multiselect("Class set", sets, key="flt_clsset", placeholder="Any",
                                 help="Exact class names (only known for runs with a summary row, "
                                      "YOLO weights or an NHLA class mapping).")
        status = st.segmented_control("Status", ["All", "Complete", "Partial"], default="All", key="flt_status",
                                      help="Complete runs have final weights; partial runs only have logs/checkpoints.")
        only_study = st.toggle("Has Optuna study", key="flt_study")
        only_onnx = st.toggle("Has ONNX export", key="flt_onnx")
        text = st.text_input("Search name", key="flt_text", placeholder="e.g. TILE, 2026-09")
    out = df[df["family"].isin(chosen or families)]
    if arch_sel:
        out = out[out["arch"].isin(arch_sel)]
    if count_sel:
        labels = out["n_classes"].map(lambda n: "Unknown" if pd.isna(n) else class_count_label(int(n)))
        out = out[labels.isin(count_sel)]
    if set_sel:
        out = out[out["classes"].isin(set_sel)]
    if status == "Complete":
        out = out[out["status"] == "complete"]
    elif status == "Partial":
        out = out[out["status"] == "partial"]
    if only_study:
        out = out[out["has_study"]]
    if only_onnx:
        out = out[out["has_onnx"]]
    if text:
        mask = out["name"].str.contains(text, case=False, regex=False)
        mask |= out["date"].astype(str).str.contains(text, regex=False)
        out = out[mask]
    return out
