"""Plotly chart builders for training curves, metrics and Optuna studies."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ui
from functions import run_registry as reg

SERIES_COLORS = {"train": ui.SERIES[0], "val": ui.SERIES[1], "test": ui.SERIES[2],
                 "box": ui.SERIES[0], "mask": ui.SERIES[1], "": ui.SERIES[0]}
_SPLITS = {"train", "val", "test", "valid", "validation"}
_SKIP_GROUPS = {"time", "epoch", "step"}


def split_tag(tag: str) -> tuple[str, str]:
    """Map a scalar tag/column to (chart group, series within the chart).

    ``Loss/Train`` -> (loss, train), ``val/macro_f1`` -> (macro_f1, val),
    ``train_loss`` -> (loss, train), ``metrics/mAP50(M)`` -> (mAP50, mask),
    ``lr/pg0`` -> (lr, pg0).
    """
    t = tag.strip()
    suffix = ""
    m = re.match(r"^(.*)\(([BM])\)$", t)
    if m:
        t, suffix = m.group(1), {"B": "box", "M": "mask"}[m.group(2)]
    if "/" in t:
        a, b = t.split("/", 1)
        al, bl = a.lower(), b.lower()
        if al == "lr":
            group, series = "lr", b
        elif al in _SPLITS:
            group, series = b, al
        elif al == "metrics":
            group, series = b, ""
        elif bl in _SPLITS:
            group, series = a, bl
        else:
            group, series = a, b
    else:
        m = re.match(r"^(train|val|test)_(.+)$", t, re.IGNORECASE)
        group, series = (m.group(2), m.group(1).lower()) if m else (t, "")
    series = suffix or series
    if series == "validation" or series == "valid":
        series = "val"
    return group.lower() if group.lower() in {"loss", "lr"} else group, series


def _lower_is_better(group: str) -> bool:
    return "loss" in group.lower()


def curves_long(run: reg.Run, ultra_label: str | None = None) -> pd.DataFrame:
    """All per-epoch series for a run as long frame: source, group, series, step, value."""
    frames = []
    for d in run.tb_dirs:
        df = ui.tb_scalars(d)
        if df.empty:
            continue
        df = df.copy()
        df["source"] = "TensorBoard · " + ui.rel(d).split("/")[-1]
        frames.append(df[["source", "tag", "step", "value"]])
    for p in run.history_files:
        try:
            h = ui.history(p)
        except Exception:
            continue
        cols = [c for c in h.columns if c != "epoch" and pd.api.types.is_numeric_dtype(h[c])]
        if not cols:
            continue
        long = h.melt(id_vars="epoch", value_vars=cols, var_name="tag", value_name="value")
        long = long.rename(columns={"epoch": "step"})
        long["source"] = "History · " + p.name
        frames.append(long[["source", "tag", "step", "value"]])
    labels = [ultra_label] if ultra_label else [k for k in run.ultra_results if not k.startswith("optuna/")][:1]
    for label in labels:
        if label not in run.ultra_results:
            continue
        r = ui.ultra_results(run.ultra_results[label])
        cols = [c for c in r.columns if c not in ("epoch", "time") and pd.api.types.is_numeric_dtype(r[c])]
        long = r.melt(id_vars="epoch", value_vars=cols, var_name="tag", value_name="value")
        long = long.rename(columns={"epoch": "step"})
        long["source"] = "Ultralytics · " + label
        frames.append(long[["source", "tag", "step", "value"]])
    if not frames:
        return pd.DataFrame(columns=["source", "tag", "group", "series", "step", "value"])
    out = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    split = out["tag"].map(split_tag)
    out["group"] = split.map(lambda x: x[0])
    out["series"] = split.map(lambda x: x[1])
    return out[~out["group"].str.lower().isin(_SKIP_GROUPS)]


def curve_figure(df: pd.DataFrame, group: str, height: int = 280) -> go.Figure:
    """One small-multiple panel: every series of one group, best point marked."""
    fig = go.Figure()
    sub = df[df["group"] == group]
    series_names = list(dict.fromkeys(sub["series"]))
    lower = _lower_is_better(group)
    for i, name in enumerate(series_names):
        s = sub[sub["series"] == name].sort_values("step")
        color = SERIES_COLORS.get(name, ui.SERIES[i % len(ui.SERIES)])
        fig.add_trace(go.Scatter(
            x=s["step"], y=s["value"], mode="lines", name=name or group,
            line=dict(color=color, width=2), hovertemplate="%{y:.4f}<extra>" + (name or group) + "</extra>",
        ))
    # Mark the best value of the series that decides model selection (val if present).
    target = "val" if "val" in series_names else (series_names[-1] if series_names else None)
    if target is not None and group != "lr":
        s = sub[sub["series"] == target]
        if not s.empty:
            idx = s["value"].idxmin() if lower else s["value"].idxmax()
            best = s.loc[idx]
            fig.add_trace(go.Scatter(
                x=[best["step"]], y=[best["value"]], mode="markers", showlegend=False,
                marker=dict(size=10, color=SERIES_COLORS.get(target, ui.SERIES[0]),
                            line=dict(color=ui.SURFACE, width=2)),
                hovertemplate=f"best {target}: %{{y:.4f}} @ epoch %{{x}}<extra></extra>",
            ))
            fig.add_annotation(x=best["step"], y=best["value"], text=f"{best['value']:.3f}",
                               showarrow=False, yshift=14, font=dict(color=ui.TEXT, size=11))
    fig.update_layout(title=group, height=height, hovermode="x unified",
                      showlegend=len(series_names) >= 2, xaxis_title="epoch")
    return fig


def _group_order(groups: list[str]) -> list[str]:
    def rank(g: str) -> tuple[int, str]:
        gl = g.lower()
        if gl == "loss":
            return 0, g
        if "loss" in gl:
            return 1, g
        if gl == "lr":
            return 9, g
        return 2, g
    return sorted(groups, key=rank)


def render_curves(df: pd.DataFrame, columns: int = 3) -> None:
    if df.empty:
        ui.empty("No training curves found for this run (no TensorBoard events, history or results.csv).")
        return
    for source, sdf in df.groupby("source", sort=False):
        st.markdown(f"**{source}**")
        groups = _group_order(list(dict.fromkeys(sdf["group"])))
        for start in range(0, len(groups), columns):
            cols = st.columns(columns)
            for col, group in zip(cols, groups[start:start + columns]):
                with col:
                    st.plotly_chart(curve_figure(sdf, group), width="stretch",
                                    key=f"curve-{source}-{group}")


def per_class_figure(pc: pd.DataFrame) -> go.Figure:
    metrics = list(dict.fromkeys(pc["metric"]))
    fig = go.Figure()
    for i, metric in enumerate(metrics):
        s = pc[pc["metric"] == metric]
        fig.add_trace(go.Bar(x=s["class"], y=s["value"], name=metric, marker_color=ui.SERIES[i],
                             marker_line=dict(color=ui.SURFACE, width=2), marker_cornerradius=4,
                             hovertemplate="%{x}: %{y:.3f}<extra>" + metric + "</extra>"))
    fig.update_layout(barmode="group", title="Per-class metrics", height=340, yaxis_range=[0, 1],
                      bargap=0.25, bargroupgap=0.08, showlegend=len(metrics) >= 2)
    return fig


def confusion_figure(cm: list[list[float]], labels: list[str]) -> go.Figure:
    z = pd.DataFrame(cm, index=labels, columns=labels)
    norm = z.div(z.sum(axis=1).replace(0, 1), axis=0)
    fig = px.imshow(norm, text_auto=False, color_continuous_scale=["#1d2a3c", "#3987e5", "#cde2fb"],
                    zmin=0, zmax=1, aspect="auto", labels=dict(x="Predicted", y="Actual", color="Row %"))
    fig.update_traces(text=z.values, texttemplate="%{text}", customdata=norm.values,
                      hovertemplate="actual %{y} → predicted %{x}<br>%{text} (%{customdata:.0%})<extra></extra>")
    fig.update_layout(title="Confusion matrix (row-normalized color, counts shown)", height=420)
    return fig


def timeline_figure(df: pd.DataFrame) -> go.Figure:
    """Headline metric over time, one trace per family (color + symbol)."""
    fig = go.Figure()
    d = df.dropna(subset=["metric", "date"])
    for fam, color in ui.FAMILY_COLORS.items():
        s = d[d["family"] == fam]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s["date"], y=s["metric"], mode="markers", name=fam,
            marker=dict(size=11, color=color, symbol=ui.FAMILY_SYMBOLS[fam], line=dict(color=ui.SURFACE, width=2)),
            customdata=s[["name", "metric_label"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}: %{y:.3f}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
        ))
    fig.update_layout(title="Headline metric by run date", height=360, yaxis_title="score",
                      hovermode="closest", showlegend=True)
    return fig


def style_optuna(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(template="plotly_dark+oak", height=height)
    return fig
