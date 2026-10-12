import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import charts
import ui
from functions import run_registry as reg

ui.hero("Compare runs", "Overlay training curves and line up metrics and hyperparameters.")

runs = [r for r in ui.load_runs() if r.tb_dirs or r.history_files or r.ultra_results or r.summary_row]
by_key = {r.key: r for r in runs}
default = [k for k in st.session_state.get("compare_keys", []) if k in by_key]
if not default:
    # Most recent scored runs from the family of the latest one, so the overlay is meaningful.
    scored = ui.load_runs_frame().dropna(subset=["metric", "date"])
    scored = scored[scored["key"].isin(by_key)].sort_values("date", ascending=False)
    if not scored.empty:
        default = list(scored[scored["family"] == scored.iloc[0]["family"]]["key"].head(3))

keys = st.multiselect(
    "Runs (up to 8)", list(by_key), default=default, max_selections=len(ui.SERIES),
    format_func=lambda k: f"{by_key[k].name} · {reg.FAMILY_LABELS[by_key[k].family]}",
)
st.session_state["compare_keys"] = keys
if len(keys) < 2:
    ui.empty("Pick at least two runs.")
    st.stop()

selected = [by_key[k] for k in keys]
# Color follows the run (selection order), so removing a run never repaints the others.
colors = {r.key: ui.SERIES[i] for i, r in enumerate(selected)}
families = {r.family for r in selected}
class_sets = {reg.class_label(r.n_classes, r.class_names) for r in selected}
if len(families) == 1 and len(class_sets) > 1:
    st.warning("These runs predict different class sets (" + " | ".join(sorted(class_sets)) + "). "
               "mIoU and loss are not directly comparable; for segmentation, compare "
               "`mean iou defects` (Background excluded) in the table below.", icon=":material/warning:")
if len(families) > 1:
    st.warning("These runs come from different families, so their metrics measure different things. "
               "Curves are only overlaid where the metric names match.", icon=":material/warning:")

curves = {r.key: charts.curves_long(r) for r in selected}
pairs = sorted({(g, s) for df in curves.values() if not df.empty for g, s in zip(df["group"], df["series"])})
if not pairs:
    ui.empty("None of these runs have training curves.")
else:
    common = [p for p in pairs if sum(((curves[k]["group"] == p[0]) & (curves[k]["series"] == p[1])).any()
                                      for k in keys if not curves[k].empty) >= 2] or pairs
    labels = [f"{g} · {s}" if s else g for g, s in common]
    pref = next((i for i, (g, s) in enumerate(common) if s == "val" and "loss" not in g.lower()), 0)
    a, b = st.columns([3, 1])
    pick = a.selectbox("Metric", range(len(common)), index=pref, format_func=lambda i: labels[i])
    smooth = b.slider("Smoothing", 0.0, 0.95, 0.0, 0.05, help="Exponential moving average weight")
    group, series = common[pick]

    fig = go.Figure()
    for r in selected:
        df = curves[r.key]
        if df.empty:
            continue
        s = df[(df["group"] == group) & (df["series"] == series)].sort_values("step")
        if s.empty:
            continue
        s = s.groupby("step", as_index=False)["value"].mean()
        y = s["value"].ewm(alpha=1 - smooth).mean() if smooth else s["value"]
        fig.add_trace(go.Scatter(x=s["step"], y=y, mode="lines", name=r.name,
                                 line=dict(color=colors[r.key], width=2),
                                 hovertemplate="%{y:.4f}<extra>" + r.name + "</extra>"))
    fig.update_layout(title=labels[pick], height=440, hovermode="x unified", xaxis_title="epoch",
                      legend=dict(orientation="h", y=-0.2, yanchor="top"))
    st.plotly_chart(fig, width="stretch")

st.markdown("### Side by side")
rows = {}
for r in selected:
    value, label = reg.headline_metric(r)
    col = {"family": reg.FAMILY_LABELS[r.family], "arch": r.arch, "variant": r.variant,
           "classes": reg.class_label(r.n_classes, r.class_names),
           "started": r.ts.strftime("%Y-%m-%d %H:%M") if r.ts else "legacy",
           f"score ({label})" if label else "score": ui.fmt_metric(value, 4)}
    col.update({k: ui.fmt_metric(v, 4) for k, v in reg.scalar_metrics(r).items()})
    col.update({f"hp · {k}": str(v) for k, v in reg.hyperparameters(r).items()})
    rows[r.name] = col
table = pd.DataFrame(rows)
only_diff = st.toggle("Only rows that differ", value=True)
if only_diff:
    table = table[table.nunique(axis=1, dropna=False) > 1]


def _highlight(row: pd.Series) -> list[str]:
    return ["background-color: rgba(224,160,64,0.12)" if row.nunique(dropna=False) > 1 else "" for _ in row]


st.dataframe(table.fillna("—").style.apply(_highlight, axis=1), width="stretch",
             height=min(36 * len(table) + 40, 700))
