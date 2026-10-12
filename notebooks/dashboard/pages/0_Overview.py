import pandas as pd
import streamlit as st

import charts
import ui

ui.hero("Oak model dashboard", "Every training run, its Optuna study, curves and metrics, in one place.")

df = ui.load_runs_frame()
studies = ui.load_studies()

if df.empty:
    ui.empty("No runs found under models/, checkpoints/ or logs/tensorboard/.")
    st.stop()

complete = df[df["status"] == "complete"]
c = st.columns(5)
c[0].metric("Runs", len(df), help="Distinct runs across models/, checkpoints/ and logs/tensorboard/")
c[1].metric("With final weights", len(complete))
c[2].metric("Optuna studies", len(studies))
c[3].metric("Completed trials", int(sum(s.n_complete for s in studies)),
            help=f"{sum(s.n_trials for s in studies)} trials in total, including pruned/failed")
c[4].metric("ONNX exports", int(df["has_onnx"].sum()))

st.markdown("### Best run per family")
cols = st.columns(len(ui.FAMILY_COLORS))
for col, (family, color) in zip(cols, ui.FAMILY_COLORS.items()):
    fam = df[(df["family"] == family) & df["metric"].notna()]
    with col:
        if fam.empty:
            ui.card(family, "—", "no scored runs", color)
            continue
        n_opts = sorted(int(n) for n in fam["n_classes"].dropna().unique())
        if len(n_opts) > 1:
            # Scores are only comparable within one class set, so pick which one to rank.
            labels = {ui.class_count_label(n): n for n in n_opts}
            pick = st.segmented_control("Classes", list(labels), default=list(labels)[-1],
                                        key=f"best-n-{family}", label_visibility="collapsed")
            if pick is not None:
                fam = fam[fam["n_classes"] == labels[pick]]
        best = fam.loc[fam["metric"].idxmax()]
        ui.card(f"{family} · {best['metric_label']}", ui.fmt_metric(best["metric"]),
                f"{best['name']} · {best['classes']}", color)
        if st.button("Open", key=f"best-{family}", icon=":material/arrow_forward:", type="tertiary"):
            ui.open_run(best["key"])

st.markdown("### ")
left, right = st.columns([3, 2])
with left:
    st.plotly_chart(charts.timeline_figure(df), width="stretch")
    st.caption("Metrics differ by family (mIoU, mAP50-95, macro F1), so compare points within one family.")
with right:
    st.markdown("**Recent runs**")
    recent = df.dropna(subset=["date"]).sort_values("date", ascending=False).head(10)
    event = st.dataframe(
        recent[["name", "family", "metric", "status"]],
        hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
        column_config={
            "name": st.column_config.TextColumn("Run", width="large"),
            "family": "Family",
            "metric": st.column_config.NumberColumn("Score", format="%.3f"),
            "status": "Status",
        },
        key="recent",
    )
    if event.selection.rows:
        ui.open_run(recent.iloc[event.selection.rows[0]]["key"])
    st.caption("Select a row to open the run.")

st.markdown("### Runs by family")
counts = df.groupby(["family", "status"]).size().unstack(fill_value=0)
counts = counts.reindex([f for f in ui.FAMILY_COLORS if f in counts.index])
counts["total"] = counts.sum(axis=1)
st.dataframe(counts, width="stretch")
