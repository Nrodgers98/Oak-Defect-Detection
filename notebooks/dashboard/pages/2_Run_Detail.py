from pathlib import Path

import pandas as pd
import streamlit as st

import charts
import optuna_view
import ui
from functions import run_registry as reg

runs = ui.load_runs()
if not runs:
    ui.empty("No runs found.")
    st.stop()

keys = [r.key for r in runs]
by_key = {r.key: r for r in runs}
requested = st.query_params.get("run") or st.session_state.get("run_key")
index = keys.index(requested) if requested in keys else 0

key = st.selectbox(
    "Run", keys, index=index, key="detail_select",
    format_func=lambda k: f"{by_key[k].name}  ·  {reg.FAMILY_LABELS[by_key[k].family]}",
)
st.session_state["run_key"] = key
st.query_params["run"] = key
run = by_key[key]
family = reg.FAMILY_LABELS[run.family]

ui.hero(run.name, run.ts.strftime("Started %Y-%m-%d %H:%M:%S") if run.ts else "Legacy run (no timestamp)")
badge_items = [(family, ui.FAMILY_COLORS[family]), (run.arch, None), (run.variant, None),
               (f"classes · {reg.class_label(run.n_classes, run.class_names)}", None),
               ("final weights" if run.status == "complete" else "partial · no final weights", None)]
if run.onnx:
    badge_items.append(("ONNX exported", None))
if run.study:
    badge_items.append((f"Optuna · {run.study.study_name}" + (" (inferred)" if run.study.confidence == "inferred" else ""), None))
ui.badges(badge_items)

value, label = reg.headline_metric(run)
scalars = reg.scalar_metrics(run)
row = run.summary_row or {}
c = st.columns(4)
c[0].metric(label or "Score", ui.fmt_metric(value))
curves = charts.curves_long(run)
epochs = row.get("num_epochs_actual") or (int(curves["step"].nunique()) if not curves.empty else "—")
c[1].metric("Epochs", epochs, help="From the training summary CSV, else the number of logged steps")
c[2].metric("Trials", run.study.n_trials if run.study else "—")
c[3].metric("Size on disk", f"{run.size_bytes / 1e6:,.0f} MB")

tab_curves, tab_metrics, tab_optuna, tab_files = st.tabs(
    [":material/show_chart: Training curves", ":material/analytics: Metrics",
     ":material/tune: Optuna", ":material/folder_open: Artifacts"])

with tab_curves:
    ultra_labels = [k for k in run.ultra_results if not k.startswith("optuna/")]
    chosen = None
    if len(ultra_labels) > 1:
        chosen = st.segmented_control("Ultralytics run", ultra_labels, default=ultra_labels[0], key="ultra_pick")
    charts.render_curves(curves if chosen in (None, ultra_labels[0] if ultra_labels else None)
                         else charts.curves_long(run, ultra_label=chosen))

with tab_metrics:
    if scalars:
        items = list(scalars.items())
        for start in range(0, len(items), 6):
            cols = st.columns(6)
            for col, (k, v) in zip(cols, items[start:start + 6]):
                col.metric(k.replace("_", " "), ui.fmt_metric(v, 4))
    else:
        ui.empty("No scalar test metrics recorded for this run (no summary CSV row or metrics JSON).")

    pc = reg.per_class_metrics(run)
    cm = reg.confusion_matrix(run)
    if pc is not None or cm is not None:
        left, right = st.columns(2)
        if pc is not None:
            with left:
                st.plotly_chart(charts.per_class_figure(pc), width="stretch")
                with st.expander("Per-class table"):
                    st.dataframe(pc.pivot(index="class", columns="metric", values="value"),
                                 width="stretch")
        if cm is not None:
            with right:
                st.plotly_chart(charts.confusion_figure(*cm), width="stretch")

    images = run.images()
    if images:
        st.markdown("**Evaluation plots**")
        cols = st.columns(3)
        for i, img in enumerate(images[:12]):
            with cols[i % 3]:
                st.image(str(img), caption=ui.rel(img), width="stretch")

    params = reg.hyperparameters(run)
    if params:
        st.markdown("**Hyperparameters**")
        st.dataframe(pd.DataFrame({"parameter": list(params), "value": [str(v) for v in params.values()]}),
                     hide_index=True, width="stretch", height=min(36 * len(params) + 40, 520))

with tab_optuna:
    if run.study:
        if run.study.confidence == "inferred":
            st.info("This study is linked by name and time overlap, not an explicit record. "
                    "Notebooks often resume an earlier study, so check the trial dates.", icon=":material/link:")
        if st.button("Open in Optuna page", icon=":material/open_in_new:", type="tertiary"):
            ui.open_study(run.study.db_path, run.study.study_name)
        optuna_view.render_study(run.study.db_path, run.study.study_name, key=f"detail-{run.key}", run=run)
    else:
        ui.empty("No Optuna study linked to this run.")

with tab_files:
    groups = {
        "Weights": run.weights,
        "ONNX": run.onnx,
        "TensorBoard logs": run.tb_dirs,
        "Training history": run.history_files,
        "Ultralytics results": list(run.ultra_results.values()),
        "Metrics files": run.all_metric_paths(),
    }
    dirs = [("Model dir", run.model_dir), ("Checkpoint dir", run.checkpoint_dir)]
    for label_, path in dirs:
        if path:
            n_ckpt = len(list(path.glob("*.pt"))) if label_ == "Checkpoint dir" else None
            extra = f" · {n_ckpt} checkpoints" if n_ckpt else ""
            st.markdown(f"**{label_}**{extra}")
            st.code(str(path.resolve()), language=None)
    for label_, paths in groups.items():
        if not paths:
            continue
        rows = []
        for p in paths:
            p = Path(p)
            size = p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.glob("*") if f.is_file())
            rows.append({"path": ui.rel(p), "size_mb": size / 1e6})
        st.markdown(f"**{label_}**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"path": st.column_config.TextColumn("Path", width="large"),
                                    "size_mb": st.column_config.NumberColumn("Size", format="%.2f MB")})
    if run.study:
        st.markdown("**Optuna storage**")
        st.code(f"sqlite:///{run.study.db_path.resolve()}  ·  study: {run.study.study_name}", language=None)
