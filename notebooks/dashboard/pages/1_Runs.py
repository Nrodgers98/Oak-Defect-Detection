import streamlit as st

import ui

ui.hero("Runs", "Filter in the sidebar. Select rows to open a run or compare several.")

df = ui.load_runs_frame()
view = ui.run_filters(df).sort_values("date", ascending=False, na_position="last")

c = st.columns(4)
c[0].metric("Showing", f"{len(view)} / {len(df)}")
c[1].metric("Complete", int((view["status"] == "complete").sum()))
c[2].metric("Linked to a study", int(view["has_study"].sum()))
c[3].metric("Total size", f"{view['size_mb'].sum() / 1000:.1f} GB")

if view.empty:
    ui.empty("No runs match the current filters.")
    st.stop()

max_metric = float(view["metric"].max()) if view["metric"].notna().any() else 1.0
event = st.dataframe(
    view[["name", "family", "arch", "variant", "n_classes", "classes", "date", "status", "metric", "metric_label",
          "has_curves", "has_study", "trials", "has_onnx", "size_mb"]],
    hide_index=True, width="stretch", height=620,
    on_select="rerun", selection_mode="multi-row", key="runs_table",
    column_config={
        "name": st.column_config.TextColumn("Run", width="large", pinned=True),
        "family": "Family",
        "arch": "Arch",
        "variant": "Variant",
        "date": st.column_config.DatetimeColumn("Started", format="YYYY-MM-DD HH:mm"),
        "status": "Status",
        "metric": st.column_config.ProgressColumn("Score", format="%.3f", min_value=0.0,
                                                  max_value=max(max_metric, 1e-9)),
        "metric_label": st.column_config.TextColumn("Metric", width="small"),
        "n_classes": st.column_config.NumberColumn("# Cls", format="%d",
                                                   help="Foreground classes (segmentation excludes Background)"),
        "classes": st.column_config.TextColumn("Classes", width="medium"),
        "has_curves": st.column_config.CheckboxColumn("Curves"),
        "has_study": st.column_config.CheckboxColumn("Optuna"),
        "trials": st.column_config.NumberColumn("Trials"),
        "has_onnx": st.column_config.CheckboxColumn("ONNX"),
        "size_mb": st.column_config.NumberColumn("Size", format="%.0f MB"),
    },
)

selected = view.iloc[event.selection.rows] if event.selection.rows else view.iloc[0:0]
b1, b2, _ = st.columns([1, 1, 4])
if b1.button("Open run", icon=":material/insights:", disabled=len(selected) != 1, type="primary",
             width="stretch"):
    ui.open_run(selected.iloc[0]["key"])
if b2.button(f"Compare ({len(selected)})", icon=":material/compare_arrows:", disabled=len(selected) < 2,
             width="stretch"):
    st.session_state["compare_keys"] = list(selected["key"])
    st.switch_page(ui.PAGES["compare"])
st.caption("Score is the family's headline metric (see the Metric column). It is not comparable across families, "
           "and segmentation mIoU (which includes Background) is only comparable between runs with the same classes.")
