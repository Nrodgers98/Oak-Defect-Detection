from pathlib import Path

import pandas as pd
import streamlit as st

import optuna_view
import ui

ui.hero("Optuna studies", "Read-only view of every study in logs/optuna/*.db.")

studies = ui.load_studies()
if not studies:
    ui.empty("No Optuna databases found in logs/optuna/.")
    st.stop()

runs = ui.load_runs()
linked: dict[tuple[str, str], list[str]] = {}
for r in runs:
    if r.study:
        linked.setdefault((str(r.study.db_path), r.study.study_name), []).append(r.name)

table = pd.DataFrame([{
    "db": s.db_path.name,
    "study": s.study_name,
    "trials": s.n_trials,
    "complete": s.n_complete,
    "first_trial": s.first_start,
    "last_trial": s.last_start,
    "runs": ", ".join(linked.get((str(s.db_path), s.study_name), [])),
    "_db_path": str(s.db_path),
} for s in studies]).sort_values("last_trial", ascending=False, na_position="last")

with st.expander(f"All studies ({len(table)})", expanded="optuna_study" not in st.session_state):
    event = st.dataframe(
        table.drop(columns=["_db_path"]), hide_index=True, width="stretch",
        on_select="rerun", selection_mode="single-row", key="study_table",
        column_config={
            "db": "Database", "study": st.column_config.TextColumn("Study", width="large"),
            "trials": st.column_config.NumberColumn("Trials"),
            "complete": st.column_config.ProgressColumn("Complete", format="%d",
                                                        min_value=0, max_value=int(table["trials"].max() or 1)),
            "first_trial": st.column_config.DatetimeColumn("First trial", format="YYYY-MM-DD HH:mm"),
            "last_trial": st.column_config.DatetimeColumn("Last trial", format="YYYY-MM-DD HH:mm"),
            "runs": st.column_config.TextColumn("Linked runs", width="large"),
        },
    )
    if event.selection.rows:
        picked = table.iloc[event.selection.rows[0]]
        st.session_state["optuna_db"] = picked["_db_path"]
        st.session_state["optuna_study"] = picked["study"]

dbs = sorted({str(s.db_path) for s in studies})
db_default = st.session_state.get("optuna_db", dbs[0])
a, b = st.columns([1, 2])
db = a.selectbox("Database", dbs, index=dbs.index(db_default) if db_default in dbs else 0,
                 format_func=lambda p: Path(p).name)
names = [s.study_name for s in studies if str(s.db_path) == db]
names.sort(key=lambda n: next((s.last_start for s in studies if s.study_name == n and str(s.db_path) == db), None)
           or pd.Timestamp.min, reverse=True)
study_default = st.session_state.get("optuna_study")
study_name = b.selectbox("Study", names, index=names.index(study_default) if study_default in names else 0)
st.session_state["optuna_db"], st.session_state["optuna_study"] = db, study_name

owners = linked.get((db, study_name), [])
if owners:
    st.caption("Linked runs:")
    cols = st.columns(min(len(owners), 4))
    for i, name in enumerate(owners[:8]):
        run = next(r for r in runs if r.name == name)
        if cols[i % len(cols)].button(name, key=f"own-{name}", icon=":material/insights:", type="tertiary"):
            ui.open_run(run.key)

owner_run = next((r for r in runs if r.name in owners), None)
optuna_view.render_study(Path(db), study_name, key=f"opt-{Path(db).stem}-{study_name}", run=owner_run)
