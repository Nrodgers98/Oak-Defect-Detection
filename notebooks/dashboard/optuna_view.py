"""Optuna study panel used by the Optuna page and the Run detail page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import charts
import ui


def _safe_plot(builder, *args, height: int = 380, **kwargs) -> None:
    try:
        fig = builder(*args, **kwargs)
    except Exception as exc:  # optuna raises for too few trials, single params, etc.
        ui.empty(f"Not available: {exc}")
        return
    st.plotly_chart(charts.style_optuna(fig, height), width="stretch")


def render_study(db: Path, study_name: str, key: str, run=None) -> None:
    from optuna.importance import get_param_importances
    from optuna.trial import TrialState
    import optuna.visualization as ov

    try:
        study = ui.study_obj(db, study_name)
        trials = ui.trials_df(db, study_name)
    except Exception as exc:
        ui.empty(f"Could not open study {study_name}: {exc}")
        return

    states = trials["state"].value_counts() if "state" in trials else pd.Series(dtype=int)
    complete = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]
    direction = study.direction.name.lower() if len(study.directions) == 1 else "multi-objective"

    c = st.columns(5)
    best_value = study.best_value if complete and len(study.directions) == 1 else None
    c[0].metric("Best value", ui.fmt_metric(best_value, 4), help=f"Direction: {direction}")
    c[1].metric("Trials", len(trials))
    c[2].metric("Complete", int(states.get("COMPLETE", 0)))
    c[3].metric("Pruned", int(states.get("PRUNED", 0)))
    c[4].metric("Failed / running", int(states.get("FAIL", 0) + states.get("RUNNING", 0)))
    ui.badges([(f"DB · {ui.rel(db)}", None), (f"study · {study_name}", None), (f"{direction}", None)])

    if not complete:
        ui.empty("No completed trials yet, so there is nothing to plot.")
    else:
        if complete and len(study.directions) == 1:
            best = study.best_trial
            with st.expander(f"Best trial #{best.number} · value {best.value:.4f}", expanded=True):
                st.json({"params": best.params, "user_attrs": best.user_attrs}, expanded=True)

        params = sorted({p for t in complete for p in t.params})
        tabs = st.tabs(["History", "Importance", "Parallel", "Slice", "Contour", "Timeline"])
        with tabs[0]:
            _safe_plot(ov.plot_optimization_history, study)
        with tabs[1]:
            if len(complete) < 2 or len(params) < 2:
                ui.empty("Parameter importance needs at least 2 completed trials and 2 parameters.")
            else:
                try:
                    imp = get_param_importances(study)
                    imp_df = pd.DataFrame({"param": list(imp), "importance": list(imp.values())})
                    import plotly.graph_objects as go
                    fig = go.Figure(go.Bar(
                        x=imp_df["importance"], y=imp_df["param"], orientation="h",
                        marker=dict(color=ui.SERIES[0], cornerradius=4),
                        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
                    ))
                    fig.update_layout(title="Hyperparameter importance (fANOVA)", height=60 + 36 * len(imp_df),
                                      yaxis=dict(autorange="reversed"), xaxis_title="importance")
                    st.plotly_chart(fig, width="stretch")
                except Exception as exc:
                    ui.empty(f"Not available: {exc}")
        with tabs[2]:
            _safe_plot(ov.plot_parallel_coordinate, study, height=420)
        with tabs[3]:
            chosen = st.multiselect("Parameters", params, default=params[:4], key=f"{key}-slice")
            if chosen:
                _safe_plot(ov.plot_slice, study, params=chosen, height=360)
        with tabs[4]:
            if len(params) < 2:
                ui.empty("Contour plots need at least two parameters.")
            else:
                a, b = st.columns(2)
                px_ = a.selectbox("X", params, index=0, key=f"{key}-cx")
                py_ = b.selectbox("Y", params, index=1, key=f"{key}-cy")
                if px_ != py_:
                    _safe_plot(ov.plot_contour, study, params=[px_, py_], height=460)
        with tabs[5]:
            _safe_plot(ov.plot_timeline, study)

    st.markdown("**Trials**")
    show = trials.copy()
    show.columns = [c.replace("params_", "").replace("user_attrs_", "attr: ") for c in show.columns]
    st.dataframe(show.sort_values("value", ascending=study.direction.name == "MINIMIZE", na_position="last")
                 if "value" in show and len(study.directions) == 1 else show,
                 hide_index=True, width="stretch",
                 column_config={"value": st.column_config.NumberColumn("value", format="%.4f"),
                                "duration_min": st.column_config.NumberColumn("minutes", format="%.1f"),
                                "datetime_start": st.column_config.DatetimeColumn("started", format="MM-DD HH:mm")},
                 key=f"{key}-trials")

    # YOLO pith runs keep one Ultralytics run per trial: optuna/trial_NNNN/results.csv
    trial_runs = {k: v for k, v in (run.ultra_results.items() if run else []) if k.startswith("optuna/")}
    if trial_runs:
        st.markdown("**Per-trial training curves**")
        label = st.selectbox("Trial folder", sorted(trial_runs), key=f"{key}-trialdir")
        charts.render_curves(charts.curves_long(run, ultra_label=label))
