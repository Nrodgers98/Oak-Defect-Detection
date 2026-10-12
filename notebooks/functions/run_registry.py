"""Discover training runs on disk and link their artifacts together.

Training notebooks write model weights, checkpoints, TensorBoard logs, Optuna
studies and metrics to separate trees that are only tied together by naming
conventions (and those conventions drift: ``ResNet34_`` infixes, case changes,
``_v1`` renames, timestamps a second apart). This module rebuilds that index
read-only so the model dashboard can show everything about one run in one place.

Nothing here imports streamlit; the dashboard wraps the loaders in its own cache.
"""

from __future__ import annotations

import csv
import json
import re
import os
import pickle
import sqlite3
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

TS_TOLERANCE = timedelta(seconds=2)
INFERRED_STUDY_WINDOW = timedelta(days=2)
INFERRED_STUDY_TS_WINDOW = timedelta(minutes=5)

FAMILIES = ("segmentation", "yolo_seg", "yolo_pith", "nhla")
FAMILY_LABELS = {
    "segmentation": "Segmentation",
    "yolo_seg": "YOLO Seg",
    "yolo_pith": "YOLO Pith",
    "nhla": "NHLA Grade",
}

_TS_RE = re.compile(r"^(?P<prefix>.*?)[_-](?P<ts>\d{8}_\d{6})$")
_LEGACY_RE = re.compile(r"^(?P<prefix>.*?)_v\d+$", re.IGNORECASE)
_LOOSE_METRIC_RE = re.compile(
    r"^(?P<run>.+?)_(?P<kind>confusion_matrix|roc_curves|metrics)\.(?P<ext>png|jpg|json)$",
    re.IGNORECASE,
)
_WEIGHT_NAMES = ("model_best.pt", "best.pt", "face_classifier.pt")
_DROP_TOKENS = {"resnet34"}


# ---------------------------------------------------------------------------
# Name parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedName:
    raw: str
    prefix: str  # normalized: lowercase, drop tokens removed
    ts: datetime | None

    @property
    def tokens(self) -> set[str]:
        return set(self.prefix.split("_")) - {""}


def _normalize_prefix(prefix: str) -> str:
    tokens = [t for t in re.split(r"[_\-\s]+", prefix.lower()) if t and t not in _DROP_TOKENS]
    return "_".join(tokens)


def parse_run_name(name: str) -> ParsedName:
    """Split a run-like name into a normalized prefix and an optional timestamp.

    ``STACK_ResNet34_DeepLabV3_FullImage_Optuna_20260803_140657`` and
    ``STACK_DeepLabV3_FullImage_Optuna_20260803_140658`` both normalize to
    prefix ``stack_deeplabv3_fullimage_optuna`` with near-equal timestamps.
    ``*_v1`` names and names without a timestamp are "legacy" (ts=None).
    """
    stem = name.strip()
    match = _TS_RE.match(stem)
    if match:
        try:
            ts = datetime.strptime(match.group("ts"), "%Y%m%d_%H%M%S")
            return ParsedName(name, _normalize_prefix(match.group("prefix")), ts)
        except ValueError:
            pass
    legacy = _LEGACY_RE.match(stem)
    if legacy:
        stem = legacy.group("prefix")
    return ParsedName(name, _normalize_prefix(stem), None)


def family_for(prefix: str) -> str:
    if "nhla" in prefix:
        return "nhla"
    if "yolo" in prefix and "pith" in prefix:
        return "yolo_pith"
    if "yolo" in prefix:
        return "yolo_seg"
    return "segmentation"


def arch_and_variant(prefix: str, family: str) -> tuple[str, str]:
    tokens = prefix.split("_")
    if family == "nhla":
        side = next((t for t in tokens if t.endswith("side")), "")
        return "NHLA Classifier", side.replace("side", "-side") if side else "—"
    if family == "yolo_pith":
        return "YOLO Detect", "Pith" + (" (Optuna)" if "optuna" in tokens else "")
    if family == "yolo_seg":
        size = next((t for t in tokens if t in {"n", "s", "m", "l", "x"}), "")
        dataset = "_".join(tokens[: tokens.index("yolo")]).upper() if "yolo" in tokens else ""
        return f"YOLO-{size.upper()} Seg" if size else "YOLO Seg", dataset or "—"
    arch = "Model"
    for key, label in (("unetplusplus", "UNet++"), ("deeplabv3", "DeepLabV3"), ("unet", "UNet")):
        if key in tokens:
            arch = label
            break
    arch_idx = next((i for i, t in enumerate(tokens) if t in {"unetplusplus", "deeplabv3", "unet"}), len(tokens))
    dataset = "_".join(tokens[:arch_idx]).upper() or "—"
    mode = "Tiled" if "tiled" in tokens else ("Patches" if "patches" in tokens else "Full image")
    return arch, f"{dataset} · {mode}"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class StudyRef:
    db_path: Path
    study_name: str
    n_trials: int = 0
    n_complete: int = 0
    first_start: datetime | None = None
    last_start: datetime | None = None
    confidence: str = "exact"  # "exact" or "inferred"

    @property
    def parsed(self) -> ParsedName:
        return parse_run_name(self.study_name)


@dataclass
class Run:
    key: str
    name: str
    prefix: str
    ts: datetime | None
    family: str
    arch: str
    variant: str
    model_dir: Path | None = None
    checkpoint_dir: Path | None = None
    weights: list[Path] = field(default_factory=list)
    tb_dirs: list[Path] = field(default_factory=list)
    ultra_results: dict[str, Path] = field(default_factory=dict)  # label -> results.csv
    history_files: list[Path] = field(default_factory=list)
    metrics_dirs: list[Path] = field(default_factory=list)
    metrics_files: list[Path] = field(default_factory=list)
    onnx: list[Path] = field(default_factory=list)
    summary_row: dict[str, str] | None = None
    study: StudyRef | None = None
    size_bytes: int = 0
    # Foreground classes the model predicts (segmentation excludes Background).
    n_classes: int | None = None
    class_names: list[str] | None = None
    aliases: set[str] = field(default_factory=set)

    @property
    def status(self) -> str:
        return "complete" if self.weights else "partial"

    @property
    def is_legacy(self) -> bool:
        return self.ts is None

    def all_metric_paths(self) -> list[Path]:
        paths = list(self.metrics_files)
        for d in self.metrics_dirs:
            paths.extend(_file_paths(d))
        return paths

    def images(self) -> list[Path]:
        """Evaluation plots worth showing (confusion matrices, ROC/PR curves, ...)."""
        keep = ("confusion", "roc", "pr_curve", "f1_curve", "results.png", "detection_visual")
        out = [p for p in self.all_metric_paths() if p.suffix.lower() in {".png", ".jpg"}
               and any(k in p.name.lower() for k in keep)]
        if self.model_dir:
            for csv_path in self.ultra_results.values():
                for name in ("results.png", "confusion_matrix_normalized.png", "MaskPR_curve.png", "BoxPR_curve.png"):
                    candidate = csv_path.parent / name
                    if candidate.exists():
                        out.append(candidate)
        seen, unique = set(), []
        for p in out:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _files(path: Path) -> tuple[tuple[Path, int], ...]:
    """Every file under ``path`` with its size, from a single walk.

    The repo usually sits on a slow bind mount where each ``stat`` costs ~1 ms, so
    each directory is walked once per scan and reused (``discover_runs`` clears it).
    """
    out: list[tuple[Path, int]] = []
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        try:
                            size = entry.stat(follow_symlinks=False).st_size
                        except OSError:
                            size = 0
                        out.append((Path(entry.path), size))
        except OSError:
            continue
    out.sort()
    return tuple(out)


def _file_paths(path: Path | None, name: str | None = None, suffix: str | None = None) -> list[Path]:
    if path is None or not path.exists():
        return []
    return [p for p, _ in _files(path)
            if (name is None or p.name == name) and (suffix is None or p.suffix == suffix)]


def _event_dirs(path: Path) -> list[Path]:
    return sorted({p.parent for p in _file_paths(path) if p.name.startswith("events.out.tfevents")})


def _dir_size(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return sum(size for _, size in _files(path))


class _Index:
    """Groups artifacts into runs keyed by (normalized prefix, timestamp)."""

    def __init__(self) -> None:
        self.runs: list[Run] = []

    def find(self, parsed: ParsedName) -> Run | None:
        for run in self.runs:
            if run.prefix != parsed.prefix:
                continue
            if run.ts is None and parsed.ts is None:
                return run
            if run.ts is not None and parsed.ts is not None and abs(run.ts - parsed.ts) <= TS_TOLERANCE:
                return run
        return None

    def find_legacy(self, prefix: str) -> Run | None:
        return next((r for r in self.runs if r.prefix == prefix and r.ts is None), None)

    def get_or_create(self, name: str, create: bool = True) -> Run | None:
        parsed = parse_run_name(name)
        run = self.find(parsed)
        if run is None and create:
            family = family_for(parsed.prefix)
            arch, variant = arch_and_variant(parsed.prefix, family)
            key = parsed.prefix + ("_" + parsed.ts.strftime("%Y%m%d_%H%M%S") if parsed.ts else "")
            run = Run(key=key, name=name, prefix=parsed.prefix, ts=parsed.ts,
                      family=family, arch=arch, variant=variant)
            self.runs.append(run)
        if run is not None:
            run.aliases.add(name)
        return run

    def attach_secondary(self, name: str) -> Run | None:
        """Attach metrics/ONNX artifacts: exact match, else fall back to the legacy run of the same prefix."""
        parsed = parse_run_name(name)
        run = self.find(parsed) or self.find_legacy(parsed.prefix)
        if run is not None:
            run.aliases.add(name)
        return run


def _collect_ultralytics(run: Run) -> None:
    if run.model_dir is None:
        return
    for csv_path in _file_paths(run.model_dir, name="results.csv"):
        label = csv_path.parent.relative_to(run.model_dir).as_posix() or "."
        run.ultra_results[label] = csv_path


def _collect_weights(run: Run) -> None:
    if run.model_dir is None:
        return
    found = [p for p in _file_paths(run.model_dir, suffix=".pt") if p.name in _WEIGHT_NAMES
             and "optuna" not in p.relative_to(run.model_dir).parts
             and "checkpoints" not in p.relative_to(run.model_dir).parts]
    # Shallowest first so model_dir/best.pt beats runs/weights/best.pt.
    run.weights = sorted(found, key=lambda p: (len(p.relative_to(run.model_dir).parts), str(p)))


def load_summary_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def list_studies(db_path: Path) -> list[StudyRef]:
    """List studies in an Optuna SQLite DB without going through Optuna (fast, read-only)."""
    uri = f"file:{db_path.resolve()}?mode=ro"
    out: list[StudyRef] = []
    try:
        with sqlite3.connect(uri, uri=True) as conn:
            rows = conn.execute(
                """
                SELECT s.study_name,
                       (SELECT COUNT(*) FROM trials t WHERE t.study_id = s.study_id),
                       (SELECT COUNT(*) FROM trials t WHERE t.study_id = s.study_id AND t.state = 'COMPLETE'),
                       (SELECT MIN(datetime_start) FROM trials t WHERE t.study_id = s.study_id),
                       (SELECT MAX(datetime_start) FROM trials t WHERE t.study_id = s.study_id)
                FROM studies s ORDER BY s.study_id
                """
            ).fetchall()
    except sqlite3.Error:
        return out
    for name, n, n_complete, first, last in rows:
        out.append(StudyRef(
            db_path=db_path, study_name=name, n_trials=n or 0, n_complete=n_complete or 0,
            first_start=_parse_dt(first), last_start=_parse_dt(last),
        ))
    return out


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def discover_studies(root: Path) -> list[StudyRef]:
    studies: list[StudyRef] = []
    for db in sorted((root / "logs" / "optuna").glob("*.db")):
        studies.extend(list_studies(db))
    return studies


def _link_studies(runs: list[Run], studies: list[StudyRef], root: Path) -> None:
    by_name = {s.study_name: s for s in studies}
    claimed: set[tuple[Path, str]] = set()

    def claim(run: Run, study: StudyRef, confidence: str) -> None:
        run.study = StudyRef(**{**study.__dict__, "confidence": confidence})
        claimed.add((study.db_path, study.study_name))

    # 1. Explicit link from the summary CSV.
    for run in runs:
        name = (run.summary_row or {}).get("tp_study_name")
        if name and name in by_name:
            claim(run, by_name[name], "exact")

    # 2. Same timestamp (study names may use a different prefix, e.g. the tiled notebook).
    for run in runs:
        if run.study or run.ts is None:
            continue
        candidates = [s for s in studies if s.parsed.ts and abs(s.parsed.ts - run.ts) <= TS_TOLERANCE]
        candidates.sort(key=lambda s: -len(s.parsed.tokens & run_tokens(run)))
        if candidates:
            claim(run, candidates[0], "exact")

    # 3. Legacy runs: study named exactly like the run prefix.
    for run in runs:
        if run.study or run.ts is not None:
            continue
        match = next((s for s in studies if s.parsed.ts is None and s.parsed.prefix == run.prefix), None)
        if match:
            claim(run, match, "exact")

    # 4. Inferred: notebooks resume older studies, so a run often belongs to a study whose
    #    trials were running when the run started. Rank by name overlap, then time distance.
    for run in runs:
        if run.study or run.ts is None:
            continue
        best: tuple[tuple[int, float], StudyRef] | None = None
        for study in studies:
            overlap = len(study.parsed.tokens & run_tokens(run))
            if overlap < 2 or study.first_start is None or study.last_start is None:
                continue
            delta = _distance_to_span(run.ts, study.first_start - INFERRED_STUDY_TS_WINDOW,
                                      study.last_start + INFERRED_STUDY_TS_WINDOW)
            # Timestamped studies belong to the run with that timestamp; only share one
            # when this run started mid-study. Untimestamped studies are reused by design.
            is_claimed = (study.db_path, study.study_name) in claimed
            if is_claimed and study.parsed.ts is not None and delta > timedelta(0):
                continue
            window = INFERRED_STUDY_TS_WINDOW if study.parsed.ts else INFERRED_STUDY_WINDOW
            if delta > window:
                continue
            score = (overlap, -delta.total_seconds())
            if best is None or score > best[0]:
                best = (score, study)
        if best:
            claim(run, best[1], "inferred")


def _distance_to_span(t: datetime, start: datetime, end: datetime) -> timedelta:
    if t < start:
        return start - t
    if t > end:
        return t - end
    return timedelta(0)


def run_tokens(run: Run) -> set[str]:
    return set(run.prefix.split("_")) - {""}


def discover_runs(root: Path, studies: list[StudyRef] | None = None) -> list[Run]:
    """Build the run index for a repo root. Pure filesystem reads; never writes."""
    root = Path(root)
    _files.cache_clear()
    models_dir = root / "models"
    ckpt_dir = root / "checkpoints"
    tb_root = root / "logs" / "tensorboard"
    metrics_root = root / "data" / "metrics"
    index = _Index()

    # Primary artifacts: model dirs, checkpoint dirs, TensorBoard dirs.
    if models_dir.exists():
        for d in sorted(p for p in models_dir.iterdir() if p.is_dir() and p.name != "onnx"):
            run = index.get_or_create(d.name)
            run.name = d.name
            run.model_dir = d
    if ckpt_dir.exists():
        for d in sorted(p for p in ckpt_dir.iterdir() if p.is_dir()):
            run = index.get_or_create(d.name)
            run.checkpoint_dir = d
            history = d / "training_history.json"
            if history.exists():
                run.history_files.append(history)
    if tb_root.exists():
        for d in sorted(p for p in tb_root.iterdir() if p.is_dir()):
            event_dirs = _event_dirs(d)
            if not event_dirs:
                continue
            run = index.get_or_create(d.name)
            run.tb_dirs.extend(event_dirs)

    # Secondary artifacts attach to an existing run (or the legacy run of the same prefix).
    summary_rows = load_summary_rows(metrics_root / "TRAINING_RESULTS_SUMMARY.csv")
    for row in summary_rows:
        names = []
        if row.get("checkpoint_path"):
            names.append(Path(row["checkpoint_path"]).parent.name)
        if row.get("run_name"):
            names.append(row["run_name"])
        for name in names:
            run = index.attach_secondary(name)
            if run is not None:
                run.summary_row = row
                break

    if metrics_root.exists():
        for p in sorted(metrics_root.iterdir()):
            if p.is_dir():
                run = index.attach_secondary(p.name) or index.get_or_create(p.name)
                run.metrics_dirs.append(p)
            elif p.is_file():
                m = _LOOSE_METRIC_RE.match(p.name)
                if m:
                    run = index.attach_secondary(m.group("run"))
                    if run is not None:
                        run.metrics_files.append(p)

    onnx_dir = models_dir / "onnx"
    if onnx_dir.exists():
        for p in sorted(onnx_dir.glob("*.onnx")):
            run = index.attach_secondary(p.stem)
            if run is not None:
                run.onnx.append(p)

    for run in index.runs:
        if run.model_dir is None and run.checkpoint_dir is not None:
            run.name = run.checkpoint_dir.name
        elif run.model_dir is None and run.tb_dirs:
            run.name = sorted(run.aliases, key=len)[0]
        _collect_weights(run)
        _collect_ultralytics(run)
        if run.model_dir is not None:
            # Model-local logs (NHLA writes tensorboard/ and history.json inside the run dir).
            for d in _event_dirs(run.model_dir):
                if d not in run.tb_dirs:
                    run.tb_dirs.append(d)
            for name in ("history.json", "training_history.json"):
                for p in run.model_dir.glob(f"*/{name}"):
                    run.history_files.append(p)
                if (run.model_dir / name).exists():
                    run.history_files.append(run.model_dir / name)
        for d in run.metrics_dirs:
            run.history_files.extend(sorted(d.glob("*_history.csv")))
        run.size_bytes = _dir_size(run.model_dir) + _dir_size(run.checkpoint_dir)
        run.n_classes, run.class_names = class_info(run)

    # Drop groups that only consist of metrics dirs with nothing in them.
    runs = [r for r in index.runs if r.model_dir or r.checkpoint_dir or r.tb_dirs or r.metrics_dirs]

    if studies is None:
        studies = discover_studies(root)
    _link_studies(runs, studies, root)

    runs.sort(key=lambda r: (r.ts or datetime.min, r.name), reverse=True)
    return runs


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

class _ShapeOnlyUnpickler(pickle.Unpickler):
    """Reads a torch checkpoint's pickle without loading tensor data or running code.

    Tensors become their shape tuple; any class other than plain containers becomes
    an inert stub that keeps its pickled state (that is where Ultralytics keeps
    ``names``). Nothing outside a small allowlist is ever imported or called.
    """

    _SAFE = {("collections", "OrderedDict"), ("builtins", "set"), ("builtins", "frozenset"),
             ("builtins", "dict"), ("builtins", "list"), ("builtins", "tuple"), ("builtins", "slice")}

    class _Stub:
        def __init__(self, *args, **kwargs):
            self.state = None

        def __setstate__(self, state):
            self.state = state

        def __call__(self, *args, **kwargs):
            return self

    @staticmethod
    def _tensor_shape(storage, offset, size, *args, **kwargs):
        return tuple(size)

    def find_class(self, module, name):
        if module.startswith("torch") and name.startswith("_rebuild"):
            return self._tensor_shape
        if (module, name) in self._SAFE:
            return super().find_class(module, name)
        return self._Stub

    def persistent_load(self, pid):
        return None


def peek_checkpoint(path: Path) -> Any:
    """Checkpoint structure with tensors replaced by shapes; None if unreadable."""
    try:
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if n.endswith("data.pkl"))
            with archive.open(name) as handle:
                return _ShapeOnlyUnpickler(handle).load()
    except Exception:
        return None


def _head_size(state_dict: Any) -> int | None:
    """Output size of the last segmentation/classification layer in a state dict."""
    if not isinstance(state_dict, dict):
        return None
    for key in reversed(list(state_dict)):
        shape = state_dict[key]
        if (key.endswith("bias") and isinstance(shape, tuple) and len(shape) == 1
                and any(h in key for h in ("segmentation_head", "classifier", "fc.", "head."))):
            return int(shape[0])
    return None


def _yolo_names(ckpt: Any) -> list[str] | None:
    if not isinstance(ckpt, dict):
        return None
    for key in ("ema", "model"):
        state = getattr(ckpt.get(key), "state", None)
        if isinstance(state, tuple):  # (state, slotstate)
            state = state[0]
        names = state.get("names") if isinstance(state, dict) else None
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names)]
        if isinstance(names, list):
            return [str(n) for n in names]
    return None


def _checkpoint_candidates(run: Run) -> list[Path]:
    paths = list(run.weights)
    if run.checkpoint_dir is not None:
        paths += _file_paths(run.checkpoint_dir, suffix=".pt")[:1]
    return paths


def class_info(run: Run) -> tuple[int | None, list[str] | None]:
    """(number of foreground classes, class names if known) for a run."""
    row = run.summary_row or {}
    if row.get("class_names"):
        try:
            names = [str(n) for n in json.loads(row["class_names"])]
            if run.family == "segmentation":
                names = [n for n in names if n.lower() != "background"]
            return len(names), names
        except json.JSONDecodeError:
            pass

    if run.family == "nhla":
        for p in run.all_metric_paths() + _file_paths(run.model_dir, name="class_mapping.json"):
            if p.name == "class_mapping.json":
                mapping = load_json(p)
                if isinstance(mapping, dict) and mapping:
                    names = [k for k, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
                    return len(names), names

    for path in _checkpoint_candidates(run):
        ckpt = peek_checkpoint(path)
        if ckpt is None:
            continue
        if run.family in ("yolo_seg", "yolo_pith"):
            names = _yolo_names(ckpt)
            if names:
                return len(names), names
            continue
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else None
        size = _head_size(state)
        if size:
            # Segmentation heads include the Background channel.
            return (size - 1 if run.family == "segmentation" else size), None
    return None, None


def class_label(n_classes: int | None, names: list[str] | None) -> str:
    if n_classes is None:
        return "unknown"
    return f"{n_classes}: {', '.join(names)}" if names else f"{n_classes} (names unknown)"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_tb_scalars(log_dir: Path) -> pd.DataFrame:
    """All scalar series in a TensorBoard dir as a long frame: tag, step, value, wall_time."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(log_dir), size_guidance={"scalars": 0, "images": 0, "histograms": 0,
                                                       "tensors": 0, "audio": 0, "compressedHistograms": 0})
    acc.Reload()
    rows = []
    for tag in acc.Tags().get("scalars", []):
        for ev in acc.Scalars(tag):
            rows.append((tag, ev.step, ev.value, ev.wall_time))
    return pd.DataFrame(rows, columns=["tag", "step", "value", "wall_time"])


def load_ultralytics_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_history(path: Path) -> pd.DataFrame:
    """training_history.json (dict of lists) or *_history.csv as an epoch-indexed frame."""
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            lengths = {len(v) for v in data.values() if isinstance(v, list)}
            n = max(lengths) if lengths else 0
            df = pd.DataFrame({k: list(v) + [np.nan] * (n - len(v)) for k, v in data.items() if isinstance(v, list)})
        else:
            df = pd.DataFrame(data)
    if "epoch" not in df.columns:
        df.insert(0, "epoch", range(1, len(df) + 1))
    return df


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def parse_array_cell(value: Any) -> list[float] | None:
    """Parse JSON lists and numpy-printed arrays like ``"[0.96 0.73 ...]"``."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value]
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [float(v) for v in parsed]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    numbers = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan", text.strip("[]() "))
    if not numbers:
        return None
    return [float(n) for n in numbers]


def readonly_storage(db_path: Path):
    from optuna.storages import RDBStorage

    url = f"sqlite:///file:{Path(db_path).resolve()}?mode=ro&uri=true"
    return RDBStorage(url, skip_compatibility_check=True, skip_table_creation=True)


def load_study(db_path: Path, study_name: str):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.load_study(study_name=study_name, storage=readonly_storage(db_path))


def load_trials_df(db_path: Path, study_name: str) -> pd.DataFrame:
    study = load_study(db_path, study_name)
    df = study.trials_dataframe(attrs=("number", "value", "state", "datetime_start", "duration",
                                       "params", "user_attrs"))
    if "duration" in df.columns:
        df["duration_min"] = df["duration"].dt.total_seconds() / 60.0
        df = df.drop(columns=["duration"])
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _to_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _primary_metrics_json(run: Run) -> dict | None:
    candidates = [p for p in run.all_metric_paths() if p.suffix == ".json"
                  and (p.name.endswith("_metrics.json") or p.name in {"metrics.json", "locked_test_metrics.json"})]
    if run.model_dir:
        candidates += _file_paths(run.model_dir, name="metrics.json")
    for p in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        data = load_json(p)
        if isinstance(data, dict):
            return data
    return None


def _best_from_results(run: Run, columns: Iterable[str]) -> float | None:
    preferred = [k for k in ("final_detection", "runs", "runs-2", ".") if k in run.ultra_results]
    others = [k for k in run.ultra_results if k not in preferred and not k.startswith("optuna/")]
    for label in preferred + others:
        try:
            df = load_ultralytics_results(run.ultra_results[label])
        except Exception:
            continue
        for col in columns:
            if col in df.columns and df[col].notna().any():
                return float(df[col].max())
    return None


def _best_from_history(run: Run, columns: Iterable[str]) -> float | None:
    for p in run.history_files:
        try:
            df = load_history(p)
        except Exception:
            continue
        for col in columns:
            if col in df.columns and df[col].notna().any():
                return float(df[col].max())
    return None


def headline_metric(run: Run) -> tuple[float | None, str]:
    """The single number that best summarizes a run, plus its label."""
    row = run.summary_row or {}
    if run.family == "segmentation":
        value = _to_float(row.get("mean_iou"))
        if value is not None:
            return value, "Test mIoU"
        return _best_from_history(run, ("val_iou",)), "Best val IoU"
    if run.family == "yolo_seg":
        data = _primary_metrics_json(run) or {}
        value = _to_float(data.get("seg_map50_95"))
        if value is not None:
            return value, "Mask mAP50-95"
        return _best_from_results(run, ("metrics/mAP50-95(M)", "metrics/mAP50-95(B)")), "Mask mAP50-95"
    if run.family == "yolo_pith":
        return _best_from_results(run, ("metrics/mAP50-95(B)",)), "Box mAP50-95"
    if run.family == "nhla":
        data = _primary_metrics_json(run) or {}
        value = _to_float(data.get("macro_f1"))
        if value is not None:
            return value, "Macro F1"
        return _best_from_history(run, ("val_macro_f1",)), "Best val macro F1"
    return None, ""


def scalar_metrics(run: Run) -> dict[str, float]:
    """Flat dict of scalar metrics from the summary row and metrics JSON files."""
    out: dict[str, float] = {}
    row = run.summary_row or {}
    for key in ("mean_iou", "accuracy", "f1_macro", "optuna_best_value"):
        v = _to_float(row.get(key))
        if v is not None:
            out[key] = v
    iou = parse_array_cell(row.get("iou_per_class"))
    if run.family == "segmentation" and iou and len(iou) > 1:
        # mean_iou includes Background; this one is comparable across class sets.
        out["mean_iou_defects"] = float(np.mean(iou[1:]))
    for key, v in row.items():
        if key.startswith("metric_"):
            f = _to_float(v)
            if f is not None:
                out[key.removeprefix("metric_")] = f
    data = _primary_metrics_json(run) or {}
    for key, v in data.items():
        if isinstance(v, bool):
            continue
        f = _to_float(v)
        if f is not None and key not in out:
            out[key] = f
    return out


def per_class_metrics(run: Run) -> pd.DataFrame | None:
    """Per-class metrics (IoU/F1/ROC-AUC or precision/recall/F1) as a tidy frame."""
    row = run.summary_row or {}
    class_names = None
    if row.get("class_names"):
        try:
            class_names = json.loads(row["class_names"])
        except json.JSONDecodeError:
            class_names = None
    series: dict[str, list[float]] = {}
    for key, label in (("iou_per_class", "IoU"), ("f1_per_class", "F1"), ("roc_auc_per_class", "ROC-AUC")):
        values = parse_array_cell(row.get(key))
        if values:
            series[label] = values
    data = _primary_metrics_json(run) or {}
    for key, label in (("per_class_precision", "Precision"), ("per_class_recall", "Recall"), ("per_class_f1", "F1")):
        values = parse_array_cell(data.get(key))
        if values and label not in series:
            series[label] = values
    if not series:
        return None
    if class_names is None:
        report = data.get("classification_report") if isinstance(data, dict) else None
        if isinstance(report, dict):
            class_names = [k for k in report if k not in {"accuracy", "macro avg", "weighted avg"}]
    n = max(len(v) for v in series.values())
    if not class_names or len(class_names) != n:
        class_names = [f"class {i}" for i in range(n)]
    rows = []
    for metric, values in series.items():
        for name, v in zip(class_names, values):
            rows.append({"class": name, "metric": metric, "value": v})
    return pd.DataFrame(rows)


def confusion_matrix(run: Run) -> tuple[list[list[float]], list[str]] | None:
    data = _primary_metrics_json(run) or {}
    cm = data.get("confusion_matrix")
    if not isinstance(cm, list) or not cm:
        return None
    report = data.get("classification_report") or {}
    labels = [k for k in report if k not in {"accuracy", "macro avg", "weighted avg"}]
    if len(labels) != len(cm):
        labels = [str(i) for i in range(len(cm))]
    return cm, labels


def hyperparameters(run: Run) -> dict[str, Any]:
    """Hyperparameters from the summary row, best_params.json, args.yaml and pith training config."""
    params: dict[str, Any] = {}
    row = run.summary_row or {}
    for key, value in row.items():
        if key.startswith("tp_") and value not in ("", None):
            params[key.removeprefix("tp_")] = value
    for key in ("optimizer_name", "learning_rate", "weight_decay", "batch_size", "num_epochs_actual",
                "loss_name", "image_height", "image_width", "raw_data_folder", "transfer_learning"):
        if row.get(key) not in ("", None):
            params.setdefault(key, row[key])
    for p in run.all_metric_paths():
        if p.name in {"best_params.json", "optuna_best.json"}:
            data = load_json(p) or {}
            data = data.get("params", data) if isinstance(data, dict) else {}
            for k, v in data.items():
                params.setdefault(k, v)
    if run.model_dir:
        for name in ("best_params.json", "final_training_config.json"):
            p = run.model_dir / name
            if p.exists():
                data = load_json(p) or {}
                if isinstance(data, dict):
                    for k, v in data.items():
                        if not isinstance(v, (dict, list)):
                            params.setdefault(k, v)
        args_yaml = next(iter(_file_paths(run.model_dir, name="args.yaml")), None)
        if args_yaml is not None:
            for k, v in _read_simple_yaml(args_yaml).items():
                if k in {"model", "epochs", "batch", "imgsz", "optimizer", "lr0", "lrf", "weight_decay",
                         "patience", "freeze", "mosaic", "close_mosaic", "cos_lr"}:
                    params.setdefault(k, v)
    return params


def _read_simple_yaml(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            if ":" in line and not line.startswith((" ", "#")):
                k, _, v = line.partition(":")
                out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def runs_frame(runs: list[Run]) -> pd.DataFrame:
    """One row per run for tables and filters."""
    rows = []
    for run in runs:
        value, label = headline_metric(run)
        rows.append({
            "key": run.key,
            "name": run.name,
            "family": FAMILY_LABELS[run.family],
            "arch": run.arch,
            "variant": run.variant,
            "date": run.ts,
            "status": run.status,
            "metric": value,
            "metric_label": label,
            "has_tb": bool(run.tb_dirs),
            "has_study": run.study is not None,
            "study": run.study.study_name if run.study else None,
            "trials": run.study.n_trials if run.study else 0,
            "has_onnx": bool(run.onnx),
            "has_curves": bool(run.tb_dirs or run.history_files or run.ultra_results),
            "size_mb": run.size_bytes / 1e6,
            "n_classes": run.n_classes,
            "classes": class_label(run.n_classes, run.class_names),
            "class_names": ", ".join(run.class_names) if run.class_names else None,
        })
    return pd.DataFrame(rows)
