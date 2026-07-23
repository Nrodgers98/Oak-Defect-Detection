from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _normalize_key(key: str) -> str:
    cleaned = []
    for ch in str(key).strip().lower():
        if ch.isalnum() or ch == "_":
            cleaned.append(ch)
        elif ch in (" ", "-", "/", ".", "(", ")"):
            cleaned.append("_")
    collapsed = "".join(cleaned)
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed.strip("_") or "unknown"


def _jsonify(value: Any) -> str | Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def build_run_row(
    *,
    notebook_name: str,
    model_name: str,
    raw_data_folder: Any = None,
    image_height: Any = None,
    image_width: Any = None,
    patch_size: Any = None,
    batch_size: Any = None,
    num_workers: Any = None,
    num_epochs_config: Any = None,
    num_epochs_actual: Any = None,
    optimizer_name: Any = None,
    learning_rate: Any = None,
    weight_decay: Any = None,
    scheduler_name: Any = None,
    scheduler_step_size: Any = None,
    scheduler_gamma: Any = None,
    scheduler_t_max: Any = None,
    scheduler_eta_min: Any = None,
    scheduler_warmup_epochs: Any = None,
    momentum: Any = None,
    betas: Any = None,
    eps: Any = None,
    patience: Any = None,
    early_stopping_min_delta: Any = None,
    loss_name: Any = None,
    seed: Any = None,
    mixed_precision: Any = None,
    gradient_clip: Any = None,
    augment_enabled: Any = None,
    augment_name: Any = None,
    num_classes: Any = None,
    class_names: Any = None,
    train_split_size: Any = None,
    val_split_size: Any = None,
    test_split_size: Any = None,
    accuracy: Any = None,
    f1_macro: Any = None,
    f1_per_class: Any = None,
    mean_iou: Any = None,
    iou_per_class: Any = None,
    roc_auc_per_class: Any = None,
    checkpoint_path: Any = None,
    training_history_path: Any = None,
    onnx_path: Any = None,
    run_name: Any = None,
    transfer_learning: Any = None,
    transfer_checkpoint_path: Any = None,
    freeze_encoder_epochs: Any = None,
    optuna_enabled: Any = None,
    optuna_n_trials: Any = None,
    optuna_best_value: Any = None,
    optuna_best_params: Any = None,
    extra_training_params: dict[str, Any] | None = None,
    extra_metrics: dict[str, Any] | None = None,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "date_time_utc": datetime.now(timezone.utc).isoformat(),
        "raw_data_folder": _jsonify(raw_data_folder),
        "notebook_name": notebook_name,
        "model_name": model_name,
        "image_height": image_height,
        "image_width": image_width,
        "patch_size": patch_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "num_epochs_config": num_epochs_config,
        "num_epochs_actual": num_epochs_actual,
        "optimizer_name": optimizer_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "scheduler_name": scheduler_name,
        "scheduler_step_size": scheduler_step_size,
        "scheduler_gamma": scheduler_gamma,
        "scheduler_t_max": scheduler_t_max,
        "scheduler_eta_min": scheduler_eta_min,
        "scheduler_warmup_epochs": scheduler_warmup_epochs,
        "momentum": momentum,
        "betas": _jsonify(betas),
        "eps": eps,
        "patience": patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "loss_name": loss_name,
        "seed": seed,
        "mixed_precision": mixed_precision,
        "gradient_clip": gradient_clip,
        "augment_enabled": augment_enabled,
        "augment_name": augment_name,
        "num_classes": num_classes,
        "class_names": _jsonify(class_names),
        "train_split_size": train_split_size,
        "val_split_size": val_split_size,
        "test_split_size": test_split_size,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_per_class": _jsonify(f1_per_class),
        "mean_iou": mean_iou,
        "iou_per_class": _jsonify(iou_per_class),
        "roc_auc_per_class": _jsonify(roc_auc_per_class),
        "checkpoint_path": _jsonify(checkpoint_path),
        "training_history_path": _jsonify(training_history_path),
        "onnx_path": _jsonify(onnx_path),
        "run_name": run_name,
        "transfer_learning": transfer_learning,
        "transfer_checkpoint_path": _jsonify(transfer_checkpoint_path),
        "freeze_encoder_epochs": freeze_encoder_epochs,
        "optuna_enabled": optuna_enabled,
        "optuna_n_trials": optuna_n_trials,
        "optuna_best_value": optuna_best_value,
        "optuna_best_params": _jsonify(optuna_best_params),
    }

    for group_name, group in (
        ("tp", extra_training_params or {}),
        ("metric", extra_metrics or {}),
        ("info", extra_info or {}),
    ):
        for key, value in group.items():
            norm = _normalize_key(key)
            row[f"{group_name}_{norm}"] = _jsonify(value)

    return row


def append_row_to_csv(csv_path: Path | str, row: dict[str, Any]) -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    serialized_row = {k: _jsonify(v) for k, v in row.items()}
    for key, value in list(serialized_row.items()):
        if value is None:
            serialized_row[key] = ""

    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(serialized_row.keys()))
            writer.writeheader()
            writer.writerow(serialized_row)
        return csv_path

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)
        existing_headers = list(reader.fieldnames or [])

    merged_headers = list(existing_headers)
    for key in serialized_row.keys():
        if key not in merged_headers:
            merged_headers.append(key)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=merged_headers)
        writer.writeheader()
        for existing_row in existing_rows:
            writer.writerow(existing_row)
        writer.writerow(serialized_row)

    return csv_path


def infer_split_sizes(project_root: Path) -> tuple[Any, Any, Any]:
    candidates = [
        project_root / "data" / "organized-data" / "split.json",
        project_root / "data" / "yolo-format" / "split.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                with candidate.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                train_size = len(payload.get("train", []))
                val_size = len(payload.get("val", []))
                test_size = len(payload.get("test", []))
                return train_size, val_size, test_size
            except Exception:
                continue
    return None, None, None


def guess_actual_epochs(history: Any) -> Any:
    if isinstance(history, dict):
        for key in ("train_loss", "val_loss", "val_iou", "train_iou"):
            values = history.get(key)
            if isinstance(values, list):
                return len(values)
    return None


def first_available(local_vars: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in local_vars:
            return local_vars[name]
    return None
