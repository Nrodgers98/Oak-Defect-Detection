"""Reusable helpers for grouped YOLO detection experiments."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

FILENAME_RE = re.compile(r"^(?P<group>[^_]+)_(?P<timestamp>\d+)_(?P<face>Top_Col|Bot_Col)\.tiff?$", re.IGNORECASE)


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_image_identity(path: str | Path) -> dict[str, Any]:
    name = Path(path).name
    match = FILENAME_RE.match(name)
    if not match:
        raise ValueError(f"Unsupported board filename: {name!r}")
    timestamp = int(match.group("timestamp"))
    return {
        "image_id": Path(path).stem,
        "group_id": match.group("group"),
        "timestamp": timestamp,
        "face": match.group("face").split("_")[0],
        "filename": name,
    }


def xywhn_to_xyxy(box: Iterable[float], width: int, height: int) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in box)
    if len(values) != 4:
        raise ValueError("Expected normalized (cx, cy, w, h)")
    cx, cy, box_width, box_height = values
    return (
        (cx - box_width / 2) * width,
        (cy - box_height / 2) * height,
        (cx + box_width / 2) * width,
        (cy + box_height / 2) * height,
    )


def xyxy_to_xywhn(box: Iterable[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    return (
        ((x1 + x2) / 2) / width,
        ((y1 + y2) / 2) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def clip_box_to_tile(
    box: Iterable[float], tile: tuple[int, int, int, int], min_retained_fraction: float = 0.25,
    min_width: float = 2.0, min_height: float = 2.0,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = (float(value) for value in box)
    tile_y1, tile_y2, tile_x1, tile_x2 = (float(value) for value in tile)
    original_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    clipped = (
        max(x1, tile_x1), max(y1, tile_y1),
        min(x2, tile_x2), min(y2, tile_y2),
    )
    clipped_area = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
    if original_area <= 0 or clipped_area / original_area < min_retained_fraction:
        return None
    if clipped[2] - clipped[0] < min_width or clipped[3] - clipped[1] < min_height:
        return None
    return clipped


def restore_tile_box(box: Iterable[float], tile: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    tile_y1, _, tile_x1, _ = tile
    return x1 + tile_x1, y1 + tile_y1, x2 + tile_x1, y2 + tile_y1


def box_iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def tile_windows(height: int, width: int, tile_height: int, tile_width: int, overlap: float) -> list[tuple[int, int, int, int]]:
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if tile_height <= 0 or tile_width <= 0:
        raise ValueError("tile dimensions must be positive")
    step_y = max(1, int(round(tile_height * (1 - overlap))))
    step_x = max(1, int(round(tile_width * (1 - overlap))))

    def windows(length: int, tile: int, step: int) -> list[tuple[int, int]]:
        if length <= tile:
            return [(0, length)]
        starts = list(range(0, max(1, length - tile + 1), step))
        last = length - tile
        if starts[-1] != last:
            starts.append(last)
        return [(start, min(start + tile, length)) for start in dict.fromkeys(starts)]

    return [(y1, y2, x1, x2) for y1, y2 in windows(height, tile_height, step_y) for x1, x2 in windows(width, tile_width, step_x)]


def read_tiff_metadata(path: str | Path) -> dict[str, Any]:
    with Image.open(path) as image:
        tags = image.tag_v2
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "frames": getattr(image, "n_frames", 1),
            "compression": tags.get(259),
            "photometric": tags.get(262),
            "samples_per_pixel": tags.get(277),
            "bits_per_sample": tags.get(258),
            "file_size_bytes": Path(path).stat().st_size,
        }


def write_json(path: str | Path, payload: Any) -> None:
    def json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(json_safe(payload), indent=2, default=str), encoding="utf-8")


def percentile_summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {}
    return {key: float(value) for key, value in zip(("min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"), np.percentile(array, [0, 1, 5, 25, 50, 75, 95, 99, 100]))}


def finite_or_error(value: float, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def read_yolo_boxes(label_path: str | Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line_number, line in enumerate(Path(label_path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number} must contain five fields")
        class_id = int(fields[0])
        cx, cy, width, height = (finite_or_error(value, f"{label_path}:{line_number}") for value in fields[1:])
        boxes.append((class_id, cx, cy, width, height))
    return boxes


def tile_label_rows(
    label_path: str | Path,
    image_width: int,
    image_height: int,
    tile: tuple[int, int, int, int],
    min_retained_fraction: float,
    min_width: float,
    min_height: float,
) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for class_id, cx, cy, width, height in read_yolo_boxes(label_path):
        clipped = clip_box_to_tile(
            xywhn_to_xyxy((cx, cy, width, height), image_width, image_height),
            tile,
            min_retained_fraction,
            min_width,
            min_height,
        )
        if clipped is None:
            continue
        tile_y1, tile_y2, tile_x1, tile_x2 = tile
        local_box = (clipped[0] - tile_x1, clipped[1] - tile_y1, clipped[2] - tile_x1, clipped[3] - tile_y1)
        tile_width = tile_x2 - tile_x1
        tile_height = tile_y2 - tile_y1
        rows.append((class_id, *xyxy_to_xywhn(local_box, tile_width, tile_height)))
    return rows


def match_detections(
    predictions: Iterable[tuple[float, float, float, float, float]],
    truth: Iterable[tuple[float, float, float, float]],
    iou_threshold: float = 0.5,
) -> dict[str, int]:
    remaining = list(truth)
    true_positive = false_positive = 0
    for prediction in sorted(predictions, key=lambda item: item[4], reverse=True):
        best_index = -1
        best_iou = 0.0
        for index, target in enumerate(remaining):
            overlap = box_iou(prediction[:4], target)
            if overlap > best_iou:
                best_index, best_iou = index, overlap
        if best_iou >= iou_threshold:
            true_positive += 1
            remaining.pop(best_index)
        else:
            false_positive += 1
    return {"tp": true_positive, "fp": false_positive, "fn": len(remaining)}


def binary_detection_metrics(matches: dict[str, int]) -> dict[str, float]:
    tp, fp, fn = (float(matches[key]) for key in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
