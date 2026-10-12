"""Precompute native-resolution image and mask tiles for segmentation."""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _positive_int(value: int, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _edge_mode(value: str) -> str:
    if value not in {"shrink", "flush"}:
        raise ValueError("edge_mode must be 'shrink' or 'flush'")
    return value


def compute_tile_windows(
    length: int, max_tile: int, step: int, edge_mode: str = "shrink"
) -> list[tuple[int, int]]:
    """Return deterministic one-dimensional tile windows covering ``length``."""
    length = _positive_int(length, "length")
    max_tile = _positive_int(max_tile, "max_tile")
    step = _positive_int(step, "step")
    edge_mode = _edge_mode(edge_mode)

    if length <= max_tile:
        return [(0, length)]
    if edge_mode == "shrink":
        return [(start, min(start + max_tile, length)) for start in range(0, length, step)]

    last_start = length - max_tile
    starts = list(range(0, max(1, last_start + 1), step))
    if last_start % step != 0:
        starts.append(last_start)
    starts = list(dict.fromkeys(starts))
    return [(start, start + max_tile) for start in starts]


def tessellate_boxes(
    height: int,
    width: int,
    max_x: int,
    max_y: int,
    step_x: int | None = None,
    step_y: int | None = None,
    edge_mode: str = "shrink",
) -> list[tuple[int, int, int, int]]:
    """Return row-major ``(ymin, ymax, xmin, xmax)`` tile boxes."""
    height = _positive_int(height, "height")
    width = _positive_int(width, "width")
    max_x = _positive_int(max_x, "max_x")
    max_y = _positive_int(max_y, "max_y")
    step_x = max_x if step_x is None else _positive_int(step_x, "step_x")
    step_y = max_y if step_y is None else _positive_int(step_y, "step_y")
    edge_mode = _edge_mode(edge_mode)
    y_windows = compute_tile_windows(height, max_y, step_y, edge_mode)
    x_windows = compute_tile_windows(width, max_x, step_x, edge_mode)
    return [
        (ymin, ymax, xmin, xmax)
        for ymin, ymax in y_windows
        for xmin, xmax in x_windows
    ]


def _config(max_x, max_y, step_x, step_y, edge_mode, drop_background_train_tiles, min_tile_defect_pixels):
    return {
        "max_x": int(max_x), "max_y": int(max_y),
        "step_x": None if step_x is None else int(step_x),
        "step_y": None if step_y is None else int(step_y),
        "edge_mode": edge_mode,
        "drop_background_train_tiles": bool(drop_background_train_tiles),
        "min_tile_defect_pixels": int(min_tile_defect_pixels),
    }


def _stats(split_counts: dict[str, int], board_counts: dict[str, int], sizes: list[tuple[int, int]]) -> dict[str, Any]:
    counts = list(board_counts.values())
    distribution: dict[str, int] = {}
    for height, width in sizes:
        key = f"{height}x{width}"
        distribution[key] = distribution.get(key, 0) + 1
    return {
        "tiles_per_split": split_counts,
        "tiles_per_board": {
            "min": min(counts) if counts else 0,
            "mean": sum(counts) / len(counts) if counts else 0.0,
            "max": max(counts) if counts else 0,
        },
        "tile_size_distribution": distribution,
    }


def create_tiled_dataset(
    organized_folder: str | Path,
    output_folder: str | Path,
    max_x: int,
    max_y: int,
    step_x: int | None = None,
    step_y: int | None = None,
    edge_mode: str = "shrink",
    drop_background_train_tiles: bool = False,
    min_tile_defect_pixels: int = 0,
    force: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Write a tiled dataset matching the organized-data directory contract."""
    max_x = _positive_int(max_x, "max_x")
    max_y = _positive_int(max_y, "max_y")
    step_x = None if step_x is None else _positive_int(step_x, "step_x")
    step_y = None if step_y is None else _positive_int(step_y, "step_y")
    edge_mode = _edge_mode(edge_mode)
    min_tile_defect_pixels = int(min_tile_defect_pixels)
    if min_tile_defect_pixels < 0:
        raise ValueError("min_tile_defect_pixels must be non-negative")
    if not isinstance(drop_background_train_tiles, bool):
        raise ValueError("drop_background_train_tiles must be a boolean")

    organized = Path(organized_folder)
    output = Path(output_folder)
    config = _config(max_x, max_y, step_x, step_y, edge_mode, drop_background_train_tiles, min_tile_defect_pixels)
    config_path = output / "tiling_config.json"
    stats_path = output / "tiling_stats.json"
    if not force and config_path.exists() and stats_path.exists():
        with config_path.open(encoding="utf-8") as handle:
            if json.load(handle) == config:
                with stats_path.open(encoding="utf-8") as handle:
                    return str(output), json.load(handle)

    with (organized / "split.json").open(encoding="utf-8") as handle:
        source_split = json.load(handle)
    splits = {name: list(source_split.get(name, [])) for name in ("train", "val", "test")}
    if output.exists():
        shutil.rmtree(output)
    images_out = output / "images"
    masks_out = output / "combined_masks"
    images_out.mkdir(parents=True)
    masks_out.mkdir()
    manifest: dict[str, dict[str, Any]] = {}
    tiled_split = {"train": [], "val": [], "test": []}
    split_counts = {"train": 0, "val": 0, "test": 0}
    board_counts: dict[str, int] = {}
    sizes: list[tuple[int, int]] = []

    for split_name, image_ids in splits.items():
        for image_id in image_ids:
            image = np.asarray(Image.open(organized / "images" / f"{image_id}_Col.tif").convert("RGB"))
            mask = np.asarray(Image.open(organized / "combined_masks" / f"{image_id}_mask.png"))
            if image.shape[:2] != mask.shape[:2]:
                raise ValueError(f"image and mask shapes differ for {image_id}")
            height, width = image.shape[:2]
            boxes = tessellate_boxes(height, width, max_x, max_y, step_x, step_y, edge_mode)
            x_step = max_x if step_x is None else step_x
            x_count = len(compute_tile_windows(width, max_x, x_step, edge_mode))
            board_counts[image_id] = 0
            for index, (ymin, ymax, xmin, xmax) in enumerate(boxes):
                row, col = divmod(index, x_count)
                tile_id = f"{image_id}__r{row:02d}_c{col:02d}"
                tile_mask = mask[ymin:ymax, xmin:xmax]
                if split_name == "train" and drop_background_train_tiles and int(np.count_nonzero(tile_mask)) < min_tile_defect_pixels:
                    continue
                tile_image = image[ymin:ymax, xmin:xmax]
                Image.fromarray(tile_image).save(images_out / f"{tile_id}_Col.tif")
                Image.fromarray(tile_mask.astype(np.uint8), mode="L").save(masks_out / f"{tile_id}_mask.png")
                tiled_split[split_name].append(tile_id)
                split_counts[split_name] += 1
                board_counts[image_id] += 1
                sizes.append((ymax - ymin, xmax - xmin))
                manifest[tile_id] = {
                    "parent_board": image_id, "split": split_name,
                    "box": [ymin, ymax, xmin, xmax], "parent_size": [height, width],
                }

    for split_name in ("train", "val", "test"):
        tiled_split[f"num_{split_name}"] = len(tiled_split[split_name])
    result_stats = _stats(split_counts, board_counts, sizes)
    with (output / "split.json").open("w", encoding="utf-8") as handle:
        json.dump(tiled_split, handle, indent=2)
    with (output / "tile_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(result_stats, handle, indent=2)
    return str(output), result_stats
