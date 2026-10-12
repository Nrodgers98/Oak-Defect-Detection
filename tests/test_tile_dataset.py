import json

import numpy as np
import pytest
from PIL import Image

from notebooks.functions.tile_dataset import compute_tile_windows, create_tiled_dataset, tessellate_boxes


def test_acceptance_window_geometry():
    assert tessellate_boxes(2000, 5030, 3000, 1000) == [
        (0, 1000, 0, 3000), (0, 1000, 3000, 5030),
        (1000, 2000, 0, 3000), (1000, 2000, 3000, 5030),
    ]
    assert tessellate_boxes(2000, 5030, 3000, 9999) == [
        (0, 2000, 0, 3000), (0, 2000, 3000, 5030)
    ]


def test_windows_support_overlap_and_flush_edges():
    assert compute_tile_windows(10, 6, 4) == [(0, 6), (4, 10), (8, 10)]
    assert compute_tile_windows(10, 6, 4, "flush") == [(0, 6), (4, 10)]
    assert compute_tile_windows(6, 6, 2) == [(0, 6)]
    assert compute_tile_windows(12, 4, 4) == [(0, 4), (4, 8), (8, 12)]
    assert compute_tile_windows(5, 10, 2) == [(0, 5)]
    assert compute_tile_windows(11, 6, 3, "flush") == [(0, 6), (3, 9), (5, 11)]


@pytest.mark.parametrize(
    "length,max_tile,step,edge_mode",
    [(0, 4, 4, "shrink"), (10, 0, 4, "shrink"), (10, 4, 0, "shrink"), (10, 4, 4, "invalid")],
)
def test_window_arguments_are_validated(length, max_tile, step, edge_mode):
    with pytest.raises(ValueError):
        compute_tile_windows(length, max_tile, step, edge_mode)


def test_round_trip_and_skip(tmp_path):
    organized = tmp_path / "organized"
    (organized / "images").mkdir(parents=True)
    (organized / "combined_masks").mkdir()
    split = {"train": ["board_a"], "val": ["board_b"], "test": [], "num_train": 1, "num_val": 1, "num_test": 0}
    (organized / "split.json").write_text(json.dumps(split), encoding="utf-8")
    for board_id in ("board_a", "board_b"):
        image = np.zeros((3, 5, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(5, dtype=np.uint8)
        mask = np.zeros((3, 5), dtype=np.uint8)
        mask[1, 4] = 2
        Image.fromarray(image).save(organized / "images" / f"{board_id}_Col.tif")
        Image.fromarray(mask).save(organized / "combined_masks" / f"{board_id}_mask.png")

    output = tmp_path / "tiles"
    result_path, stats = create_tiled_dataset(organized, output, 3, 2)
    assert result_path == str(output)
    assert stats["tiles_per_split"] == {"train": 4, "val": 4, "test": 0}
    tiled_split = json.loads((output / "split.json").read_text(encoding="utf-8"))
    assert len(tiled_split["train"]) == 4
    manifest = json.loads((output / "tile_manifest.json").read_text(encoding="utf-8"))
    assert manifest[tiled_split["train"][0]]["parent_board"] == "board_a"
    assert manifest[tiled_split["train"][0]]["parent_size"] == [3, 5]
    assert (output / "images" / f"{tiled_split['train'][0]}_Col.tif").exists()
    assert (output / "combined_masks" / f"{tiled_split['train'][0]}_mask.png").exists()
    assert create_tiled_dataset(organized, output, 3, 2)[1] == stats


def test_train_background_tiles_can_be_dropped(tmp_path):
    organized = tmp_path / "organized"
    (organized / "images").mkdir(parents=True)
    (organized / "combined_masks").mkdir()
    (organized / "split.json").write_text(
        json.dumps({"train": ["board"], "val": [], "test": []}), encoding="utf-8"
    )
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    mask = np.zeros((2, 4), dtype=np.uint8)
    mask[:, 3] = 1
    Image.fromarray(image).save(organized / "images" / "board_Col.tif")
    Image.fromarray(mask).save(organized / "combined_masks" / "board_mask.png")

    _, stats = create_tiled_dataset(
        organized,
        tmp_path / "tiles",
        max_x=2,
        max_y=2,
        drop_background_train_tiles=True,
        min_tile_defect_pixels=1,
    )
    assert stats["tiles_per_split"] == {"train": 1, "val": 0, "test": 0}
