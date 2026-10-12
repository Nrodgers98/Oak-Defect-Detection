import numpy as np
import json

from notebooks.functions.yolo_detection import (
    box_iou,
    clip_box_to_tile,
    binary_detection_metrics,
    match_detections,
    parse_image_identity,
    restore_tile_box,
    tile_label_rows,
    tile_windows,
    xywhn_to_xyxy,
    xyxy_to_xywhn,
    write_json,
)


def test_normalized_pixel_box_round_trip():
    normalized = (0.5, 0.5, 0.4, 0.2)
    pixel_box = xywhn_to_xyxy(normalized, 100, 50)
    assert pixel_box == (30.0, 20.0, 70.0, 30.0)
    assert xyxy_to_xywhn(pixel_box, 100, 50) == normalized


def test_tile_clip_and_restore():
    clipped = clip_box_to_tile(
        (10, 10, 90, 40),
        (0, 50, 0, 50),
        min_retained_fraction=0.25,
        min_width=2,
        min_height=2,
    )
    assert clipped == (10.0, 10.0, 50.0, 40.0)
    assert restore_tile_box((0, 10, 40, 40), (0, 50, 50, 100)) == (
        50.0,
        10.0,
        90.0,
        40.0,
    )


def test_tile_windows_cover_image_without_gaps():
    windows = tile_windows(7, 11, 4, 6, 0.25)
    coverage = np.zeros((7, 11), dtype=np.int8)
    for y1, y2, x1, x2 in windows:
        coverage[y1:y2, x1:x2] += 1
    assert np.all(coverage > 0)


def test_iou_and_filename_identity():
    assert box_iou((0, 0, 10, 10), (5, 5, 15, 15)) == 25 / 175
    identity = parse_image_identity("12_1789597583418_Top_Col.tif")
    assert identity["group_id"] == "12"
    assert identity["face"] == "Top"
    assert identity["timestamp"] == 1789597583418


def test_tile_label_rows_and_detection_metrics(tmp_path):
    label_path = tmp_path / "sample.txt"
    label_path.write_text("0 0.5 0.5 0.8 0.4\n", encoding="utf-8")
    rows = tile_label_rows(label_path, 100, 50, (0, 50, 0, 60), 0.25, 2, 2)
    assert rows[0][0] == 0
    assert rows[0][3] < 1.0

    matches = match_detections([(0, 0, 10, 10, 0.9), (20, 20, 30, 30, 0.8)], [(0, 0, 10, 10)])
    assert matches == {"tp": 1, "fp": 1, "fn": 0}
    assert binary_detection_metrics(matches)["precision"] == 0.5


def test_write_json_normalizes_tuple_keys(tmp_path):
    output_path = tmp_path / "summary.json"
    write_json(output_path, {"dimensions": {(768, 2048): np.int64(400)}})
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["dimensions"]["(768, 2048)"] == 400