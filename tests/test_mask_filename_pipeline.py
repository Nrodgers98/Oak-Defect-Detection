import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from functions.filter_by_mask_count import filter_classes_by_mask_count
from functions.mask_filenames import parse_class_mask_filename
from functions.model_tools import create_combined_masks, reorganize_dataset, validate_dataset


def test_parse_class_mask_filename_supports_current_and_legacy_names():
    cases = {
        "board_Top_Col_Knot.tif": ("board_Top_Col", "Knot"),
        "board_Top_Col_Bin_Rot.tif": ("board_Top_Col", "Rot"),
        "board_Top_Bin_Col_Sapwood.tif": ("board_Top_Col", "Sapwood"),
    }

    for filename, expected in cases.items():
        assert parse_class_mask_filename(filename) == expected

    assert parse_class_mask_filename("board_Top_Col.tif") is None


def test_data_prep_handles_mixed_class_mask_suffixes(tmp_path):
    raw_dir = tmp_path / "raw"
    filtered_dir = tmp_path / "filtered"
    organized_dir = tmp_path / "organized"
    raw_dir.mkdir()

    samples = [
        ("board1_Top", "Knot", "_Col_Knot"),
        ("board2_Top", "Rot", "_Col_Bin_Rot"),
        ("board3_Top", "Sapwood", "_Bin_Col_Sapwood"),
    ]
    for board, _class_name, mask_suffix in samples:
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            raw_dir / f"{board}_Col.tif"
        )
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 2:5] = 255
        Image.fromarray(mask).save(raw_dir / f"{board}{mask_suffix}.tif")

    result = filter_classes_by_mask_count(
        source_folder=raw_dir,
        mask_threshold=1,
        output_folder=filtered_dir,
    )
    assert set(result["classes_moved"]) == {"Knot", "Rot", "Sapwood"}
    assert result["color_images_copied"] == 3

    reorganize_dataset(filtered_dir, organized_dir)
    valid_samples, issues = validate_dataset(organized_dir)
    assert len(valid_samples) == 3
    assert issues == []

    class_mapping = create_combined_masks(organized_dir)
    assert set(class_mapping) == {"Knot", "Rot", "Sapwood"}
    for board, class_name, _mask_suffix in samples:
        combined = np.asarray(
            Image.open(organized_dir / "combined_masks" / f"{board}_mask.png")
        )
        assert np.all(combined[2:5, 2:5] == class_mapping[class_name])