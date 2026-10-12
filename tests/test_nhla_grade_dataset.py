import json
from pathlib import Path

import pandas as pd
from PIL import Image

from notebooks.NHLA_Classification import run_optuna_tuning
from notebooks.functions.nhla_grade_dataset import build_nhla_grade_dataset


def test_dataset_extraction_and_validation_rule(tmp_path):
    board = "B001_2024-08-12_10.05.50.102"
    xml_path = tmp_path / f"{board}.xml"
    xml_path.write_text(
        """<?xml version=\"1.0\" encoding=\"utf-8\"?>
<NHLA_BoardInfo xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">
  <NHLA_Grade_Vision>7</NHLA_Grade_Vision>
  <NHLA_Grade_Vision_Top>7</NHLA_Grade_Vision_Top>
  <NHLA_Grade_Vision_Bot>4</NHLA_Grade_Vision_Bot>
  <NHLA_Grade_Validation>0</NHLA_Grade_Validation>
  <NHLA_Grade_Validation_Top>0</NHLA_Grade_Validation_Top>
  <NHLA_Grade_Validation_Bot>0</NHLA_Grade_Validation_Bot>
</NHLA_BoardInfo>
"""
    )

    # Use realistic board image names with top/bottom markers.
    for suffix in ("_top.png", "_bottom.png"):
        (tmp_path / f"{board}{suffix}").write_bytes(b"fake")

    rows = build_nhla_grade_dataset(str(tmp_path), str(tmp_path / "nhla_dataset"), save_csvs=True)

    assert len(rows["faces"]) == 2
    assert len(rows["boards"]) == 1

    top = next(r for r in rows["faces"] if r["face"] == "top")
    bottom = next(r for r in rows["faces"] if r["face"] == "bottom")

    assert top["grade_label"] == "F1F"
    assert bottom["grade_label"] == "2A Com"

    board = rows["boards"][0]
    assert board["top_grade_label"] == "F1F"
    assert board["bottom_grade_label"] == "2A Com"


def test_validation_overrides_vision(tmp_path):
    board = "B002_2024-08-12_10.05.50.102"
    xml_path = tmp_path / f"{board}.xml"
    xml_path.write_text(
        """<?xml version=\"1.0\" encoding=\"utf-8\"?>
<NHLA_BoardInfo xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">
  <NHLA_Grade_Vision>7</NHLA_Grade_Vision>
  <NHLA_Grade_Vision_Top>7</NHLA_Grade_Vision_Top>
  <NHLA_Grade_Vision_Bot>4</NHLA_Grade_Vision_Bot>
  <NHLA_Grade_Validation>0</NHLA_Grade_Validation>
  <NHLA_Grade_Validation_Top>6</NHLA_Grade_Validation_Top>
  <NHLA_Grade_Validation_Bot>5</NHLA_Grade_Validation_Bot>
</NHLA_BoardInfo>
"""
    )

    (tmp_path / f"{board}_top.png").write_bytes(b"fake")
    (tmp_path / f"{board}_bot.png").write_bytes(b"fake")

    rows = build_nhla_grade_dataset(str(tmp_path), str(tmp_path / "nhla_dataset"), save_csvs=True)
    top = next(r for r in rows["faces"] if r["face"] == "top")
    bottom = next(r for r in rows["faces"] if r["face"] == "bottom")

    assert top["grade_label"] == "Select"
    assert bottom["grade_label"] == "1 Com"
    assert top["grade_source"] == "validation"
    assert bottom["grade_source"] == "validation"


def _write_board(root, board, vision, top, bot, validation):
    (root / f"{board}.xml").write_text(
        f"""<?xml version=\"1.0\" encoding=\"utf-8\"?>
<NHLA_BoardInfo>
  <NHLA_Grade_Vision>{vision}</NHLA_Grade_Vision>
  <NHLA_Grade_Vision_Top>{top}</NHLA_Grade_Vision_Top>
  <NHLA_Grade_Vision_Bot>{bot}</NHLA_Grade_Vision_Bot>
  <NHLA_Grade_Validation>{validation}</NHLA_Grade_Validation>
</NHLA_BoardInfo>
"""
    )
    (root / f"{board}_Top.jpg").write_bytes(b"fake")
    (root / f"{board}_Bot.jpg").write_bytes(b"fake")


def test_board_validation_is_not_used_as_face_grade(tmp_path):
    # Real scanner exports carry only a board-level validation grade.
    _write_board(tmp_path, "B010_2025-08-18_10.02.34.808", vision=7, top=5, bot=7, validation=5)

    rows = build_nhla_grade_dataset(str(tmp_path), str(tmp_path / "nhla_dataset"))
    faces = {r["face"]: r for r in rows["faces"]}

    assert faces["top"]["grade_label"] == "1 Com"
    assert faces["bottom"]["grade_label"] == "F1F"
    assert faces["bottom"]["grade_source"] == "vision"
    assert rows["boards"][0]["board_grade_label"] == "1 Com"
    assert rows["boards"][0]["vision_validated"] is False


def test_validated_face_csv_keeps_only_boards_where_grader_agrees(tmp_path):
    _write_board(tmp_path, "B011_2025-08-18_10.02.34.808", vision=5, top=5, bot=4, validation=5)
    _write_board(tmp_path, "B012_2025-08-18_10.02.36.604", vision=7, top=7, bot=7, validation=4)
    _write_board(tmp_path, "B013_2025-08-18_10.02.38.100", vision=4, top=4, bot=4, validation=0)

    rows = build_nhla_grade_dataset(str(tmp_path), str(tmp_path / "nhla_dataset"))

    validated = pd.read_csv(tmp_path / "nhla_dataset" / "nhla_face_labels_validated.csv")
    assert set(validated["board_id"]) == {"B011_2025-08-18_10.02.34.808"}
    assert dict(zip(validated["face"], validated["grade_label"])) == {"top": "1 Com", "bottom": "2A Com"}
    assert len(rows["validated_faces"]) == 2


def test_relative_scan_root_generates_absolute_image_paths(tmp_path, monkeypatch):
    project_root = tmp_path / "project_root"
    project_root.mkdir()
    scan_root = project_root / "data" / "sample_set"
    scan_root.mkdir(parents=True)

    board = "B003_2024-08-12_10.05.50.102"
    (scan_root / f"{board}.xml").write_text(
        """<?xml version=\"1.0\" encoding=\"utf-8\"?>
<NHLA_BoardInfo xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">
  <NHLA_Grade_Vision>7</NHLA_Grade_Vision>
  <NHLA_Grade_Vision_Top>7</NHLA_Grade_Vision_Top>
  <NHLA_Grade_Vision_Bot>4</NHLA_Grade_Vision_Bot>
  <NHLA_Grade_Validation>0</NHLA_Grade_Validation>
  <NHLA_Grade_Validation_Top>0</NHLA_Grade_Validation_Top>
  <NHLA_Grade_Validation_Bot>0</NHLA_Grade_Validation_Bot>
</NHLA_BoardInfo>
"""
    )
    (scan_root / f"{board}_Top.jpg").write_bytes(b"fake")
    (scan_root / f"{board}_Bot.jpg").write_bytes(b"fake")

    monkeypatch.chdir(tmp_path)
    rows = build_nhla_grade_dataset(
        str(Path("project_root") / "data" / "sample_set"),
        str(project_root / "data" / "sample_set" / "nhla_dataset"),
        save_csvs=True,
    )

    top_row = next(r for r in rows["faces"] if r["face"] == "top")
    bottom_row = next(r for r in rows["faces"] if r["face"] == "bottom")

    assert Path(top_row["image_path"]).is_absolute()
    assert Path(bottom_row["image_path"]).is_absolute()
    assert Path(top_row["image_path"]).exists()
    assert Path(bottom_row["image_path"]).exists()


def test_optuna_tuning_smoke(tmp_path):
    board_rows = [
        {"board_id": "B001", "face": "top", "image_path": str(tmp_path / "B001_top.png"), "grade_label": "FAS"},
        {"board_id": "B001", "face": "bottom", "image_path": str(tmp_path / "B001_bottom.png"), "grade_label": "F1F"},
        {"board_id": "B002", "face": "top", "image_path": str(tmp_path / "B002_top.png"), "grade_label": "FAS"},
        {"board_id": "B002", "face": "bottom", "image_path": str(tmp_path / "B002_bottom.png"), "grade_label": "F1F"},
    ]

    for row in board_rows:
        img = Image.new("RGB", (64, 64), color=(255, 0, 0) if "top" in row["image_path"] else (0, 255, 0))
        img.save(row["image_path"])

    csv_path = tmp_path / "nhla_face_labels.csv"
    pd.DataFrame(board_rows).to_csv(csv_path, index=False)

    output_dir = tmp_path / "optuna_results"
    study, best_params = run_optuna_tuning(
        csv_path=str(csv_path),
        mode="face",
        output_root=str(output_dir),
        n_trials=1,
        epochs=1,
        seed=42,
        val_split=0.5,
        image_cache_dir=str(tmp_path / "cache"),
    )

    assert len(study.trials) == 1
    assert "lr" in best_params
    assert "batch_size" in best_params
    best_params_path = output_dir / "best_params.json"
    assert best_params_path.exists()

    with best_params_path.open("r") as fh:
        payload = json.load(fh)
    assert payload["lr"] == best_params["lr"]
