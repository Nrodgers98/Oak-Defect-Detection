from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


DEFAULT_GRADE_MAP = {
    0: "Unknown",
    1: "3B Com",
    2: "3A Com",
    3: "2B Com",
    4: "2A Com",
    5: "1 Com",
    6: "Select",
    7: "F1F",
    8: "FAS",
}

FACE_KEYWORDS = {
    "top": {"top", "t", "face1", "front", "upper"},
    "bottom": {"bottom", "bot", "b", "face2", "back", "lower"},
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def safe_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    raw = str(value).strip()
    if raw in {"", "null", "None", "nan"}:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _find_element_text(root: ET.Element, tags: Iterable[str]) -> Optional[str]:
    for tag in tags:
        for candidate in root.iter(tag):
            text = candidate.text
            if text is not None and text.strip() not in {"", "0"}:
                return text.strip()
    return None


def _find_grade_value(root: ET.Element, face: str, validation: bool = False) -> Optional[int]:
    suffix = "Top" if face == "top" else "Bot"
    tag_names = [
        f"NHLA_Grade_Validation_{suffix}",
        f"NHLA_Grade_Vision_{suffix}",
        f"NHLA_Grade_Validation",
        f"NHLA_Grade_Vision",
    ]
    if validation:
        tag_names = [
            f"NHLA_Grade_Validation_{suffix}",
            f"NHLA_Grade_Validation",
            f"NHLA_Grade_Vision_{suffix}",
            f"NHLA_Grade_Vision",
        ]

    for tag_name in tag_names:
        for element in root.iter(tag_name):
            value = safe_int(element.text)
            if value is None:
                continue
            if validation and value == 0:
                continue
            return value

    if validation:
        return None

    # If the XML has only a zero-valued validation tag and no vision tag,
    # we still want the vision value to be considered valid when the grader
    # hasn't overridden it.
    for tag_name in [f"NHLA_Grade_Vision_{suffix}", "NHLA_Grade_Vision"]:
        for element in root.iter(tag_name):
            value = safe_int(element.text)
            if value is not None:
                return value

    return None


def grade_label_from_value(raw_value: Optional[int], grade_map: Optional[Dict[int, str]] = None) -> str:
    mapping = DEFAULT_GRADE_MAP if grade_map is None else grade_map
    if raw_value is None:
        return "Missing"
    return mapping.get(raw_value, f"grade_{raw_value}")


def resolve_grade_for_face(root: ET.Element, face: str, grade_map: Optional[Dict[int, str]] = None) -> Tuple[Optional[int], str, str]:
    """Return (grade_value, grade_label, source)."""
    validation_value = _find_grade_value(root, face, validation=True)
    vision_value = _find_grade_value(root, face, validation=False)

    if validation_value is not None and validation_value > 0:
        raw_value = validation_value
        source = "validation"
    elif vision_value is not None:
        raw_value = vision_value
        source = "vision"
    else:
        raw_value = validation_value if validation_value is not None else vision_value
        source = "missing"

    label = grade_label_from_value(raw_value, grade_map)
    return raw_value, label, source


def resolve_board_grade(root: ET.Element, grade_map: Optional[Dict[int, str]] = None) -> Tuple[Optional[int], str, str]:
    validation_value = None
    for tag in ["NHLA_Grade_Validation"]:
        for element in root.iter(tag):
            value = safe_int(element.text)
            if value is not None and value > 0:
                validation_value = value
                break
            if value is not None and value == 0:
                validation_value = None
        if validation_value is not None:
            break

    vision_value = None
    for tag in ["NHLA_Grade_Vision"]:
        for element in root.iter(tag):
            value = safe_int(element.text)
            if value is not None:
                vision_value = value
                break

    if validation_value is not None:
        raw_value = validation_value
        source = "validation"
    elif vision_value is not None:
        raw_value = vision_value
        source = "vision"
    else:
        raw_value = validation_value if validation_value is not None else vision_value
        source = "missing"

    label = grade_label_from_value(raw_value, grade_map)
    return raw_value, label, source


def _normalize_face_name(name: str) -> str:
    lower = name.lower()
    if any(token in lower for token in FACE_KEYWORDS["top"]):
        return "top"
    if any(token in lower for token in FACE_KEYWORDS["bottom"]):
        return "bottom"
    return "unknown"


def _match_image_for_face(board_stem: str, face: str, image_files: List[Path]) -> Optional[Path]:
    board_prefix = board_stem.lower()
    face_keywords = FACE_KEYWORDS.get(face, {face})

    candidates: List[Tuple[int, Path]] = []
    for path in image_files:
        lower_name = path.name.lower()
        if not lower_name.endswith(tuple(IMAGE_SUFFIXES)):
            continue

        shared_prefix = board_prefix in lower_name
        face_match = any(keyword in lower_name for keyword in face_keywords)
        if not shared_prefix:
            continue
        if not face_match:
            continue

        score = 0
        score += 30 if board_prefix in lower_name else 0
        score += 20 if face_match else 0
        if "top" in lower_name and face == "top":
            score += 15
        if "bot" in lower_name and face == "bottom":
            score += 15
        candidates.append((score, path))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _discover_board_images(scan_root: Path) -> List[Path]:
    return sorted(
        (p for p in scan_root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: str(p)
    )


# Longest keywords first so "bottom"/"back" win over the ambiguous single-letter "b".
_FACE_SUFFIXES: List[Tuple[str, str]] = sorted(
    ((face, kw) for face, kws in FACE_KEYWORDS.items() for kw in kws),
    key=lambda item: len(item[1]),
    reverse=True,
)


def _index_images_by_board(image_files: List[Path]) -> Dict[str, Dict[str, Path]]:
    """Build board_key -> {face: path} in O(N) instead of matching every board
    against every image (O(N*M)), which is what made large scans effectively hang."""
    index: Dict[str, Dict[str, Path]] = {}
    for path in image_files:
        stem_lower = path.stem.lower()
        for face, keyword in _FACE_SUFFIXES:
            for sep in ("_", "-", ""):
                suffix = f"{sep}{keyword}"
                if suffix and stem_lower.endswith(suffix) and len(stem_lower) > len(suffix):
                    board_key = stem_lower[: -len(suffix)]
                    index.setdefault(board_key, {}).setdefault(face, path)
                    break
            else:
                continue
            break
    return index


def _board_stem_from_xml(xml_path: Path) -> str:
    return xml_path.stem


def _board_metadata_from_xml(xml_path: Path) -> Dict[str, object]:
    root = ET.parse(xml_path).getroot()
    board_id = _board_stem_from_xml(xml_path)
    board_grade_raw, board_grade_label, board_grade_source = resolve_board_grade(root)

    records = []
    for face in ("top", "bottom"):
        raw_value, label, source = resolve_grade_for_face(root, face)
        records.append({
            "board_id": board_id,
            "face": face,
            "grade_value": raw_value,
            "grade_label": label,
            "grade_source": source,
            "board_grade_value": board_grade_raw,
            "board_grade_label": board_grade_label,
            "board_grade_source": board_grade_source,
        })

    return {
        "board_id": board_id,
        "xml_path": str(xml_path),
        "board_grade_value": board_grade_raw,
        "board_grade_label": board_grade_label,
        "board_grade_source": board_grade_source,
        "faces": records,
    }


def build_nhla_grade_dataset(
    scan_root: str,
    output_dir: Optional[str] = None,
    grade_map: Optional[Dict[int, str]] = None,
    save_csvs: bool = True,
) -> Dict[str, List[Dict[str, object]]]:
    """Assemble side-level and board-level CSV data from a folder of XML + images."""
    root = Path(scan_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Scan directory does not exist: {scan_root}")

    xml_files = sorted(root.rglob("*.xml"), key=lambda p: str(p))
    image_files = [_p.resolve() for _p in _discover_board_images(root)]
    image_index = _index_images_by_board(image_files)
    grade_map = DEFAULT_GRADE_MAP if grade_map is None else grade_map

    face_rows: List[Dict[str, object]] = []
    board_rows: List[Dict[str, object]] = []

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError:
            continue

        root_element = tree.getroot()
        board_id = _board_stem_from_xml(xml_file)
        board_grade_raw, board_grade_label, board_grade_source = resolve_board_grade(root_element, grade_map)
        board_images = image_index.get(board_id.lower(), {})

        face_grades: Dict[str, Tuple[Optional[int], str]] = {}
        for face in ("top", "bottom"):
            grade_raw, grade_label, grade_source = resolve_grade_for_face(root_element, face, grade_map)
            face_image = board_images.get(face)
            face_grades[face] = (grade_raw, grade_label)
            face_rows.append({
                "board_id": board_id,
                "face": face,
                "image_path": str(face_image) if face_image is not None else "",
                "grade_value": grade_raw,
                "grade_label": grade_label,
                "grade_source": grade_source,
                "board_grade_value": board_grade_raw,
                "board_grade_label": board_grade_label,
                "board_grade_source": board_grade_source,
                "xml_path": str(xml_file),
            })

        top_image = board_images.get("top")
        bottom_image = board_images.get("bottom")
        top_grade_value, top_grade_label = face_grades["top"]
        bottom_grade_value, bottom_grade_label = face_grades["bottom"]

        board_rows.append({
            "board_id": board_id,
            "top_image_path": str(top_image) if top_image is not None else "",
            "bottom_image_path": str(bottom_image) if bottom_image is not None else "",
            "top_grade_value": top_grade_value,
            "top_grade_label": top_grade_label,
            "bottom_grade_value": bottom_grade_value,
            "bottom_grade_label": bottom_grade_label,
            "board_grade_value": board_grade_raw,
            "board_grade_label": board_grade_label,
            "board_grade_source": board_grade_source,
            "xml_path": str(xml_file),
        })

    if save_csvs:
        output = Path(output_dir).expanduser().resolve() if output_dir is not None else (root / "nhla_dataset").resolve()
        output.mkdir(parents=True, exist_ok=True)

        face_csv = output / "nhla_face_labels.csv"
        board_csv = output / "nhla_board_labels.csv"

        with face_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "board_id",
                    "face",
                    "image_path",
                    "grade_value",
                    "grade_label",
                    "grade_source",
                    "board_grade_value",
                    "board_grade_label",
                    "board_grade_source",
                    "xml_path",
                ],
            )
            writer.writeheader()
            writer.writerows(face_rows)

        with board_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "board_id",
                    "top_image_path",
                    "bottom_image_path",
                    "top_grade_value",
                    "top_grade_label",
                    "bottom_grade_value",
                    "bottom_grade_label",
                    "board_grade_value",
                    "board_grade_label",
                    "board_grade_source",
                    "xml_path",
                ],
            )
            writer.writeheader()
            writer.writerows(board_rows)

        grade_map_path = output / "grade_mapping.json"
        with grade_map_path.open("w") as fh:
            json.dump({str(k): v for k, v in grade_map.items()}, fh, indent=2)

    return {"faces": face_rows, "boards": board_rows}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NHLA grade CSVs from a folder of XML and board images.")
    parser.add_argument("--scan-root", required=True, help="Folder containing XML files and board images.")
    parser.add_argument("--output-dir", default=None, help="Directory to write output CSVs. Defaults to scan-root/nhla_dataset.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rows = build_nhla_grade_dataset(args.scan_root, args.output_dir)
    print(f"Face rows: {len(rows['faces'])}")
    print(f"Board rows: {len(rows['boards'])}")
