"""Helpers for recognizing per-class color-mask TIFF filenames."""

from pathlib import Path
import re


_CLASS_MASK_SUFFIX = re.compile(
    r"^(?P<image_prefix>.+?)(?:_Col_Bin_|_Bin_Col_|_Col_)(?P<class_name>.+)$",
    re.IGNORECASE,
)


def parse_class_mask_filename(path: str | Path) -> tuple[str, str] | None:
    """Return ``(image_stem, class_name)`` for a class-mask TIFF.

    Supports legacy ``*_Col_Bin_<class>.tif`` and ``*_Bin_Col_<class>.tif``
    names, as well as ``*_Col_<class>.tif``. The returned image stem always
    ends in ``_Col`` so it matches the corresponding color-image stem.
    """
    mask_path = Path(path)
    if mask_path.suffix.lower() not in {".tif", ".tiff"}:
        return None

    match = _CLASS_MASK_SUFFIX.match(mask_path.stem)
    if match is None:
        return None

    image_stem = f"{match.group('image_prefix')}_Col"
    class_name = match.group("class_name")
    return image_stem, class_name