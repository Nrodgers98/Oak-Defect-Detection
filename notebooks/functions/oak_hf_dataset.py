"""
Download the Oak Defect Detection dataset from Hugging Face into ``data/raw-data``.

The Hub repo stores line-scan TIFFs under ``raw-data/`` in the same flat naming
convention expected by ``prepare_dataset`` (``*_Col.tif``, ``*_Col_Bin_*.tif``).

Dataset: https://huggingface.co/datasets/nrodgers98/Oak-Defect-Detection
License: CC BY-NC 4.0 (see dataset card).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

HF_OAK_DEFECT_DATASET_REPO = "nrodgers98/Oak-Defect-Detection"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def default_raw_data_path() -> Path:
    """``<project>/data/raw-data`` (same default as ``prepare_dataset``)."""
    return _repo_root() / "data" / "raw-data"


def ensure_oak_raw_data(
    dest_dir: str | Path | None = None,
    *,
    repo_id: str = HF_OAK_DEFECT_DATASET_REPO,
    force: bool = False,
) -> Path:
    """
    Ensure raw TIFFs from the Hugging Face dataset are available under *dest_dir*.

    If *dest_dir* already contains at least one ``*.tif`` and *force* is False,
    returns immediately without re-downloading. Otherwise pulls a Hub snapshot
    (cached by ``huggingface_hub``) and copies ``raw-data/*.tif`` into *dest_dir*.

    Parameters
    ----------
    dest_dir
        Target directory (defaults to ``data/raw-data`` at the project root).
    repo_id
        Hugging Face dataset repository id.
    force
        If True, remove existing ``*.tif`` files in *dest_dir* and re-copy from the Hub.

    Returns
    -------
    pathlib.Path
        Resolved path to *dest_dir*.
    """
    dest = Path(dest_dir) if dest_dir is not None else default_raw_data_path()
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if not force and any(dest.glob("*.tif")):
        return dest

    if force:
        for f in dest.glob("*.tif"):
            f.unlink()

    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["raw-data/*.tif"],
        )
    )
    src = snapshot_dir / "raw-data"
    if not src.is_dir():
        raise RuntimeError(f"Missing 'raw-data' in Hub snapshot at {snapshot_dir}")

    for f in src.glob("*.tif"):
        shutil.copy2(f, dest / f.name)

    return dest


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Download Oak Defect Detection raw-data from Hugging Face")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Output directory (default: {default_raw_data_path()})",
    )
    p.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Delete existing *.tif in output and re-copy from the Hub",
    )
    p.add_argument(
        "--repo",
        default=HF_OAK_DEFECT_DATASET_REPO,
        help="Hugging Face dataset repo id",
    )
    args = p.parse_args()
    out = ensure_oak_raw_data(args.output, repo_id=args.repo, force=args.force)
    print(f"Oak raw data ready at: {out}")
