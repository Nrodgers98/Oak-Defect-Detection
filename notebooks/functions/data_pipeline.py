"""
Consolidated data-preparation pipeline.

Call ``prepare_dataset()`` from any notebook to run (or skip) the full
data-organization workflow in one shot.
"""

from pathlib import Path

from functions.model_tools import (
    analyze_dataset,
    reorganize_dataset,
    validate_dataset,
    move_invalid_samples,
    create_combined_masks,
    create_train_val_split,
)
from functions.rename_files import rename_defect_files
from functions.filter_by_mask_count import filter_classes_by_mask_count


def _recover_class_mapping(organized_folder: str) -> dict:
    """
    Reconstruct the class_mapping dict from an already-organized dataset
    by reading the mask subdirectories.

    Returns e.g. {"BlackRot": 1, "Knot": 2}
    """
    masks_dir = Path(organized_folder) / "masks"
    mask_classes = sorted([d.name for d in masks_dir.iterdir() if d.is_dir()])
    return {name: idx + 1 for idx, name in enumerate(mask_classes)}


def prepare_dataset(
    raw_data_folder: str = "../data/raw-data",
    mask_threshold: int = 200,
    val_split: float = 0.16,
    test_split: float = 0.2,
    filtered_folder: str = "../data/filtered-data",
    organized_folder: str = "../data/organized-data",
    seed: int = 42,
    force: bool = False,
) -> tuple:
    """
    Run the full data-preparation pipeline, or skip it if the organized
    dataset already exists.

    Steps executed (in order):
      1. Rename inconsistent defect file names (e.g. Black_Rot → BlackRot)
      2. Filter out classes with fewer than *mask_threshold* masks
      3. Reorganize filtered data into images/ and masks/ folders
      4. Validate the reorganized dataset and move invalid samples
      5. Create combined multi-class masks
      6. Create reproducible train / val / test splits

    Parameters
    ----------
    raw_data_folder : str
        Path to the raw line-scan data directory.
    mask_threshold : int
        Minimum number of masks a defect class must have to be kept.
    val_split : float
        Fraction of data reserved for validation.
    test_split : float
        Fraction of data reserved for testing (0.0 to disable).
    filtered_folder : str
        Intermediate folder for class-filtered data.
    organized_folder : str
        Final organized-data folder (images/, masks/, combined_masks/, split.json).
    seed : int
        Random seed for reproducible splits.
    force : bool
        If *True*, re-run the full pipeline even when organized data already
        exists.  If *False* (default), skip processing when ``split.json``
        and ``combined_masks/`` are already present.

    Returns
    -------
    (organized_folder, class_mapping)
        *organized_folder* is the path string to the ready-to-use data root.
        *class_mapping* is a dict mapping class names to integer labels
        (e.g. ``{"BlackRot": 1, "Knot": 2}``).  Background is always 0.
    """
    organized_path = Path(organized_folder)
    split_exists = (organized_path / "split.json").exists()
    masks_exist = (organized_path / "combined_masks").exists()

    # ── Skip logic ────────────────────────────────────────────────────
    if not force and split_exists and masks_exist:
        class_mapping = _recover_class_mapping(organized_folder)
        print("=" * 60)
        print("PREPARE_DATASET — SKIPPED (data already prepared)")
        print("=" * 60)
        print(f"  Organized data: {organized_folder}")
        print(f"  Classes: {class_mapping}")
        print(f"  (pass force=True to re-run the full pipeline)")
        print("=" * 60)
        return organized_folder, class_mapping

    # ── Step 1: Normalize defect file names ───────────────────────────
    print("\n[1/6] Renaming inconsistent defect files …")
    rename_defect_files(raw_data_folder)

    # ── Step 2: Filter rare classes ───────────────────────────────────
    print(f"\n[2/6] Filtering classes with < {mask_threshold} masks …")
    filter_classes_by_mask_count(
        source_folder=raw_data_folder,
        mask_threshold=mask_threshold,
        output_folder=filtered_folder,
    )

    # ── Step 3: Reorganize into images/ + masks/ ──────────────────────
    print("\n[3/6] Reorganizing dataset …")
    reorganize_dataset(filtered_folder, organized_folder)

    # ── Step 4: Validate and remove bad samples ───────────────────────
    print("\n[4/6] Validating dataset …")
    _valid_samples, issues = validate_dataset(organized_folder)
    if issues:
        move_invalid_samples(organized_folder, issues, output_folder=filtered_folder)

    # ── Step 5: Create combined multi-class masks ─────────────────────
    print("\n[5/6] Creating combined masks …")
    class_mapping = create_combined_masks(organized_folder)

    # ── Step 6: Train / val / test split ──────────────────────────────
    print("\n[6/6] Creating train/val/test split …")
    create_train_val_split(
        organized_folder,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )

    print("\n" + "=" * 60)
    print("PREPARE_DATASET — COMPLETE")
    print("=" * 60)
    print(f"  Organized data: {organized_folder}")
    print(f"  Classes: {class_mapping}")
    print("=" * 60)

    return organized_folder, class_mapping
