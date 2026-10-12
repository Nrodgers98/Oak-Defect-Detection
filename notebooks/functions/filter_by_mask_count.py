import shutil
from pathlib import Path
from collections import defaultdict

from functions.mask_filenames import parse_class_mask_filename


def filter_classes_by_mask_count(source_folder: str = None, mask_threshold: int = 50, output_folder: str = None) -> dict:
    """
    Copy recognized color class masks from classes with count >= threshold
    into filtered-data, along with their associated color images. Supports
    ``_Col_<class>``, ``_Col_Bin_<class>``, and ``_Bin_Col_<class>`` masks.
    All files are copied to a flat structure (no subfolders) in filtered-data.
    
    Args:
        source_folder: Path to source directory containing mask files.
                      Defaults to ../data/raw-data relative to this file.
        mask_threshold: Minimum number of masks to filter (default 50).
                       Classes with count >= threshold will be copied to filtered-data.
        output_folder: Path to destination folder for filtered masks/images.
                      Defaults to ../data/filtered-data relative to source_folder.
    
    Returns:
        Dictionary with filtering results:
        {
            'classes_moved': ['ClassName1', 'ClassName2', ...],
            'classes_kept': ['ClassName1', 'ClassName2', ...],
            'files_moved': {'ClassName': count, ...},
            'files_kept': {'ClassName': count, ...},
            'color_images_copied': int
        }
    """
    if source_folder is None:
        # Default to raw-data directory
        source_path = Path(__file__).parent.parent.parent / "data" / "raw-data"
    else:
        source_path = Path(source_folder)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source folder not found: {source_path}")
    
    if output_folder is None:
        output_folder = source_path.parent / "filtered-data"
    else:
        output_folder = Path(output_folder)
    
    # This folder is a generated intermediate; reset it so a rerun cannot mix
    # files from a previous raw dataset with the current filtered results.
    source_resolved = source_path.resolve()
    output_resolved = output_folder.resolve()
    if (
        source_resolved == output_resolved
        or source_resolved in output_resolved.parents
        or output_resolved in source_resolved.parents
    ):
        raise ValueError("source_folder and output_folder must not overlap")
    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Count masks by class
    masks_by_class = defaultdict(list)
    images_by_stem = defaultdict(list)  # Track exact associated color images
    
    for file in source_path.glob("*.tif"):
        filename = file.name
        
        parsed_mask = parse_class_mask_filename(file)
        if parsed_mask is not None:
            _image_stem, class_name = parsed_mask
            masks_by_class[class_name].append(filename)
        elif file.stem.endswith("_Col"):
            images_by_stem[file.stem].append(filename)
    
    # Separate classes by mask count
    classes_to_move = {}
    classes_to_keep = {}
    
    for class_name, files in masks_by_class.items():
        mask_count = len(files)
        if mask_count >= mask_threshold:
            classes_to_move[class_name] = files
        else:
            classes_to_keep[class_name] = files
    
    # Copy files
    files_moved = defaultdict(int)
    files_kept = defaultdict(int)
    color_images_copied = 0
    
    print("="*70)
    print(f"FILTERING CLASSES WITH >= {mask_threshold} MASKS")
    print("="*70)
    
    # Track which color images we've copied to avoid duplicates
    copied_images = set()
    
    # Copy files from classes meeting threshold
    for class_name, files in classes_to_move.items():
        # Copy mask files and track timestamps
        image_stems_to_copy = set()
        for filename in files:
            src = source_path / filename
            dst = output_folder / filename
            try:
                shutil.copy2(str(src), str(dst))
                files_moved[class_name] += 1
                
                parsed_mask = parse_class_mask_filename(filename)
                if parsed_mask is not None:
                    image_stems_to_copy.add(parsed_mask[0])
            except Exception as e:
                print(f"Error copying {filename}: {e}")
        
        # Copy associated color images
        for image_stem in image_stems_to_copy:
            for color_img in images_by_stem[image_stem]:
                if color_img not in copied_images:
                    src = source_path / color_img
                    dst = output_folder / color_img
                    try:
                        shutil.copy2(str(src), str(dst))
                        copied_images.add(color_img)
                        color_images_copied += 1
                    except Exception as e:
                        print(f"Error copying color image {color_img}: {e}")
    
    # Count kept files
    for class_name, files in classes_to_keep.items():
        files_kept[class_name] = len(files)
    
    # Print results
    print(f"\nClasses COPIED (>= {mask_threshold} masks):")
    for class_name in sorted(classes_to_move.keys()):
        mask_count = len(classes_to_move[class_name])
        moved_count = files_moved[class_name]
        print(f"  - {class_name}: {mask_count} masks → copied to {output_folder.name}/")
    
    print(f"\nClasses KEPT (< {mask_threshold} masks):")
    for class_name in sorted(classes_to_keep.keys()):
        mask_count = len(classes_to_keep[class_name])
        print(f"  - {class_name}: {mask_count} masks")
    
    print("="*70)
    print(f"\nTotal mask files copied: {sum(files_moved.values())}")
    print(f"Total color images copied: {color_images_copied}")
    print(f"Filtered data location: {output_folder}")
    print("="*70)
    
    return {
        'classes_moved': list(classes_to_move.keys()),
        'classes_kept': list(classes_to_keep.keys()),
        'files_moved': dict(files_moved),
        'files_kept': dict(files_kept),
        'color_images_copied': color_images_copied,
        'output_folder': str(output_folder)
    }


if __name__ == "__main__":
    # Default usage - scans raw-data by default
    result = filter_classes_by_mask_count(
        mask_threshold=50
    )
    print("\nFiltering complete!")
    print(f"Moved classes: {result['classes_moved']}")
    print(f"Kept classes: {result['classes_kept']}")
