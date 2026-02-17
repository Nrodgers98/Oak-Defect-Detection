import shutil
from pathlib import Path
from collections import defaultdict


def filter_classes_by_mask_count(source_folder: str, mask_threshold: int = 50, output_folder: str = None) -> dict:
    """
    Copy classes with MORE than a certain number of masks to a filtered data folder,
    along with their associated color images.
    
    Args:
        source_folder: Path to source directory containing mask files
        mask_threshold: Maximum number of masks to filter (default 50).
                       Classes with MORE masks will be copied to filtered-data.
        output_folder: Path to destination folder for filtered classes.
                      Defaults to /data/filtered-data
    
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
    source_path = Path(source_folder)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source folder not found: {source_path}")
    
    if output_folder is None:
        output_folder = source_path.parent / "filtered-data"
    else:
        output_folder = Path(output_folder)
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Count masks by class
    masks_by_class = defaultdict(list)
    images_by_timestamp = defaultdict(list)  # Track color images for syncing
    
    for file in source_path.glob("*.tif"):
        filename = file.name
        
        # Extract timestamp (e.g., "1-29-26_3.28.48.206" from full filename)
        timestamp = filename.split("_")[0] + "_" + "_".join(filename.split("_")[1:3])
        
        if "_Bin_" in filename:
            # Extract class name from mask files
            class_name = filename.split("_Bin_")[1].replace(".tif", "")
            masks_by_class[class_name].append(filename)
        elif "_Col.tif" in filename:
            # Track color images by timestamp
            images_by_timestamp[timestamp].append(filename)
    
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
        class_output_dir = output_folder / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy mask files and track timestamps
        timestamps_to_copy = set()
        for filename in files:
            src = source_path / filename
            dst = class_output_dir / filename
            try:
                shutil.copy2(str(src), str(dst))
                files_moved[class_name] += 1
                
                # Extract timestamp for color image matching
                parts = filename.split("_")
                timestamp = parts[0] + "_" + parts[1]
                timestamps_to_copy.add(timestamp)
            except Exception as e:
                print(f"Error copying {filename}: {e}")
        
        # Copy associated color images
        for timestamp in timestamps_to_copy:
            for color_img in images_by_timestamp[timestamp]:
                if color_img not in copied_images:
                    src = source_path / color_img
                    dst = class_output_dir / color_img
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
    # Default usage
    source_folder = "../data/output_scv2"
    result = filter_classes_by_mask_count(
        source_folder=source_folder,
        mask_threshold=50
    )
    print("\nFiltering complete!")
    print(f"Moved classes: {result['classes_moved']}")
    print(f"Kept classes: {result['classes_kept']}")
