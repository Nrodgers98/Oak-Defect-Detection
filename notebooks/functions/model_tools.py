import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import numpy as np
import json
from sklearn.model_selection import train_test_split




## Function to analyze the dataset for Color Images and masks, broken up by class
def analyze_dataset(folder_path):
    """
    Analyze your current dataset structure
    """
    path = Path(folder_path)
    
    # Statistics
    stats = {
        'total_files': 0,
        'color_images': [],
        'masks_by_class': defaultdict(list),
        'other_files': []
    }
    
    for file in path.glob("*.tif"):
        stats['total_files'] += 1
        filename = file.name
        
        if "_Bin_" in filename:
            # Extract class name
            class_name = filename.split("_Bin_")[1].replace(".tif", "")
            stats['masks_by_class'][class_name].append(filename)
        elif "_Col.tif" in filename:
            stats['color_images'].append(filename)
        else:
            stats['other_files'].append(filename)
    
    # Print report
    print("="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    print(f"Total files: {stats['total_files']}")
    print(f"\nColor images (_Col.tif): {len(stats['color_images'])}")
    
    print(f"\nMask classes found:")
    for class_name, files in sorted(stats['masks_by_class'].items()):
        print(f"  - {class_name}: {len(files)} masks")
    
    print(f"\nOther files (ignored): {len(stats['other_files'])}")
    if stats['other_files'][:5]:
        print("  Examples:", stats['other_files'][:5])
    
    print("="*60)
    
    return stats


## Function to reorganize the dataset
def reorganize_dataset(source_folder, output_folder):
    """
    Organize data into proper structure:
    output/
      ├── images/
      └── masks/
          ├── Knot/
          ├── Stain/
          └── ...
    """
    source_path = Path(source_folder)
    output_path = Path(output_folder)
    
    # Clear existing output folder if it exists
    if output_path.exists():
        shutil.rmtree(output_path)
        print(f"Cleared existing folder: {output_path}")
    
    # Create directories
    images_dir = output_path / "images"
    masks_dir = output_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {'images': 0, 'masks': defaultdict(int)}
    
    # Process all files
    for file in source_path.glob("*.tif"):
        filename = file.name
        
        # Copy color images (ends with _Col.tif)
        if filename.endswith("_Col.tif"):
            shutil.copy2(file, images_dir / filename)
            stats['images'] += 1
        
        # Copy Col masks (contains _Col_Bin_ but NOT _Lum_Bin_)
        elif "_Col_Bin_" in filename and "_Lum_Bin_" not in filename:
            class_name = filename.split("_Col_Bin_")[1].replace(".tif", "")
            class_dir = masks_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            shutil.copy2(file, class_dir / filename)
            stats['masks'][class_name] += 1
    
    # Report
    print("="*60)
    print("REORGANIZATION COMPLETE")
    print("="*60)
    print(f"Images copied: {stats['images']}")
    print(f"\nMasks by class:")
    for class_name, count in sorted(stats['masks'].items()):
        print(f"  {class_name}: {count}")
    print("="*60)
    
    return stats


# Function to validate the dataset after reorganization
def validate_dataset(organized_folder):
    """
    Check that each image has corresponding masks
    and validate mask properties
    """
    base_path = Path(organized_folder)
    images_dir = base_path / "images"
    masks_dir = base_path / "masks"
    
    # Get all images
    image_files = sorted(images_dir.glob("*.tif"))
    
    # Get all mask classes
    mask_classes = [d.name for d in masks_dir.iterdir() if d.is_dir()]
    
    print("="*60)
    print("DATASET VALIDATION")
    print("="*60)
    print(f"Total images: {len(image_files)}")
    print(f"Mask classes: {mask_classes}")
    print()
    
    issues = []
    valid_samples = []
    
    for img_file in image_files:
        # Extract base name
        base_name = img_file.stem.replace("_Col", "")
        
        # Find corresponding masks
        masks_found = {}
        for class_name in mask_classes:
            mask_pattern = f"*{base_name}*_Bin_{class_name}.tif"
            mask_files = list((masks_dir / class_name).glob(mask_pattern))
            
            if mask_files:
                masks_found[class_name] = mask_files[0]
        
        # Check if at least one mask exists
        if not masks_found:
            issues.append(f"No masks for: {img_file.name}")
            continue
        
        # Validate dimensions
        img = Image.open(img_file)
        img_size = img.size
        
        for class_name, mask_file in masks_found.items():
            mask = Image.open(mask_file)
            if mask.size != img_size:
                issues.append(f"Size mismatch: {img_file.name} vs {mask_file.name}")
        
        valid_samples.append({
            'image': img_file,
            'masks': masks_found,
            'size': img_size
        })
    
    # Print results
    print(f"Valid samples: {len(valid_samples)}")
    print(f"Issues found: {len(issues)}")
    
    if issues:
        print("\nIssues:")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    
    print("="*60)
    
    return valid_samples, issues



def move_invalid_samples(organized_folder, issues, output_folder="../data/filtered-data"):
    """
    Move images and masks that have issues to a separate filtered-data folder
    
    Args:
        organized_folder: Path to the organized dataset
        issues: List of issue strings (e.g., ["No masks for: image1.tif", ...])
        output_folder: Path where invalid samples will be moved
    """
    base_path = Path(organized_folder)
    output_path = Path(output_folder)
    
    images_dir = base_path / "images"
    masks_dir = base_path / "masks"
    
    # Create output directories
    filtered_images_dir = output_path / "images"
    filtered_masks_dir = output_path / "masks"
    filtered_images_dir.mkdir(parents=True, exist_ok=True)
    filtered_masks_dir.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    
    # Extract filenames from issues
    for issue in issues:
        # Parse different issue types
        if "No masks for:" in issue:
            filename = issue.split("No masks for: ")[1].strip()
            img_file = images_dir / filename
            
            if img_file.exists():
                shutil.move(str(img_file), str(filtered_images_dir / filename))
                moved_count += 1
                print(f"Moved image: {filename}")
        
        elif "Size mismatch:" in issue:
            # Extract image filename from "Size mismatch: image.tif vs mask.tif"
            parts = issue.split("Size mismatch: ")[1].split(" vs ")
            img_filename = parts[0].strip()
            mask_filename = parts[1].strip()
            
            # Move image
            img_file = images_dir / img_filename
            if img_file.exists():
                shutil.move(str(img_file), str(filtered_images_dir / img_filename))
                moved_count += 1
                print(f"Moved image: {img_filename}")
            
            # Move mask
            for class_dir in masks_dir.iterdir():
                if class_dir.is_dir():
                    mask_file = class_dir / mask_filename
                    if mask_file.exists():
                        class_output = filtered_masks_dir / class_dir.name
                        class_output.mkdir(exist_ok=True)
                        shutil.move(str(mask_file), str(class_output / mask_filename))
                        moved_count += 1
                        print(f"Moved mask: {mask_filename}")
                        break
    
    print("="*60)
    print(f"Moved {moved_count} files to {output_folder}")
    print("="*60)
    
    return moved_count



def create_combined_masks(organized_folder):
    """
    Combine multiple binary masks into single multi-class mask
    Background = 0, Class1 = 1, Class2 = 2, etc.
    """
    base_path = Path(organized_folder)
    images_dir = base_path / "images"
    masks_dir = base_path / "masks"
    combined_masks_dir = base_path / "combined_masks"
    combined_masks_dir.mkdir(exist_ok=True)
    
    # Get mask classes
    mask_classes = sorted([d.name for d in masks_dir.iterdir() if d.is_dir()])
    class_mapping = {class_name: idx + 1 for idx, class_name in enumerate(mask_classes)}
    
    print("="*60)
    print("CREATING COMBINED MASKS")
    print("="*60)
    print("Class mapping:")
    print("  0: Background")
    for class_name, idx in class_mapping.items():
        print(f"  {idx}: {class_name}")
    print()
    
    # Process each image
    image_files = sorted(images_dir.glob("*.tif"))
    
    for img_file in tqdm(image_files, desc="Processing"):
        base_name = img_file.stem.replace("_Col", "")
        
        # Load image to get dimensions
        img = Image.open(img_file)
        width, height = img.size
        
        # Create empty combined mask
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Overlay each class mask
        for class_name, class_idx in class_mapping.items():
            mask_pattern = f"*{base_name}*_Bin_{class_name}.tif"
            mask_files = list((masks_dir / class_name).glob(mask_pattern))
            
            if mask_files:
                mask = np.array(Image.open(mask_files[0]))
                
                # Handle RGB masks (take first channel)
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]
                
                # Set pixels to class index where mask is active
                combined_mask[mask > 127] = class_idx
        
        # Save combined mask
        output_file = combined_masks_dir / f"{base_name}_mask.png"
        Image.fromarray(combined_mask).save(output_file)
    
    print(f"\nCombined masks saved to: {combined_masks_dir}")
    print("="*60)
    
    return class_mapping


def create_train_val_split(organized_folder, val_split=0.2, test_split=0.0, seed=42):
    """
    Split dataset into train, validation, (optional) test sets.
    test_split: fraction for test set (0.0 disables test split)
    """
    base_path = Path(organized_folder)
    images_dir = base_path / "images"

    # all image ids (without _Col)
    image_files = sorted([f.stem.replace("_Col", "") for f in images_dir.glob("*.tif")])

    # if no test split, do simple train/val
    if test_split and test_split > 0.0:
        # split into train+val+test
        train_val_ids, test_ids = train_test_split(image_files, test_size=test_split, random_state=seed)
        # adjust val fraction relative to train_val set
        #val_rel = val_split / (1.0 - test_split)
        train_ids, val_ids = train_test_split(train_val_ids, test_size=val_split, random_state=seed)
    else:
        train_ids, val_ids = train_test_split(image_files, test_size=val_split, random_state=seed)
        test_ids = []

    split_info = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "num_train": len(train_ids),
        "num_val": len(val_ids),
        "num_test": len(test_ids)
    }

    split_file = base_path / "split.json"
    with open(split_file, "w") as f:
        json.dump(split_info, f, indent=2)

    print("="*60)
    print("TRAIN/VAL/TEST SPLIT")
    print("="*60)
    print(f"Train samples: {len(train_ids)}")
    print(f"Val samples:   {len(val_ids)}")
    if test_ids:
        print(f"Test samples:  {len(test_ids)}")
    print(f"Split saved to: {split_file}")
    print("="*60)

    return split_info