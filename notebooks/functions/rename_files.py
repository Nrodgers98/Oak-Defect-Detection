import os
from pathlib import Path


def rename_defect_files(directory: str = None) -> dict:
    """
    Rename files by applying specific naming transformations to defect types.
    
    Transforms:
    - Black_Rot → BlackRot
    - Stain_Minor → StainMinor
    - Yellow_Rot → YellowRot
    
    Args:
        directory: Path to directory containing files to rename. 
                   Defaults to data/raw-data/ in the project root.
    
    Returns:
        Dictionary with renamed file info: {old_name: new_name}
    """
    if directory is None:
        # Use default data directory
        directory = Path(__file__).parent / "data" / "raw-data"
    else:
        directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Mapping of old patterns to new patterns
    rename_mappings = {
        "Black_Rot": "BlackRot",
        "Stain_Minor": "StainMinor",
        "Yellow_Rot": "YellowRot",  
    }
    
    renamed_files = {}
    
    # Iterate through all files in the directory
    for file_path in directory.iterdir():
        if file_path.is_file():
            old_name = file_path.name
            new_name = old_name
            
            # Apply each rename mapping
            for old_pattern, new_pattern in rename_mappings.items():
                new_name = new_name.replace(old_pattern, new_pattern)
            
            # Only rename if the name changed
            if new_name != old_name:
                new_path = file_path.parent / new_name
                try:
                    file_path.rename(new_path)
                    renamed_files[old_name] = new_name
                    print(f"Renamed: {old_name} → {new_name}")
                except Exception as e:
                    print(f"Error renaming {old_name}: {e}")
    
    return renamed_files


if __name__ == "__main__":
    # Run the rename function
    result = rename_defect_files()
    print(f"\nTotal files renamed: {len(result)}")
    
    if result:
        print("\nRename summary:")
        for old_name, new_name in sorted(result.items()):
            print(f"  {old_name} → {new_name}")
