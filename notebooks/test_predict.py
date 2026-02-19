"""
Simple test script for predict_image.py
"""
import sys
from pathlib import Path
from datetime import datetime

# Add notebooks directory to path so we can import
sys.path.insert(0, str(Path(__file__).parent))

from predict_image import predict_image, load_image, visualize_prediction
from PIL import Image

# Ensure predictions directory exists
predictions_dir = Path(__file__).parent.parent / "data" / "predictions"
predictions_dir.mkdir(parents=True, exist_ok=True)

# Test configuration
TEST_IMAGE = "../data/organized-data/images/1-29-26_3.28.48.206_Bot_Col.tif"
FULL_IMAGE_MODEL = "../models/UNet_FullImage/best_model.pt"
PATCH_MODEL = "../models/UNet_Patches/model_best.pt"

print("=" * 60)
print("Testing Full-Image Model")
print("=" * 60)

try:
    # Test full-image model (output_path=None will auto-generate timestamped filename)
    mask = predict_image(
        image_path=TEST_IMAGE,
        model_path=FULL_IMAGE_MODEL,
        output_path=None,  # Will auto-generate timestamped filename in data/predictions/
        use_patch_based=False,
        device='cuda'  # Change to 'cpu' if no GPU
    )
    
    # Create visualization
    image = load_image(TEST_IMAGE)
    vis = visualize_prediction(image, mask)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_path = predictions_dir / f"visualization_full_{timestamp}.png"
    Image.fromarray(vis).save(vis_path)
    print("✓ Full-image model test completed!")
    print("  - Prediction saved to data/predictions/ with timestamp")
    print(f"  - Visualization saved: {vis_path}")
    
except Exception as e:
    print(f"✗ Error testing full-image model: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Testing Patch-Based Model")
print("=" * 60)

try:
    # Test patch-based model (output_path=None will auto-generate timestamped filename)
    mask = predict_image(
        image_path=TEST_IMAGE,
        model_path=PATCH_MODEL,
        output_path=None,  # Will auto-generate timestamped filename in data/predictions/
        use_patch_based=True,
        tile_size=512,
        device='cuda'  # Change to 'cpu' if no GPU
    )
    
    # Create visualization
    image = load_image(TEST_IMAGE)
    vis = visualize_prediction(image, mask)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_path = predictions_dir / f"visualization_patch_{timestamp}.png"
    Image.fromarray(vis).save(vis_path)
    print("✓ Patch-based model test completed!")
    print("  - Prediction saved to data/predictions/ with timestamp")
    print(f"  - Visualization saved: {vis_path}")
    
except Exception as e:
    print(f"✗ Error testing patch-based model: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
