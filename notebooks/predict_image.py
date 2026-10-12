"""
Prediction script for oak defect detection models.

This script loads a trained UNet model and generates prediction masks for board images.
Supports both patch-based and full-image models.

Example usage:
    # Command line:
    python predict_image.py --image path/to/image.tif --model path/to/model.pt --output prediction.png
    
    # Programmatic:
    from predict_image import predict_image
    mask = predict_image(
        image_path='path/to/image.tif',
        model_path='path/to/model.pt',
        output_path='prediction.png',
        use_patch_based=False  # Set True for patch-based models
    )
    
    # For Streamlit:
    import streamlit as st
    from predict_image import predict_image, visualize_prediction
    from PIL import Image
    
    uploaded_file = st.file_uploader("Upload board image", type=['tif', 'tiff', 'png', 'jpg'])
    model_path = st.text_input("Model path", "models/UNet_FullImage/best_model.pt")
    
    if uploaded_file and st.button("Predict"):
        mask = predict_image(uploaded_file.name, model_path)
        vis = visualize_prediction(Image.open(uploaded_file), mask)
        st.image(vis, caption="Prediction overlay")
"""

import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Mapping, Tuple, Optional, Union
from datetime import datetime
import segmentation_models_pytorch as smp
import time

# Default ImageNet normalization (matches training)
DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Generic labels avoid assuming a checkpoint-specific class ordering.
CLASS_NAMES = [f"Class {class_id}" for class_id in range(4)]


def apply_confidence_threshold(
    pred_mask: np.ndarray,
    prob_map: np.ndarray,
    thresholds: Optional[Mapping[int, float]] = None,
) -> np.ndarray:
    """Suppress low-confidence winning classes to background (class 0)."""
    if thresholds is None:
        return pred_mask
    if prob_map.ndim != 3 or prob_map.shape[1:] != pred_mask.shape:
        raise ValueError("prob_map must have shape (classes, height, width)")

    thresholded_mask = pred_mask.copy()
    for class_id, threshold_percent in thresholds.items():
        if class_id < 0 or class_id >= prob_map.shape[0]:
            raise ValueError(f"class_id {class_id} is outside the probability map")
        if not 0 <= threshold_percent <= 100:
            raise ValueError("confidence thresholds must be percentages from 0 to 100")
        low_confidence = (
            (thresholded_mask == class_id)
            & (prob_map[class_id] < (threshold_percent / 100.0))
        )
        thresholded_mask[low_confidence] = 0
    return thresholded_mask


def load_image(image_path: Union[str, Path, any]) -> np.ndarray:
    """
    Load an image and convert to RGB numpy array.
    
    Args:
        image_path: Path to the image file or file-like object (e.g., Streamlit upload)
        
    Returns:
        Image as HxWx3 numpy array (uint8)
    """
    img = Image.open(image_path).convert('RGB')
    return np.array(img)


def preprocess_image(image: np.ndarray, mean: np.ndarray = DEFAULT_MEAN, 
                     std: np.ndarray = DEFAULT_STD, device: str = 'cuda') -> torch.Tensor:
    """
    Preprocess image for model inference.
    
    Args:
        image: HxWx3 numpy array (uint8)
        mean: Normalization mean
        std: Normalization std
        device: Device to place tensor on
        
    Returns:
        Preprocessed tensor [1, 3, H, W]
    """
    # Convert to float and normalize
    img = image.astype(np.float32) / 255.0
    img = (img - mean) / std
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device).float()
    return img_tensor


def tessellate_image(
    image: np.ndarray, max_x: int, max_y: int
) -> Tuple[list, list]:
    """Split an image into non-overlapping crops and native-coordinate boxes."""
    if image.ndim < 2:
        raise ValueError("image must have at least two dimensions (H, W)")
    if not isinstance(max_x, (int, np.integer)) or max_x <= 0:
        raise ValueError("max_x must be a positive integer")
    if not isinstance(max_y, (int, np.integer)) or max_y <= 0:
        raise ValueError("max_y must be a positive integer")

    height, width = image.shape[:2]
    crops = []
    boxes = []
    for ymin in range(0, height, int(max_y)):
        ymax = min(ymin + int(max_y), height)
        for xmin in range(0, width, int(max_x)):
            xmax = min(xmin + int(max_x), width)
            crops.append(image[ymin:ymax, xmin:xmax])
            boxes.append((ymin, ymax, xmin, xmax))
    return crops, boxes


def load_model(model_path: Union[str, Path], num_classes: Optional[int] = None,
               encoder_name: str = 'resnet34', device: str = 'cuda') -> torch.nn.Module:
    """
    Load a trained UNet model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint (.pt file)
        num_classes: Number of output classes (if None, will auto-detect from checkpoint)
        encoder_name: Encoder name (default: resnet34)
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    device = device if torch.cuda.is_available() and 'cuda' in device else 'cpu'
    
    # Normalize path to string for architecture heuristics
    model_path = Path(model_path)
    model_path_str = str(model_path).lower()

    # Load checkpoint first to inspect it
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Determine state_dict and num_classes
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Check if num_classes is stored in checkpoint metadata
            if num_classes is None and 'num_classes' in checkpoint:
                num_classes = checkpoint['num_classes']
        else:
            # Assume the dict itself is the state_dict
            state_dict = checkpoint
    else:
        # Assume it's directly a state_dict
        state_dict = checkpoint
    
    # Auto-detect num_classes from state_dict if not provided
    if num_classes is None:
        # Check the segmentation head output layer shape
        # The segmentation head weight should be [num_classes, channels, kernel, kernel]
        seg_head_key = None
        for key in state_dict.keys():
            if 'segmentation_head.0.weight' in key or 'segmentation_head.weight' in key:
                seg_head_key = key
                break
        
        if seg_head_key:
            num_classes = state_dict[seg_head_key].shape[0]
            print(f"Auto-detected num_classes={num_classes} from checkpoint")
        else:
            # Fallback: try to infer from other layers or use default
            print("Warning: Could not auto-detect num_classes, using default of 4")
            num_classes = 4
    
    # Choose architecture based on model path / checkpoint metadata
    arch: str = "unet"

    # Heuristic based on folder / file name
    if "deeplab" in model_path_str or "deep_lab" in model_path_str:
        arch = "deeplabv3"
    elif "unetplusplus" in model_path_str or "unet_plus_plus" in model_path_str or "unet++" in model_path_str:
        arch = "unet++"
    elif "unet" in model_path_str:
        arch = "unet"

    # Allow explicit architecture hint from checkpoint if present
    if isinstance(checkpoint, dict) and "arch" in checkpoint:
        arch = str(checkpoint["arch"]).lower()

    # Prefer the encoder recorded in the checkpoint (e.g. Optuna-tuned resnet50)
    if isinstance(checkpoint, dict) and checkpoint.get("encoder"):
        encoder_name = str(checkpoint["encoder"])

    # Create model architecture with detected/provided num_classes
    if arch in ("deeplabv3", "deeplab", "deeplab_v3"):
        model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
        )
    elif arch in ("unet++", "unetplusplus", "unet_plus_plus"):
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
        )
    else:
        # Default: plain UNet
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,  # We're loading trained weights
            in_channels=3,
            classes=num_classes,
        )
    
    # Load the state_dict
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    return model


def predict_full_image(model: torch.nn.Module, image: np.ndarray, 
                      device: str = 'cuda', mean: np.ndarray = DEFAULT_MEAN,
                      std: np.ndarray = DEFAULT_STD) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict on a full image (for full-image trained models).
    
    Args:
        model: Trained model
        image: HxWx3 numpy array (uint8)
        device: Device to run inference on
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Tuple of (pred_mask, prob_map)
        - pred_mask: HxW uint8 class predictions
        - prob_map: CxHxW float32 probability map
    """
    device = device if torch.cuda.is_available() and 'cuda' in device else 'cpu'
    
    # Preprocess image
    img = image
    orig_h, orig_w = img.shape[:2]

    # Many encoder/decoder stacks require H and W to be divisible by 32 (5 pooling levels).
    # Automatically pad up to the nearest multiple of 32 to avoid runtime errors,
    # then crop predictions back down to the original size.
    align = 32
    pad_h = (align - orig_h % align) % align
    pad_w = (align - orig_w % align) % align

    if pad_h or pad_w:
        img = cv2.copyMakeBorder(
            img,
            top=0,
            bottom=pad_h,
            left=0,
            right=pad_w,
            borderType=cv2.BORDER_REFLECT_101,
        )

    img_tensor = preprocess_image(img, mean=mean, std=std, device=device)
    
    # Run inference
    with torch.no_grad():
        output = model(img_tensor)  # [1, C, H_pad, W_pad]
        probs = torch.nn.functional.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).squeeze(0)
    
    # Convert to numpy
    pred_mask = pred.cpu().numpy().astype(np.uint8)
    prob_map = probs.squeeze(0).cpu().numpy().astype(np.float32)

    # Crop back to original size if we padded
    if pad_h or pad_w:
        pred_mask = pred_mask[:orig_h, :orig_w]
        prob_map = prob_map[:, :orig_h, :orig_w]
    
    return pred_mask, prob_map


def predict_patch_based(model: torch.nn.Module, image: np.ndarray,
                        device: str = 'cuda', tile_size: int = 512,
                        stride: Optional[int] = None, batch_size: int = 4,
                        mean: np.ndarray = DEFAULT_MEAN, std: np.ndarray = DEFAULT_STD) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict on an image using sliding window (for patch-based trained models).
    
    Args:
        model: Trained model
        image: HxWx3 numpy array (uint8)
        device: Device to run inference on
        tile_size: Size of patches
        stride: Stride for sliding window (default: tile_size)
        batch_size: Batch size for processing patches
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Tuple of (pred_mask, prob_map)
        - pred_mask: HxW uint8 class predictions
        - prob_map: CxHxW float32 probability map
    """
    # Try to import sliding_window_inference from functions
    try:
        from functions.sliding_inference import sliding_window_inference
    except ImportError:
        # If that fails, try importing the function directly
        import sys
        from pathlib import Path
        functions_path = Path(__file__).parent / 'functions'
        if functions_path.exists():
            sys.path.insert(0, str(Path(__file__).parent))
        from functions.sliding_inference import sliding_window_inference
    
    pred_mask, prob_map = sliding_window_inference(
        model=model,
        image=image,
        device=device,
        tile_size=tile_size,
        stride=stride,
        batch_size=batch_size,
        mean=mean,
        std=std
    )
    
    return pred_mask, prob_map


def predict_image_tiled(
    model: torch.nn.Module,
    image: np.ndarray,
    max_crop_width: int = 3000,
    max_crop_height: int = 1000,
    device: str = 'cuda',
    use_patch_based: bool = False,
    tile_size: int = 512,
    stride: Optional[int] = None,
    batch_size: int = 4,
    mean: np.ndarray = DEFAULT_MEAN,
    std: np.ndarray = DEFAULT_STD,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run independent inference on non-overlapping crops and stitch results."""
    crops, boxes = tessellate_image(image, max_crop_width, max_crop_height)
    height, width = image.shape[:2]
    stitched_prob_map = None

    for crop, (ymin, ymax, xmin, xmax) in zip(crops, boxes):
        if use_patch_based:
            _, tile_prob_map = predict_patch_based(
                model, crop, device=device, tile_size=tile_size,
                stride=stride, batch_size=batch_size, mean=mean, std=std,
            )
        else:
            _, tile_prob_map = predict_full_image(
                model, crop, device=device, mean=mean, std=std,
            )

        tile_height, tile_width = ymax - ymin, xmax - xmin
        if tile_prob_map.shape[1:] != (tile_height, tile_width):
            raise ValueError(
                "Tile inference returned a probability map with shape "
                f"{tile_prob_map.shape[1:]}, expected {(tile_height, tile_width)}"
            )
        if stitched_prob_map is None:
            stitched_prob_map = np.zeros(
                (tile_prob_map.shape[0], height, width), dtype=np.float32
            )
        elif stitched_prob_map.shape[0] != tile_prob_map.shape[0]:
            raise ValueError("All tiles must produce the same number of classes")
        stitched_prob_map[:, ymin:ymax, xmin:xmax] = tile_prob_map

    pred_mask = np.argmax(stitched_prob_map, axis=0).astype(np.uint8)
    return pred_mask, stitched_prob_map


def predict_image(image_path: Union[str, Path], model_path: Union[str, Path],
                 output_path: Optional[Union[str, Path]] = None,
                 use_patch_based: bool = False, tile_size: int = 512,
                 stride: Optional[int] = None, batch_size: int = 4,
                 use_tiled: bool = False, max_crop_width: int = 3000,
                 max_crop_height: int = 1000,
                 num_classes: Optional[int] = None, encoder_name: str = 'resnet34',
                 device: str = 'cuda', return_probabilities: bool = False,
                 overlay_alpha: float = 0.5, save_output: bool = True,
                 return_timing: bool = False,
                 confidence_thresholds: Optional[Mapping[int, float]] = None
                 ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Main prediction function: load image and model, generate prediction mask.
    
    Args:
        image_path: Path to input image
        model_path: Path to model checkpoint
        output_path: Optional path to save prediction mask
        use_patch_based: If True, use sliding window inference (for patch models)
        use_tiled: If True, crop the original image before inference
        max_crop_width: Maximum crop width for tiled inference
        max_crop_height: Maximum crop height for tiled inference
        tile_size: Patch size for sliding window (only if use_patch_based=True)
        stride: Stride for sliding window (only if use_patch_based=True)
        batch_size: Batch size for patch processing (only if use_patch_based=True)
        num_classes: Number of classes (if None, will auto-detect from checkpoint)
        encoder_name: Encoder name (default: resnet34)
        device: Device to run inference on
        return_probabilities: If True, also return probability map
        confidence_thresholds: Optional per-class minimum confidence percentages
        overlay_alpha: Transparency for overlay visualization (0.0 = transparent, 1.0 = opaque)
        save_output: If True, save prediction mask and overlay to disk
        return_timing: If True, also return inference time in milliseconds
        
    Returns:
        If return_probabilities=False: pred_mask (HxW uint8)
        If return_probabilities=True: (pred_mask, prob_map) tuple
    """
    # Load image
    print(f"Loading image from {image_path}")
    image = load_image(image_path)
    print(f"Image shape: {image.shape}")
    
    # Load model
    print(f"Loading model from {model_path}")
    model = load_model(model_path, num_classes=num_classes, 
                      encoder_name=encoder_name, device=device)
    print("Model loaded successfully")
    
    # Run prediction (measure only the model inference part, not image/model loading)
    start_time = time.perf_counter()
    if use_tiled:
        print("Using tiled inference")
        pred_mask, prob_map = predict_image_tiled(
            model, image, max_crop_width=max_crop_width,
            max_crop_height=max_crop_height, device=device,
            use_patch_based=use_patch_based, tile_size=tile_size,
            stride=stride, batch_size=batch_size,
        )
    elif use_patch_based:
        print("Using patch-based inference (sliding window)")
        pred_mask, prob_map = predict_patch_based(
            model, image, device=device, tile_size=tile_size,
            stride=stride, batch_size=batch_size
        )
    else:
        print("Using full-image inference")
        pred_mask, prob_map = predict_full_image(model, image, device=device)
    end_time = time.perf_counter()
    inference_time_ms = (end_time - start_time) * 1000.0

    pred_mask = apply_confidence_threshold(
        pred_mask, prob_map, confidence_thresholds
    )
    
    print(f"Prediction complete. Mask shape: {pred_mask.shape}")
    print(f"Class distribution: {np.bincount(pred_mask.flatten())}")
    
    # Save outputs if requested
    if save_output:
        # Determine output path
        # Get project root (assuming script is in notebooks/ directory)
        script_dir = Path(__file__).parent
        project_root = script_dir.parent if script_dir.name == "notebooks" else Path.cwd()
        
        if output_path is None:
            # Generate timestamped filename in data/predictions/
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = project_root / "data" / "predictions" / f"prediction_{timestamp}.png"
        else:
            output_path = Path(output_path)
            # If only filename provided (no directory), save to data/predictions/
            if not output_path.is_absolute() and (output_path.parent == Path(".") or str(output_path.parent) == "."):
                output_path = project_root / "data" / "predictions" / output_path.name
            elif not output_path.is_absolute():
                # Relative path - make it relative to project root
                output_path = project_root / output_path
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as PNG (uint8 mask)
        Image.fromarray(pred_mask, mode='L').save(output_path)
        print(f"Prediction mask saved to {output_path}")
        
        # Automatically create and save visualization overlay
        vis_overlay = visualize_prediction(image, pred_mask, alpha=overlay_alpha)
        vis_path = output_path.parent / f"{output_path.stem}_overlay{output_path.suffix}"
        Image.fromarray(vis_overlay).save(vis_path)
        print(f"Colored overlay saved to {vis_path}")
    
    if return_probabilities:
        if return_timing:
            return pred_mask, prob_map, inference_time_ms
        return pred_mask, prob_map
    else:
        if return_timing:
            return pred_mask, inference_time_ms
        return pred_mask


def visualize_prediction(image: np.ndarray, pred_mask: np.ndarray,
                        class_names: Optional[list] = None, alpha: float = 0.5) -> np.ndarray:
    """
    Create a visualization overlay of prediction on image.
    Background (class 0) is transparent, other classes are colored.
    
    Args:
        image: Original image (HxWx3 uint8)
        pred_mask: Prediction mask (HxW uint8)
        class_names: List of class names (if None, uses default)
        alpha: Transparency for overlay (0.0 = transparent, 1.0 = opaque)
        
    Returns:
        Visualization image (HxWx3 uint8) with colored overlay on original image
    """
    # Determine number of classes from mask
    num_classes = int(pred_mask.max() + 1)
    
    # Color map for classes (RGB). Background (0) remains transparent.
    colors = np.array([
        [0, 0, 0],        # Background (0) - transparent
        [0, 0, 255],      # Class 1 - blue
        [255, 0, 0],      # Class 2 - red
        [255, 255, 0],    # Class 3 - yellow
        [0, 180, 0],      # Class 4 - green
        [255, 0, 255],    # Class 5 - magenta
        [0, 255, 255],    # Class 6 - cyan
    ], dtype=np.uint8)
    
    # Extend colors if we have more classes than defined
    if num_classes > len(colors):
        # Generate additional colors
        import colorsys
        additional_colors = []
        for i in range(num_classes - len(colors)):
            hue = (i + 0.5) / (num_classes - len(colors))
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            additional_colors.append([int(c * 255) for c in rgb])
        colors = np.vstack([colors, np.array(additional_colors, dtype=np.uint8)])
    else:
        colors = colors[:num_classes]
    
    # Start with original image
    overlay = image.copy().astype(np.float32)
    
    # Create colored mask - ensure pred_mask values are within valid range
    pred_mask_clipped = np.clip(pred_mask, 0, len(colors) - 1)
    
    # Only overlay non-background classes (class 0 is background/transparent)
    for class_id in range(1, num_classes):
        mask = (pred_mask_clipped == class_id)
        if mask.any():
            # Blend colored mask with original image only where this class is present
            overlay[mask] = overlay[mask] * (1 - alpha) + colors[class_id] * alpha
    
    return overlay.astype(np.uint8)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict defects on board images')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None, 
                       help='Path to save prediction mask (default: auto-generate timestamped filename in data/predictions/)')
    parser.add_argument('--patch-based', action='store_true', 
                       help='Use patch-based inference (sliding window)')
    parser.add_argument('--tiled', action='store_true',
                       help='Crop the original image before inference')
    parser.add_argument('--max-crop-width', type=int, default=3000,
                       help='Maximum crop width for tiled inference')
    parser.add_argument('--max-crop-height', type=int, default=1000,
                       help='Maximum crop height for tiled inference')
    parser.add_argument('--tile-size', type=int, default=512,
                       help='Tile size for patch-based inference')
    parser.add_argument('--stride', type=int, default=None,
                       help='Stride for sliding window (default: tile_size)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for patch processing')
    parser.add_argument('--num-classes', type=int, default=None,
                       help='Number of classes (if not provided, will auto-detect from checkpoint)')
    parser.add_argument('--encoder', type=str, default='resnet34',
                       help='Encoder name')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--visualize', type=str, default=None,
                       help='Custom path to save visualization overlay (default: auto-saved as _overlay.png)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Transparency for overlay (0.0 = transparent, 1.0 = opaque, default: 0.5)')
    
    args = parser.parse_args()
    
    # Run prediction (automatically creates overlay)
    result = predict_image(
        image_path=args.image,
        model_path=args.model,
        output_path=args.output,
        use_patch_based=args.patch_based,
        use_tiled=args.tiled,
        max_crop_width=args.max_crop_width,
        max_crop_height=args.max_crop_height,
        tile_size=args.tile_size,
        stride=args.stride,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        encoder_name=args.encoder,
        device=args.device,
        return_probabilities=False,
        overlay_alpha=args.alpha
    )
    
    # Create custom visualization if requested with different alpha
    if args.visualize:
        image = load_image(args.image)
        vis = visualize_prediction(image, result, alpha=args.alpha)
        Image.fromarray(vis).save(args.visualize)
        print(f"Custom visualization saved to {args.visualize}")
