"""
Streamlit app for oak defect detection.

This app allows users to:
- Select a model from the models directory
- Select an image from organized-data/images
- Generate and display prediction masks
"""

import streamlit as st
import sys
from pathlib import Path
from typing import Mapping, Optional
import numpy as np
from PIL import Image

# Add notebooks directory to path
sys.path.insert(0, str(Path(__file__).parent))

import predict_image as pred_module
predict_image = pred_module.predict_image
load_image = pred_module.load_image
visualize_prediction = pred_module.visualize_prediction

try:
    import onnxruntime as ort
except ImportError:
    ort = None


@st.cache_data(show_spinner=False)
def get_model_class_count(model_path: str, backend: str) -> int:
    """Return the selected model's number of segmentation output classes."""
    if backend == "ONNX (.onnx)":
        if ort is None:
            raise RuntimeError("ONNX Runtime is not available")
        session = ort.InferenceSession(model_path)
        output_shape = session.get_outputs()[0].shape
        if len(output_shape) < 2 or not isinstance(output_shape[1], (int, np.integer)):
            raise ValueError("Could not determine the ONNX model's output class count")
        return int(output_shape[1])

    import torch

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("num_classes"), int):
        return int(checkpoint["num_classes"])
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    for key, value in state_dict.items():
        if "segmentation_head.0.weight" in key or "segmentation_head.weight" in key:
            return int(value.shape[0])
    raise ValueError("Could not determine the PyTorch model's output class count")

# Page configuration
st.set_page_config(
    page_title="Oak Defect Detection",
    layout="wide"
)

# Title
st.title("Oak Defect Detection")
st.markdown("Select a model and image to generate defect predictions")

# Get project root (assuming script is in notebooks/ directory)
script_dir = Path(__file__).parent
project_root = script_dir.parent if script_dir.name == "notebooks" else Path.cwd()

# Paths
models_dir = project_root / "models"
default_images_dir = project_root / "data" / "organized-data" / "images"
# Default directory for demo boards to predict
default_to_predict_dir = project_root / "data" / "to_predict"
images_dir = default_images_dir

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # Device selection (for PyTorch backend)
    import torch
    device_available = torch.cuda.is_available()
    device = st.selectbox(
        "Device",
        ["cuda", "cpu"],
        index=0 if device_available else 1,
        disabled=not device_available,
        help="Select GPU (cuda) or CPU for PyTorch inference"
    )
    if not device_available and device == "cuda":
        device = "cpu"
        st.warning("CUDA not available, using CPU")
    
    # Backend selection
    backend_options = ["PyTorch (.pt/.pth)"]
    if ort is not None:
        backend_options.append("ONNX (.onnx)")
    backend = st.selectbox(
        "Model Backend",
        backend_options,
        help="Choose whether to run models with PyTorch or ONNX Runtime",
    )
    if backend == "ONNX (.onnx)" and ort is None:
        st.warning("ONNX Runtime is not installed. Falling back to PyTorch backend.")
        backend = "PyTorch (.pt/.pth)"
    
    # Image directory selection
    st.subheader("Image Directory")
    image_dir_mode = st.radio(
        "Image source",
        options=["Organized dataset", "Demo boards (to_predict)", "Custom path"],
        index=1,
        help="Choose where to load images from.",
    )
    if image_dir_mode == "Organized dataset":
        images_dir = default_images_dir
    elif image_dir_mode == "Demo boards (to_predict)":
        images_dir = default_to_predict_dir
    else:
        custom_dir_str = st.text_input(
            "Custom image directory",
            value=str(default_to_predict_dir),
            help="Enter an absolute or project-relative path containing images.",
        )
        # Interpret relative paths as relative to project root
        custom_path = Path(custom_dir_str)
        if not custom_path.is_absolute():
            custom_path = project_root / custom_path
        images_dir = custom_path

    st.caption(f"Current image directory: `{images_dir}`")

    # Overlay transparency
    alpha = st.slider(
        "Overlay Transparency",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Transparency of the colored overlay (0.0 = transparent, 1.0 = opaque)"
    )
    
    # Patch-based settings (only relevant for PyTorch backend)
    use_patch_based = st.checkbox(
        "Use Patch-Based Inference",
        value=False,
        help="Enable for patch-based models (sliding window inference, PyTorch only)"
    )
    
    if use_patch_based:
        tile_size = st.number_input("Tile Size", min_value=256, max_value=1024, value=512, step=64)
        stride = st.number_input("Stride", min_value=64, max_value=512, value=512, step=64)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=16, value=4, step=1)
    else:
        tile_size = 512
        stride = None
        batch_size = 4

    st.subheader("Tiled / Crop-Based Inference")
    use_tiled = st.checkbox(
        "Enable Tiled Inference",
        value=False,
        help="Crop the original board before inference and stitch tile predictions back together.",
    )
    max_crop_width = st.number_input(
        "Max Crop Width (X)", min_value=1, max_value=20000, value=3000, step=100
    )
    max_crop_height = st.number_input(
        "Max Crop Height (Y)", min_value=1, max_value=20000, value=1000, step=100
    )

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.header("Model Selection")
    
    # Find all model files based on selected backend
    model_files = []
    if models_dir.exists():
        if backend == "PyTorch (.pt/.pth)":
            # Look for segmentation checkpoints in model subfolders; skip ONNX exports and
            # YOLO / NHLA runs, which this app can't load
            non_segmentation = ("yolo", "nhla")
            for model_folder in models_dir.iterdir():
                if (
                    model_folder.is_dir()
                    and model_folder.name != "onnx"
                    and not any(tag in model_folder.name.lower() for tag in non_segmentation)
                ):
                    for model_file in model_folder.glob("*.pt"):
                        model_files.append(model_file)
                    # Also check for .pth files
                    for model_file in model_folder.glob("*.pth"):
                        model_files.append(model_file)
        else:
            # ONNX backend: look for .onnx files in models/onnx
            onnx_dir = models_dir / "onnx"
            if onnx_dir.exists():
                for model_file in onnx_dir.glob("*.onnx"):
                    model_files.append(model_file)
    
    if not model_files:
        if backend == "PyTorch (.pt/.pth)":
            st.error(f"No PyTorch model files (.pt, .pth) found in {models_dir}")
        else:
            st.error(f"No ONNX model files (.onnx) found in {models_dir / 'onnx'}")
        st.stop()
    
    # Create model selection dropdown
    model_options = {}
    for model_file in sorted(model_files):
        # Create a readable name: folder_name/model_file
        rel_path = model_file.relative_to(project_root)
        display_name = f"{model_file.parent.name} / {model_file.name}"
        model_options[display_name] = str(rel_path)
    
    selected_model_display = st.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        help="Choose a trained model from the models directory"
    )
    
    selected_model_path = model_options[selected_model_display]
    full_model_path = project_root / selected_model_path
    
    st.info(f"**Selected Model:** `{selected_model_path}`")
    
    # Show model info
    if full_model_path.exists():
        st.success(f"✓ Model file found ({full_model_path.stat().st_size / (1024*1024):.2f} MB)")
    else:
        st.error(f"✗ Model file not found: {full_model_path}")

try:
    model_class_count = get_model_class_count(str(full_model_path), backend)
except Exception as e:
    st.error(f"Could not determine the selected model's class count: {e}")
    st.stop()

with st.sidebar:
    st.subheader("Class Names and Minimum Confidence")
    class_names = []
    confidence_thresholds = {}
    for class_id in range(model_class_count):
        default_name = (
            pred_module.CLASS_NAMES[class_id]
            if class_id < len(pred_module.CLASS_NAMES)
            else f"Class {class_id}"
        )
        class_name = st.text_input(
            f"Class {class_id} name",
            value=default_name,
            key=f"class_name_{class_id}",
        ).strip() or f"Class {class_id}"
        class_names.append(class_name)
        if class_id > 0:
            confidence_thresholds[class_id] = float(st.number_input(
                f"{class_name} minimum score (%)",
                min_value=0,
                max_value=100,
                value=0,
                step=1,
                key=f"class_threshold_{class_id}",
            ))

with col2:
    st.header("Image Selection")
    
    # Find all images
    image_files = []
    if images_dir.exists():
        # Support common image formats
        for ext in ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]:
            image_files.extend(list(images_dir.glob(ext)))
            image_files.extend(list(images_dir.glob(ext.upper())))
    
    if not image_files:
        st.error(f"No image files found in {images_dir}")
        st.stop()
    
    # Create image selection dropdown
    image_options = {}
    for image_file in sorted(image_files):
        display_name = image_file.name
        image_options[display_name] = image_file
    
    selected_image_name = st.selectbox(
        "Select Image",
        options=list(image_options.keys()),
        help="Choose an image from the organized-data/images directory"
    )
    
    selected_image_path = image_options[selected_image_name]
    
    st.info(f"**Selected Image:** `{selected_image_name}`")

    if use_tiled and selected_image_path.exists():
        try:
            image_width, image_height = Image.open(selected_image_path).size
            tile_columns = int(np.ceil(image_width / max_crop_width))
            tile_rows = int(np.ceil(image_height / max_crop_height))
            tile_count = tile_columns * tile_rows
            st.metric(
                "Tiled inference",
                f"{tile_columns}x{tile_rows} = {tile_count} tiles",
                help="The original image is cropped before preprocessing.",
            )
        except Exception as e:
            st.warning(f"Could not calculate tile breakdown: {e}")
    
    # Display selected image
    if selected_image_path.exists():
        try:
            preview_image = Image.open(selected_image_path)
            st.image(preview_image, caption="Selected Image", use_container_width=True)
        except Exception as e:
            st.warning(f"Could not preview image: {e}")

@st.cache_resource
def get_onnx_session(model_path: str):
    if ort is None:
        raise RuntimeError("ONNX Runtime is not available")
    return ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


def predict_with_onnx_array(
    image: np.ndarray,
    onnx_model_path: str,
    return_probabilities: bool = False,
):
    """
    Run inference with an ONNX model.

    Supports two ONNX model formats (chosen from the first input's declared type,
    not only the tensor name):
    - **Graph-internal preprocessing** (ONNX input type ``tensor(uint8)``):
      Feed raw RGB pixels as ``uint8`` NCHW ``[1, 3, H, W]``. The graph may
      handle normalization / softmax internally.
    - **External preprocessing** (input type ``tensor(float)`` / float16, etc.):
      Use ``preprocess_image`` (ImageNet-style float32 NCHW) and apply softmax
      + argmax in NumPy after inference.
    """
    session = get_onnx_session(onnx_model_path)
    inp_meta = session.get_inputs()[0]
    input_name = inp_meta.name
    onnx_type = (getattr(inp_meta, "type", None) or "").lower()
    expects_uint8 = "uint8" in onnx_type
    inference_image = image

    input_shape = getattr(inp_meta, "shape", None)
    if input_shape and len(input_shape) >= 4:
        expected_height, expected_width = input_shape[-2:]
        if (
            isinstance(expected_height, (int, np.integer))
            and isinstance(expected_width, (int, np.integer))
            and expected_height > 0
            and expected_width > 0
            and image.shape[:2] != (expected_height, expected_width)
        ):
            inference_image = np.asarray(
                Image.fromarray(image).resize(
                    (int(expected_width), int(expected_height)),
                    resample=Image.BILINEAR,
                )
            )

    if expects_uint8:
        # ── uint8 NCHW (preprocessing may be inside the ONNX graph) ─────────
        img_chw = np.transpose(inference_image, (2, 0, 1))[np.newaxis, ...]  # [1, 3, H, W]
        img_input = img_chw.astype(np.uint8, copy=False)

        outputs = session.run(None, {input_name: img_input})[0]  # [1, C, H, W]
        prob_map = np.asarray(outputs[0], dtype=np.float32)
        if np.nanmax(prob_map) > 1.0:
            prob_map = prob_map / 100.0
        prob_map = np.clip(prob_map, 0.0, 1.0)
    else:
        # ── Float preprocessed input (e.g. tensor(float)) ─────────
        img_tensor = pred_module.preprocess_image(inference_image, device="cpu")
        inputs = img_tensor.numpy()  # [1, 3, H, W]

        outputs = session.run(None, {input_name: inputs})[0]  # [1, C, H, W]

        logits = outputs
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        prob_map = probs[0].astype(np.float32)

    # ONNX graphs may emit H×W that differs slightly from the file (padding / fixed stride).
    h, w = image.shape[:2]
    if prob_map.shape[1:] != (h, w):
        prob_map = np.stack([
            np.asarray(
                Image.fromarray(class_map, mode="F").resize(
                    (w, h), resample=Image.BILINEAR
                ),
                dtype=np.float32,
            )
            for class_map in prob_map
        ])

    pred = prob_map.argmax(axis=0).astype(np.uint8)
    if return_probabilities:
        return pred, prob_map
    return pred


def predict_with_onnx(image_path: str, onnx_model_path: str):
    """Run ONNX inference for an image path."""
    return predict_with_onnx_array(load_image(image_path), onnx_model_path)


def predict_with_onnx_tiled(
    image: np.ndarray,
    onnx_model_path: str,
    max_crop_width: int = 3000,
    max_crop_height: int = 1000,
    confidence_thresholds: Optional[Mapping[int, float]] = None,
) -> np.ndarray:
    """Run ONNX inference on crops and reconstruct the native-size mask."""
    crops, boxes = pred_module.tessellate_image(
        image, max_crop_width, max_crop_height
    )
    height, width = image.shape[:2]
    stitched_prob_map = None

    for crop, (ymin, ymax, xmin, xmax) in zip(crops, boxes):
        _, tile_prob_map = predict_with_onnx_array(
            crop, onnx_model_path, return_probabilities=True
        )
        tile_height, tile_width = ymax - ymin, xmax - xmin
        if tile_prob_map.shape[1:] != (tile_height, tile_width):
            raise ValueError("ONNX tile probabilities do not match crop dimensions")
        if stitched_prob_map is None:
            stitched_prob_map = np.zeros(
                (tile_prob_map.shape[0], height, width), dtype=np.float32
            )
        elif stitched_prob_map.shape[0] != tile_prob_map.shape[0]:
            raise ValueError("All ONNX tiles must produce the same number of classes")
        stitched_prob_map[:, ymin:ymax, xmin:xmax] = tile_prob_map

    stitched_mask = stitched_prob_map.argmax(axis=0).astype(np.uint8)
    return pred_module.apply_confidence_threshold(
        stitched_mask, stitched_prob_map, confidence_thresholds
    )


# Prediction button
st.markdown("---")
predict_button = st.button("🔮 Generate Prediction", type="primary", use_container_width=True)

# Run prediction
if predict_button:
    if not full_model_path.exists():
        st.error(f"Model file not found: {full_model_path}")
        st.stop()
    
    if not selected_image_path.exists():
        st.error(f"Image file not found: {selected_image_path}")
        st.stop()
    
    # Determine if patch-based based on model name
    is_patch_model = "Patch" in selected_model_display or "patch" in selected_model_path.lower()
    use_patch = use_patch_based or is_patch_model
    
    with st.spinner("Running prediction... This may take a while."):
        try:
            # Suppress print statements during prediction
            import io
            import contextlib
            import time
            
            f = io.StringIO()
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                if backend == "PyTorch (.pt/.pth)":
                    # Run prediction (don't save to disk, just return the mask and timing)
                    pred_mask, inference_time_ms = predict_image(
                        image_path=str(selected_image_path),
                        model_path=str(full_model_path),
                        output_path=None,  # Don't save
                        use_patch_based=use_patch,
                        tile_size=tile_size if use_patch else 512,
                        stride=stride if use_patch else None,
                        batch_size=batch_size if use_patch else 4,
                        use_tiled=use_tiled,
                        max_crop_width=int(max_crop_width),
                        max_crop_height=int(max_crop_height),
                        confidence_thresholds=confidence_thresholds,
                        num_classes=None,  # Auto-detect
                        encoder_name='resnet34',
                        device=device,
                        return_probabilities=False,
                        overlay_alpha=alpha,
                        save_output=False,  # Don't save files, just return results
                        return_timing=True
                    )
                else:
                    # ONNX backend: use selected .onnx file directly
                    onnx_model_path = full_model_path
                    if not onnx_model_path.exists():
                        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
                    
                    start_time = time.perf_counter()
                    original_image_for_prediction = load_image(str(selected_image_path))
                    if use_tiled:
                        pred_mask = predict_with_onnx_tiled(
                            image=original_image_for_prediction,
                            onnx_model_path=str(onnx_model_path),
                            max_crop_width=int(max_crop_width),
                            max_crop_height=int(max_crop_height),
                            confidence_thresholds=confidence_thresholds,
                        )
                    else:
                        pred_mask, prob_map = predict_with_onnx_array(
                            image=original_image_for_prediction,
                            onnx_model_path=str(onnx_model_path),
                            return_probabilities=True,
                        )
                        pred_mask = pred_module.apply_confidence_threshold(
                            pred_mask, prob_map, confidence_thresholds
                        )
                    end_time = time.perf_counter()
                    inference_time_ms = (end_time - start_time) * 1000.0
        
            # Load original image
            original_image = load_image(str(selected_image_path))
            
            # Create overlay
            overlay_image = visualize_prediction(original_image, pred_mask, class_names=class_names, alpha=alpha)
            
            # Display results
            st.success("Prediction complete!")
            st.write(f"**Model processing time:** {inference_time_ms:.1f} ms")
            
            # Create two columns for displaying results
            col_pred, col_overlay = st.columns(2)
            
            with col_pred:
                st.subheader("Raw Prediction Mask")
                # Convert mask to displayable format (grayscale)
                mask_display = Image.fromarray(pred_mask, mode='L')
                st.image(mask_display, caption="Prediction Mask (Class IDs)", use_container_width=True)
                
                # Show class distribution
                unique, counts = np.unique(pred_mask, return_counts=True)
                class_dist = dict(zip(unique, counts))
                st.write("**Class Distribution:**")
                for class_id, count in sorted(class_dist.items()):
                    percentage = (count / pred_mask.size) * 100
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                    st.write(f"- {class_name} (ID {class_id}): {count:,} pixels ({percentage:.2f}%)")
            
            with col_overlay:
                st.subheader("Colored Overlay")
                overlay_display = Image.fromarray(overlay_image)
                st.image(overlay_display, caption="Prediction Overlay on Original Image", use_container_width=True)
                
                # Color legend
                st.write("**Color Legend:**")
                st.write(f"- **Original image**: {class_names[0]} (ID 0, transparent)")
                overlay_colors = ["Blue", "Red", "Yellow", "Green", "Magenta", "Cyan"]
                for class_id in range(1, len(class_names)):
                    color_name = (
                        overlay_colors[class_id - 1]
                        if class_id - 1 < len(overlay_colors)
                        else f"Color {class_id}"
                    )
                    st.write(f"- **{color_name}**: {class_names[class_id]} (ID {class_id})")
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Oak Defect Detection App
    </div>
    """,
    unsafe_allow_html=True
)
