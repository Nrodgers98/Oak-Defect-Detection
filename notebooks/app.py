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
images_dir = project_root / "data" / "organized-data" / "images"

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

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.header("Model Selection")
    
    # Find all model files based on selected backend
    model_files = []
    if models_dir.exists():
        if backend == "PyTorch (.pt/.pth)":
            # Look for PyTorch checkpoints in all model subfolders except ONNX folder
            for model_folder in models_dir.iterdir():
                if model_folder.is_dir() and model_folder.name != "onnx":
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


def predict_with_onnx(image_path: str, onnx_model_path: str):
    # Reuse existing helpers; keep tensor on CPU for ONNX Runtime
    image = load_image(image_path)
    img_tensor = pred_module.preprocess_image(image, device="cpu")
    inputs = img_tensor.numpy()  # [1, 3, H, W]
    
    session = get_onnx_session(onnx_model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: inputs})[0]  # [1, C, H, W]
    
    # Softmax + argmax in NumPy
    logits = outputs
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)[0].astype(np.uint8)
    return pred


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
                    pred_mask = predict_with_onnx(
                        image_path=str(selected_image_path),
                        onnx_model_path=str(onnx_model_path),
                    )
                    end_time = time.perf_counter()
                    inference_time_ms = (end_time - start_time) * 1000.0
        
            # Load original image
            original_image = load_image(str(selected_image_path))
            
            # Create overlay
            overlay_image = visualize_prediction(original_image, pred_mask, alpha=alpha)
            
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
                    st.write(f"- Class {class_id}: {count:,} pixels ({percentage:.2f}%)")
            
            with col_overlay:
                st.subheader("Colored Overlay")
                overlay_display = Image.fromarray(overlay_image)
                st.image(overlay_display, caption="Prediction Overlay on Original Image", use_container_width=True)
                
                # Color legend
                st.write("**Color Legend:**")
                st.write("- **Blue**: Class 1")
                st.write("- **Red**: Class 2")
                st.write("- **Yellow**: Class 3")
                st.write("- **Background**: Original image (transparent)")
                
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
