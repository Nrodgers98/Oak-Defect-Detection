# Streamlit App for Oak Defect Detection

## Running the App

To run the Streamlit app, use the following command from the project root:

```bash
streamlit run notebooks/app.py
```

Or from the notebooks directory:

```bash
cd notebooks
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Features

- **Model Selection**: Choose from available PyTorch (`.pt`/`.pth`) or ONNX (`.onnx`) models
- **Backend Selection**: Switch between PyTorch and ONNX Runtime inference engines
- **Image Selection**: Select images from `data/organized-data/images/`
- **Real-time Prediction**: Generate prediction masks with a single click
- **Dual Display**: View both the raw prediction mask and colored overlay
- **Customizable Settings**: Adjust overlay transparency and inference parameters

## Usage

1. **Choose Model Backend** (sidebar): Select either:
   - **PyTorch (.pt/.pth)**: Uses PyTorch for inference (supports patch-based models)
   - **ONNX (.onnx)**: Uses ONNX Runtime for faster inference (full-image models only)
2. **Select a Model**: The dropdown will automatically show:
   - `.pt`/`.pth` files from `models/` (excluding `models/onnx/`) when PyTorch backend is selected
   - `.onnx` files from `models/onnx/` when ONNX backend is selected
3. **Select an Image**: Pick an image from the organized data directory
4. **Adjust Settings** (optional):
   - Device (CUDA/CPU) - only applies to PyTorch backend
   - Overlay transparency
   - Patch-based inference settings (PyTorch backend only)
5. **Click "Generate Prediction"**: The app will run inference and display results
6. **View Results**: 
   - Left side: Raw prediction mask (grayscale class IDs)
   - Right side: Colored overlay on original image

## Color Scheme

- **Background (Class 0)**: Transparent (original image shows through)
- **Class 1**: Blue
- **Class 2**: Red  
- **Class 3**: Yellow

## Notes

- **PyTorch Backend**: 
  - Supports both full-image and patch-based models
  - Automatically detects patch-based models from the model name
  - Can use GPU (CUDA) or CPU
- **ONNX Backend**:
  - Currently supports full-image models only
  - Requires ONNX Runtime to be installed (`pip install onnxruntime` or `onnxruntime-gpu`)
  - Automatically uses GPU if available, falls back to CPU
- Predictions are displayed in real-time without saving to disk
- Large images may take some time to process
- To use ONNX models, first export them from your training notebooks (see main README)