# Oak Defect Detection

A GPU‑accelerated deep learning project for semantic segmentation of defects in green rough oak planks. The workflow cleans and standardizes line‑scan data, builds multi‑class masks, trains a ResNet34‑UNet model, and evaluates with IoU, F1, accuracy, ROC curves, and confusion matrices.

## Deep learning GPU environment

This repository is designed to run inside a VS Code dev container with NVIDIA GPU support.

**Included stack (reference)**  
- CUDA 12.5, cuDNN 9.1  
- PyTorch 2.10, TensorFlow 2.16, Keras 3.3, scikit‑learn 1.4  
- Python 3.10, NumPy 1.24, Pandas 2.2, Matplotlib 3.10  
- JupyterLab, TensorBoard, Optuna  
- segmentation_models_pytorch, Albumentations, OpenCV, Streamlit  

Based on [NVIDIA's TensorFlow 24.06 container](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-24-06.html).

**Requirements**  
- NVIDIA GPU (Pascal or newer) with driver ≥545  
- Docker with GPU support  
- VS Code with the Dev Containers extension  

Linux users should also install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Project structure

```
Oak-Defect-Detection/
├── .devcontainer/               # Dev container configuration
├── data/                        # Raw, filtered, and organized datasets
├── logs/                        # TensorBoard logs
├── models/                      # Saved model files
├── notebooks/                   # Training, evaluation, experiments
│   ├── functions/               # Custom helper modules (incl. ``oak_hf_dataset.py``)
│   ├── Unet_FullImage.ipynb     # Full-image UNet training pipeline
│   ├── Unet_FullImage Optuna.ipynb  # Hyperparameter optimization with Optuna
│   ├── Unet_Patches.ipynb       # Patch-based UNet training pipeline
│   ├── environment_test.ipynb  # Environment verification and GPU testing
│   └── app.py        # Interactive web app for model inference
├── .gitignore
├── LICENSE
└── README.md
```

## Oak defect detection workflow (high level)

1. **Data audit and normalization**  
   Analyze raw line‑scan imagery and normalize defect class naming.

2. **Class filtering**  
   Move rare classes below a mask-count threshold to reduce imbalance.

3. **Reorganization and validation**  
   Align images/masks, validate samples, and quarantine invalid data.

4. **Mask composition**  
   Merge per‑class masks into a single multi‑class mask per image.

5. **Train/val/test split**  
   Persist split metadata for repeatable experiments.

6. **Model training**  
   Train a ResNet34‑UNet with Albumentations augmentations and TensorBoard logging.

7. **Evaluation**  
   Select best checkpoint by validation IoU, then compute test metrics:
   mean IoU, F1, accuracy, ROC curves, and confusion matrix.

## Custom functions

Notebook pipelines use project‑specific utilities from `notebooks/functions`, including:
- `analyze_dataset`, `reorganize_dataset`, `validate_dataset`, `move_invalid_samples`
- `create_combined_masks`, `create_train_val_split`
- `rename_defect_files`, `filter_classes_by_mask_count`
- `oak_hf_dataset.ensure_oak_raw_data` — fetch published TIFFs from Hugging Face into `data/raw-data`

These make the preprocessing steps repeatable and reduce manual error.

## Key libraries

| Library | Purpose | Notes |
| --- | --- | --- |
| PyTorch | Training, inference, dataloaders | Core deep learning framework and CUDA support |
| segmentation_models_pytorch | UNet + encoder zoo | Provides ResNet34‑UNet and pretrained encoders |
| Albumentations | Augmentations + normalization | Fast, flexible image transforms for segmentation |
| OpenCV | Image utilities | Used for file I/O and preprocessing support |
| NumPy | Array ops | Efficient tensor/array manipulation |
| Pillow | Image reading | Handles TIFF/PNG image loading |
| scikit‑learn | Metrics | ROC, F1, accuracy, confusion matrix |
| Matplotlib | Plotting | Curves and confusion matrix visualization |
| TensorBoard | Logging | Visual tracking of training metrics |
| Optuna | Hyperparameter optimization | Automated hyperparameter tuning with TPE sampler |
| Streamlit | Web interface | Interactive app for model inference and visualization |
| huggingface_hub | Dataset download | Pulls `nrodgers98/Oak-Defect-Detection` into `data/raw-data` |

## Data layout (expected)

```
data/
├── raw-data/                    # Raw line-scan output
├── filtered-data/               # Removed/rare/invalid samples
└── organized-data/
    ├── images/                  # RGB images
    ├── combined_masks/          # Multi-class masks
    └── split.json               # Train/val/test split
```

## Public dataset (Hugging Face)

The project’s raw line-scan TIFFs are published as **[nrodgers98/Oak-Defect-Detection](https://huggingface.co/datasets/nrodgers98/Oak-Defect-Detection)** on Hugging Face (image segmentation, ~1.5k samples, CC BY-NC 4.0). The repo stores files under a `raw-data/` tree using the same `*_Col.tif` / `*_Col_Bin_<Class>.tif` naming used by the preprocessing pipeline.

### Populate `data/raw-data` from the Hub

Install dependencies (includes `huggingface_hub`; see `requirements.txt`), then either:

**Option A — from a notebook** (run before `prepare_dataset`):

```python
from functions.oak_hf_dataset import ensure_oak_raw_data
from functions.data_pipeline import prepare_dataset

RAW_DIR = ensure_oak_raw_data()  # downloads once; skips if *.tif already present
DATA_ROOT, class_mapping = prepare_dataset(raw_data_folder=str(RAW_DIR))

# Optional: pick a custom folder name under data/
# RAW_DIR = ensure_oak_raw_data(raw_folder_name="my-raw-oak-data")
# DATA_ROOT, class_mapping = prepare_dataset(raw_data_folder=str(RAW_DIR))
```

**Option B — command line** from the repository root:

```bash
python notebooks/functions/oak_hf_dataset.py

# Optional custom folder under data/
# python notebooks/functions/oak_hf_dataset.py --raw-folder-name my-raw-oak-data
```

Use `python notebooks/functions/oak_hf_dataset.py --force` to replace existing `data/raw-data/*.tif` with a fresh copy from the Hub. Set `HF_TOKEN` if you need higher rate limits or access to a private fork.

**Option C — `datasets` API** (loads an `Image` column; useful for exploration, not a drop-in for the TIFF pipeline):

```python
from datasets import load_dataset
ds = load_dataset("nrodgers98/Oak-Defect-Detection", split="train")
```

For training with this repository’s notebooks, Option A is recommended so `prepare_dataset` receives the on-disk TIFF layout it expects.

## How to run

### Training notebooks

1. Open the repository in a VS Code dev container.  
2. Run one of the training notebooks from top to bottom:
   - `notebooks/Unet_FullImage.ipynb` — Full-image UNet training
   - `notebooks/Unet_FullImage Optuna.ipynb` — Hyperparameter optimization
   - `notebooks/Unet_Patches.ipynb` — Patch-based UNet training
3. Check TensorBoard logs in `logs/` and saved models in `models/`.

### Exporting models to ONNX (for lightweight inference)

After training, you can export any trained PyTorch model to ONNX format for faster inference. Each training notebook includes an ONNX export code block at the end that saves models to `models/onnx/`. Simply run the export cell after training completes to generate the `.onnx` file.

### Streamlit inference app

Run the interactive web app for model inference:

```bash
streamlit run notebooks/app.py
```

The app will open at `http://localhost:8501` and allows you to:
- Select trained models from `models/`
- Choose images from `data/organized-data/images/`
- Generate and visualize prediction masks in real-time
- Adjust overlay transparency and inference settings

#### Backend selection: PyTorch vs ONNX

The Streamlit app can run models with either **PyTorch** or **ONNX Runtime**:

- **Model Backend (sidebar)**: choose between:
  - **PyTorch (.pt/.pth)**: runs checkpoints saved under `models/` (excluding `models/onnx/`).
  - **ONNX (.onnx)**: runs ONNX exports saved under `models/onnx/` using ONNX Runtime.
- **Model picker behavior**:
  - When PyTorch is selected, the model dropdown lists only `.pt` / `.pth` files.
  - When ONNX is selected, the model dropdown lists only `.onnx` files from `models/onnx/`.
- The rest of the UI (image selection, overlays, class distributions) works identically for both backends.

See `STREAMLIT_README.md` for detailed usage instructions.

## Metrics reported

- Mean IoU (per‑class and averaged)
- F1 score (per‑class and macro)
- Accuracy (pixel‑level)
- ROC curves (one‑vs‑rest)
- Confusion matrix (percentage‑normalized)

## Notes

- Model checkpoints are stored under `checkpoints/` during training.  
- The best checkpoint is saved to `models/` for test inference.  
- Update class mappings and thresholds as new defect classes are added.

