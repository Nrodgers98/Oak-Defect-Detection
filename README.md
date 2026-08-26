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
├── data/                        # Raw data, NHLA exports, and organized datasets
│   ├── 26-8-18 Test/            # Example archive with XML + board image exports
│   ├── filtered-data/            # Filtered/invalid samples
│   └── organized-data/           # Legacy segmentation dataset layout
├── logs/                        # TensorBoard logs
├── models/                      # Saved model files
├── notebooks/                   # Training, evaluation, experiments
│   ├── functions/               # Project helper modules
│   ├── NHLA_Classification.py   # NHLA classification training script
│   ├── NHLA_Classification.ipynb # NHLA notebook pipeline
│   ├── nhla_grade_dataset.py    # XML + image extractor for NHLA grades
│   ├── Unet_FullImage.ipynb     # Full-image UNet training pipeline
│   ├── Unet_FullImage Optuna.ipynb  # Hyperparameter optimization with Optuna
│   ├── Unet_Patches.ipynb       # Patch-based UNet training pipeline
│   ├── environment_test.ipynb  # Environment verification and GPU testing
│   └── app.py                   # Interactive web app for model inference
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Oak defect detection workflow (high level)

### Segmentation pipeline
1. **Data audit and normalization**  
   Analyze raw line-scan imagery and normalize defect class naming.

2. **Class filtering**  
   Move rare classes below a mask-count threshold to reduce imbalance.

3. **Reorganization and validation**  
   Align images/masks, validate samples, and quarantine invalid data.

4. **Mask composition**  
   Merge per-class masks into a single multi-class mask per image.

5. **Train/val/test split**  
   Persist split metadata for repeatable experiments.

6. **Model training**  
   Train a ResNet34-UNet with Albumentations augmentations and TensorBoard logging.

7. **Evaluation**  
   Select the best checkpoint by validation IoU, then compute test metrics:
   mean IoU, F1, accuracy, ROC curves, and confusion matrix.

### NHLA classification workflow
This repository includes a dedicated NHLA grading pipeline for the scanner export data. The workflow is split into a 1-side classifier and a 2-side board-level classifier so each model can be trained and evaluated independently.

1. **Archive export ingestion**  
   A scanner export folder contains XML files plus the matching board images. The XML stores the top/bottom vision grades and any grader override.

2. **XML-to-CSV extraction**  
   The script `notebooks/functions/nhla_grade_dataset.py` reads the XML metadata, resolves the valid grade per face, associates each board image to its top and bottom view, and produces:
   - `nhla_face_labels.csv`
   - `nhla_board_labels.csv`
   - `grade_mapping.json`

3. **Grade prioritization rule**  
   The extraction logic follows the scanner grade precedence:
   - if `NHLA_Grade_Validation` is present and non-zero, use it
   - otherwise fall back to `NHLA_Grade_Vision`
   - this enforces the same validation override behavior used by the grader

4. **Two classification models**  
   - **1-side model**: predicts the grade from a single face image.
   - **2-side model**: predicts the board grade from both top and bottom images together.

5. **Optuna tuning and final training**  
   The shared training logic in `notebooks/NHLA_Classification.py` supports configurable encoders (`resnet18`, `resnet34`, `resnet50`), a tunable Optuna search space, and a final training pass that uses the best Optuna parameters rather than the default configuration values. The Optuna database remains the source of truth for the trial history.

6. **Artifact layout**  
   Final model runs keep the model directory clean:
   - `models/NHLA_Classification_1Side/checkpoints/` stores periodic epoch snapshots every N epochs
   - `models/NHLA_Classification_1Side/model_best.pt` stores the best model checkpoint
   - TensorBoard logs are stored under `logs/tensorboard/...`
   - metrics and class mapping artifacts are saved under a configurable `metrics_dir`, such as `metrics/NHLA_Classification_1Side/`
   - the Optuna database is stored under `logs/optuna/...` and not copied into the model output folder

7. **Evaluation and deployment**  
   Each run saves classification metrics including accuracy, balanced accuracy, macro/weighted F1, confusion matrix, classification report, and optional plots. These artifacts can be compared across the 1-side and 2-side workflows before deployment.

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

### NHLA classification workflow

1. Prepare your scanner archive folder. It should contain:
   - XML files with the NHLA grade metadata
   - corresponding top and bottom board images

2. Run the XML extraction pipeline from the repo root:

```bash
python notebooks/functions/nhla_grade_dataset.py --scan-root "data/26-8-18 Test"
```

This creates a folder such as:

```text
data/26-8-18 Test/nhla_dataset/
├── nhla_face_labels.csv
├── nhla_board_labels.csv
├── grade_mapping.json
```

3. Train the face-only model:

```bash
python notebooks/NHLA_Classification_1Side.py \
  --csv-path "data/26-8-18 Test/nhla_dataset/nhla_face_labels.csv" \
  --output-dir "models/NHLA_Classification_1Side" \
  --epochs 12 \
  --batch-size 16 \
  --lr 1e-3 \
  --val-split 0.2 \
  --seed 42 \
  --checkpoint-interval 5 \
  --tensorboard-dir "logs/tensorboard/NHLA_Classification_1Side" \
  --metrics-dir "metrics/NHLA_Classification_1Side"
```

4. Train the paired board model:

```bash
python notebooks/NHLA_Classification_2Side.py \
  --csv-path "data/26-8-18 Test/nhla_dataset/nhla_board_labels.csv" \
  --output-dir "models/NHLA_Classification_2Side" \
  --epochs 12 \
  --batch-size 8 \
  --lr 1e-3 \
  --val-split 0.2 \
  --seed 42 \
  --checkpoint-interval 5 \
  --tensorboard-dir "logs/tensorboard/NHLA_Classification_2Side" \
  --metrics-dir "metrics/NHLA_Classification_2Side"
```

5. Optional Optuna tuning before final training:

```bash
python notebooks/NHLA_Classification_1Side.py \
  --csv-path "data/26-8-18 Test/nhla_dataset/nhla_face_labels.csv" \
  --output-dir "models/NHLA_Classification_1Side" \
  --optuna-trials 5 \
  --tune-epochs 4 \
  --checkpoint-interval 5
```

6. Open the notebook version for interactive execution:
   - `notebooks/NHLA_Classification_1Side.ipynb`
   - `notebooks/NHLA_Classification_2Side.ipynb`
   - `notebooks/NHLA_Classification.py`

### Segmentation training notebooks

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

### Segmentation metrics
- Mean IoU (per-class and averaged)
- F1 score (per-class and macro)
- Accuracy (pixel-level)
- ROC curves (one-vs-rest)
- Confusion matrix (percentage-normalized)

### NHLA classification metrics
- Accuracy
- Balanced accuracy
- Macro precision / recall / F1
- Weighted precision / recall / F1
- Micro F1
- Per-class precision / recall / F1
- Confusion matrix
- One-vs-rest ROC AUC
- Classification report

All classification metrics are logged to TensorBoard and saved as JSON files in the model output directory.

## Notes

- Model checkpoints are stored under `checkpoints/` during training.  
- The best checkpoint is saved to `models/` for test inference.  
- Update class mappings and thresholds as new defect classes are added.

