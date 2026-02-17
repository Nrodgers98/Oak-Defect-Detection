# Oak Defect Detection

A GPU‑accelerated deep learning project for semantic segmentation of defects in green rough oak planks. The workflow cleans and standardizes line‑scan data, builds multi‑class masks, trains a ResNet34‑UNet model, and evaluates with IoU, F1, accuracy, ROC curves, and confusion matrices.

## Deep learning GPU environment

This repository is designed to run inside a VS Code dev container with NVIDIA GPU support.

**Included stack (reference)**  
- CUDA 12.5, cuDNN 9.1  
- PyTorch 2.10, TensorFlow 2.16, Keras 3.3, scikit‑learn 1.4  
- Python 3.10, NumPy 1.24, Pandas 2.2, Matplotlib 3.10  
- JupyterLab, TensorBoard, Optuna  

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
│   ├── functions/               # Custom helper modules
│   └── Unet_FullImage.ipynb      # Full-image UNet training pipeline
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

These make the preprocessing steps repeatable and reduce manual error.

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

## How to run

1. Open the repository in a VS Code dev container.  
2. Run `notebooks/Unet_FullImage.ipynb` from top to bottom.  
3. Check TensorBoard logs in `logs/` and saved models in `models/`.

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

