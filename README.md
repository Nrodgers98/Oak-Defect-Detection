# Oak Defect Detection

Deep learning for green rough oak planks imaged by an industrial line-scan scanner. The project has three tracks:

| Track | Task | Models |
| --- | --- | --- |
| **Semantic segmentation** | Pixel-wise defect maps (e.g. Knot, Rot, Stain, Sapwood, BlackRot) | smp **UNet / UNet++ / DeepLabV3** (ResNet34/50), full-image or native-resolution tiles, exported to ONNX |
| **YOLO** | Defect instance segmentation; **pith** bounding-box detection | Ultralytics YOLOv8-seg, YOLO26n detection |
| **NHLA grade classification** | Lumber grade per face (1-side) and per board (2-side) from scanner XML exports | torchvision ResNet18/34/50 |

Every training pipeline is tuned with **Optuna**, logs to **TensorBoard**, and writes timestamped run folders. A Streamlit **inference app** and a Streamlit **model dashboard** sit on top.

---

## Repository layout

```
Oak-Defect-Detection/
├── .devcontainer/                      # GPU dev container (image, ports, post-create install)
├── notebooks/                          # Entry points — run with notebooks/ as the working directory
│   ├── DeepLabv3_FullImage_Optuna.ipynb    # Segmentation: DeepLabV3, full image, Optuna
│   ├── Unet++_FullImage_Optuna.ipynb       # Segmentation: UNet++, full image, Optuna
│   ├── Tiled_Optuna.ipynb                  # Segmentation: UNet / UNet++ / DeepLabV3 on native-resolution tiles
│   ├── YOLO_FullImage.ipynb                # YOLOv8-seg defect instance segmentation
│   ├── YOLO_Pith_Detection_Optuna.ipynb    # YOLO pith detection (full board or tiled), Optuna + calibration
│   ├── NHLA_Classification_1Side.ipynb     # NHLA grade, single face
│   ├── NHLA_Classification_2Side.ipynb     # NHLA grade, whole board (top + bottom)
│   ├── NHLA_Classification.py              # Shared NHLA training library (models, datasets, Optuna, metrics)
│   ├── NHLA_Classification_1Side.py        # CLI: 1-side training
│   ├── NHLA_Classification_2Side.py        # CLI: 2-side training
│   ├── NHLA_Classification_Evaluate.py     # CLI: compare 1-side vs 2-side metrics
│   ├── predict_image.py                    # Segmentation inference library + CLI
│   ├── app.py                              # Streamlit inference app (PyTorch + ONNX Runtime)
│   ├── dashboard/                          # Streamlit model dashboard (runs, Optuna studies, curves, metrics)
│   └── functions/                          # Helper modules (see below)
├── presentations/                      # Marp slide deck + assets
├── tests/                              # pytest suite
├── requirements.txt
└── LICENSE
```

`data/`, `models/`, `checkpoints/`, `logs/`, `notebooks/runs/` and all `*.pt` / `*.pth` / `*.onnx` / `*.db` files are gitignored — they are produced locally.

---

## Environment

The project runs in a VS Code **dev container** built on [`gperdrizet/deeplearning-gpu`](https://hub.docker.com/r/gperdrizet/deeplearning-gpu) (NVIDIA TensorFlow 24.06 base: CUDA 12.5, cuDNN 9.1, Python 3.10+).

**Requirements:** NVIDIA GPU (Pascal or newer, driver ≥ 545), Docker with GPU support, VS Code + Dev Containers extension. On Linux, also install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

On creation the container runs:

```bash
pip install -r requirements.txt
pip install --no-deps ultralytics==8.4.21   # --no-deps so it can't override the pinned torch/numpy
```

Key pinned packages: PyTorch 2.10 / torchvision 0.25, segmentation_models_pytorch 0.5, Albumentations 2.0, Optuna 4.7, TensorBoard 2.16, ONNX 1.20 / onnxruntime-gpu 1.23, Streamlit 1.54, Plotly, huggingface_hub.

**Forwarded ports**

| Port | Service | Command |
| --- | --- | --- |
| 6006 | TensorBoard | `tensorboard --logdir logs/tensorboard --bind_all` |
| 8080 | Optuna dashboard | `optuna-dashboard sqlite:///logs/optuna/<study>.db --host 0.0.0.0` |
| 8501 | Inference app | `streamlit run notebooks/app.py` |
| 8502 | Model dashboard | `cd notebooks/dashboard && streamlit run Home.py` |

---

## Data

### Public dataset (Hugging Face)

The raw line-scan TIFFs are published as **[nrodgers98/Oak-Defect-Detection](https://huggingface.co/datasets/nrodgers98/Oak-Defect-Detection)** (image segmentation, ~1.5k samples, CC BY-NC 4.0).

```python
# From a notebook (cwd = notebooks/)
from functions.oak_hf_dataset import ensure_oak_raw_data
RAW_DIR = ensure_oak_raw_data(raw_folder_name="raw-data")   # skips download if data/raw-data/*.tif exists
```

```bash
# From the repo root
python notebooks/functions/oak_hf_dataset.py [--raw-folder-name NAME] [-o PATH] [--force] [--repo REPO_ID]
```

Set `HF_TOKEN` for higher rate limits or a private fork. The `datasets` API (`load_dataset("nrodgers98/Oak-Defect-Detection")`) works for exploration but is not a drop-in for the TIFF pipeline.

### File naming

- Color image: `*_Col.tif`
- Class mask: `*_Col_<Class>.tif`, `*_Col_Bin_<Class>.tif` or `*_Bin_Col_<Class>.tif`

`functions/mask_filenames.parse_class_mask_filename` is the single parser for mask names.

### Preprocessing (`functions/data_pipeline.prepare_dataset`)

One call runs the whole chain:

1. **Rename** legacy class names (`Black_Rot` → `BlackRot`, …).
2. **Filter** classes with fewer than `mask_threshold` masks into `data/filtered-data/`.
3. **Reorganize** images and per-class masks.
4. **Validate** and quarantine invalid samples.
5. **Combine** per-class masks into one multi-class mask per image. Class IDs are assigned alphabetically by mask folder starting at 1; 0 is background.
6. **Split** into train / val / test (default 64 / 16 / 20) and write `split.json`.

```
data/organized-data/
├── images/            # *_Col.tif
├── masks/<Class>/     # per-class binary masks
├── combined_masks/    # *_mask.png, multi-class
└── split.json
```

> **Note:** if `split.json` and `combined_masks/` already exist the step is skipped, so a different `mask_threshold` or raw folder has no effect until you pass `force=True`.

### Tiling (optional, `functions/tile_dataset.create_tiled_dataset`)

Crops boards into native-resolution tiles under `data/tiles/x{W}_y{H}_sx{stepX}_sy{stepY}_{edge_mode}/`, with a `tile_manifest.json` plus config/stats JSON.

- Tile boxes are `(ymin, ymax, xmin, xmax)` in row-major order.
- `edge_mode="shrink"` makes the last tile smaller; `"flush"` shifts it back so it stays full size.
- Background-only train tiles can be dropped; every val/test tile is kept.

`predict_image.tessellate_image` uses the same geometry at inference time (checked by `tests/test_tiled_inference.py`).

---

## Segmentation

| Notebook | Architecture | Input | Run prefix |
| --- | --- | --- | --- |
| `DeepLabv3_FullImage_Optuna.ipynb` | DeepLabV3 (encoder tuned: ResNet34/50) | Full board, resized | `STACK_` |
| `Unet++_FullImage_Optuna.ipynb` | UNet++ (ResNet34) | Full board, 384×1024 | `CC_` |
| `Tiled_Optuna.ipynb` | `MODEL_ARCH` = `Unet` / `UnetPlusPlus` / `DeepLabV3` | Native-resolution tiles (default 1088×1088, flush) | `TILE_` |

**Shared recipe**

- Encoders pretrained on ImageNet, cross-entropy loss.
- Albumentations: flips, shift/scale/rotate, brightness/contrast/gamma, noise/blur, ImageNet normalization.
- **Optuna:** TPE sampler (multivariate, seed 42) maximizing validation mean IoU, with Median or Hyperband pruning. Typical search space: learning rate, weight decay, optimizer (Adam/AdamW), plus scheduler (cosine/step/none) and encoder in the DeepLab and Tiled notebooks.
- Studies live in SQLite under `logs/optuna/`, and notebooks can resume the most recent matching study.
- **Final training** uses the best trial's parameters with early stopping, optionally followed by transfer learning from a previous checkpoint.
- **Test evaluation:** pixel accuracy, per-class and macro F1, per-class IoU and mean IoU excluding background, a row-normalized confusion matrix, and one-vs-rest ROC/AUC. Plots are saved to `data/metrics/`.
- The Tiled notebook can also stitch tiles back into whole boards for board-level metrics.
- **ONNX export:** a wrapper takes `uint8 [B,3,H,W]`, does /255 and ImageNet normalization inside the graph, and outputs softmax × 100. It uses opset 12 with dynamic batch/H/W, is verified with onnxruntime, and is saved to `models/onnx/<run>.onnx`.
- **Summary CSV:** one row per run is appended to `data/metrics/TRAINING_RESULTS_SUMMARY.csv` via `functions/metrics_export`.

---

## YOLO

### `YOLO_FullImage.ipynb` — defect instance segmentation

- Converts `organized-data` combined masks to YOLO polygon labels in `data/yolo-format/`.
- Trains Ultralytics **YOLOv8-seg** (n/m/l/x; default `l`) at 1024 px with `rect`, halving the batch on OOM. Transfer learning is optional.
- Reports box and mask mAP50 / mAP50-95.
- Output: `models/STACK_YOLO_<size>_FullImage_<ts>/`.

### `YOLO_Pith_Detection_Optuna.ipynb` — pith detection

Single-class **Pith** bounding-box detector on Top/Bot face TIFFs.

- **Input:** a pre-built YOLO dataset in `data/bbox_clean_yolo/` (`images/`, `labels/`, `class_mapping.json`, `dataset_manifest.json`). An empty label file means a verified negative.
- **Split:** grouped by board ID (64/16/20), balancing positives. Saved with a fingerprint to `data/yolo_pith_detection/split_manifest.json`.
- **Modes:** `full` (whole board) or `tiled`. Tiled mode uses 1024×768 tiles with 25% overlap. Boxes are clipped into tiles via `functions/yolo_detection.py`, and negative train tiles are sampled 1:1.
- **Model:** YOLO26n detection (`yolo26n.pt`).
- **Optuna** (opt-in via `RUN_REAL_OPTUNA`) tunes `lr0`, `lrf`, `weight_decay` and the optimizer. It maximizes a weighted score of recall, AP50, AP50-95, precision, board sensitivity and F1, minus a penalty for false positives on negative boards.
- **Calibration:** a grid search on val over confidence and NMS IoU, with constraints recall ≥ 0.90 and ≤ 0.25 FP per negative board. This is followed by a locked test run.
- **Outputs:**
  - `models/<RUN_NAME>/final_detection/weights/best.pt`
  - `deployment_config.json` (thresholds, tile geometry, versions)
  - audits, operating points, test metrics and visual-review overlays in `data/metrics/<RUN_NAME>/`

Expensive stages are controlled by `RUN_*` flags and won't overwrite existing outputs unless the matching `FORCE_*` flag is set.

---

## NHLA grade classification

### 1. Build label CSVs from a scanner export

A scanner export folder holds XML grade metadata plus the matching top and bottom board images.

```bash
python notebooks/functions/nhla_grade_dataset.py --scan-root "data/<export-folder>" [--output-dir DIR] [--num-workers N]
```

Writes to `<scan-root>/nhla_dataset/` by default:

| File | Contents |
| --- | --- |
| `nhla_face_labels.csv` | One row per face (top/bottom) with grade, grade source, image path |
| `nhla_board_labels.csv` | One row per board with top and bottom image paths and board grade |
| `nhla_face_labels_validated.csv` | Only faces of boards where the grader confirmed the vision grade (used by the 1-side notebook) |
| `grade_mapping.json` | Grade ID → name (Unknown, 3B Com, 3A Com, 2B Com, 2A Com, 1 Com, Select, F1F, FAS, Cull) |

**Grade rules**

- **Board grade:** `NHLA_Grade_Validation` if present and non-zero, otherwise `NHLA_Grade_Vision`.
- **Face grade:** the per-face `NHLA_Grade_Validation_{Top|Bot}` if non-zero, otherwise `NHLA_Grade_Vision_{Top|Bot}`, otherwise the board-level vision grade.

### 2. Train

`NHLA_Classification.py` is the shared library. The 1-side and 2-side scripts and notebooks are thin wrappers around it.

- **1-side:** `FaceClassifier`, a torchvision ResNet18/34/50 with a new classification head.
- **2-side:** `DualFaceClassifier` with three fusion options:
  - `shared` (default): one backbone per face, features concatenated. It can start from a 1-side checkpoint (`--init-checkpoint`).
  - `stacked`: faces stacked vertically into one image.
  - `channels`: 6-channel input.
- **Image cache:** images are decoded once into a sharded JPEG cache (`functions/nhla_image_cache.py`), scaled by 0.375 and padded to a 512×1888 canvas.
  - Location: `$NHLA_CACHE_DIR` or `~/.cache/nhla_images`. Keep it on a Linux filesystem, not a Windows mount.
  - It can be very large (≈115 GB for the full board set).
- **Training:** AdamW, cosine LR, bf16 autocast; the best epoch is chosen by validation macro-F1. The split is grouped by board, so both faces of a board stay together.
- **Optuna:** TPE + MedianPruner on val macro-F1. Searches learning rate, batch size, weight decay and encoder, and can tune on a stratified subset of boards (`--tune-subset-fraction`).

```bash
# 1-side (face) — runs 5 Optuna trials first by default; pass --optuna-trials 0 to skip
python notebooks/NHLA_Classification_1Side.py \
  --csv-path "data/<export>/nhla_dataset/nhla_face_labels.csv" \
  --output-dir models/NHLA_Classification_1Side \
  --tensorboard-dir logs/tensorboard/NHLA_Classification_1Side \
  --metrics-dir data/metrics/NHLA_Classification_1Side \
  --epochs 12 --batch-size 16 --optuna-trials 5 --tune-epochs 4

# 2-side (board)
python notebooks/NHLA_Classification_2Side.py \
  --csv-path "data/<export>/nhla_dataset/nhla_board_labels.csv" \
  --output-dir models/NHLA_Classification_2Side \
  --two-side-fusion shared \
  --init-checkpoint models/NHLA_Classification_1Side/model_best.pt
```

Other flags: `--lr`, `--weight-decay`, `--val-split`, `--seed`, `--encoder-name {resnet18,resnet34,resnet50}`, `--[no-]pretrained`, `--checkpoint-interval`, `--num-workers`, `--image-cache-dir`, `--image-scale`, `--canvas H W`, and `--cuda-device-index` (2-side only). Run each script with `--help` for the full list.

The notebooks (`NHLA_Classification_{1,2}Side.ipynb`) do the same with timestamped output dirs and a persistent Optuna DB at `logs/optuna/NHLA_Classification.db`.

**Outputs**

- In the output dir: `model_best.pt` and `checkpoints/epoch_NNN.pt`.
- In the metrics dir: `class_mapping.json`, `<run>_metrics.json`, `<run>_history.csv`, `<run>_confusion_matrix.jpg` and `best_params.json`.
- TensorBoard logs.

**Metrics:** accuracy, balanced accuracy, macro / weighted / micro precision, recall and F1; per-class P/R/F1; confusion matrix; classification report; one-vs-rest ROC-AUC.

> Training uses a train/val split only; there is no held-out test set, so reported metrics come from the validation set used for model selection.

### 3. Compare

```bash
python notebooks/NHLA_Classification_Evaluate.py \
  --face-metrics  data/metrics/<1side-run>/<run>_metrics.json \
  --board-metrics data/metrics/<2side-run>/<run>_metrics.json
```

---

## Inference

### Streamlit app — `streamlit run notebooks/app.py` (port 8501)

- **Backends:** PyTorch (`models/*/*.pt|.pth`) or ONNX Runtime (`models/onnx/*.onnx`).
- **Architectures:** UNet, UNet++ and DeepLabV3, detected from the checkpoint or its path.
- **Image sources:** `data/organized-data/images/`, `data/to_predict/` (demo boards) or a custom path.
- **Inference modes:** full image, patch-based sliding window, and tiled/crop-based for full-resolution boards (both backends).
- **Settings:** editable class names, per-class minimum confidence, overlay transparency, CUDA/CPU.
- **Results:** inference time, the raw class mask with per-class pixel distribution, and a colored overlay with legend.

### Command line — `notebooks/predict_image.py`

```bash
python notebooks/predict_image.py --image path/to/board_Col.tif --model models/<run>/model_best.pt \
  [--tiled --max-crop-width 3000 --max-crop-height 1000] [--patch-based --tile-size 224 --stride 112] \
  [--output mask.png] [--alpha 0.5] [--device cuda]
```

The mask and a matching `_overlay.png` are saved to `data/predictions/prediction_<timestamp>.png` by default (`--visualize PATH` overrides the overlay path).

---

## Model dashboard — `cd notebooks/dashboard && streamlit run Home.py` (port 8502)

Launch it from that directory so its `.streamlit/config.toml` (theme, port) applies. It reads everything on disk through `functions/run_registry.py` (read-only):

| Page | Shows |
| --- | --- |
| **Overview** | KPIs, best run per family (Segmentation, YOLO Seg, YOLO Pith, NHLA), timeline, recent runs |
| **Runs** | Filterable table of every run: architecture, classes, status, score, Optuna link, ONNX export |
| **Run Detail** | Training curves (TensorBoard / Ultralytics / history), scalar and per-class metrics, confusion matrix, hyperparameters, linked study, artifacts |
| **Optuna** | Every study in `logs/optuna/*.db`: history, parallel coordinates, parameter importance |
| **Compare** | Overlay curves of up to 8 runs; side-by-side metrics/hyperparameters with a diff-only toggle |

Nothing on disk records which model dir, checkpoint dir, TensorBoard dir, Optuna study and ONNX file belong together. The registry links them by normalized run-name prefix plus timestamp (±2 s). Studies are linked by the summary-CSV study name, then by timestamp, then (marked *inferred*) by name and time overlap. A fresh clone has no runs, so the dashboard starts empty.

---

## Run artifacts

```
models/<PREFIX>_<Arch>_<Variant>_<YYYYMMDD_HHMMSS>/model_best.pt
checkpoints/<same name>/                    # periodic checkpoints + history
models/onnx/<run>.onnx
logs/tensorboard/<run>/
logs/optuna/<study>.db                      # source of truth for trial history; reopen to continue a study
data/metrics/                               # per-run JSON/PNG + TRAINING_RESULTS_SUMMARY.csv
```

---

## Helper modules (`notebooks/functions/`)

| Module | Purpose |
| --- | --- |
| `oak_hf_dataset` | Download the HF dataset into `data/<folder>` (also a CLI) |
| `data_pipeline` | `prepare_dataset()`, the full preprocessing chain |
| `rename_files`, `filter_by_mask_count`, `model_tools`, `mask_filenames` | Individual preprocessing steps (rename, class filter, reorganize/validate/combine/split, mask-name parsing) |
| `tile_dataset` | Native-resolution tile windows and tiled dataset creation |
| `sliding_inference` | Batched overlapping-window inference |
| `metrics_export` | Build and append rows to the training summary CSV |
| `yolo_detection` | Box format conversion, tile clipping/restoring, IoU matching, detection metrics, TIFF metadata |
| `nhla_grade_dataset` | Scanner XML → NHLA label CSVs (also a CLI) |
| `nhla_image_cache` | Sharded decode-once image cache for NHLA training |
| `run_registry` | Read-only index of runs, studies and metrics for the dashboard |

Inside `notebooks/`, modules import each other as `functions.*`. Tests import them as `notebooks.functions.*`.

---

## Tests

```bash
pip install pytest
python -m pytest tests
```

Covers mask filename parsing and the preprocessing chain, tile geometry and tiled inference stitching, YOLO box utilities and detection metrics, NHLA XML extraction and grade rules (plus an Optuna smoke test), and run-registry linking.

---

## License

Code is released under the [MIT License](LICENSE). The Hugging Face dataset is licensed separately under CC BY-NC 4.0.
