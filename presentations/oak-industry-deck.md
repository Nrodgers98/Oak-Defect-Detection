---
marp: true
theme: default
paginate: true
size: 16:9
footer: Oak defect detection — semantic segmentation
---

<!--
  Numbers: UNet Optuna rows from saved notebook outputs; DeepLabv3 Optuna from ONNX on test split (see Results slide footnote).
  To refresh CM PNGs: re-run evaluation cells, or regenerate DeepLab figure from `models/onnx/deeplabv3_fullimage_optuna.onnx` + `split.json` test IDs.
-->

# Oak defect detection

**Pixel-wise defect maps from line-scan imagery**

Nicholas Rodgers · Computer Vision Engineer · **Ten Oaks, LLC**

<!--
  About 30s: You solve grading/sorting for green rough oak using dense labels, not just boxes.
-->

---

## Why this problem

- Mill throughput and consistent grading depend on **reliable defect cues** on every board.
- Many defects are **irregular in shape** — **semantic segmentation** gives a full surface map.
- **Goal**: a **reproducible** pipeline from raw scans to **identify** defects in oak planks that is **compatible** with our scanner manufacturer’s (**EBI**) ONNX runtime.

<!--
  Speaker: Connect to their world — recovery, rework, customer claims — without overclaiming autonomy. Next slide is the EBI contract detail.
-->

---

## EBI-oriented ONNX contract (segmentation)

What EBI’s interpreter expects — implemented in **`EBIExportWrapper`** in export cells (e.g. [`notebooks/Unet_FullImage.ipynb`](../notebooks/Unet_FullImage.ipynb), [`notebooks/DeepLabv3_FullImage.ipynb`](../notebooks/DeepLabv3_FullImage.ipynb)):

- **Task shape**: **Dense semantic segmentation** — per-pixel class **scores** over the board. **No** instance NMS, **no** bounding-box head, **no** extra decoder post-processing on the host
- **Input tensor**: **uint8** RGB, shape **`[B, 3, H, W]`**, raw **0–255** (CHW, batched).
- **Bundled into the ONNX file** (host passes raw images only): rescale pixels to [0, 1], apply **ImageNet** normalization, run the segmentation model, then **softmax** and **×100** so every pixel has **C channels of class scores from 0–100** (percent scale).
- **Output tensor**: **float32** **`[B, C, H, W]`** with `C` = number of classes; values interpretable as **percent** per channel (approximately **~100** when summed across `C` at each pixel).
- **Naming**: some exports use ONNX input name **`image`** (EBI-oriented path in `notebooks/app.py`); others use **`input`** for the same numeric contract — **match whatever your EBI toolchain prescribes**.

<!--
  Speaker: If challenged on “official EBI spec,” point to your manufacturer packet; this slide reflects what we implemented to satisfy their interpreter + our export wrapper.
-->

---

## What we built

- **Trained models**: semantic segmentation (multiple backbones / heads explored).
- **Rigorous evaluation**: held-out test, IoU / F1 / accuracy / ROC / confusion matrix.
- **Exports**: **ONNX** for lightweight integration; verified runtime signatures where noted in notebooks.
- **Demo**: **Streamlit** app — load a checkpoint, run inference, visualize overlays (`notebooks/app.py`).

<!--
  Speaker: Emphasize traceability — same splits and metrics across experiments.
-->

---

## Data

- **Source**: line-scan RGB **TIFFs** (industrial-style plank imagery).
- **Public release**: [Hugging Face — nrodgers98/Oak-Defect-Detection](https://huggingface.co/datasets/nrodgers98/Oak-Defect-Detection) (~1.5k samples, **CC BY-NC 4.0**).
- **Typical classes**: Background, **BlackRot**, **Knot**, **Stain** — **not every experiment uses the same set** (e.g. some Optuna notebooks drop **Stain** or other rare classes). **Always pair a metric with the notebook + class list** when you present numbers.

<!--
  Speaker: Mention license if industry cares about redistribution; point to Hub for reproducibility. Mixed class sets are a feature of honest R&D — say you standardize per experiment.
-->

---

## Pipeline (raw scans → training)

1. **Raw TIFFs** → audit & **normalize** defect names  
2. **Filter** rare classes (explicit threshold)  
3. **Reorganize** — aligned image/mask pairs, validate, quarantine bad samples  
4. **Combined** multi-class masks + **train/val/test** split (JSON)  
5. **Train** — checkpoint on best **val IoU**

- Automation: `notebooks/functions/` (`data_pipeline`, filtering, splits, HF download helper).

<!--
  Speaker: One sentence each box — you’re showing engineering discipline, not just model hacking.
-->

---

## Modeling approach

- **Task**: multi-class **semantic segmentation** (per-pixel class).
- **Training**: pretrained encoders; **Albumentations**; full-image (and patch workflows in separate notebooks).
- **Model selection**: best checkpoint by **validation IoU**, then **frozen test split** for reported scores.

<!--
  Speaker: This slide defends why inspection stakeholders should trust the headline numbers.
-->

---

## Architectures explored

| Approach | Notes |
|----------|--------|
| **ResNet34–UNet** | Full image, patch training variant, **Optuna** tuning notebooks |
| **ResNet34–U-Net++** | Full image + Optuna variant |
| **ResNet34–DeepLabV3** | Full image + Optuna; ONNX export path in notebook |
| **YOLO backbone + dense head** | Pixel logits for deployment-style export (see `YOLO+SegHead.ipynb`) |

<!--
  Speaker: You scoped the solution space — classical proven encoders plus a YOLO-feature experiment.
-->

---

## Hyperparameters (Optuna)

- **Tool**: [Optuna](https://optuna.org/) (TPE-style search) on selected notebooks.
- **Examples of tuned knobs**: batch size, learning rate, weight decay, optimizer choice — *exact ranges in each `*_Optuna.ipynb`*.
- **Objective**: maximize **validation IoU** (early stopping on plateau / best epoch per trial).

<!--
  Speaker: Skip trial counts unless asked — offer “in appendix” for detail.
-->

---

## How we measure quality

- **Segmentation**: per-class **IoU**, **mean IoU** (often reported excluding background — **say which** when you quote numbers).
- **Classification-style**: macro / per-class **F1**, pixel **accuracy**.
- **Diagnostics**: one-vs-rest **ROC-AUC**, **confusion matrix** (normalized).
- **Rule**: metrics computed on **test** images only after choosing the best val checkpoint.

<!--
  Speaker: This slide preempts “how do we know it’s not cherry-picked?”
-->

---

## Results — test metrics (from notebooks)

**Breadth over a single hero model**: same evaluation pattern (**held-out test**, row-normalized **confusion matrix**, IoU / F1 / accuracy). Class sets **differ by experiment** — always cite the notebook.

| Notebook | Classes (defects) | Mean IoU (excl. bg) | Macro F1 | Pixel acc. |
|----------|-------------------|---------------------|----------|------------|
| [`Unet_FullImage Optuna.ipynb`](../notebooks/Unet_FullImage%20Optuna.ipynb) | BlackRot, Knot | 0.465 | 0.736 | 0.946 |
| [`Unet++_FullImage_Optuna.ipynb`](../notebooks/Unet++_FullImage_Optuna.ipynb) | BlackRot, Knot, Stain | 0.477 | 0.720 | 0.934 |
| [`DeepLabv3_FullImage_Optuna.ipynb`](../notebooks/DeepLabv3_FullImage_Optuna.ipynb) | BlackRot, Knot, Stain | 0.509 | 0.740 | 0.923 |

*UNet rows: metrics from **saved** notebook outputs. DeepLabv3 Optuna: same **test split** and preprocessing size (384×1024), metrics and confusion matrix from **`models/onnx/deeplabv3_fullimage_optuna.onnx`** (notebook had no embedded evaluation figure). Re-run the notebook evaluation cell to align with a fresh `.pt` checkpoint if needed.*

<!--
  Speaker: All three rows are Optuna-tuned. UNet Optuna is three-way (Background, BlackRot, Knot). UNet++ and DeepLab are four-way (Background + three defects).
-->

---

## Confusion matrix — `Unet_FullImage Optuna.ipynb`

![width:640px](assets/cm_unet_fullimage_optuna.png)

*Background, BlackRot, Knot — % of true class per row (test set).*

<!--
  Speaker: Diagonal dominance vs off-diagonal shows where the model confuses knot with rot, etc.
-->

---

## Confusion matrix — `Unet++_FullImage_Optuna.ipynb`

![width:640px](assets/cm_unetplusplus_fullimage_optuna.png)

*Background, BlackRot, Knot, Stain — same normalization.*

<!--
  Speaker: Optuna-tuned U-Net++ with full four defect channels.
-->

---

## Confusion matrix — `DeepLabv3_FullImage_Optuna.ipynb`

![width:640px](assets/cm_deeplabv3_fullimage_optuna.png)

*Background, BlackRot, Knot, Stain — test set, row-normalized (same convention as other notebooks). Figure built from exported ONNX on 384×1024 inputs; notebook evaluation cell can replace this after a full run.*

<!--
  Speaker: DeepLabV3 + Optuna; CM artifact matches the EBI-style ONNX export used on the line scanner path.
-->

---

## Deployment path

- **ONNX**: export + **ONNX Runtime** verification in DeepLabV3-style notebooks (preprocessing / output scaling as implemented there).
- **Streamlit**: toggle **PyTorch** vs **ONNX Runtime**; GPU when available.
- **Integration story**: same class colormap and tensor shapes documented in app + export cells.

<!--
  Speaker: Position as prototype suitable for IT/plant discussion — not a claimed certified production system unless you own that.
-->

---

## Demo (30 seconds)

1. `streamlit run notebooks/app.py`
2. Pick backend (**PyTorch** or **ONNX**).
3. Select image → **Generate prediction** → show **overlay** and class map.

<!--
  Speaker: If live demo is risky, use a screenshot slide instead — swap order with Results figure.
-->

---

## Limitations and next steps

- **Data**: class imbalance; rare defects need more labels or revised taxonomy.
- **Generalization**: new lines, species, or lighting → expect **domain shift**; plan calibration data.
- **Operations**: human review for edge cases; optional **active learning** / feedback loops.

<!--
  Speaker: Audiences trust you faster when you name limits before they do.
-->

---

## Thank you

- **Repo**: Oak Defect Detection (this workspace)
- **Dataset**: [nrodgers98/Oak-Defect-Detection](https://huggingface.co/datasets/nrodgers98/Oak-Defect-Detection)
- **Questions**

<!--
  Put email or LinkedIn on this slide if appropriate for the forum.
-->

---

<!-- _class: lead -->
## Backup — stack (if asked)

PyTorch · segmentation_models_pytorch · Albumentations · TensorBoard · Optuna · scikit-learn · Streamlit · ONNX Runtime · Hugging Face Hub

---

<!-- _class: lead -->
## Backup — project layout

`data/` (raw, filtered, organized) · `notebooks/` (training + `functions/`) · `models/` · `logs/` · Streamlit `app.py`
