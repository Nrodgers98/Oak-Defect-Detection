from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from torchvision.io import decode_jpeg
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notebooks.functions.nhla_image_cache import ensure_image_cache

SUPPORTED_ENCODERS = ["resnet18", "resnet34", "resnet50"]
ENCODER_WEIGHTS = {
    "resnet18": ResNet18_Weights.IMAGENET1K_V1,
    "resnet34": ResNet34_Weights.IMAGENET1K_V1,
    "resnet50": ResNet50_Weights.IMAGENET1K_V1,
}


def build_resnet_backbone(encoder_name: str, pretrained: bool = True, in_channels: int = 3):
    encoder_name = encoder_name.lower()
    if encoder_name not in SUPPORTED_ENCODERS:
        raise ValueError(f"Unsupported encoder '{encoder_name}'. Choose from: {SUPPORTED_ENCODERS}")

    weights = ENCODER_WEIGHTS[encoder_name] if pretrained else None
    if encoder_name == "resnet18":
        backbone = models.resnet18(weights=weights)
    elif encoder_name == "resnet34":
        backbone = models.resnet34(weights=weights)
    elif encoder_name == "resnet50":
        backbone = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported encoder '{encoder_name}'")

    if in_channels != 3:
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return backbone


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cuda_device_index: int = 0) -> torch.device:
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if cuda_device_index < 0 or cuda_device_index >= device_count:
            raise ValueError(
                f"Invalid cuda_device_index={cuda_device_index}. Available CUDA devices: 0..{device_count - 1}"
            )
        torch.cuda.set_device(cuda_device_index)
        device_name = torch.cuda.get_device_name(cuda_device_index)
        print(f"Using device: cuda:{cuda_device_index} ({device_name})")
        return torch.device(f"cuda:{cuda_device_index}")

    print("Using device: cpu (CUDA unavailable)")
    return torch.device("cpu")


class FaceDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        from PIL import Image
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        label = int(row["label_index"])
        return image, label


class BoardDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        from PIL import Image

        top = Image.open(row["top_image_path"]).convert("RGB")
        bottom = Image.open(row["bottom_image_path"]).convert("RGB")
        top = self.transform(top)
        bottom = self.transform(bottom)
        label = int(row["label_index"])
        return top, bottom, label


DEFAULT_IMAGE_SCALE = 0.375          # native 1 mm/px along the board -> ~2.7 mm/px
DEFAULT_CANVAS = (512, 1888)         # (height, width); fits 99% of faces at DEFAULT_IMAGE_SCALE
_IMAGENET_MEAN_RGB = (124, 116, 104)  # padding value; normalizes to ~0


class CachedImageDataset(Dataset):
    """Serves uint8 CHW faces from the JPEG shard cache built by ``ensure_image_cache``.

    Each face is decoded at its cached (fixed-scale) size and pasted top-left onto a
    ``canvas`` (height, width) filled with the ImageNet mean colour, so every board keeps
    its true physical size. ``image_indices`` is ``(N,)`` for face mode or ``(N, 2)``
    (top, bottom) for board mode. Only NumPy arrays are held, so DataLoader workers don't
    slowly copy a large DataFrame.
    """

    def __init__(self, cache_dir: str | Path, image_indices: np.ndarray, labels: np.ndarray, canvas: Tuple[int, int] = DEFAULT_CANVAS):
        self.cache_dir = Path(cache_dir)
        self.image_indices = np.asarray(image_indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.canvas = canvas
        self._index = None

    def __len__(self) -> int:
        return len(self.labels)

    def _image(self, index: int) -> torch.Tensor:
        if self._index is None:
            self._index = np.load(self.cache_dir / "index.npy")
        shard, offset, length, height, width = (int(v) for v in self._index[index])
        fd = os.open(self.cache_dir / "shards" / f"{shard:05d}.bin", os.O_RDONLY)
        try:
            payload = os.pread(fd, length, offset)
        finally:
            os.close(fd)
        face = decode_jpeg(torch.frombuffer(bytearray(payload), dtype=torch.uint8))
        canvas = torch.tensor(_IMAGENET_MEAN_RGB, dtype=torch.uint8).view(3, 1, 1).repeat(1, *self.canvas)
        canvas[:, :height, :width] = face
        return canvas

    def __getitem__(self, idx: int):
        row = self.image_indices[idx]
        label = int(self.labels[idx])
        if row.ndim == 0:
            return self._image(int(row)), label
        return self._image(int(row[0])), self._image(int(row[1])), label


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255.0
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255.0


def prepare_images(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a batch to *device*; uint8 batches from the cache are ImageNet-normalized on the GPU."""
    images = images.to(device, non_blocking=True)
    if images.dtype == torch.uint8:
        images = (images.float() - _IMAGENET_MEAN.to(device)) / _IMAGENET_STD.to(device)
    return images.contiguous(memory_format=torch.channels_last)


def image_path_columns(mode: str) -> List[str]:
    return ["image_path"] if mode == "face" else ["top_image_path", "bottom_image_path"]


def build_image_cache_for_csv(
    csv_path: str,
    mode: str = "face",
    image_scale: float = DEFAULT_IMAGE_SCALE,
    canvas: Tuple[int, int] = DEFAULT_CANVAS,
    cache_root: str | None = None,
    num_workers: int | None = None,
):
    """Build (or reuse) the fixed-scale image cache for the dataset that *csv_path* belongs to.

    The cache covers every image referenced by any label CSV in the same folder
    (face, validated face, board), so 1-side and 2-side training share one cache.
    """
    csv_path = Path(csv_path).resolve()
    paths = set()
    for sibling in sorted(csv_path.parent.glob("*.csv")):
        columns = pd.read_csv(sibling, nrows=0).columns
        image_columns = [c for c in ("image_path", "top_image_path", "bottom_image_path") if c in columns]
        if image_columns:
            frame = pd.read_csv(sibling, usecols=image_columns)
            paths.update(frame.stack().dropna().astype(str))
    paths.discard("")
    return ensure_image_cache(
        sorted(paths),
        scale=image_scale,
        max_height=canvas[0],
        max_width=canvas[1],
        cache_root=cache_root,
        name_hint=csv_path.parent.parent.name,
        num_workers=num_workers,
    )


def subsample_boards(df: pd.DataFrame, fraction: float, seed: int = 42) -> pd.DataFrame:
    """Keep a label-stratified *fraction* of boards (all rows of each kept board)."""
    if fraction >= 1.0:
        return df
    board_labels = df.groupby("board_id")["label_name"].first()
    kept = board_labels.groupby(board_labels, group_keys=False).apply(
        lambda group: group.sample(n=max(1, round(len(group) * fraction)), random_state=seed)
    )
    return df[df["board_id"].isin(kept.index)].reset_index(drop=True)


class FaceClassifier(nn.Module):
    def __init__(self, num_classes: int, encoder_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.backbone = build_resnet_backbone(encoder_name=encoder_name, pretrained=pretrained, in_channels=3)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


TWO_SIDE_FUSIONS = ["shared", "stacked", "channels"]


class DualFaceClassifier(nn.Module):
    """Board-level classifier over the top and bottom faces.

    ``fusion``:
      * ``"shared"`` (default): one 3-channel backbone runs on each face; the two pooled
        feature vectors are concatenated and classified. Can be initialised from a 1-side
        ``FaceClassifier`` checkpoint via ``load_face_backbone``.
      * ``"stacked"``: top and bottom are stacked vertically into one image (3 channels).
      * ``"channels"``: legacy 6-channel input (top RGB + bottom RGB); first conv is random.
    """

    def __init__(self, num_classes: int, encoder_name: str = "resnet18", pretrained: bool = True, fusion: str = "shared"):
        super().__init__()
        if fusion not in TWO_SIDE_FUSIONS:
            raise ValueError(f"Unsupported fusion '{fusion}'. Choose from: {TWO_SIDE_FUSIONS}")
        self.fusion = fusion
        self.backbone = build_resnet_backbone(encoder_name=encoder_name, pretrained=pretrained, in_channels=6 if fusion == "channels" else 3)
        features = self.backbone.fc.in_features
        if fusion == "shared":
            self.backbone.fc = nn.Identity()
            self.head = nn.Linear(2 * features, num_classes)
        else:
            self.backbone.fc = nn.Linear(features, num_classes)

    def forward(self, top: torch.Tensor, bottom: torch.Tensor) -> torch.Tensor:
        if self.fusion == "shared":
            features = self.backbone(torch.cat([top, bottom], dim=0))
            top_features, bottom_features = features.chunk(2, dim=0)
            return self.head(torch.cat([top_features, bottom_features], dim=1))
        if self.fusion == "stacked":
            return self.backbone(torch.cat([top, bottom], dim=2))
        return self.backbone(torch.cat([top, bottom], dim=1))

    def load_face_backbone(self, checkpoint_path: str | Path) -> None:
        """Copy backbone weights (everything except the classifier) from a 1-side checkpoint."""
        state = torch.load(checkpoint_path, map_location="cpu")
        backbone_state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.") and not k.startswith("backbone.fc.")}
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        missing = [k for k in missing if not k.startswith("fc.")]
        if missing or unexpected:
            raise ValueError(f"Checkpoint {checkpoint_path} does not match this backbone: missing={missing[:5]} unexpected={unexpected[:5]}")


def encoder_from_checkpoint(checkpoint_path: str | Path) -> str:
    """Infer resnet18/34/50 from a FaceClassifier state dict."""
    keys = torch.load(checkpoint_path, map_location="cpu").keys()
    if any(k.startswith("backbone.layer1.0.conv3.") for k in keys):
        return "resnet50"
    if any(k.startswith("backbone.layer3.5.") for k in keys):
        return "resnet34"
    return "resnet18"


def build_class_mapping(labels: List[str]) -> Dict[str, int]:
    unique_labels = sorted(set(labels))
    return {label: idx for idx, label in enumerate(unique_labels)}


def load_dataset(csv_path: str, mode: str):
    df = pd.read_csv(csv_path)
    if mode == "face":
        required = ["board_id", "image_path", "grade_label"]
    else:
        required = ["board_id", "top_image_path", "bottom_image_path", "board_grade_label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns for {mode}: {missing}")

    df = df.dropna(subset=["board_id"]).copy()
    df = df[df["board_id"].astype(str).str.strip() != ""]

    if mode == "face":
        df = df.dropna(subset=["image_path", "grade_label"]).copy()
        df["label_name"] = df["grade_label"].astype(str)
    else:
        df = df.dropna(subset=["top_image_path", "bottom_image_path", "board_grade_label"]).copy()
        df["label_name"] = df["board_grade_label"].astype(str)

    return df


def split_board_ids(df: pd.DataFrame, val_split: float = 0.2, seed: int = 42):
    if len(df["board_id"].unique()) < 2:
        raise ValueError("Need at least 2 boards for train/val split.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df["board_id"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def compute_metrics(y_true: List[int], y_pred: List[int], y_prob: np.ndarray | None, labels: List[str]) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["macro_precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["macro_recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["weighted_precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["weighted_recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["per_class_precision"] = precision_score(y_true, y_pred, average=None, labels=list(range(len(labels))), zero_division=0).tolist()
    metrics["per_class_recall"] = recall_score(y_true, y_pred, average=None, labels=list(range(len(labels))), zero_division=0).tolist()
    metrics["per_class_f1"] = f1_score(y_true, y_pred, average=None, labels=list(range(len(labels))), zero_division=0).tolist()
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).tolist()
    metrics["classification_report"] = classification_report(y_true, y_pred, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0)

    if len(labels) > 2 and y_prob is not None:
        y_bin = label_binarize(y_true, classes=list(range(len(labels))))
        # Macro one-vs-rest AUC over the classes present in y_true (rare grades can be absent from a split).
        present = [c for c in range(y_bin.shape[1]) if 0 < y_bin[:, c].sum() < len(y_bin)]
        if present:
            metrics["roc_auc_ovr_macro"] = float(np.mean([roc_auc_score(y_bin[:, c], y_prob[:, c]) for c in present]))
        else:
            metrics["roc_auc_ovr_macro"] = None
    else:
        metrics["roc_auc_ovr_macro"] = None

    return metrics


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, class_names: List[str], is_board_model: bool = False, use_amp: bool = False):
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            if is_board_model:
                top, bottom, targets = batch
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    outputs = model(prepare_images(top, device), prepare_images(bottom, device))
            else:
                images, targets = batch
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    outputs = model(prepare_images(images, device))
            logits = outputs.float()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            targets = targets.cpu().tolist()
            all_targets.extend(targets)
            all_preds.extend(preds)
            all_probs.append(probs)

    prob_array = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, len(class_names)))
    return compute_metrics(all_targets, all_preds, prob_array if prob_array.size else None, class_names)


def release_loaders(*loaders: DataLoader) -> None:
    """Stop persistent workers and free cached pinned host buffers.

    Full-resolution batches are large and WSL2 limits page-locked memory, so buffers
    cached by earlier runs in the same process (e.g. Optuna trials) must be returned.
    """
    for loader in loaders:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
        loader._iterator = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch._C._host_emptyCache()


def train_model(
    csv_path: str,
    output_dir: str,
    mode: str,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    val_split: float = 0.2,
    use_optuna: bool = False,
    optuna_trials: int = 10,
    encoder_name: str = "resnet18",
    pretrained: bool = True,
    checkpoint_name: str | None = None,
    log_name: str | None = None,
    metrics_run_name: str | None = None,
    tensorboard_dir: str | None = None,
    metrics_dir: str | None = None,
    checkpoint_interval: int = 5,
    save_trial_artifacts: bool = True,
    image_size: int = 224,
    num_workers: int = 2,
    cuda_device_index: int = 0,
    use_image_cache: bool = True,
    image_cache_dir: str | None = None,
    subset_fraction: float = 1.0,
    use_amp: bool = True,
    epoch_callback=None,
    image_scale: float = DEFAULT_IMAGE_SCALE,
    canvas: Tuple[int, int] = DEFAULT_CANVAS,
    two_side_fusion: str = "shared",
    init_checkpoint: str | None = None,
):
    """Train a face or board classifier.

    ``use_image_cache`` reads faces resized by the fixed ``image_scale`` and padded onto a
    ``canvas`` (height, width) from a decode-once JPEG cache (see
    ``functions/nhla_image_cache.py``); it's built the first time it's needed. Without the
    cache, images are read from disk and squashed to ``image_size`` x ``image_size``.
    ``two_side_fusion`` / ``init_checkpoint`` configure the board model (see
    ``DualFaceClassifier``); ``init_checkpoint`` is a 1-side ``model_best.pt``.
    ``subset_fraction`` < 1 trains and validates on a label-stratified subset of boards.
    ``epoch_callback(epoch, val_macro_f1)`` is called after every epoch (used for Optuna pruning).
    """
    set_seed(seed)
    device = resolve_device(cuda_device_index=cuda_device_index)
    torch.backends.cudnn.benchmark = True  # fixed input shape -> autotune conv kernels once
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if save_trial_artifacts:
        tensorboard_root = Path(tensorboard_dir) if tensorboard_dir else output_root / "tensorboard"
        tensorboard_root.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = output_root / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metrics_root = Path(metrics_dir) if metrics_dir else output_root / "metrics"
        metrics_root.mkdir(parents=True, exist_ok=True)
        log_dir = tensorboard_root / (log_name or f"{mode}_{seed}")
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        tensorboard_root = None
        checkpoint_dir = None
        metrics_root = None
        log_dir = None

    df = load_dataset(csv_path, mode)
    class_mapping = build_class_mapping(df["label_name"].tolist())
    df["label_index"] = df["label_name"].map(class_mapping)

    if use_image_cache:
        cache_dir, path_to_index = build_image_cache_for_csv(csv_path, mode, image_scale=image_scale, canvas=canvas, cache_root=image_cache_dir)
        index_columns = []
        for column in image_path_columns(mode):
            df[f"{column}_cache_idx"] = df[column].map(path_to_index)
            index_columns.append(f"{column}_cache_idx")
        df = df.dropna(subset=index_columns).reset_index(drop=True)
        df[index_columns] = df[index_columns].astype(np.int64)

    df = subsample_boards(df, subset_fraction, seed=seed)

    train_df, val_df = split_board_ids(df, val_split=val_split, seed=seed)
    class_names = [k for k, _ in sorted(class_mapping.items(), key=lambda x: x[1])]

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # No pinning: WSL2 caps page-locked memory at ~1 GB, which batch 32 x workers x prefetch
    # at this canvas overflows. Unpinned uint8 copies cost ~10 ms/batch, negligible vs. compute.
    pin_memory = False
    persistent_workers = num_workers > 0
    use_amp = use_amp and device.type == "cuda"

    def make_dataset(frame: pd.DataFrame) -> Dataset:
        if use_image_cache:
            indices = frame[index_columns].to_numpy()
            return CachedImageDataset(cache_dir, indices[:, 0] if mode == "face" else indices, frame["label_index"].to_numpy(), canvas=canvas)
        return FaceDataset(frame, transform=transform) if mode == "face" else BoardDataset(frame, transform=transform)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    train_loader = DataLoader(make_dataset(train_df), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(make_dataset(val_df), shuffle=False, **loader_kwargs)
    if mode == "face":
        model = FaceClassifier(len(class_names), encoder_name=encoder_name, pretrained=pretrained)
    else:
        model = DualFaceClassifier(len(class_names), encoder_name=encoder_name, pretrained=pretrained, fusion=two_side_fusion)
        if init_checkpoint:
            model.load_face_backbone(init_checkpoint)
    model = model.to(device).to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    writer = SummaryWriter(log_dir=str(log_dir)) if save_trial_artifacts else None

    best_val_f1 = -1.0
    best_state = None
    hist = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_macro_f1": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            targets = batch[-1].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(*(prepare_images(images, device) for images in batch[:-1]))
                loss = criterion(outputs.float(), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)

        train_loss /= len(train_df)
        model.eval()
        val_loss = 0.0
        val_targets = []
        val_preds = []
        val_probs = []

        with torch.no_grad():
            for batch in val_loader:
                targets = batch[-1].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    outputs = model(*(prepare_images(images, device) for images in batch[:-1]))
                outputs = outputs.float()
                loss = criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().tolist()
                val_targets.extend(targets.cpu().tolist())
                val_preds.extend(preds)
                val_probs.append(probs)

        val_loss /= len(val_df)
        val_probs = np.concatenate(val_probs, axis=0)
        metrics = compute_metrics(val_targets, val_preds, val_probs, class_names)

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["val_accuracy"].append(metrics["accuracy"])
        hist["val_macro_f1"].append(metrics["macro_f1"])

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/accuracy", metrics["accuracy"], epoch)
            writer.add_scalar("val/balanced_accuracy", metrics["balanced_accuracy"], epoch)
            writer.add_scalar("val/macro_f1", metrics["macro_f1"], epoch)
            writer.add_scalar("val/macro_precision", metrics["macro_precision"], epoch)
            writer.add_scalar("val/macro_recall", metrics["macro_recall"], epoch)

        if metrics["macro_f1"] > best_val_f1:
            best_val_f1 = metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if save_trial_artifacts and checkpoint_dir is not None and epoch % checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            torch.save({key: value.cpu().clone() for key, value in model.state_dict().items()}, checkpoint_path)

        if save_trial_artifacts and checkpoint_dir is not None and best_state is not None and metrics["macro_f1"] == best_val_f1:
            torch.save({k: v.cpu().clone() for k, v in best_state.items()}, output_root / "model_best.pt")

        scheduler.step()

        if epoch_callback is not None:
            try:
                epoch_callback(epoch, metrics["macro_f1"])
            except BaseException:  # e.g. optuna.TrialPruned
                if writer is not None:
                    writer.close()
                release_loaders(train_loader, val_loader)
                raise

    if writer is not None:
        writer.flush()
        writer.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    eval_metrics = evaluate_model(model, val_loader, device, class_names, is_board_model=(mode == "board"), use_amp=use_amp)
    release_loaders(train_loader, val_loader)

    if save_trial_artifacts:
        torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()}, output_root / "model_best.pt")
        with (metrics_root / "class_mapping.json").open("w") as f:
            json.dump(class_mapping, f, indent=2)

        metrics_json_path = metrics_root / (f"{metrics_run_name or mode}_metrics.json" if metrics_run_name else f"{mode}_metrics.json")
        with metrics_json_path.open("w") as f:
            json.dump(eval_metrics, f, indent=2)

        history_path = metrics_root / (f"{metrics_run_name or mode}_history.csv" if metrics_run_name else f"{mode}_history.csv")
        pd.DataFrame(hist).to_csv(history_path, index=False)

        if len(class_names) > 0:
            conf_matrix = np.array(eval_metrics["confusion_matrix"], dtype=int)
            plt.figure(figsize=(8, 6))
            plt.imshow(conf_matrix, cmap="Blues")
            plt.title(f"{mode.title()} Confusion Matrix")
            plt.colorbar()
            tick_labels = [str(label) for label in class_names]
            plt.xticks(range(len(tick_labels)), tick_labels, rotation=45)
            plt.yticks(range(len(tick_labels)), tick_labels)
            for row_idx in range(conf_matrix.shape[0]):
                for col_idx in range(conf_matrix.shape[1]):
                    plt.text(col_idx, row_idx, str(conf_matrix[row_idx, col_idx]), ha="center", va="center", color="black")
            plt.tight_layout()
            plt.savefig(metrics_root / (f"{metrics_run_name or mode}_confusion_matrix.jpg" if metrics_run_name else f"{mode}_confusion_matrix.jpg"), dpi=200)
            plt.close()

        return {"class_mapping": class_mapping, "metrics": eval_metrics, "history": hist, "metrics_path": str(metrics_json_path)}

    return {"class_mapping": class_mapping, "metrics": eval_metrics, "history": hist, "metrics_path": None}


def objective(
    trial,
    csv_path: str,
    mode: str,
    output_root: str,
    epochs: int = 8,
    seed: int = 42,
    val_split: float = 0.2,
    pretrained: bool = True,
    search_space: dict | None = None,
    image_size: int = 224,
    num_workers: int = 2,
    cuda_device_index: int = 0,
    subset_fraction: float = 1.0,
    image_cache_dir: str | None = None,
    image_scale: float = DEFAULT_IMAGE_SCALE,
    canvas: Tuple[int, int] = DEFAULT_CANVAS,
    two_side_fusion: str = "shared",
    init_checkpoint: str | None = None,
):
    if search_space is None:
        search_space = {
            "lr": {"low": 1e-5, "high": 1e-2, "log": True},
            "batch_size": [4, 8, 16, 32],
            "weight_decay": {"low": 1e-6, "high": 1e-2, "log": True},
            "encoder_name": SUPPORTED_ENCODERS,
        }

    params = {}
    if init_checkpoint:
        # The encoder is fixed by the 1-side checkpoint the board model starts from.
        search_space = {k: v for k, v in search_space.items() if k != "encoder_name"}
        params["encoder_name"] = encoder_from_checkpoint(init_checkpoint)
    for key, spec in search_space.items():
        if key == "lr":
            params[key] = trial.suggest_float("lr", float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
        elif key == "batch_size":
            params[key] = trial.suggest_categorical("batch_size", list(spec))
        elif key == "weight_decay":
            params[key] = trial.suggest_float("weight_decay", float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
        elif key == "encoder_name":
            params[key] = trial.suggest_categorical("encoder_name", list(spec))
        else:
            raise ValueError(f"Unsupported Optuna search key: {key}")

    def report_and_prune(epoch: int, macro_f1: float) -> None:
        trial.report(macro_f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    run_dir = Path(output_root) / f"trial_{trial.number}"
    result = train_model(
        csv_path=csv_path,
        output_dir=str(run_dir),
        mode=mode,
        epochs=epochs,
        batch_size=int(params["batch_size"]),
        learning_rate=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
        seed=seed,
        val_split=val_split,
        use_optuna=True,
        optuna_trials=1,
        encoder_name=params["encoder_name"],
        pretrained=pretrained,
        tensorboard_dir=None,
        metrics_dir=None,
        checkpoint_interval=epochs,
        save_trial_artifacts=False,
        image_size=image_size,
        num_workers=num_workers,
        cuda_device_index=cuda_device_index,
        image_cache_dir=image_cache_dir,
        subset_fraction=subset_fraction,
        epoch_callback=report_and_prune,
        image_scale=image_scale,
        canvas=canvas,
        two_side_fusion=two_side_fusion,
        init_checkpoint=init_checkpoint,
    )
    trial.set_user_attr("macro_f1", result["metrics"]["macro_f1"])
    trial.set_user_attr("accuracy", result["metrics"]["accuracy"])
    trial.set_user_attr("encoder_name", params["encoder_name"])
    return float(result["metrics"]["macro_f1"])


def run_optuna_tuning(
    csv_path: str,
    mode: str,
    output_root: str,
    n_trials: int = 10,
    epochs: int = 8,
    seed: int = 42,
    val_split: float = 0.2,
    pretrained: bool = True,
    search_space: dict | None = None,
    study_name: str | None = None,
    db_path: str | None = None,
    image_size: int = 224,
    num_workers: int = 2,
    cuda_device_index: int = 0,
    subset_fraction: float = 1.0,
    image_cache_dir: str | None = None,
    image_scale: float = DEFAULT_IMAGE_SCALE,
    canvas: Tuple[int, int] = DEFAULT_CANVAS,
    two_side_fusion: str = "shared",
    init_checkpoint: str | None = None,
):
    """Tune with Optuna; ``subset_fraction`` < 1 tunes on a stratified subset of boards for speed."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    study_name = study_name or f"nhla_{mode}_tuning_{datetime.now():%Y%m%d_%H%M%S}"
    storage = f"sqlite:///{db_path}" if db_path else None
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(
            trial,
            csv_path,
            mode,
            str(output_root),
            epochs=epochs,
            seed=seed,
            val_split=val_split,
            pretrained=pretrained,
            search_space=search_space,
            image_size=image_size,
            num_workers=num_workers,
            cuda_device_index=cuda_device_index,
            subset_fraction=subset_fraction,
            image_cache_dir=image_cache_dir,
            image_scale=image_scale,
            canvas=canvas,
            two_side_fusion=two_side_fusion,
            init_checkpoint=init_checkpoint,
        ),
        n_trials=n_trials,
        gc_after_trial=True,
    )

    best_params = dict(study.best_params)
    if init_checkpoint:
        best_params["encoder_name"] = encoder_from_checkpoint(init_checkpoint)
    best_params_path = output_root / "best_params.json"
    with best_params_path.open("w") as fh:
        json.dump(best_params, fh, indent=2)

    print(f"Optuna tuning complete for {mode} mode. Best params: {best_params}")
    return study, best_params


def main():
    parser = argparse.ArgumentParser(description="Train NHLA classification models with TensorBoard logging and optional Optuna tuning.")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to nhla_face_labels.csv or nhla_board_labels.csv")
    parser.add_argument("--mode", type=str, default="face", choices=["face", "board"], help="Classification mode")
    parser.add_argument("--output-dir", type=str, default="models/nhla_classification", help="Where to save model and logs")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split proportion")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--optuna-trials", type=int, default=0, help="Optional Optuna trials. Set to 0 to disable.")
    parser.add_argument("--encoder-name", type=str, default="resnet18", choices=SUPPORTED_ENCODERS, help="Backbone encoder to use for the classifier")
    parser.add_argument("--pretrained", dest="pretrained", action=argparse.BooleanOptionalAction, default=True, help="Use pretrained ImageNet weights")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size (square)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--cuda-device-index", type=int, default=0, help="CUDA device index to use (0, 1, ...) when CUDA is available")
    args = parser.parse_args()

    if args.optuna_trials > 0:
        study, best_params = run_optuna_tuning(
            csv_path=args.csv_path,
            mode=args.mode,
            output_root=args.output_dir,
            n_trials=args.optuna_trials,
            epochs=min(8, max(2, args.epochs // 2)),
            seed=args.seed,
            val_split=args.val_split,
            pretrained=args.pretrained,
            image_size=args.image_size,
            num_workers=args.num_workers,
            cuda_device_index=args.cuda_device_index,
        )
        print("Best Optuna params:", best_params)
        args.batch_size = int(best_params["batch_size"])
        args.lr = float(best_params["lr"])
        args.weight_decay = float(best_params["weight_decay"])
        args.encoder_name = best_params["encoder_name"]

    result = train_model(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        val_split=args.val_split,
        use_optuna=args.optuna_trials > 0,
        optuna_trials=args.optuna_trials,
        encoder_name=args.encoder_name,
        pretrained=args.pretrained,
        image_size=args.image_size,
        num_workers=args.num_workers,
        cuda_device_index=args.cuda_device_index,
    )
    print(json.dumps({"mode": args.mode, "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
