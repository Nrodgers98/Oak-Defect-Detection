from __future__ import annotations

import argparse
import json
import os
import random
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
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

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


class FaceClassifier(nn.Module):
    def __init__(self, num_classes: int, encoder_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.backbone = build_resnet_backbone(encoder_name=encoder_name, pretrained=pretrained, in_channels=3)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class DualFaceClassifier(nn.Module):
    def __init__(self, num_classes: int, encoder_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.backbone = build_resnet_backbone(encoder_name=encoder_name, pretrained=pretrained, in_channels=6)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, top: torch.Tensor, bottom: torch.Tensor) -> torch.Tensor:
        stacked = torch.cat([top, bottom], dim=1)
        return self.backbone(stacked)


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
        if y_bin.shape[1] == 1:
            metrics["roc_auc_ovr_macro"] = None
        else:
            metrics["roc_auc_ovr_macro"] = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    else:
        metrics["roc_auc_ovr_macro"] = None

    return metrics


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, class_names: List[str], is_board_model: bool = False):
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            if is_board_model:
                top, bottom, targets = batch
                top = top.to(device)
                bottom = bottom.to(device)
                outputs = model(top, bottom)
            else:
                images, targets = batch
                images = images.to(device)
                outputs = model(images)
            logits = outputs
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            targets = targets.cpu().tolist()
            all_targets.extend(targets)
            all_preds.extend(preds)
            all_probs.append(probs)

    prob_array = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, len(class_names)))
    return compute_metrics(all_targets, all_preds, prob_array if prob_array.size else None, class_names)


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
):
    set_seed(seed)
    device = resolve_device(cuda_device_index=cuda_device_index)
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

    train_df, val_df = split_board_ids(df, val_split=val_split, seed=seed)
    class_names = [k for k, _ in sorted(class_mapping.items(), key=lambda x: x[1])]

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    if mode == "face":
        train_loader = DataLoader(
            FaceDataset(train_df, transform=transform),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        val_loader = DataLoader(
            FaceDataset(val_df, transform=transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        model = FaceClassifier(len(class_names), encoder_name=encoder_name, pretrained=pretrained).to(device)
    else:
        train_loader = DataLoader(
            BoardDataset(train_df, transform=transform),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        val_loader = DataLoader(
            BoardDataset(val_df, transform=transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        model = DualFaceClassifier(len(class_names), encoder_name=encoder_name, pretrained=pretrained).to(device)

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
            if mode == "face":
                images, targets = batch
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
            else:
                top, bottom, targets = batch
                top, bottom, targets = top.to(device), bottom.to(device), targets.to(device)
                outputs = model(top, bottom)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
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
                if mode == "face":
                    images, targets = batch
                    images, targets = images.to(device), targets.to(device)
                    outputs = model(images)
                else:
                    top, bottom, targets = batch
                    top, bottom, targets = top.to(device), bottom.to(device), targets.to(device)
                    outputs = model(top, bottom)

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

    if writer is not None:
        writer.flush()
        writer.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    eval_metrics = evaluate_model(model, val_loader, device, class_names, is_board_model=(mode == "board"))

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
):
    if search_space is None:
        search_space = {
            "lr": {"low": 1e-5, "high": 1e-2, "log": True},
            "batch_size": [4, 8, 16, 32],
            "weight_decay": {"low": 1e-6, "high": 1e-2, "log": True},
            "encoder_name": SUPPORTED_ENCODERS,
        }

    params = {}
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
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    study_name = study_name or f"nhla_{mode}_tuning_{datetime.now():%Y%m%d_%H%M%S}"
    storage = f"sqlite:///{db_path}" if db_path else None
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(),
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
        ),
        n_trials=n_trials,
    )

    best_params = dict(study.best_params)
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
