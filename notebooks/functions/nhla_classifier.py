from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


DEFAULT_IMAGE_SIZE = (224, 224)


class FaceDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform: transforms.Compose | None = None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform or default_transform()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = int(row["label_index"])
        return image, label


class BoardDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform: transforms.Compose | None = None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform or default_transform()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        row = self.df.iloc[idx]
        top_path = row["top_image_path"]
        bottom_path = row["bottom_image_path"]

        top = self.transform(Image.open(top_path).convert("RGB"))
        bottom = self.transform(Image.open(bottom_path).convert("RGB"))
        label = int(row["label_index"])
        return top, bottom, label


def default_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(DEFAULT_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class FaceClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class DualFaceClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, top: torch.Tensor, bottom: torch.Tensor) -> torch.Tensor:
        stacked = torch.cat([top, bottom], dim=1)
        return self.backbone(stacked)


def resolve_csv(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "face":
        required = ["board_id", "face", "image_path", "grade_label"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Face CSV missing required columns: {missing}")
        df = df.dropna(subset=["image_path", "grade_label"]).copy()
        df["label_name"] = df["grade_label"].astype(str)
        return df

    required = ["board_id", "top_image_path", "bottom_image_path", "board_grade_label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Board CSV missing required columns: {missing}")
    df = df.dropna(subset=["top_image_path", "bottom_image_path", "board_grade_label"]).copy()
    df["label_name"] = df["board_grade_label"].astype(str)
    return df


def build_label_mapping(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted(df["label_name"].unique().tolist())
    mapping = {label: idx for idx, label in enumerate(labels)}
    reverse = {idx: label for label, idx in mapping.items()}
    return mapping, reverse


def split_by_board(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    ss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(ss.split(df, groups=df["board_id"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def train_face_classifier(csv_path: str, output_dir: str, epochs: int = 5, batch_size: int = 8, lr: float = 1e-3, val_size: float = 0.2, seed: int = 42):
    df = pd.read_csv(csv_path)
    df = resolve_csv(df, mode="face")
    if df.empty:
        raise ValueError("No valid rows found in the face CSV.")

    label_mapping, reverse_mapping = build_label_mapping(df)
    df["label_index"] = df["label_name"].map(label_mapping)
    train_df, val_df = split_by_board(df, test_size=val_size, random_state=seed)

    train_ds = FaceDataset(train_df)
    val_ds = FaceDataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = FaceClassifier(len(label_mapping)).to("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        device = next(model.parameters()).device
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(targets.cpu().tolist())

        val_acc = accuracy_score(all_targets, all_preds)
        print(f"Epoch {epoch + 1}/{epochs} - validation accuracy: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "face_classifier.pt")
    with (out / "class_map.json").open("w") as fh:
        json.dump({k: v for k, v in label_mapping.items()}, fh, indent=2)
    with (out / "reverse_class_map.json").open("w") as fh:
        json.dump({str(k): v for k, v in reverse_mapping.items()}, fh, indent=2)

    return {"label_mapping": label_mapping, "reverse_mapping": reverse_mapping, "val_acc": best_val_acc}


def train_board_classifier(csv_path: str, output_dir: str, epochs: int = 5, batch_size: int = 4, lr: float = 1e-3, val_size: float = 0.2, seed: int = 42):
    df = pd.read_csv(csv_path)
    df = resolve_csv(df, mode="board")
    if df.empty:
        raise ValueError("No valid rows found in the board CSV.")

    label_mapping, reverse_mapping = build_label_mapping(df)
    df["label_index"] = df["label_name"].map(label_mapping)
    train_df, val_df = split_by_board(df, test_size=val_size, random_state=seed)

    train_ds = BoardDataset(train_df)
    val_ds = BoardDataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualFaceClassifier(len(label_mapping)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for top_images, bottom_images, targets in train_loader:
            top_images = top_images.to(device)
            bottom_images = bottom_images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(top_images, bottom_images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for top_images, bottom_images, targets in val_loader:
                top_images = top_images.to(device)
                bottom_images = bottom_images.to(device)
                outputs = model(top_images, bottom_images)
                batch_preds = torch.argmax(outputs, dim=1)
                preds.extend(batch_preds.cpu().tolist())
                labels.extend(targets.cpu().tolist())

        val_acc = accuracy_score(labels, preds)
        print(f"Epoch {epoch + 1}/{epochs} - validation accuracy: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "board_classifier.pt")
    with (out / "class_map.json").open("w") as fh:
        json.dump({k: v for k, v in label_mapping.items()}, fh, indent=2)
    with (out / "reverse_class_map.json").open("w") as fh:
        json.dump({str(k): v for k, v in reverse_mapping.items()}, fh, indent=2)

    return {"label_mapping": label_mapping, "reverse_mapping": reverse_mapping, "val_acc": best_acc}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an NHLA grade classifier from board-face or board-level CSVs.")
    parser.add_argument("--csv-path", type=str, required=True, help="CSV file generated by nhla_grade_dataset.py")
    parser.add_argument("--mode", type=str, default="face", choices=["face", "board"], help="Model type to train")
    parser.add_argument("--output-dir", type=str, default="models/nhla_model", help="Where to save the trained model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = _parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if args.mode == "face":
        if "nhla_face_labels" not in csv_path.name:
            print("Warning: face mode is intended for nhla_face_labels.csv")
        report = train_face_classifier(str(csv_path), args.output_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, val_size=args.val_size, seed=args.seed)
    else:
        if "nhla_board_labels" not in csv_path.name:
            print("Warning: board mode is intended for nhla_board_labels.csv")
        report = train_board_classifier(str(csv_path), args.output_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, val_size=args.val_size, seed=args.seed)

    print(json.dumps({"mode": args.mode, "report": report}, indent=2))


if __name__ == "__main__":
    main()
