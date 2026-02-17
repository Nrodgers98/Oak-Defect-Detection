from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import json
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PatchSegmentationDataset(Dataset):
    """
    Patch-based segmentation dataset:
      - train: returns random patches (patches_per_image per image)
      - val: returns tiled patches covering each image (stride controls overlap)
    """
    def __init__(self, data_root, split='train', transform=None, patch_size=512, patches_per_image=4, stride=None):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.patch_size = int(patch_size)
        self.patches_per_image = int(patches_per_image)
        self.stride = stride if stride is not None else self.patch_size  # non-overlap by default

        split_file = self.data_root / "split.json"
        with open(split_file, 'r') as f:
            split_info = json.load(f)
        self.image_ids = split_info[split]

        self.images_dir = self.data_root / "images"
        self.masks_dir = self.data_root / "combined_masks"

        # For validation, precompute tiles (list of (img_id, y, x))
        if self.split == 'val':
            self.tiles = []
            for img_id in self.image_ids:
                img_path = self.images_dir / f"{img_id}_Col.tif"
                img = np.array(Image.open(img_path).convert('RGB'))
                H, W = img.shape[:2]
                stride = self.stride
                for y in range(0, max(1, H - self.patch_size + 1), stride):
                    for x in range(0, max(1, W - self.patch_size + 1), stride):
                        self.tiles.append((img_id, y, x))
                # ensure coverage of right/bottom edges
                if (H - self.patch_size) % stride != 0:
                    y = max(0, H - self.patch_size)
                    for x in range(0, max(1, W - self.patch_size + 1), stride):
                        self.tiles.append((img_id, y, x))
                if (W - self.patch_size) % stride != 0:
                    x = max(0, W - self.patch_size)
                    for y in range(0, max(1, H - self.patch_size + 1), stride):
                        self.tiles.append((img_id, y, x))
                    # corner
                    if (H - self.patch_size) % stride != 0:
                        self.tiles.append((img_id, max(0, H - self.patch_size), max(0, W - self.patch_size)))

        print(f"Loaded {split} dataset: {len(self.image_ids)} images; patch_size={self.patch_size}")

    def __len__(self):
        if self.split == 'train':
            return len(self.image_ids) * self.patches_per_image
        else:
            return len(self.tiles)

    def _load_image_and_mask(self, img_id):
        img_path = self.images_dir / f"{img_id}_Col.tif"
        mask_path = self.masks_dir / f"{img_id}_mask.png"
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path))
        return image, mask

    def _pad_if_needed(self, arr, target_h, target_w):
        h, w = arr.shape[:2]
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        if pad_h == 0 and pad_w == 0:
            return arr
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        if arr.ndim == 3:
            return cv2.copyMakeBorder(arr, top, bottom, left, right, borderType=cv2.BORDER_REFLECT)
        else:
            return cv2.copyMakeBorder(arr, top, bottom, left, right, borderType=cv2.BORDER_REFLECT)

    def __getitem__(self, idx):
        if self.split == 'train':
            img_idx = idx // self.patches_per_image
            img_id = self.image_ids[img_idx]
            image, mask = self._load_image_and_mask(img_id)
            H, W = image.shape[:2]
            # pad if smaller than patch
            if H < self.patch_size or W < self.patch_size:
                image = self._pad_if_needed(image, self.patch_size, self.patch_size)
                mask = self._pad_if_needed(mask, self.patch_size, self.patch_size)
                H, W = image.shape[:2]
            y = np.random.randint(0, H - self.patch_size + 1)
            x = np.random.randint(0, W - self.patch_size + 1)
            img_patch = image[y:y+self.patch_size, x:x+self.patch_size]
            mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
        else:
            img_id, y, x = self.tiles[idx]
            image, mask = self._load_image_and_mask(img_id)
            H, W = image.shape[:2]
            # pad to ensure tile available
            if H < self.patch_size or W < self.patch_size:
                image = self._pad_if_needed(image, self.patch_size, self.patch_size)
                mask = self._pad_if_needed(mask, self.patch_size, self.patch_size)
            img_patch = image[y:y+self.patch_size, x:x+self.patch_size]
            mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]

        if self.transform:
            transformed = self.transform(image=img_patch, mask=mask_patch)
            img_patch = transformed['image']
            mask_patch = transformed['mask']

        return img_patch, mask_patch.long()

# Patch-aware augmentations (do NOT include Resize)
def get_training_augmentation_patches():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.OneOf([A.RandomBrightnessContrast(p=1), A.RandomGamma(p=1)], p=0.3),
        A.OneOf([A.GaussNoise(p=1), A.GaussianBlur(p=1)], p=0.2),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])

def get_validation_augmentation_patches():
    return A.Compose([
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])

def create_dataloaders_patch(data_root, batch_size=4, patch_size=224, patches_per_image=4, stride=None, num_workers=4):
    train_dataset = PatchSegmentationDataset(
        data_root=data_root, split='train',
        transform=get_training_augmentation_patches(),
        patch_size=patch_size, patches_per_image=patches_per_image, stride=stride
    )
    val_dataset = PatchSegmentationDataset(
        data_root=data_root, split='val',
        transform=get_validation_augmentation_patches(),
        patch_size=patch_size, patches_per_image=1, stride=stride
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader