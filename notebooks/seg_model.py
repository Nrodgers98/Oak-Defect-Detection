# ============================================================
# CONFIGURATION - SET THIS ONCE
# ============================================================

RAW_DATA_FOLDER = r"E:\Zakaria\output scv2"  # Your current folder with all .tif files
ORGANIZED_FOLDER = r"E:\Zakaria\test\segdata"  # Where to save organized data

BATCH_SIZE = 16
IMG_SIZE = 1024
NUM_EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# IMPORTS
# ============================================================

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


# ============================================================
# STEP 1: ANALYZE RAW DATA
# ============================================================

def analyze_raw_data():
    path = Path(RAW_DATA_FOLDER)
    
    stats = {
        'color_images': [],
        'masks_by_class': defaultdict(list),
    }
    
    for file in path.glob("*.tif"):
        filename = file.name
        
        if "_Bin_" in filename:
            class_name = filename.split("_Bin_")[1].replace(".tif", "")
            stats['masks_by_class'][class_name].append(filename)
        elif "_Col.tif" in filename:
            stats['color_images'].append(filename)
    
    print("="*60)
    print("STEP 1: ANALYZING RAW DATA")
    print("="*60)
    print(f"Color images: {len(stats['color_images'])}")
    print(f"\nMask classes:")
    for class_name, files in sorted(stats['masks_by_class'].items()):
        print(f"  {class_name}: {len(files)} masks")
    print("="*60)
    
    return stats


# ============================================================
# STEP 2: REORGANIZE FILES
# ============================================================

def reorganize_data():
    source_path = Path(RAW_DATA_FOLDER)
    output_path = Path(ORGANIZED_FOLDER)
    
    images_dir = output_path / "images"
    masks_dir = output_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {'images': 0, 'masks': defaultdict(int)}
    
    print("\n" + "="*60)
    print("STEP 2: REORGANIZING FILES")
    print("="*60)
    
    for file in source_path.glob("*.tif"):
        filename = file.name
        
        if "_Bin_" in filename:
            class_name = filename.split("_Bin_")[1].replace(".tif", "")
            class_dir = masks_dir / class_name
            class_dir.mkdir(exist_ok=True)
            shutil.copy2(file, class_dir / filename)
            stats['masks'][class_name] += 1
            
        elif "_Col.tif" in filename:
            shutil.copy2(file, images_dir / filename)
            stats['images'] += 1
    
    print(f"✓ Copied {stats['images']} images")
    print(f"✓ Copied masks:")
    for class_name, count in sorted(stats['masks'].items()):
        print(f"    {class_name}: {count}")
    print("="*60)


# ============================================================
# STEP 3: CREATE COMBINED MASKS
# ============================================================

def create_combined_masks():
    base_path = Path(ORGANIZED_FOLDER)
    images_dir = base_path / "images"
    masks_dir = base_path / "masks"
    combined_dir = base_path / "combined_masks"
    combined_dir.mkdir(exist_ok=True)
    
    mask_classes = sorted([d.name for d in masks_dir.iterdir() if d.is_dir()])
    class_mapping = {class_name: idx + 1 for idx, class_name in enumerate(mask_classes)}
    
    print("\n" + "="*60)
    print("STEP 3: CREATING COMBINED MASKS")
    print("="*60)
    print("Class mapping:")
    print("  0: Background")
    for class_name, idx in class_mapping.items():
        print(f"  {idx}: {class_name}")
    
    image_files = sorted(images_dir.glob("*.tif"))
    
    for img_file in tqdm(image_files, desc="Processing"):
        base_name = img_file.stem.replace("_Col", "")
        img = Image.open(img_file)
        width, height = img.size
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        
        for class_name, class_idx in class_mapping.items():
            class_folder = masks_dir / class_name
            mask_files = list(class_folder.glob(f"*{base_name}*_Bin_{class_name}.tif"))
            
            if mask_files:
                mask = np.array(Image.open(mask_files[0]))
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]
                combined_mask[mask > 127] = class_idx
        
        output_file = combined_dir / f"{base_name}_mask.png"
        Image.fromarray(combined_mask).save(output_file)
    
    print(f"✓ Created {len(image_files)} combined masks")
    print("="*60)
    
    return len(class_mapping) + 1  # +1 for background


# ============================================================
# STEP 4: VERIFY & CREATE SPLIT
# ============================================================

def verify_and_split():
    base_path = Path(ORGANIZED_FOLDER)
    combined_dir = base_path / "combined_masks"
    images_dir = base_path / "images"
    
    # Verify classes
    all_classes = set()
    for mask_file in combined_dir.glob("*.png"):
        mask = np.array(Image.open(mask_file))
        all_classes.update(np.unique(mask).tolist())
    
    num_classes = len(all_classes)
    
    print("\n" + "="*60)
    print("STEP 4: VERIFICATION")
    print("="*60)
    print(f"Classes in masks: {sorted(all_classes)}")
    print(f"Total classes: {num_classes}")
    
    # Create split
    image_files = [f.stem.replace("_Col", "") for f in images_dir.glob("*.tif")]
    train_ids, val_ids = train_test_split(image_files, test_size=0.2, random_state=42)
    
    split_info = {'train': train_ids, 'val': val_ids}
    with open(base_path / "split.json", 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"✓ Train: {len(train_ids)} images")
    print(f"✓ Val: {len(val_ids)} images")
    print("="*60)
    
    return num_classes


# ============================================================
# DATASET & TRANSFORMS
# ============================================================

class SegmentationDataset(Dataset):
    def __init__(self, split='train', transform=None):
        self.data_root = Path(ORGANIZED_FOLDER)
        self.transform = transform
        
        with open(self.data_root / "split.json", 'r') as f:
            split_info = json.load(f)
        
        self.image_ids = split_info[split]
        self.images_dir = self.data_root / "images"
        self.masks_dir = self.data_root / "combined_masks"
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        img_path = self.images_dir / f"{img_id}_Col.tif"
        image = np.array(Image.open(img_path).convert('RGB'))
        
        mask_path = self.masks_dir / f"{img_id}_mask.png"
        mask = np.array(Image.open(mask_path))
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return image, mask.long()


def get_train_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# ============================================================
# TRAINER
# ============================================================

class Trainer:
    def __init__(self, model, train_loader, val_loader, num_classes):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        self.best_val_loss = float('inf')
        Path('checkpoints').mkdir(exist_ok=True)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [TRAIN]')
        for images, masks in pbar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [VAL]')
            for images, masks in pbar:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        print("\n" + "="*60)
        print("STEP 5: TRAINING")
        print("="*60)
        print(f"Device: {DEVICE}")
        print(f"Epochs: {NUM_EPOCHS}")
        print(f"Classes: {self.num_classes}")
        print("="*60)
        
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            self.scheduler.step(val_loss)
            
            print(f"\nEpoch {epoch}/{NUM_EPOCHS}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'checkpoints/best.pth')
                print(f"  ✓ Saved best model")
            
            print("-"*60)
        
        print("\n" + "="*60)
        print(f"TRAINING COMPLETE - Best Loss: {self.best_val_loss:.4f}")
        print("="*60)


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Step 1: Analyze
    analyze_raw_data()
    
    # Step 2: Reorganize
    reorganize_data()
    
    # Step 3: Create combined masks
    num_classes = create_combined_masks()
    
    # Step 4: Verify and split
    num_classes = verify_and_split()
    
    # Step 5: Prepare dataloaders
    train_dataset = SegmentationDataset('train', get_train_transform())
    val_dataset = SegmentationDataset('val', get_val_transform())
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    
    # Step 6: Create model
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=3,
        classes=num_classes
    )
    
    # Step 7: Train
    trainer = Trainer(model, train_loader, val_loader, num_classes)
    trainer.train()