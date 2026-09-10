# -*- coding: utf-8 -*-
"""SegFormer Training.ipynb
"""

# =============================================================================
# STEP 0: INSTALLING THE LIBRARY
# =============================================================================
print("Installing the library Hugging Face (transformers, evaluate) dan accelerate...")
# !pip install transformers accelerate evaluate "evaluate[image]"

# =============================================================================
# STEP 1: IMPORT THE LIBRARY
# =============================================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
from google.colab import drive

from transformers import AutoModelForSemanticSegmentation, AutoConfig
import evaluate
# ------------------------------------

# Mount Google Drive
print("Mounting Google Drive...")
try:
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")
except Exception as e:
    print(f"Error mounting Google Drive: {e}")

# =====================================================================================
# 2. DATASET LOADER
# =====================================================================================

class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

        mask[mask == 255] = 1

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask.long()

# =====================================================================================
# 3. UTILITY FUNCTION
# =====================================================================================

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, batch_size, train_transform, val_transform, num_workers=4, pin_memory=True):
    train_ds = RoadDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_ds = RoadDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, val_loader

# --- Mean IoU Evaluation ---
def check_accuracy(loader, model, metric, num_classes, device="cuda"):
    """
    Calculating Mean IoU (mIoU) using the `evaluate` library.
    """
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating"):
            x = x.to(device)
            y = y.to(device) # (N, H, W), torch.long

            # Forward pass
            outputs = model(pixel_values=x)
            logits = outputs.logits # (N, num_classes, H_out, W_out)

            # Upsample logits ke ukuran label (mask)
            # The SegFormer model outputs logits at H/4, W/4
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=y.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            preds = torch.argmax(upsampled_logits, dim=1) # (N, H, W)
            metric.add_batch(predictions=preds, references=y)

    model.train()

    try:
        results = metric.compute(
            num_labels=num_classes,
            ignore_index=255,
            reduce_labels=False 
        )
        print(f"\nValidation mIoU: {results['mean_iou']:.4f}")
        print(f"Validation Acc: {results['mean_accuracy']:.4f}")
        print(f"IoU Background: {results['per_category_iou'][0]:.4f}")
        print(f"IoU Road: {results['per_category_iou'][1]:.4f}\n")
        return results['mean_iou']
    except Exception as e:
        print(f"Error computing metric: {e}")
        return 0

# --- ArgMax function ---
def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    os.makedirs(folder, exist_ok=True)

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device) # (N, H, W)

        with torch.no_grad():
            outputs = model(pixel_values=x)
            logits = outputs.logits

            # Upsample logits
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=y.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            preds = torch.argmax(upsampled_logits, dim=1) # (N, H, W)

        torchvision.utils.save_image(
            preds.float().unsqueeze(1) * 255, # (N, 1, H, W)
            f"{folder}/pred_{idx}.png"
        )

        torchvision.utils.save_image(
            y.float().unsqueeze(1) * 255, # (N, 1, H, W)
            f"{folder}/target_{idx}.png"
        )

        if idx >= 0:
            break

    model.train()

# =====================================================================================
# 4. TRAINING OBJECTIVES
# =====================================================================================

def train_fn(loader, model, optimizer, scaler, device):
    loop = tqdm(loader, desc="Training")
    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device) # (N, H, W), torch.long

        with torch.amp.autocast("cuda"):
            outputs = model(pixel_values=data, labels=targets)
            loss = outputs.loss
            # =========================================================

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())

# =====================================================================================
# 5. MAIN FUNCTIONS
# =====================================================================================

def main():
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8 
    NUM_EPOCHS = 25
    NUM_WORKERS = 2
    IMAGE_HEIGHT = 750
    IMAGE_WIDTH = 750
    PIN_MEMORY = True
    LOAD_MODEL = False
    NUM_CLASSES = 2 # 0: background, 1: road

    # --- PATH SETTINGS ---\n
    BASE_DATASET_DIR = "/content/drive/MyDrive/Road Detection_Data_/Dataset/750 Dataset 3"
    TRAIN_IMG_DIR = os.path.join(BASE_DATASET_DIR, "train_images")
    TRAIN_MASK_DIR = os.path.join(BASE_DATASET_DIR, "train_masks")
    VAL_IMG_DIR = os.path.join(BASE_DATASET_DIR, "valid_images")
    VAL_MASK_DIR = os.path.join(BASE_DATASET_DIR, "valid_masks")

    MODEL_OUTPUT_DIR = "/content/drive/MyDrive/Road Detection_Data_/Model"
    MODEL_FILENAME = "segformer_b0_roaddetection_1.pth.tar"
    MODEL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME)

    SAVED_IMAGES_DIR = "/content/drive/MyDrive/Road Detection_Data_/Hasil Prediksi/SegFormer Test"

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAVED_IMAGES_DIR, exist_ok=True)

    # Data augmentation
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize( 
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    print("=> Loading the pre-trained SegFormer (B0) model...")

    model_id = "nvidia/segformer-b0-finetuned-ade-512-512"

    config = AutoConfig.from_pretrained(model_id, num_labels=NUM_CLASSES)

    # Load model
    model = AutoModelForSemanticSegmentation.from_pretrained(
        model_id,
        config=config,
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    print(f"The SegFormer model is pre-trained on {NUM_CLASSES} classes.")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- Initialisation of Metrics (Mean IoU) ---
    metric = evaluate.load("mean_iou")
    # ------------------------------------
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(MODEL_SAVE_PATH), model)

    print("--- Starting the Initial Accuracy Check (mIoU) ---")
    check_accuracy(val_loader, model, metric, NUM_CLASSES, device=DEVICE)

    scaler = torch.amp.GradScaler("cuda")
    best_iou = 0

    # Loop training
    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

        metric = evaluate.load("mean_iou")
        train_fn(train_loader, model, optimizer, scaler, device=DEVICE)
        current_iou = check_accuracy(val_loader, model, metric, NUM_CLASSES, device=DEVICE)
        if current_iou > best_iou:
            best_iou = current_iou
            print(f"==> mIoU has improved! Saving the model... (mIoU: {best_iou:.4f})")
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=MODEL_SAVE_PATH)

            save_predictions_as_imgs(
                val_loader, model, folder=SAVED_IMAGES_DIR, device=DEVICE
            )

    print("--- Training finished ---")
    print(f"The best models are stored in: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()

# --- Disconnect runtime ---
try:
    from google.colab import runtime
    print("\nThe training has finished. Disconnecting the runtime connection.")
    runtime.unassign()
except ImportError:
    print("\nTraining done.")