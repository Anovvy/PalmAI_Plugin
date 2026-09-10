# -*- coding: utf-8 -*-
"""DeepLabV3+ Training.ipynb
"""

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
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
# =====================================================================================
# 1. MOUNT GOOGLE DRIVE
# =====================================================================================
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print("Not within the Google Colab environment. Bypassing GDrive mounting.")

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
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0 

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

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

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds_logits = model(x)['out']

            preds = torch.sigmoid(preds_logits)
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()

    os.makedirs(folder, exist_ok=True)

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():

            preds_logits = model(x)['out']


            preds = torch.sigmoid(preds_logits)
            preds = (preds > 0.5).float()

            for i in range(x.shape[0]):
                img_idx = idx * loader.batch_size + i
                torchvision.utils.save_image(preds[i], f"{folder}/pred_{img_idx}.png")
                torchvision.utils.save_image(y[i].unsqueeze(0), f"{folder}/target_{img_idx}.png")
        break

    model.train()

# =====================================================================================
# 4. TRAINING OBJECTIVES
# =====================================================================================

def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    loop = tqdm(loader)
    model.train()

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        # Forward
        with torch.amp.autocast("cuda"):

            predictions = model(data)
            loss_main = loss_fn(predictions['out'], targets)
            loss_aux = loss_fn(predictions['aux'], targets)
            loss = loss_main + 0.4 * loss_aux

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

# =====================================================================================
# 5. MAIN FUNCTIONS
# =====================================================================================

def main():

    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 8

    NUM_EPOCHS = 25
    NUM_WORKERS = 3

    IMAGE_HEIGHT = 750
    IMAGE_WIDTH = 750

    PIN_MEMORY = True
    LOAD_MODEL = False


    BASE_DATASET_DIR = "/content/drive/MyDrive/Road Detection_Data_/Dataset/750 Dataset 3"

    TRAIN_IMG_DIR = os.path.join(BASE_DATASET_DIR, "train_images")
    TRAIN_MASK_DIR = os.path.join(BASE_DATASET_DIR, "train_masks")
    VAL_IMG_DIR = os.path.join(BASE_DATASET_DIR, "valid_images")
    VAL_MASK_DIR = os.path.join(BASE_DATASET_DIR, "valid_masks")
    TEST_IMG_DIR = os.path.join(BASE_DATASET_DIR, "test_images")
    TEST_MASK_DIR = os.path.join(BASE_DATASET_DIR, "test_masks")

    MODEL_OUTPUT_DIR = "/content/drive/MyDrive/Road Detection_Data_/Model"
    MODEL_FILENAME = "deeplabv3+_roaddetection_1.pth.tar"
    MODEL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME)

    SAVED_IMAGES_DIR = "/content/drive/MyDrive/Road Detection_Data_/Hasil Prediksi/DeepLabV3+ Test"

    if not os.path.exists(MODEL_OUTPUT_DIR):
        os.makedirs(MODEL_OUTPUT_DIR)

    if not os.path.exists(SAVED_IMAGES_DIR):
        os.makedirs(SAVED_IMAGES_DIR)


    # ----------------------------------------------------------------

    # Data augmentation
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # --- Initialising the DeepLabV3+ Model ---
    print("=> Memuat model DeepLabV3+ ResNet50 (pretrained)...")
    # 1. Load the pre-trained model (trained on COCO)
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)

    # 2. Classifier modification
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

    model = model.to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss() # Binary Cross Entropy with Logits
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Loading data
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

    print("--- Starting the Initial Accuracy Check ---")
    check_accuracy(val_loader, model, device=DEVICE)

    # --- Use torch.amp.GradScaler("cuda") ---
    # This fixes the FutureWarning
    scaler = torch.amp.GradScaler("cuda")

    # Loop training
    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, device=DEVICE)

        check_accuracy(val_loader, model, device=DEVICE)

        save_predictions_as_imgs(
            val_loader, model, folder=SAVED_IMAGES_DIR, device=DEVICE
        )

    # Save the model once all epochs have completed
    print("--- Training finished. Saving final checkpoint ---")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, filename=MODEL_SAVE_PATH)

# =====================================================================================
# 6. ENTRY POINT
# =====================================================================================

if __name__ == "__main__":
    main()

    try:
        from google.colab import runtime
        print("Disconnecting runtime.")
        runtime.unassign()
    except ImportError:
        print("Not in the Google Colab environment. Done.")

