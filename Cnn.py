"""
brush_cnn.py
PyTorch + DirectML CNN for reverse-engineering Photoshop brush parameters.
Predicts: size (px), hardness (%), spacing (%), roundness (%)
Input:    grayscale PNG (any resolution, resized to 224x224 on load)
"""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
print(torch.cuda.is_available())

DEVICE = torch.device("cuda")

# ============================================================
# CONSTANTS
# ============================================================

IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 50
LR         = 1e-4
VAL_SPLIT  = 0.2
PATIENCE   = 8

NORM = {
    "size":      100.0,
    "hardness":  100.0,
    "roundness": 100.0,
    "spacing":   200.0,
}

# ============================================================
# DATASET INSPECTION
# ============================================================

def inspect_dataset(data_dir):
    pattern = re.compile(r"brush_s(\d+)_h(\d+)_r(\d+)_sp(\d+)\.png")
    sizes, hardnesses, roundnesses, spacings = [], [], [], []

    for fname in os.listdir(data_dir):
        if fname.startswith("._"):
            continue
        match = pattern.match(fname)
        if not match:
            continue
        sizes.append(int(match.group(1)))
        hardnesses.append(int(match.group(2)))
        roundnesses.append(int(match.group(3)))
        spacings.append(int(match.group(4)))

    if not sizes:
        print("No valid samples found in dataset.")
        return

    print(f"Total samples: {len(sizes)}")
    print(f"Size      — min: {min(sizes)}  max: {max(sizes)}  unique: {len(set(sizes))}")
    print(f"Hardness  — min: {min(hardnesses)}  max: {max(hardnesses)}  unique: {len(set(hardnesses))}")
    print(f"Roundness — min: {min(roundnesses)}  max: {max(roundnesses)}  unique: {len(set(roundnesses))}")
    print(f"Spacing   — min: {min(spacings)}  max: {max(spacings)}  unique: {len(set(spacings))}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, values, name in zip(axes,
        [sizes, hardnesses, roundnesses, spacings],
        ["Size", "Hardness", "Roundness", "Spacing"]):
        ax.hist(values, bins=20)
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig("dataset_distribution.png", dpi=150)
    plt.show()

# ============================================================
# DATASET
# ============================================================

class BrushDataset(Dataset):
    """
    Loads grayscale stroke PNGs, resizes to 224x224, parses
    labels from filenames.
    Pattern: brush_s{size}_h{hardness}_r{roundness}_sp{spacing}.png
    """

    def __init__(self, data_dir):
        self.samples = []
        pattern = re.compile(r"brush_s(\d+)_h(\d+)_r(\d+)_sp(\d+)\.png")

        png_files = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith(".png") and not f.startswith("._")
        ])

        if not png_files:
            raise FileNotFoundError(f"No PNG files found in {data_dir}")

        print(f"  Found {len(png_files)} PNG files -- loading...")
        skipped_small = 0

        for fname in png_files:
            match = pattern.match(fname)
            if not match:
                print(f"  Skipping (unrecognised filename): {fname}")
                continue

            size = int(match.group(1))
            if size < 5:
                skipped_small += 1
                continue

            label = torch.tensor([
                size                    / NORM["size"],
                int(match.group(2))     / NORM["hardness"],
                int(match.group(3))     / NORM["roundness"],
                int(match.group(4))     / NORM["spacing"],
            ], dtype=torch.float32)

            img_path  = os.path.join(data_dir, fname)
            img       = Image.open(img_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
            img_tensor = torch.tensor(
                np.array(img, dtype=np.float32) / 255.0
            ).unsqueeze(0)   # (1, 224, 224)

            self.samples.append((img_tensor, label))

        print(f"  Loaded {len(self.samples)} valid samples. "
              f"(skipped {skipped_small} with size < 5px)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_dataloaders(data_dir):
    """Split dataset into train/val and return DataLoaders."""
    dataset = BrushDataset(data_dir)
    n_val   = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val

    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0
    )
    val_loader = DataLoader(
        val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f"  Train: {n_train} samples  |  Val: {n_val} samples")
    return train_loader, val_loader, val_set

# ============================================================
# MODEL
# ============================================================

class BrushCNN(nn.Module):
    """
    3-block CNN encoder + global average pool + regression head.

    Block 1  224->112  local texture / edge sharpness (hardness)
    Block 2  112->56   stroke width (size)
    Block 3  56->28    periodicity / spacing patterns
    GAP               collapse spatial dims to feature vector
    Head              FC layers -> 4 normalized param predictions
    """

    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)          # 224 -> 112
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)          # 112 -> 56
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)          # 56 -> 28
        )

        self.gap = nn.AdaptiveAvgPool2d(1)   # (batch, 128, 1, 1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = self.head(x)
        return x
# ============================================================
# WEIGHTED LOSS
# ============================================================

def weighted_mse(pred, target):
    """
    Per-parameter weighted MSE.
    Order: [size, hardness, roundness, spacing]
    Adjust weights if one parameter trains noticeably worse.
    """
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=pred.device)
    return (weights * (pred - target) ** 2).mean()

# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        preds = model(images)
        loss  = weighted_mse(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(images)

    return total_loss / len(loader.dataset)

# ============================================================
# VALIDATION
# ============================================================

def validate(model, loader):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            preds  = model(images)
            loss   = weighted_mse(preds, labels)
            total_loss += loss.item() * len(images)

    return total_loss / len(loader.dataset)

# ============================================================
# TESTING
# ============================================================

def test_model(model, val_set, num_samples=10):
    """
    Run inference on num_samples from the validation set.
    Prints predicted vs actual in real units.
    Returns list of per-sample MSE losses.
    """
    model.eval()
    param_names = ["size (px)", "hardness (%)", "roundness (%)", "spacing (%)"]
    norm_values = [NORM["size"], NORM["hardness"], NORM["roundness"], NORM["spacing"]]

    num_samples = min(num_samples, len(val_set))
    losses = []

    print(f"\n{'─'*65}")
    print(f"  {'Param':<14}  {'Predicted':>10}  {'Actual':>10}  {'Error':>10}")
    print(f"{'─'*65}")

    with torch.no_grad():
        for i in range(num_samples):
            image, label = val_set[i]
            image_in = image.unsqueeze(0).to(DEVICE)
            pred     = model(image_in).squeeze(0).cpu()

            mse = ((pred - label) ** 2).mean().item()
            losses.append(mse)

            print(f"\n  Sample {i+1}   (MSE: {mse:.5f})")
            for j, (name, scale) in enumerate(zip(param_names, norm_values)):
                pred_real  = pred[j].item()  * scale
                truth_real = label[j].item() * scale
                error      = pred_real - truth_real
                print(f"  {name:<14}  {pred_real:>10.2f}  {truth_real:>10.2f}  {error:>+10.2f}")

    print(f"\n{'─'*65}")
    print(f"  Mean test MSE over {num_samples} samples: {np.mean(losses):.5f}")
    print(f"{'─'*65}\n")

    return losses

# ============================================================
# PLOTTING
# ============================================================

def plot_loss(train_losses, val_losses, test_losses):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Brush ResNet -- Training & Test Loss", fontsize=14)

    ax = axes[0]
    ax.plot(train_losses, label="Train loss",      linewidth=2)
    ax.plot(val_losses,   label="Validation loss", linewidth=2, linestyle="--")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    x = np.arange(1, len(test_losses) + 1)
    ax.bar(x, test_losses, color="steelblue", alpha=0.8)
    ax.axhline(np.mean(test_losses), color="crimson", linestyle="--",
               linewidth=1.5, label=f"Mean: {np.mean(test_losses):.4f}")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("MSE")
    ax.set_title("Per-sample test MSE")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("brush_cnn_loss.png", dpi=150)
    plt.show()
    print("Loss plot saved -> brush_cnn_loss.png")

# ============================================================
# MAIN
# ============================================================

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR   = os.path.join(SCRIPT_DIR, "Data")

    inspect_dataset(DATA_DIR)

    print(f"\nLoading dataset from '{DATA_DIR}' ...")
    train_loader, val_loader, val_set = make_dataloaders(DATA_DIR)

    model     = BrushCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_losses      = []
    val_losses        = []
    best_val          = float("inf")
    epochs_no_improve = 0
    test_losses       = []

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(SCRIPT_DIR, "brush_cnn_best.pt")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*50}")

        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss   = validate(model, val_loader)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  Train MSE: {train_loss:.5f}  |  Val MSE: {val_loss:.5f}")

        # Checkpoint
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss":    best_val,
            }, checkpoint_path)
            print(f"  Saved best model (val_loss: {best_val:.5f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{PATIENCE} epochs")

        # Per-epoch test
        test_losses = test_model(model, val_set, num_samples=5)

        # Early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping -- no improvement for {PATIENCE} epochs.")
            print("Loading best weights...")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state"])
            break

    # Final test on best model
    print("\nFinal test on best model weights:")
    test_losses = test_model(model, val_set, num_samples=10)

    plot_loss(train_losses, val_losses, test_losses)


if __name__ == "__main__":
    main()