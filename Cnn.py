"""
brush_cnn.py
CNN for reverse-engineering Photoshop brush parameters from stroke images.
Predicts: size (px), hardness (%), spacing (%), roundness (%)
Input:    512x512 grayscale PNG
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
import keras
from keras import layers

# ============================================================
# CONSTANTS  —  match your data generation ranges
# ============================================================

IMG_SIZE = 512
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3

# Normalization maxima (map each param to [0, 1])
NORM = {
    "size": 100.0,
    "hardness": 100.0,
    "roundness": 100.0,
    "spacing": 200.0,
}

# ============================================================
# DATA LOADING
# ============================================================


def parse_filename(filename):
    """
    Extract parameters from filename.
    Expected pattern: brush_s{size}_h{hardness}_r{roundness}_sp{spacing}.png
    Returns dict of floats or None if pattern doesn't match.
    """
    pattern = r"brush_s(\d+)_h(\d+)_r(\d+)_sp(\d+)\.png"
    match = re.match(pattern, os.path.basename(filename))
    if not match:
        return None
    return {
        "size": float(match.group(1)),
        "hardness": float(match.group(2)),
        "roundness": float(match.group(3)),
        "spacing": float(match.group(4)),
    }


def normalize_params(params):
    """Normalize each parameter to [0, 1]."""
    return np.array(
        [
            params["size"] / NORM["size"],
            params["hardness"] / NORM["hardness"],
            params["roundness"] / NORM["roundness"],
            params["spacing"] / NORM["spacing"],
        ],
        dtype=np.float32,
    )


def denormalize_params(pred):
    """Reverse normalization for human-readable output."""
    keys = ["size", "hardness", "roundness", "spacing"]
    return {k: pred[i] * NORM[k] for i, k in enumerate(keys)}


def load_dataset(data_dir, val_split=0.15):
    """
    Load all PNG files from data_dir, parse labels from filenames.
    Returns (X_train, y_train, X_val, y_val) as numpy arrays.
    """
    images, labels = [], []

    png_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in {data_dir}")

    for fname in sorted(png_files):
        params = parse_filename(fname)
        if params is None:
            print(f"  Skipping (unrecognised filename): {fname}")
            continue

        img_path = os.path.join(data_dir, fname)
        img = Image.open(img_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0  # → [0, 1]
        img_array = img_array[..., np.newaxis]  # (512, 512, 1)

        images.append(img_array)
        labels.append(normalize_params(params))

    X = np.stack(images)  # (N, 512, 512, 1)
    y = np.stack(labels)  # (N, 4)

    # Shuffle before split
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(len(X) * (1 - val_split))
    return X[:split], y[:split], X[split:], y[split:]


# ============================================================
# MODEL
# ============================================================


def build_model():
    """
    3-block CNN encoder + global average pool + regression head.

    Block 1  512→256   — local texture, edge sharpness (hardness)
    Block 2  256→128   — stroke width (size)
    Block 3  128→64    — periodicity / spacing patterns
    GAP               — collapse spatial dims to feature vector
    Head              — FC layers → 4 param predictions
    """
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="stroke_image")

    # ── Block 1 ─────────────────────────────────────────────
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)  # 256×256

    # ── Block 2 ─────────────────────────────────────────────
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)  # 128×128

    # ── Block 3 ─────────────────────────────────────────────
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)  # 64×64

    # ── Global average pool ──────────────────────────────────
    x = layers.GlobalAveragePooling2D()(x)  # (batch, 128)

    # ── Regression head ──────────────────────────────────────
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)

    # Sigmoid output constrains predictions to [0, 1]
    outputs = layers.Dense(4, activation="sigmoid", name="params")(x)

    model = keras.Model(inputs, outputs, name="BrushCNN")
    return model


# ============================================================
# TRAINING STEP
# ============================================================


def train_model(model, X_train, y_train, X_val, y_val):
    """
    Compile and fit the model.
    Returns history object for loss plotting.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(LR), loss="mse", metrics=["mae"]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            "brush_cnn_best.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    return history


# ============================================================
# TESTING
# ============================================================


def test_model(model, X_val, y_val, num_samples=10):
    """
    Run inference on validation samples and print predictions
    vs ground truth for each parameter.
    Returns per-sample MSE losses.
    """
    param_names = ["size (px)", "hardness (%)", "roundness (%)", "spacing (%)"]
    norm_values = [
        NORM["size"],
        NORM["hardness"],
        NORM["roundness"],
        NORM["spacing"],
    ]

    preds = model.predict(X_val[:num_samples], verbose=0)
    truths = y_val[:num_samples]

    losses = []
    print(f"\n{'─'*65}")
    print(f"  {'Param':<14}  {'Predicted':>10}  {'Actual':>10}  {'Error':>10}")
    print(f"{'─'*65}")

    for i in range(num_samples):
        sample_mse = np.mean((preds[i] - truths[i]) ** 2)
        losses.append(sample_mse)
        print(f"\n  Sample {i+1}   (MSE: {sample_mse:.5f})")

        for j, (name, scale) in enumerate(zip(param_names, norm_values)):
            pred_real = preds[i][j] * scale
            truth_real = truths[i][j] * scale
            error = pred_real - truth_real
            print(
                f"  {name:<14}  {pred_real:>10.2f}  {truth_real:>10.2f}  {error:>+10.2f}"
            )

    print(f"\n{'─'*65}")
    print(f"  Mean test MSE over {num_samples} samples: {np.mean(losses):.5f}")
    print(f"{'─'*65}\n")

    return losses


# ============================================================
# PLOTTING
# ============================================================


def plot_loss(history, losses):
    """
    Two-panel plot:
      Left  — training vs validation loss over epochs
      Right — per-sample test MSE
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Brush CNN — Training & Test Loss", fontsize=14)

    # ── Training curve ───────────────────────────────────────
    ax = axes[0]
    ax.plot(history.history["loss"], label="Train loss", linewidth=2)
    ax.plot(
        history.history["val_loss"],
        label="Validation loss",
        linewidth=2,
        linestyle="--",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("MSE")
    ax.set_title("Training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Per-sample test MSE ──────────────────────────────────
    ax = axes[1]
    x = np.arange(1, len(losses) + 1)
    ax.bar(x, losses, color="steelblue", alpha=0.8)
    ax.axhline(
        np.mean(losses),
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {np.mean(losses):.4f}",
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("MSE")
    ax.set_title("Per-sample test MSE")
    ax.set_xticks(x)
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("brush_cnn_loss.png", dpi=150)
    plt.show()
    print("Loss plot saved → brush_cnn_loss.png")


# ============================================================
# MAIN
# ============================================================


def main():
    DATA_DIR = os.getcwd() + "/Data"  # folder containing your exported PNGs

    # ── Load data ────────────────────────────────────────────
    print(f"\nLoading dataset from '{DATA_DIR}' ...")
    X_train, y_train, X_val, y_val = load_dataset(DATA_DIR)
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")

    # ── Build model ───────────────────────────────────────────
    model = build_model()
    model.summary()

    # ── Training loop ─────────────────────────────────────────
    print("\nStarting training ...")
    history = train_model(model, X_train, y_train, X_val, y_val)

    # ── Testing loop ──────────────────────────────────────────
    print("\nRunning test predictions ...")
    losses = test_model(model, X_val, y_val, num_samples=10)

    # ── Plot ──────────────────────────────────────────────────
    plot_loss(history, losses)


if __name__ == "__main__":
    main()
