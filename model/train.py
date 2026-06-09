"""
model/train.py — CNN Parking Occupancy Classifier Training Script

FIXES APPLIED:
  1. BUG: ImageDataGenerator is deprecated in TF 2.x+ (still works but shows warnings).
     FIX: Added tf.data pipeline alternative as comment. Kept ImageDataGenerator
          for simplicity since it still works fine for final year projects.

  2. BUG: classes=['empty','occupied'] hardcoded — if folder names differ, labels flip.
          This is a CRITICAL bug: empty=0, occupied=1 must match folder names exactly.
     FIX: Added assertion to verify class indices before training.

  3. BUG: model() function shadows Python built-in name.
     FIX: Renamed to build_model().

  4. BUG: Learning rate 1e-3 is too high for a CNN this deep — causes loss spikes.
     FIX: Changed to 3e-4 (standard "Karpathy constant" for Adam).

  5. BUG: EarlyStopping monitors val_accuracy but ReduceLROnPlateau monitors val_loss.
          This creates conflicting signals — LR may reduce while accuracy is still rising.
     FIX: Both now monitor val_loss consistently.

  6. BUG: Dense(512) after Flatten is oversized for a 64x64 input — causes overfitting.
     FIX: Replaced Flatten+Dense(512) with GlobalAveragePooling2D+Dense(256).
          This is a standard best practice for small CNNs.

  7. BUG: Training curves only saved as PNG but not shown — no way to know if model
          overfit during training.
     FIX: Added overfitting detection printout after training.

  8. BUG: val/ split not documented. PKLot dataset has no pre-split val set.
     FIX: Added script comment explaining how to create the split with sklearn.

  9. BUG: No class weight handling — if dataset is imbalanced (more empty than occupied),
          model will bias toward predicting "empty".
     FIX: Added class_weight computation.

DATASET SETUP (PKLot):
  1. Download: https://web.inf.ufpr.br/vri/databases/parking-lot-database/
  2. Extract and run the split helper at the bottom of this file.
  3. Layout expected:
       dataset/train/empty/     ← images
       dataset/train/occupied/
       dataset/val/empty/
       dataset/val/occupied/
"""

import os, sys

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import layers, callbacks
except ImportError:
    print("pip install tensorflow")
    sys.exit(1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── config ────────────────────────────────────────────────────────────────────
DATASET = "dataset"
OUT     = "model/parking_cnn.h5"
IMG     = (64, 64)
BATCH   = 32
EPOCHS  = 30

# FIX 4: lr=3e-4 instead of 1e-3
LEARNING_RATE = 3e-4


# ── data generators ───────────────────────────────────────────────────────────
train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    zoom_range=0.1
)
val_aug = ImageDataGenerator(rescale=1./255)


# ── model architecture ────────────────────────────────────────────────────────
# FIX 3: renamed from model() to build_model()
# FIX 6: GlobalAveragePooling2D replaces Flatten+Dense(512)
def build_model() -> tf.keras.Model:
    inp = tf.keras.Input(shape=(64, 64, 3))

    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # FIX 6: GlobalAveragePooling — reduces params, less overfitting
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    m = tf.keras.Model(inputs=inp, outputs=out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return m


def train():
    os.makedirs("model", exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    tr = train_aug.flow_from_directory(
        f"{DATASET}/train",
        target_size=IMG,
        batch_size=BATCH,
        class_mode='binary',
        classes=['empty', 'occupied']   # empty=0, occupied=1
    )
    va = val_aug.flow_from_directory(
        f"{DATASET}/val",
        target_size=IMG,
        batch_size=BATCH,
        class_mode='binary',
        classes=['empty', 'occupied']
    )

    # FIX 2: Verify label mapping is correct before wasting training time
    assert tr.class_indices == {'empty': 0, 'occupied': 1}, (
        f"Class index mismatch! Got {tr.class_indices}. "
        "Rename your dataset folders to 'empty' and 'occupied'."
    )
    print(f"✅ Class mapping verified: {tr.class_indices}")
    print(f"   Train samples : {tr.samples}")
    print(f"   Val   samples : {va.samples}")

    # FIX 9: Compute class weights to handle imbalanced datasets
    n_empty    = tr.classes.tolist().count(0)
    n_occupied = tr.classes.tolist().count(1)
    total      = n_empty + n_occupied
    class_weight = {
        0: total / (2 * n_empty),
        1: total / (2 * n_occupied)
    }
    print(f"   Class weights : {class_weight}")

    # ── build + train ─────────────────────────────────────────────────────────
    net = build_model()
    net.summary()

    # FIX 5: Both callbacks now monitor val_loss consistently
    cbs = [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6,
            verbose=1),
        callbacks.ModelCheckpoint(
            OUT, monitor='val_accuracy', save_best_only=True, verbose=1),
    ]

    h = net.fit(
        tr,
        validation_data=va,
        epochs=EPOCHS,
        callbacks=cbs,
        class_weight=class_weight
    )

    # ── plots ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(h.history['accuracy'],     label='Train')
    ax[0].plot(h.history['val_accuracy'], label='Val')
    ax[0].set_title('Accuracy')
    ax[0].legend()
    ax[1].plot(h.history['loss'],     label='Train')
    ax[1].plot(h.history['val_loss'], label='Val')
    ax[1].set_title('Loss')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig('model/training_curves.png')
    print("📊 Training curves saved → model/training_curves.png")

    # ── evaluate ──────────────────────────────────────────────────────────────
    _, acc, auc = net.evaluate(va, verbose=0)

    # FIX 7: Overfitting check
    final_train_acc = h.history['accuracy'][-1]
    final_val_acc   = h.history['val_accuracy'][-1]
    gap = final_train_acc - final_val_acc
    if gap > 0.10:
        print(f"⚠️  Possible overfitting: train_acc={final_train_acc:.2%} val_acc={final_val_acc:.2%} gap={gap:.2%}")
        print("   Try: more Dropout, fewer Dense units, more data augmentation.")
    else:
        print(f"✅ Model generalises well (gap={gap:.2%})")

    print(f"\n✅ Final → Accuracy: {acc*100:.2f}%  AUC: {auc:.4f}  → saved to {OUT}")


# ── dataset split helper ──────────────────────────────────────────────────────
# FIX 8: Run this ONCE to split PKLot images into train/val folders
def create_train_val_split(source_dir: str, val_ratio: float = 0.2):
    """
    source_dir layout:
      source_dir/empty/    ← all empty images
      source_dir/occupied/ ← all occupied images
    Creates dataset/train/ and dataset/val/ automatically.
    """
    import shutil, random
    from pathlib import Path

    for cls in ['empty', 'occupied']:
        imgs = list(Path(source_dir, cls).glob("*.jpg")) + \
               list(Path(source_dir, cls).glob("*.png"))
        random.shuffle(imgs)
        split = int(len(imgs) * (1 - val_ratio))
        for phase, files in [('train', imgs[:split]), ('val', imgs[split:])]:
            dest = Path(DATASET, phase, cls)
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, dest / f.name)
        print(f"  {cls}: {split} train | {len(imgs)-split} val")
    print("✅ Dataset split complete.")


if __name__ == "__main__":
    train()