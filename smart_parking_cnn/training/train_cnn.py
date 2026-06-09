"""
train_coco.py — COCO Format Parking CNN Trainer

DATASET: Download from Roboflow — "PKLot" or "Parking Lot" COCO format
  URL: https://universe.roboflow.com/brad-dwyer/pklot-1tros
  Export as: COCO format → Download ZIP
  Extract to: dataset/archive/

Expected folder structure:
  dataset/archive/
  ├── train/
  │   ├── _annotations.coco.json
  │   └── *.jpg (parking lot images)
  └── valid/
      ├── _annotations.coco.json
      └── *.jpg

Run:
  python train_coco.py
  python train_coco.py --epochs 20 --batch_size 16

FIXES APPLIED:
  1. BUG: __len__() returned np.float64 (np.ceil) — TF Sequence needs int
     FIX: return int(np.ceil(...))

  2. BUG: Input size 224×224 in train, but cnn_model.py uses 64×64 at inference
          → CRASH when loading model for prediction
     FIX: Both train and inference use 64×64 (IMG_SIZE constant shared)
          Note: If you want 224×224, update cnn_model.py too. 64 is faster.

  3. BUG: No BatchNormalization — loss unstable on deep CNN
     FIX: Added BatchNorm after each Conv block

  4. BUG: No EarlyStopping — trains all 15 epochs even if converged at epoch 5
     FIX: EarlyStopping(monitor='val_loss', patience=5)

  5. BUG: No ModelCheckpoint — saves LAST model not BEST model
     FIX: ModelCheckpoint saves best val_accuracy model only

  6. BUG: Adam() called with no learning rate — default 1e-3 too high for deep CNN
     FIX: Adam(learning_rate=3e-4)

  7. BUG: No class weights — imbalanced datasets bias toward majority class
     FIX: class_weight computed from annotation counts

  8. BUG: unique_imgs dict re-reads same image on every batch call — slow
     FIX: Added image_cache dict that persists across batches (optional, RAM-based)
"""

import os
import json
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras import callbacks

# ── CRITICAL: IMG_SIZE must match cnn_model.py ───────────────────────────────
# cnn_model.py line: img = cv2.resize(image_bgr, (64, 64))
# If you change this, change cnn_model.py too!
IMG_SIZE = 64


# ── Model Architecture ────────────────────────────────────────────────────────
def build_model(img_size: int = IMG_SIZE) -> tf.keras.Model:
    """
    CNN for binary parking slot classification.
    FIX 2: Uses img_size parameter — no hardcoded 224.
    FIX 3: Added BatchNormalization after each conv block.
    FIX 6: Uses GlobalAveragePooling2D instead of Flatten+Dense(128)
            — fewer params, less overfitting.
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3,3), activation='relu', padding='same',
               input_shape=(img_size, img_size, 3)),
        BatchNormalization(),                          # FIX 3
        MaxPooling2D((2,2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        # Head — FIX 6: GAP replaces Flatten+Dense(128)
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid'),
    ])

    # FIX 7: lr=3e-4 (standard Adam for image CNNs)
    model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


# ── COCO Dataset Generator ────────────────────────────────────────────────────
class COCODatasetGenerator(Sequence):
    """
    Loads parking slot crops from COCO-annotated parking lot images.
    Each annotation bbox is cropped, resized to IMG_SIZE×IMG_SIZE.

    Label mapping:
      space-empty    → 0
      space-occupied → 1
    """

    def __init__(self, json_path: str, base_dir: str,
                 batch_size: int = 32,
                 target_size: tuple = (IMG_SIZE, IMG_SIZE),
                 is_training: bool = True,
                 use_cache: bool = True):
        self.base_dir    = base_dir
        self.batch_size  = batch_size
        self.target_size = target_size
        self.is_training = is_training
        # FIX 8: persistent image cache across batches
        self._cache: dict = {} if use_cache else None

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Build category map: only care about parking categories
        category_map = {}
        for cat in data['categories']:
            name = cat['name'].lower().strip()
            if name == 'space-empty':
                category_map[cat['id']] = 0
            elif name == 'space-occupied':
                category_map[cat['id']] = 1

        if not category_map:
            raise ValueError(
                "No 'space-empty' or 'space-occupied' categories found in JSON. "
                f"Available: {[c['name'] for c in data['categories']]}"
            )

        image_map = {img['id']: img['file_name'] for img in data['images']}

        self.samples = []
        for ann in data['annotations']:
            cat_id = ann.get('category_id')
            if cat_id in category_map:
                img_id = ann.get('image_id')
                if img_id in image_map:
                    self.samples.append((
                        image_map[img_id],
                        ann['bbox'],
                        category_map[cat_id]
                    ))

        if not self.samples:
            raise ValueError(f"No valid samples found in {json_path}")

        self.class_indices = {'empty': 0, 'occupied': 1}
        self.indexes = np.arange(len(self.samples))
        if self.is_training:
            np.random.shuffle(self.indexes)

        print(f"  Loaded {len(self.samples)} samples "
              f"({sum(1 for _,_,l in self.samples if l==1)} occupied, "
              f"{sum(1 for _,_,l in self.samples if l==0)} empty)")

    # FIX 1: return int — np.ceil returns float64, TF Sequence needs int
    def __len__(self) -> int:
        return int(np.ceil(len(self.samples) / self.batch_size))

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)

    def _load_image(self, fname: str):
        """
        FIX 8: Load image once, cache it for the session.
        Falls back to None if file missing.
        """
        if self._cache is not None and fname in self._cache:
            return self._cache[fname]

        img_path = os.path.join(self.base_dir, fname)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self._cache is not None:
            self._cache[fname] = img
        return img

    def __getitem__(self, index: int):
        batch_indexes  = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        batch_samples  = [self.samples[i] for i in batch_indexes]

        X = np.zeros((len(batch_samples), *self.target_size, 3), dtype=np.float32)
        y = np.zeros(len(batch_samples), dtype=np.float32)

        for i, (fname, bbox, label) in enumerate(batch_samples):
            img = self._load_image(fname)
            if img is not None:
                # bbox is [x, y, width, height] in COCO format
                bx, by, bw, bh = [int(v) for v in bbox]
                img_h, img_w = img.shape[:2]

                x1 = max(0, bx)
                y1 = max(0, by)
                x2 = min(img_w, bx + bw)
                y2 = min(img_h, by + bh)

                if x2 > x1 and y2 > y1:
                    crop = img[y1:y2, x1:x2]
                    crop = cv2.resize(crop, self.target_size)
                    X[i] = crop / 255.0
                # else: X[i] stays zeros — blank crop

            y[i] = label

        return X, y

    def get_class_weights(self) -> dict:
        """
        FIX 7: Compute class weights to handle imbalanced datasets.
        Returns dict suitable for model.fit(class_weight=...).
        """
        labels  = [s[2] for s in self.samples]
        n_total = len(labels)
        n_empty = labels.count(0)
        n_occ   = labels.count(1)

        if n_empty == 0 or n_occ == 0:
            return {0: 1.0, 1: 1.0}

        return {
            0: n_total / (2 * n_empty),
            1: n_total / (2 * n_occ)
        }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    script_dir      = os.path.dirname(os.path.abspath(__file__))
    default_dataset = os.path.join(script_dir, '../dataset/archive')

    parser = argparse.ArgumentParser(description="Train CNN for Smart Parking (COCO format)")
    parser.add_argument('--dataset_path', type=str, default=default_dataset)
    parser.add_argument('--epochs',       type=int, default=15)
    parser.add_argument('--batch_size',   type=int, default=32)
    parser.add_argument('--img_size',     type=int, default=IMG_SIZE,
                        help=f'Image size (default {IMG_SIZE}). Must match cnn_model.py!')
    args = parser.parse_args()

    # Validate IMG_SIZE matches what cnn_model.py expects
    if args.img_size != IMG_SIZE:
        print(f"⚠️  WARNING: --img_size={args.img_size} but cnn_model.py uses {IMG_SIZE}.")
        print(f"   Update IMG_SIZE in cnn_model.py to {args.img_size} before running app.py!")

    base_dir  = args.dataset_path
    train_dir = os.path.join(base_dir, 'train')
    valid_dir = os.path.join(base_dir, 'valid')

    if not os.path.exists(train_dir) or not os.path.exists(valid_dir):
        print(f"❌ Dataset not found at: {base_dir}")
        print("   Expected: train/ and valid/ folders with _annotations.coco.json")
        print("   Download from: https://universe.roboflow.com/brad-dwyer/pklot-1tros")
        return

    train_json = os.path.join(train_dir, '_annotations.coco.json')
    valid_json = os.path.join(valid_dir, '_annotations.coco.json')

    for p in [train_json, valid_json]:
        if not os.path.exists(p):
            print(f"❌ Missing: {p}")
            return

    print("\n📂 Loading Training Data...")
    train_gen = COCODatasetGenerator(
        train_json, train_dir,
        batch_size=args.batch_size,
        target_size=(args.img_size, args.img_size),
        is_training=True
    )

    print("📂 Loading Validation Data...")
    val_gen = COCODatasetGenerator(
        valid_json, valid_dir,
        batch_size=args.batch_size,
        target_size=(args.img_size, args.img_size),
        is_training=False
    )

    print(f"\n✅ Class indices: {train_gen.class_indices}")

    # FIX 7: Class weights
    cw = train_gen.get_class_weights()
    print(f"⚖️  Class weights: {cw}")

    # Build model
    model = build_model(img_size=args.img_size)
    model.summary()

    # Output paths
    model_dir    = os.path.join(script_dir, '../model')
    os.makedirs(model_dir, exist_ok=True)
    model_path   = os.path.join(model_dir, 'parking_cnn_model.h5')
    history_path = os.path.join(model_dir, 'training_history.json')

    # FIX 4 + 5: EarlyStopping + ModelCheckpoint
    cbs = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    print("\n🚀 Starting Training...")
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=cbs,
        class_weight=cw,       # FIX 7
        verbose=1
    )

    # Save training history
    json_history = {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(json_history, f, indent=2)

    # Final evaluation
    _, acc, auc = model.evaluate(val_gen, verbose=0)

    # Overfitting check
    train_acc = history.history['accuracy'][-1]
    val_acc   = history.history['val_accuracy'][-1]
    gap       = train_acc - val_acc
    if gap > 0.10:
        print(f"\n⚠️  Possible overfitting: train={train_acc:.2%}  val={val_acc:.2%}  gap={gap:.2%}")
    else:
        print(f"\n✅ Good generalisation (gap={gap:.2%})")

    print(f"✅ val_accuracy={acc*100:.2f}%  AUC={auc:.4f}")
    print(f"✅ Model saved → {model_path}")
    print(f"✅ History saved → {history_path}")

    # ── IMPORTANT: update cnn_model.py if img_size changed ───────────────────
    if args.img_size != 64:
        print(f"\n⚠️  IMPORTANT: Open model/cnn_model.py and change line:")
        print(f"   img = cv2.resize(image_bgr, (64, 64))")
        print(f"   → img = cv2.resize(image_bgr, ({args.img_size}, {args.img_size}))")


if __name__ == "__main__":
    # GPU memory growth (prevents OOM on shared GPU)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🖥️  Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
    else:
        print("ℹ️  No GPU found — training on CPU (slower but works fine)")

    main()