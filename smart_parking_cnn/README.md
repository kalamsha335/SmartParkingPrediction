# Smart Parking CNN Model

This folder contains a complete Image-Based Smart Parking Prediction System using a Convolutional Neural Network (CNN) in Python with TensorFlow/Keras.

## Features
* **Image Classification:** Predicts whether a parking slot is `OCCUPIED` or `EMPTY`.
* **Data Augmentation:** Robust training using `ImageDataGenerator`.
* **Model:** Custom CNN architecture with dropout and binary crossentropy.
* **Visualization:** Scripts to plot training history (loss/accuracy).

## Folder Structure
```
smart_parking_cnn/
│── dataset/
│   ├── train/ (empty & occupied subfolders)
│   ├── test/ (empty & occupied subfolders)
│── model/ (saved model & history)
│── training/
│   ├── train_cnn.py
│── prediction/
│   ├── predict_slot.py
│── visualization/
│   ├── plot_graphs.py
```

## Setup Instructions

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset:**
   Place at least 2000 images per class in `dataset/train/empty`, `dataset/train/occupied`, `dataset/test/empty`, and `dataset/test/occupied`.

3. **Train the Model:**
   ```bash
   cd training
   python train_cnn.py
   ```
   *By default, it trains for 15 epochs and saves `parking_cnn_model.h5` in the `model` directory.*

4. **Prediction / Inference:**
   ```bash
   cd prediction
   python predict_slot.py path_to_your_test_image.jpg
   ```
   *Outputs whether the slot is OCCUPIED or EMPTY.*

5. **Visualization:**
   ```bash
   cd visualization
   python plot_graphs.py
   ```
   *Reads the JSON training history and produces matplotlib graphs of accuracy and loss over the epochs.*
