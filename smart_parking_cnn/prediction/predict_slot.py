import argparse
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

def preprocess_image(image_path):
    """
    Load an image from disk, resize and format it to match
    the training preprocessing.
    """
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at path: {image_path}")
    
    # OpenCV loads images in BGR format, convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image to match the model input shape (224x224)
    img = cv2.resize(img, (224, 224))
    
    # Normalize pixel values (rescale 1./255)
    img = img.astype("float32") / 255.0
    
    # Expand dimensions to add the batch size (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    
    return img

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.join(script_dir, '../model/parking_cnn_model.h5')
    
    parser = argparse.ArgumentParser(description="Predict Parking Slot Occupancy")
    parser.add_argument('image_path', type=str, help='Path to the test image')
    parser.add_argument('--model_path', type=str, default=default_model, help='Path to the saved trained model (.h5)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please train the model first using train_cnn.py")
        return
        
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return

    # Load the trained model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    
    # Preprocess the input image
    print(f"Processing image {args.image_path}...")
    try:
        input_image = preprocess_image(args.image_path)
    except ValueError as e:
        print(e)
        return
    
    # Perform prediction
    prediction = model.predict(input_image)
    
    # The output is a sigmoid probability (0 to 1). 
    # Determine the classes based on standard ImageDataGenerator flow_from_directory mapping
    # Assuming 'empty' comes before 'occupied' alphabetically:
    # Class 0: empty
    # Class 1: occupied
    probability = prediction[0][0]
    
    if probability > 0.5:
        # Class 1
        result = "OCCUPIED"
    else:
        # Class 0
        result = "EMPTY"
    
    print("-" * 30)
    print(f"Parking Slot: {result}")
    print(f"(Probability of occupied: {probability:.4f})")
    print("-" * 30)

if __name__ == "__main__":
    main()
