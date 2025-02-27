import cv2
import numpy as np
import os
from datetime import datetime 
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing import image
from keras.models import Model

# Load pre-trained MobileNetV2 model (without the classification layer)
baseModel = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=baseModel.input, outputs=baseModel.output)

# Path to your database of Lorcana card images
databasePath = "C:/Lore Book/Card_Images"
imageSize = (224, 224)  # MobileNetV2 input size

# Function to preprocess an image for the model
def preprocess_image(imgPath):
    img = image.load_img(imgPath)
    imgArray = image.img_to_array(img)
    imgArray = np.expand_dims(imgArray, axis=0)
    imgArray = preprocess_input(imgArray)
    return imgArray

# Extract feature vector from an image using MobileNetV2
def extract_features(imgPath):
    imgArray = preprocess_image(imgPath)
    features = model.predict(imgArray)
    return features

# Build a feature database for all card images
def build_feature_database():
    featureDB = {}
    
    for filename in os.listdir(databasePath):
        if filename.endswith(".webp") or filename.endswith(".png"):
            imgPath = os.path.join(databasePath, filename)
            features = extract_features(imgPath)
            featureDB[filename] = features

    return featureDB

# Find the best match by comparing feature vectors
def find_best_match(inputFeatures, featureDB):
    bestMatch = None
    bestScore = float("inf")

    for filename, features in featureDB.items():
        distance = np.linalg.norm(inputFeatures - features)  # Euclidean distance
        if distance < bestScore:
            bestScore = distance
            bestMatch = filename

    return bestMatch

# Initialize webcam capture
cap = cv2.VideoCapture(1)  # Change the index if necessary

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Build the database
print("Building feature database...")
featureDatabase = build_feature_database()
print("Feature database ready!")

# Function to crop the image
def crop_image(frame, x, y, width, height):
    """
    Crops the frame from (x, y) with the given width and height.
    :param frame: The captured frame.
    :param x: x-coordinate of the top-left corner of the rectangle.
    :param y: y-coordinate of the top-left corner of the rectangle.
    :param width: Width of the rectangle to crop.
    :param height: Height of the rectangle to crop.
    :return: Cropped image.
    """
    return frame[y:y + height, x:x + width]

#Capture images and find the closest match in the database in a loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    cv2.imshow("Press SPACE to capture, 'q' to quit", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 32:  # Press SPACE to capture
        # Save the captured image
        outputDir="captured_cards"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        imgPath = "f{output_dir}/Card_{timestamp}.jpg"
        cv2.imwrite(imgPath, frame)
        print("Image captured!")

        # Define crop coordinates (x, y, width, height)
        x, y, width, height = 100, 100, 300, 300  # Example values for cropping (adjust as needed)
        
        # Crop the captured image
        croppedImage = crop_image(frame, x, y, width, height)
        
        # Save the cropped image
        croppedImagePath = f"{outputDir}/Cropped_card_{timestamp}.jpg"
        cv2.imwrite(croppedImagePath, croppedImage)
        print("Image cropped and saved!")

        # Extract features from the cropped image
        inputFeatures = extract_features(croppedImagePath)

        # Find the best match in the database
        matchedFilename = find_best_match(inputFeatures, featureDatabase)

        if matchedFilename:
            print(f"Best match: {matchedFilename}")
        else:
            print("No close match found.")

    elif key == ord('q'):  # Press 'q' to quit
        print("Exiting...")
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()