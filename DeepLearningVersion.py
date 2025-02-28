import cv2
import numpy as np
import os
import json
import concurrent.futures
from datetime import datetime
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

# Load MobileNetV2 model (feature extraction only)
baseModel = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=baseModel.input, outputs=baseModel.output)

databasePath = "C:/Lore Book/Card_Images"
imageSize = (224, 224)
cacheFile = "DBCardCache.json"
outputDir = "captured_cards"
os.makedirs(outputDir, exist_ok=True)

# Image preprocessing 
def preprocess_image(imgPath):
    img = image.load_img(imgPath, targetSize=imageSize)
    imgArray = image.img_to_array(img)
    imgArray = np.expand_dims(imgArray, axis=0)  # Shape: (1, 224, 224, 3)
    imgArray = preprocess_input(imgArray)  
    return imgArray

# Image preprocessing and flattening 
def extract_features(imgPath):
    imgArray = preprocess_image(imgPath)
    features = model.predict(imgArray)  
    return features.flatten()  


# Save cache
def save_cache(featureDB):
    with open(cacheFile, "w") as f:
        json.dump({k: v.tolist() for k, v in featureDB.items()}, f)

# Load cache
def load_cache():
    if os.path.exists(cacheFile):
        with open(cacheFile, "r") as f:
            try:
                return {k: np.array(v) for k, v in json.load(f).items()}
            except json.JSONDecodeError:
                print("Warning: Cache file corrupted. Rebuilding cache.")
    return {}

# Process each image
def process_image(filename):
    imgPath = os.path.join(databasePath, filename)
    return filename, extract_features(imgPath)

# Build feature database with threading
def build_feature_database():
    featureDB = load_cache()
    imageFiles = [f for f in os.listdir(databasePath) if f.endswith((".webp", ".jpg", ".jpeg", ".png"))]
    newFiles = [f for f in imageFiles if f not in featureDB]

    if newFiles:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(process_image, newFiles)
            featureDB.update(results)

        save_cache(featureDB)

    return featureDB

# Improved matching using cosine similarity
def find_best_match(inputFeatures, featureDB):
    bestMatch = None
    bestScore = -1  # Cosine similarity ranges from -1 to 1

    for filename, features in featureDB.items():
        #if not filename.startswith("005"):  
            #continue  
        
        similarity = cosine_similarity([inputFeatures], [features])[0][0]
        #print(f"Comparing with {filename}: Score = {similarity:.4f}")  # Debugging
        
        if similarity > bestScore:
            bestScore = similarity
            bestMatch = filename
            
    print(f"Comparing with {bestMatch}: Score = {bestScore}")
    return bestMatch if bestScore > 0.65 else None  # Adjust threshold as needed
# Initialize camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Building feature database...")
featureDatabase = build_feature_database()
print("Feature database ready!")

# Crop function
def crop_image(frame, x, y, width, height):
    return frame[y:y + height, x:x + width]

# Video loop
while True:
    x, y, width, height = 130, 40, 270, 390  
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    cv2.rectangle(frame, (x-3, y-3), (x + width+6, y + height+6), (0, 255, 0), 2)
    cv2.imshow("Press SPACE to capture, 'q' to quit", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # Press SPACE
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        croppedImage = crop_image(frame, x, y, width, height)
        croppedImagePath = f"{outputDir}/Cropped_card_{timestamp}.jpg"
        cv2.imwrite(croppedImagePath, croppedImage)

        # Extract features & find best match
        inputFeatures = extract_features(croppedImagePath)
        matchedFilename = find_best_match(inputFeatures, featureDatabase)

        print("Best Match Found:" if matchedFilename else "No close match found.")
        if matchedFilename:
            print(matchedFilename)

    elif key == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
