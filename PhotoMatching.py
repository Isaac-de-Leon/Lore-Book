import cv2
import numpy as np
import os
import json
import concurrent.futures
import hashlib
import tensorflow as tf
import csv
from datetime import datetime
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load MobileNetV2 model (feature extraction only)
baseModel = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=baseModel.input, outputs=baseModel.output)

databasePath = "Card_Images"
cacheFile = "DBCardCache.json"
outputDir = "captured_cards"
CSV_file = "Bulk_Add.csv"
os.makedirs(outputDir, exist_ok=True)


def extract_features(img):
    #Extract normalized feature vectors from an image.
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            return None

    if img is None or img.size == 0:
        return None

    # Convert grayscale or RGBA images to RGB
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Resize and preprocess image
    img = preprocess_input(cv2.resize(img, (224, 224)).reshape(1, 224, 224, 3))
    features = model.predict(img).flatten()
    return normalize([features])[0] if len(features) > 0 else None


def load_cache():
    #Load cached feature database.
    if os.path.exists(cacheFile):
        with open(cacheFile, "r") as f:
            try:
                return {k: np.array(v, dtype=np.float32) for k, v in json.load(f).items()}
            except json.JSONDecodeError:
                return {}
    return {}


def build_feature_database():
    #Build or load the feature database.
    featureDB = load_cache()
    imageFiles = [f for f in os.listdir(databasePath) if f.endswith((".webp", ".jpg", ".jpeg", ".png"))]
    files_to_process = [f for f in imageFiles if f not in featureDB]

    if not files_to_process:
        return featureDB

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = dict(executor.map(process_image, files_to_process))

    valid_results = {k: v for k, v in results.items() if v is not None}
    featureDB.update(valid_results)

    with open(cacheFile, "w") as f:
        json.dump({k: v.tolist() for k, v in featureDB.items()}, f)

    return featureDB


def process_image(filename):
    #Process an image and extract features.
    imgPath = os.path.join(databasePath, filename)
    img = cv2.imread(imgPath)
    return filename, extract_features(img) if img is not None else None


def find_best_matches(inputFeatures, featureDB, threshold=0.72):
    #Find best matches using cosine similarity.
    inputFeatures = normalize([inputFeatures])[0]
    matches = [(k, cosine_similarity([inputFeatures], [v])[0][0]) for k, v in featureDB.items()]
    return sorted([(f, s) for f, s in matches if s >= threshold], key=lambda x: x[1], reverse=True)

def update_csv(matchedFilename):
    #Updates CSV file when a card is confirmed
    cardSorting = matchedFilename[:-5]
    splitCard = cardSorting.split('-')
    variant = "foil" if int(splitCard[1]) > 204 else "normal"

    if variant == "normal":
        print("Is the card foil? (y/n)")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('y'):
            variant = "foil"

    updated_rows = []
    found = False

    with open(CSV_file, mode="r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0] == splitCard[0] and row[1] == splitCard[1] and row[2] == variant:
                row[3] = str(int(row[3]) + 1)  # Update count
                found = True
            updated_rows.append(row)

    if not found:
        print("New Card!")
        updated_rows.append([splitCard[0], splitCard[1], variant, "1"])

    with open(CSV_file, mode="w", newline="") as file:
        csv.writer(file).writerows(updated_rows)

    with open(CSV_file, mode="r", newline="") as file:
        reader = csv.reader(file)
        print("Current Card List:")
        for row in reader:
            print(row)


# Initialize Feature Database
featureDatabase = build_feature_database()
print("Feature database ready!")
print(f"Total entries: {len(featureDatabase)}")

__all__ = ["extract_features", "find_best_matches", "featureDatabase", "databasePath", "update_csv"]
