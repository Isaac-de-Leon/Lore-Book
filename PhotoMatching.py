import cv2
import numpy as np
import os
import json
import concurrent.futures
import csv
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Model and paths
baseModel = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=baseModel.input, outputs=baseModel.output)

activationModel = Model(inputs=model.input, outputs=baseModel.layers[-2].output)

databasePath = "Card_Images"
cacheFile = "DBCardCache.json"
outputDir = "captured_cards"
CSV_file = "Bulk_Add.csv"
os.makedirs(outputDir, exist_ok=True)

def extract_features(img):
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            return None
    if img is None or img.size == 0:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = preprocess_input(cv2.resize(img, (224, 224)).reshape(1, 224, 224, 3))
    features = model.predict(img).flatten()
    return normalize([features])[0] if len(features) > 0 else None

def visualize_activation_overlay(img, model):
    resized = cv2.resize(img, (224, 224))
    processed = preprocess_input(resized.reshape(1, 224, 224, 3))
    activations = model.predict(processed)[0]
    heatmap = activations.reshape((7, 7, -1)).mean(axis=-1)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (255 * (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

def load_cache():
    if os.path.exists(cacheFile):
        with open(cacheFile, "r") as f:
            try:
                return {k: np.array(v, dtype=np.float32) for k, v in json.load(f).items()}
            except json.JSONDecodeError:
                return {}
    return {}

def process_image(filename):
    imgPath = os.path.join(databasePath, filename)
    img = cv2.imread(imgPath)
    return filename, extract_features(img) if img is not None else (filename, None)

def build_feature_database(progress_callback=None):
    featureDB = load_cache()
    imageFiles = [f for f in os.listdir(databasePath) if f.endswith((".webp", ".jpg", ".jpeg", ".png"))]
    files_to_process = [f for f in imageFiles if f not in featureDB]
    total = len(files_to_process)
    if not files_to_process:
        if progress_callback:
            progress_callback(100, None)
        return featureDB
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for idx, result in enumerate(executor.map(process_image, files_to_process), 1):
            k, v = result
            if v is not None:
                featureDB[k] = v
            if progress_callback:
                progress_callback(int(idx / total * 100), k)
    with open(cacheFile, "w") as f:
        json.dump({k: v.tolist() for k, v in featureDB.items()}, f)
    return featureDB

def find_best_matches(inputFeatures, featureDB, threshold=0.70):
    inputFeatures = normalize([inputFeatures])[0]
    matches = [(k, cosine_similarity([inputFeatures], [v])[0][0]) for k, v in featureDB.items()]
    return sorted([(f, s) for f, s in matches if s >= threshold], key=lambda x: x[1], reverse=True)

def backup_csv():
    if os.path.exists(CSV_file):
        import shutil
        shutil.copy(CSV_file, CSV_file + ".bak")

def update_csv(matchedFilename, is_foil, count=1):
    count = int(count)
    if count < 1:
        return
    backup_csv()
    cardSorting = matchedFilename[:-5]
    splitCard = cardSorting.split('-')
    variant = "foil" if is_foil else "normal"
    updated_rows = []
    found = False
    if not os.path.exists(CSV_file):
        with open(CSV_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Set", "Card", "Variant", "Count"])
    with open(CSV_file, mode="r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0] == splitCard[0] and row[1] == splitCard[1] and row[2] == variant:
                row[3] = str(int(row[3]) + count)
                found = True
            updated_rows.append(row)
    if not found:
        updated_rows.append([splitCard[0], splitCard[1], variant, str(count)])
    with open(CSV_file, mode="w", newline="") as file:
        csv.writer(file).writerows(updated_rows)

def get_available_sets():
    sets = set()
    for fname in os.listdir(databasePath):
        if len(fname) >= 3 and fname[:3].isdigit():
            sets.add(fname[:3])
    return sorted(sets)