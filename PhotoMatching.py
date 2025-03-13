import cv2
import numpy as np
import os
import json
import concurrent.futures
import hashlib
import tensorflow as tf
import csv
from datetime import datetime
from keras.applications import MobileNetV2  # type: ignore
from keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from keras.preprocessing import image # type: ignore
from keras.models import Model # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Load MobileNetV2 model (feature extraction only)
baseModel = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=baseModel.input, outputs=baseModel.output)

databasePath = "Card_Images"
cacheFile = "DBCardCache.json"
outputDir = "captured_cards"
CSV_file = "Bulk_Add.csv"
debug = False
os.makedirs(outputDir, exist_ok=True)

# Image preprocessing 
def process_image(filename):
    imgPath = os.path.join(databasePath, filename)



    img = cv2.imread(imgPath)
    if img is None:
        print(f"ERROR: Could not read image {imgPath}")
        return filename, None  # Return None for failure

    features = extract_features(img)

    if features is None or len(features) == 0:
        print(f"ERROR: Feature extraction failed for {filename}")
        return filename, None  # Return None if extraction fails

    return filename, features

def predict_features(model, img):
    return model(img).numpy().flatten()

def extract_features(img):
    if isinstance(img, str):  
        img = cv2.imread(img)
        if img is None:
            print(f"ERROR: Could not read image from path: {img}")
            return None

    if img is None or img.size == 0:
        print("ERROR: Received an empty image!")
        return None

    # Convert grayscale or RGBA images to RGB
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Resize image
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    # Normalize and expand dimensions once
    img = preprocess_input(np.expand_dims(img, axis=0))

    # Extract features
    features = model.predict(img).flatten()
    features = normalize([features])[0]

    if len(features) == 0:
        print("ERROR: Feature extraction failed! Empty vector returned.")
        return None  

    print(f"Extracted feature vector: {features[:5]}... (Total length: {len(features)})")
    return features

def save_cache(featureDB):
    if not featureDB:
        print("Warning: Attempting to save an empty feature database! Aborting save.")
        return

    #Ensure that only numerical feature vectors are saved
    featureDB_cleaned = {k: v.tolist() for k, v in featureDB.items() if isinstance(v, np.ndarray)}

    if not featureDB_cleaned:
        print("ERROR: No valid feature vectors to save. Database is empty!")
        return

    with open(cacheFile, "w") as f:
        json.dump(featureDB_cleaned, f)

    print(f"Successfully saved {len(featureDB_cleaned)} feature vectors to cache.")

# Load cache
def load_cache():
    if os.path.exists(cacheFile):
        with open(cacheFile, "r") as f:
            try:
                cached_data = json.load(f)
                
                #Filter out invalid entries (hashes)
                featureDB = {k: np.array(v, dtype=np.float32) for k, v in cached_data.items() if isinstance(v, list) and len(v) > 0}

                if not featureDB:
                    print("Warning: Cache file exists but contains no valid feature data. Rebuilding cache.")
                    return {}

                print(f"Loaded {len(featureDB)} feature vectors from cache.")
                return featureDB

            except json.JSONDecodeError:
                print("ERROR: Cache file is corrupted. Rebuilding cache.")
                return {}

    print("No cache file found. Generating a new feature database.")
    return {}

def compute_image_hash(image_path):
    #Compute a hash of the images.
    hasher = hashlib.md5()
    with open(image_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def load_previous_hashes():
    #Load the previous stored image hashes
    if os.path.exists("ImageHashes.json"):
        with open("ImageHashes.json", "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}  # Return an empty dict if JSON is corrupt
    return {}

# Process each image
def save_image_hashes(image_hashes):
    #Save the current image hashes
    with open("ImageHashes.json", "w") as f:
        json.dump(image_hashes, f)

# Build feature database with threading
def build_feature_database():
    featureDB = load_cache()
    previous_hashes = load_previous_hashes()
    current_hashes = {}

    imageFiles = [f for f in os.listdir(databasePath) if f.endswith((".webp", ".jpg", ".jpeg", ".png"))]
    files_to_process = []

    for filename in imageFiles:
        imgPath = os.path.join(databasePath, filename)
        img_hash = compute_image_hash(imgPath)
        current_hashes[filename] = img_hash  

        if filename not in previous_hashes or previous_hashes[filename] != img_hash:
            files_to_process.append(filename)

    if not files_to_process:
        if featureDB:
            print(f"No changes detected. Using cached features ({len(featureDB)} entries).")
            return featureDB  
        else:
            print("No changes detected, but cache is empty! Rebuilding database...")

    print(f"Changes detected! Processing {len(files_to_process)} files. Rebuilding database...")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_image, files_to_process))

    valid_results = {filename: np.array(features) for filename, features in results if features is not None}

    if not valid_results:
        print("ERROR: No valid features were extracted! Database is empty.")
        return {}

    print(f"Successfully built database with {len(valid_results)} feature vectors.")

    save_cache(valid_results)
    save_image_hashes(current_hashes)  
    return valid_results


# Improved matching using cosine similarity
def find_best_matches(inputFeatures, featureDB, threshold=0.75):
    inputFeatures = normalize([inputFeatures])[0]  # Normalize input once
    matches = []

    for filename, features in featureDB.items():
        similarity = cosine_similarity([inputFeatures], [features])[0][0]

        if similarity >= threshold:
            matches.append((filename, similarity))

    return sorted(matches, key=lambda x: x[1], reverse=True) # Sort by similarity


# CSV file handling
def update_csv(matchedFilename):
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
    return

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
        if debug:
            cv2.imwrite(croppedImagePath, croppedImage)

        # Extract features & find best match
        inputFeatures = extract_features(croppedImage)
        matchedCards = find_best_matches(inputFeatures, featureDatabase)

        print("Best Match Found:" if matchedCards else "No close match found.")
        if matchedCards and len(matchedCards) > 0:
            print("Top Matches Found:")
            for i, (filename, similarity) in enumerate(matchedCards):
                print(f"{i + 1}. {filename} (Score: {similarity:.4f})")

            # Display the closest match
            top_match = matchedCards[0][0]  # Get the most accurate match
            print(f"\nMost likely match: {top_match}. Is this correct? (y/n)")
            cardImage = cv2.imread(f"{databasePath}/{top_match}")
            cv2.imshow("Matched Card", cardImage)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('y'):
                
                update_csv(top_match)
                cv2.destroyWindow("Matched Card")
                                
            elif key == ord('n') and len(matchedCards) > 1:
                print("Card not matched. Showing alternatives...")
                cv2.destroyWindow("Matched Card")
                for filename, similarity in matchedCards[1:]:  # Skip the top match
                    print(f"Alternative match: {filename} (Score: {similarity:.4f})")
                    alt_card_image = cv2.imread(f"{databasePath}/{filename}")
                    cv2.imshow("Alternative Match", alt_card_image)
                    key = cv2.waitKey(0) & 0xFF

                    if key == ord('y'):
                        print(f"Selected alternative: {filename}")
                        update_csv(filename)
                        break  # Stop once a match is confirmed
                    elif key == ord('n'):
                        cv2.destroyWindow("Alternative Match")
                        continue  # Show the next alternative
                    else:
                        print("Invalid input, skipping to next.")
                        
                        
        cv2.destroyAllWindows()
        
    elif key == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
