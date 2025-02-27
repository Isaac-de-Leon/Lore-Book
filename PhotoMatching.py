#PhotoMatching

import cv2
import numpy as np
import os

# Path to database images (folder containing card images)
database_path = "C:\Lore Book\Card_Images"

# Initialize ORB feature detector
orb = cv2.ORB_create()

# Function to capture an image from the webcam
def capture_image():
    cap = cv2.VideoCapture(1)  # Open webcam (0 = default camera)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        cv2.imshow("Press SPACE to capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Press SPACE to capture
            img_path = "captured_card.jpg"
            cv2.imwrite(img_path, frame)
            print("Image captured!")
            return img_path
           
        elif key == ord('q'):  # Quit
            cap.release()
            cv2.destroyAllWindows()
            break

# Function to find the best match in the database
def find_best_match(input_image_path):
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    keypoints1, descriptors1 = orb.detectAndCompute(input_image, None)

    best_match = None
    best_score = 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for filename in os.listdir(database_path):
        if filename.endswith(".webp") or filename.endswith(".png"):
            db_image = cv2.imread(os.path.join(database_path, filename), cv2.IMREAD_GRAYSCALE)
            keypoints2, descriptors2 = orb.detectAndCompute(db_image, None)

            if descriptors2 is None:
                continue  # Skip if no descriptors

            # Match descriptors
            matches = bf.match(descriptors1, descriptors2)
            score = len(matches)  # More matches = better match

            if score > best_score:
                best_score = score
                best_match = filename

    return best_match

# Capture image from webcam
captured_image = capture_image()

# Find the best matching card
if captured_image:
    matched_filename = find_best_match(captured_image)
    if matched_filename:
        print(f"Best match: {matched_filename}")
    else:
        print("No close match found.")
