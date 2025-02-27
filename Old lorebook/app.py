#the plan is to get a camera to scan a lorcana card and save it in a DB or .CSV
#then have a way to import the cards into dreamborn

#sick of the manual labor and just want to swipe cards to add them

#later createa better version of dreamborn? 

#step 1 how to train a AI to read cards
#step 2 database/storage of the cards
#step 3 export the collection
#step 4 make sure each step works with eachother 
#bonus, add a confidence value to the scan and possible alternatives? 


import cv2
import pytesseract
from datetime import datetime
from PIL import Image
import pandas as pd
import re
from fuzzywuzzy import fuzz, process 

# Set the path to the Tesseract executable (only required on Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def capture_card_text(camera_index=1, output_dir="captured_cards"):
    # Open the webcam
    cap = cv2.VideoCapture(1)
    print("Press Space to capture the card and 'q' to quit.")
    
    frame_x, frame_y, frame_w, frame_h = 150, 100, 350, 500
    
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to access the camera.")
            break

        # Display the live feed
        cv2.imshow("Card Reader", frame)
        
        card_image = frame[frame_y:frame_y+frame_h, frame_x:frame_x+frame_w]

        # Keypress actions
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Capture image
            # Save the frame as an image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"{output_dir}/card_{timestamp}.jpg"
            
            #cv2.imwrite(image_path, frame)
            cv2.imwrite(image_path,card_image)
            
            print(f"Card image saved: {image_path}")

            # Preprocess the image
            processed_image = preprocess_image(image_path)

            # Perform OCR
            #text = extract_text_from_image(processed_image)
            text = pytesseract.image_to_string(processed_image)
            print(f"Extracted Text:\n{text}")

            # Save text to a file
            save_text_to_file(text, f"{output_dir}/card_{timestamp}.txt")
            print(f"Extracted text saved to: {output_dir}/card_{timestamp}.txt")
            
            
            df = pd.read_csv("Lorebook1-7.csv")  # Replace with your actual file name

            # Function to extract Title and SubTitle from the input text
            def extract_title_subtitle(text):
                text = re.sub(r"\s+", " ", text.strip())  # Normalize whitespace
                parts = text.split("\n", 1)  # Split at first newline
                title = parts[0] if parts else text  # First line as Title
                subtitle = parts[1] if len(parts) > 1 else ""  # Rest as Subtitle
                return title.strip(), subtitle.strip()

            # Input text to search for
            input_text = text  # Replace with your actual input
            extracted_title, extracted_subtitle = extract_title_subtitle(input_text)

# Function to find the best matching row
            def find_best_match(title, subtitle, df):
                best_match = None
                best_score = 0

                for _, row in df.iterrows():
                    title_score = fuzz.ratio(title.lower(), str(row["Title"]).lower())
                    subtitle_score = fuzz.ratio(subtitle.lower(), str(row["SubTitle"]).lower())
                    avg_score = (title_score + subtitle_score) / 2  # Average match score

                    if avg_score > best_score:
                        best_score = avg_score
                        best_match = row

                return best_match if best_score > 80 else None  # Adjust threshold if needed

# Find the best matching row
            matched_row = find_best_match(extracted_title, extracted_subtitle, df)

# Output the result
            if matched_row is not None:
                print("Best Match Found:")
                print(matched_row)
            else:
                print("No close match found.")
            

        elif key == ord('q'):  # Quit
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    #image = image[50:150, 100:500]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to enhance text visibility
    _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

    # Save the preprocessed image for debugging (optional)
    cv2.imwrite(image_path.replace(".jpg", "_processed.jpg"), thresh)

    return thresh

def extract_text_from_image(image):
    # Use Tesseract OCR to extract text
    text = pytesseract.image_to_string(image, lang='eng')
    return text

def save_text_to_file(text, file_path):
    with open(file_path, "w") as file:
        file.write(text)
        

# Create the output directory if it doesn't exist
import os
output_dir = "captured_cards"
os.makedirs(output_dir, exist_ok=True)

# Start the card capture and OCR process
capture_card_text(camera_index=0, output_dir=output_dir)
