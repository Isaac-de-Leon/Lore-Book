import sys
import cv2
import numpy as np
import os
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QTimer
from PhotoMatching import extract_features, find_best_matches, featureDatabase, databasePath, update_csv

class CardScannerUI(QWidget):
    def __init__(self):
        super().__init__()

        # Window Settings
        self.setWindowTitle("Card Scanner")
        self.setGeometry(100, 100, 700, 600)

        # Camera Feed
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; padding: 10px;")

        # Matched Card Display
        self.match_label = QLabel(self)
        self.match_label.setAlignment(Qt.AlignCenter)
        self.match_label.setStyleSheet("border: 1px solid gray; padding: 10px;")
        self.match_label.setFixedSize(270, 390)

        # Buttons
        self.detect_button = QPushButton("Scan Card", self)
        self.detect_button.clicked.connect(self.capture_frame)

        self.next_button = QPushButton("Next Match", self)
        self.next_button.clicked.connect(self.next_match)
        self.next_button.setEnabled(False)

        self.prev_button = QPushButton("Previous Match", self)
        self.prev_button.clicked.connect(self.previous_match)
        self.prev_button.setEnabled(False)

        self.confirm_button = QPushButton("Confirm", self)
        self.confirm_button.clicked.connect(self.confirm_match)
        self.confirm_button.setEnabled(False)

        self.exit_button = QPushButton("Exit", self)
        self.exit_button.clicked.connect(self.close)

        # Scan Results Label
        self.result_label = QLabel("No Scan Performed", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-weight: bold;")

        # Layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.confirm_button)
        button_layout.addWidget(self.next_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.detect_button)
        main_layout.addWidget(self.result_label)
        main_layout.addWidget(self.match_label)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.exit_button)
        self.setLayout(main_layout)

        self.cap = cv2.VideoCapture(1)  # Change to 0 if using the default webcam
        self.timer = QTimer()
        self.timer.timeout.connect(self.display_camera_feed)
        self.timer.start(30)  # Update every 30ms

        self.crop_x, self.crop_y, self.crop_width, self.crop_height = 130, 40, 270, 390
        self.matched_cards = []
        self.current_match_index = 0

    def display_camera_feed(self):
        #Live feed from the camera with a green border indicating crop area.
        ret, frame = self.cap.read()
        if not ret:
            self.image_label.setText("Failed to access camera")
            return

        # Draw green rectangle on the frame
        cv2.rectangle(frame, (self.crop_x, self.crop_y), 
                      (self.crop_x + self.crop_width, self.crop_y + self.crop_height), 
                      (0, 255, 0), 2)

        # Convert the frame to RGB for display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Display in QLabel
        self.image_label.setPixmap(pixmap)

    def capture_frame(self):
        #Capture the frame, crop it, and match against database.
        ret, frame = self.cap.read()
        if not ret:
            self.result_label.setText("Failed to capture image.")
            return

        # Crop the image
        cropped_frame = frame[self.crop_y:self.crop_y + self.crop_height, 
                              self.crop_x:self.crop_x + self.crop_width]

        # Extract features and find matches
        input_features = extract_features(cropped_frame)
        if input_features is None:
            self.result_label.setText("Feature extraction failed.")
            return

        self.matched_cards = find_best_matches(input_features, featureDatabase)

        if not self.matched_cards:
            self.result_label.setText("No match found!")
            return

        # Reset match index
        self.current_match_index = 0
        self.display_match()

    def display_match(self):
        #Display the current match from matched_cards list.
        if not self.matched_cards:
            self.result_label.setText("No matches to display!")
            return

        match_filename, similarity = self.matched_cards[self.current_match_index]
        similarity_percentage = round(similarity * 100, 2)

        # Load and display matched card image
        match_img_path = os.path.join(databasePath, match_filename)
        if os.path.exists(match_img_path):
            pixmap = QPixmap(match_img_path)
            pixmap = pixmap.scaled(270, 390, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.match_label.setPixmap(pixmap)
            self.result_label.setText(f"Match {self.current_match_index + 1} of {len(self.matched_cards)}\n"
                                      f"{match_filename}\nConfidence: {similarity_percentage}%")

        # Enable/disable navigation buttons
        self.prev_button.setEnabled(self.current_match_index > 0)
        self.next_button.setEnabled(self.current_match_index < len(self.matched_cards) - 1)
        self.confirm_button.setEnabled(True)

    def next_match(self):
        #Navigate to the next match.
        if self.current_match_index < len(self.matched_cards) - 1:
            self.current_match_index += 1
            self.display_match()

    def previous_match(self):
        #Navigate to the previous match.
        if self.current_match_index > 0:
            self.current_match_index -= 1
            self.display_match()

    def confirm_match(self):
        #Confirm the selected match and update CSV.
        if not self.matched_cards:
            return

        match_filename, _ = self.matched_cards[self.current_match_index]
        update_csv(match_filename)

        self.result_label.setText(f"Confirmed: {match_filename}")
        self.confirm_button.setEnabled(False)  # Disable after confirmation

    def closeEvent(self, event):
        #Stop the camera when the window is closed.
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CardScannerUI()
    window.show()
    sys.exit(app.exec())
