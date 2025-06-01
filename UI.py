import sys
import cv2
import numpy as np
import os
from PySide6.QtWidgets import (
    QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QCheckBox, QDialog, QComboBox, QFormLayout, QPushButton, QLineEdit)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QTimer
from PhotoMatching import extract_features, find_best_matches, featureDatabase, databasePath, update_csv

class SettingsWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings")
        self.setGeometry(200, 200, 300, 150)

        layout = QFormLayout()

        # Dropdown
        self.dropdown = QComboBox()
        self.dropdown.addItems(["Camera 0", "Camera 1", "Camera 2"])
        layout.addRow("Select Camera:", self.dropdown)

        # Button to apply or save settings
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_settings)
        layout.addRow(apply_button)

        self.setLayout(layout)

    def apply_settings(self):
        selected_option = self.dropdown.currentText()
        print(f"Selected camera option: {selected_option}")
        self.close()



class CardScannerUI(QWidget):
    def __init__(self):
        super().__init__()

        # Window Settings
        self.setWindowTitle("Card Scanner")
        self.setGeometry(100, 100, 700, 600)
        

        # Camera Feed
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("No Image Loaded")
        self.image_label.setStyleSheet("border: 1px solid gray; padding: 10px;")

        # Matched Card Display
        self.match_label = QLabel(self)
        self.match_label.setAlignment(Qt.AlignCenter)
        self.match_label.setStyleSheet("border: 1px solid gray; padding: 10px;")
        self.match_label.setFixedSize(270, 390)

        # Buttons
        self.upload_button = QPushButton("Upload Image", self)
        self.upload_button.clicked.connect(self.upload_image)

        self.detect_button = QPushButton("Scan Card", self)
        self.detect_button.clicked.connect(self.capture_frame)
        
        self.prev_button = QPushButton("Previous Match", self)
        self.prev_button.clicked.connect(self.previous_match)
        self.prev_button.setEnabled(False)

        self.next_button = QPushButton("Next Match", self)
        self.next_button.clicked.connect(self.next_match)
        self.next_button.setEnabled(False)

        self.foil_checkbox = QCheckBox("Foil")
        self.foil_checkbox.setEnabled(False)  # Enable after scanning

        # Add this block before using self.count_input
        self.count_input = QLineEdit(self)
        self.count_input.setPlaceholderText("Count")
        self.count_input.setFixedWidth(60)
        self.count_input.setEnabled(False)  # Enable after scanning

        self.confirm_button = QPushButton("Confirm", self)
        self.confirm_button.clicked.connect(self.confirm_match)
        self.confirm_button.setEnabled(False)

        self.exit_button = QPushButton("Exit", self)
        self.exit_button.clicked.connect(self.close)

        # Scan Results Label
        self.result_label = QLabel("No Scan Performed", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-weight: bold;")
        
        # Tab Layout
        self.settings_button = QPushButton("Settings", self)
        self.settings_button.clicked.connect(self.open_settings)

        # Layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.foil_checkbox)
        button_layout.addWidget(self.count_input) 
        button_layout.addWidget(self.confirm_button)
        button_layout.addWidget(self.next_button)

        scan_layout = QVBoxLayout()
        scan_layout.addWidget(self.detect_button)
        scan_layout.addWidget(self.upload_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(scan_layout)
        main_layout.addWidget(self.result_label)
        main_layout.addWidget(self.match_label)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.exit_button)
        main_layout.addWidget(self.settings_button)
        main_layout.addWidget(self.exit_button)

        self.setLayout(main_layout)

        # Camera Setup
        self.cap = cv2.VideoCapture(1)  # Change to 0 if using the default webcam
        self.timer = QTimer()
        self.timer.timeout.connect(self.display_camera_feed)
        self.timer.start(30)  # Update every 30ms

        # Crop Box Parameters
        self.crop_x, self.crop_y, self.crop_width, self.crop_height = 130, 40, 270, 390
        self.matched_cards = []
        self.current_match_index = 0
        
        
        # Settings tab
    def open_settings(self):
        self.settings_window = SettingsWindow()
        self.settings_window.exec()
    
        
    def capture_from_webcam(self):
        #Captures an image from the live camera feed.
        ret, frame = self.cap.read()
        if not ret:
            self.result_label.setText("Camera not detected!")
            return

        # Save a temporary image and use it for scanning
        self.image_path = "temp_capture.jpg"
        cv2.imwrite(self.image_path, frame)

        self.display_image(self.image_path)
        self.detect_button.setEnabled(True)

    def display_camera_feed(self):
        #Live feed from the camera
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

    def upload_image(self):
        #Opens file dialog to select an image
        file_path, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.detect_button.setEnabled(True)  # Enable scan button

    def capture_frame(self):
        #Captures the frame, crops it, and matches against database.
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
        self.show_current_match()

        # Enable buttons
        self.prev_button.setEnabled(len(self.matched_cards) > 1)
        self.next_button.setEnabled(len(self.matched_cards) > 1)
        self.foil_checkbox.setEnabled(True)
        self.count_input.setEnabled(True) 
        self.confirm_button.setEnabled(True)

    def confirm_match(self):
        #Confirms the selected card and updates the CSV
        if not self.matched_cards:
            return

        match_filename, _ = self.matched_cards[self.current_match_index]
        is_foil = self.foil_checkbox.isChecked()
        count = self.count_input.text()
        try:
            count = int(count)
            if count < 1:
                raise ValueError
        except ValueError:
            self.result_label.setText("Please enter a valid count (positive integer).")
            return

        update_csv(match_filename, is_foil, count)  # Pass count directly

        self.result_label.setText(f"Card Confirmed: {match_filename}\nFoil: {'Yes' if is_foil else 'No'}\nCount: {count}")
        self.foil_checkbox.setChecked(False)
        self.foil_checkbox.setEnabled(False)
        self.count_input.setText("")
        self.count_input.setEnabled(False)
        self.confirm_button.setEnabled(False)

    def show_current_match(self):
        #Displays the current matched card
        if not self.matched_cards:
            self.result_label.setText("No matches to display!")
            return

        match_filename, similarity = self.matched_cards[self.current_match_index]
        similarity_percentage = round(similarity * 100, 2)

        self.result_label.setText(f"Match {self.current_match_index + 1}/{len(self.matched_cards)}:\n"
                                  f"{match_filename}\nConfidence: {similarity_percentage}%")

        match_path = os.path.join(databasePath, match_filename)
        if os.path.exists(match_path):
            self.display_match_image(match_path)

    def next_match(self):
        #Cycles through matches
        if self.matched_cards:
            self.current_match_index = (self.current_match_index + 1) % len(self.matched_cards)
            self.show_current_match()

    def previous_match(self):
        #Cycles through matches in reverse
        if self.matched_cards:
            self.current_match_index = (self.current_match_index - 1) % len(self.matched_cards)
            self.show_current_match()

    def display_match_image(self, img_path):
        #Displays matched image in the QLabel
        pixmap = QPixmap(img_path)
        pixmap = pixmap.scaled(270, 390, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.match_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CardScannerUI()
    window.show()
    sys.exit(app.exec())
