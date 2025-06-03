import sys
import cv2
import os
from PySide6.QtWidgets import (
    QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QCheckBox, QDialog, QFormLayout, QLineEdit, QListWidget, QListWidgetItem, QProgressBar
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QTimer
from PhotoMatching import (
    extract_features, find_best_matches, build_feature_database,
    databasePath, update_csv, get_available_sets, load_cache
)

class SettingsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setGeometry(200, 200, 300, 300)
        layout = QFormLayout()
        self.keep_foil_checked = QCheckBox("Keep Foil Checked")
        layout.addRow(self.keep_foil_checked)
        self.set_list = QListWidget()
        self.set_list.setSelectionMode(QListWidget.MultiSelection)
        for set_code in get_available_sets():
            item = QListWidgetItem(set_code)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if (parent and set_code in parent.selected_sets) else Qt.Unchecked)
            self.set_list.addItem(item)
        layout.addRow("Filter Sets:", self.set_list)
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_settings)
        layout.addRow(apply_button)
        self.setLayout(layout)
    def apply_settings(self):
        if self.parent():
            self.parent().keep_foil_checked = self.keep_foil_checked.isChecked()
            selected = []
            for i in range(self.set_list.count()):
                item = self.set_list.item(i)
                if item.checkState() == Qt.Checked:
                    selected.append(item.text())
            self.parent().selected_sets = selected
        self.close()

class CacheProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Building Card Cache")
        self.setModal(True)
        self.setFixedSize(400, 120)
        layout = QVBoxLayout()
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.status_label = QLabel("Starting...", self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)
    def update_progress(self, value, card_name=None):
        self.progress_bar.setValue(value)
        if card_name:
            self.status_label.setText(f"Caching: {card_name}")
        else:
            self.status_label.setText("Caching cards...")

class CardScannerUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Card Scanner")
        self.setGeometry(100, 100, 700, 600)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("No Image Loaded")
        self.image_label.setStyleSheet("border: 1px solid gray; padding: 10px;")
        self.match_label = QLabel(self)
        self.match_label.setAlignment(Qt.AlignCenter)
        self.match_label.setStyleSheet("border: 1px solid gray; padding: 10px;")
        self.match_label.setFixedSize(270, 390)
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
        self.foil_checkbox.setEnabled(False)
        self.count_input = QLineEdit(self)
        self.count_input.setPlaceholderText("Count")
        self.count_input.setFixedWidth(60)
        self.count_input.setEnabled(False)
        self.confirm_button = QPushButton("Confirm", self)
        self.confirm_button.clicked.connect(self.confirm_match)
        self.confirm_button.setEnabled(False)
        self.exit_button = QPushButton("Exit", self)
        self.exit_button.clicked.connect(self.close)
        self.result_label = QLabel("No Scan Performed", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-weight: bold;")
        self.settings_button = QPushButton("Settings", self)
        self.settings_button.clicked.connect(self.open_settings)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
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
        main_layout.addWidget(self.progress_bar)
        self.setLayout(main_layout)
        self.cap = cv2.VideoCapture(1)  # Change to 0 if using the default webcam
        self.timer = QTimer()
        self.timer.timeout.connect(self.display_camera_feed)
        self.timer.start(30)
        self.crop_x, self.crop_y, self.crop_width, self.crop_height = 130, 40, 270, 390
        self.matched_cards = []
        self.current_match_index = 0
        self.keep_foil_checked = False
        self.selected_sets = get_available_sets()
        self.featureDatabase = {}
    def open_settings(self):
        self.settings_window = SettingsWindow(self)
        self.settings_window.keep_foil_checked.setChecked(self.keep_foil_checked)
        self.settings_window.exec()
    def display_camera_feed(self):
        ret, frame = self.cap.read()
        if not ret:
            self.image_label.setText("Failed to access camera")
            return
        cv2.rectangle(frame, (self.crop_x, self.crop_y),
                      (self.crop_x + self.crop_width, self.crop_y + self.crop_height),
                      (0, 255, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.detect_button.setEnabled(True)
    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.result_label.setText("Failed to capture image.")
            return
        cropped_frame = frame[self.crop_y:self.crop_y + self.crop_height,
                              self.crop_x:self.crop_x + self.crop_width]
        input_features = extract_features(cropped_frame)
        if input_features is None:
            self.result_label.setText("Feature extraction failed.")
            return
        self.matched_cards = find_best_matches(input_features, self.featureDatabase)
        self.matched_cards = [
            (fname, score) for fname, score in self.matched_cards
            if fname[:3] in self.selected_sets
        ]
        if not self.matched_cards:
            self.result_label.setText("No match found!")
            return
        self.current_match_index = 0
        self.show_current_match()
        self.prev_button.setEnabled(len(self.matched_cards) > 1)
        self.next_button.setEnabled(len(self.matched_cards) > 1)
        self.foil_checkbox.setEnabled(True)
        self.count_input.setEnabled(True)
        self.confirm_button.setEnabled(True)
    def confirm_match(self):
        if not self.matched_cards:
            return
        match_filename, _ = self.matched_cards[self.current_match_index]
        is_foil = self.foil_checkbox.isChecked()
        try:
            count = int(self.count_input.text() or 1)
            if count < 1:
                raise ValueError
        except ValueError:
            self.result_label.setText("Please enter a valid count (positive integer).")
            return
        update_csv(match_filename, is_foil, count)
        self.result_label.setText(f"Card Confirmed: {match_filename}\nFoil: {'Yes' if is_foil else 'No'}\nCount: {count}")
        if not self.keep_foil_checked:
            self.foil_checkbox.setChecked(False)
        self.count_input.setText("")
        self.count_input.setEnabled(False)
        self.confirm_button.setEnabled(False)
    def show_current_match(self):
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
        if self.matched_cards:
            self.current_match_index = (self.current_match_index + 1) % len(self.matched_cards)
            self.show_current_match()
    def previous_match(self):
        if self.matched_cards:
            self.current_match_index = (self.current_match_index - 1) % len(self.matched_cards)
            self.show_current_match()
    def display_match_image(self, img_path):
        pixmap = QPixmap(img_path)
        pixmap = pixmap.scaled(270, 390, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.match_label.setPixmap(pixmap)
    def display_image(self, img_path):
        pixmap = QPixmap(img_path)
        pixmap = pixmap.scaled(270, 390, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.match_label.setPixmap(pixmap)
    def start_caching(self):
        # Get the current cache and list of images
        featureDB = load_cache()
        imageFiles = [f for f in os.listdir(databasePath) if f.endswith((".webp", ".jpg", ".jpeg", ".png"))]
        files_to_process = [f for f in imageFiles if f not in featureDB]

        if files_to_process:
            # Only show popup if there are new images to process
            self.cache_dialog = CacheProgressDialog(self)
            self.cache_dialog.show()
            def update_progress(value, card_name=None):
                self.cache_dialog.update_progress(value, card_name)
                QApplication.processEvents()
            def progress_callback(value, card_name=None):
                update_progress(value, card_name)
            self.featureDatabase = build_feature_database(progress_callback=progress_callback)
            self.cache_dialog.accept()
        else:
            # Just load the cache if nothing new
            self.featureDatabase = featureDB

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CardScannerUI()
    window.show()
    window.start_caching()
    sys.exit(app.exec())
