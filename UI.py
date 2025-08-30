
import sys
import os
import re
import json
import cv2
from glob import glob
from PySide6.QtWidgets import (
    QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QCheckBox,
    QDialog, QFormLayout, QLineEdit, QListWidget, QListWidgetItem, QProgressBar, QMessageBox
)
from PySide6.QtGui import QPixmap, QImage, QTextDocument
from PySide6.QtCore import Qt, QTimer
from PhotoMatching import (
    extract_features, find_best_matches, build_feature_database, databasePath,
    update_csv, get_available_sets, load_cache, visualize_activation_overlay
)

METADATA_FILE = "card_metadata.json"
TAG_SUGGESTIONS = ["Promo", "Alt Art", "Signed", "Foil", "Full Art"]

def load_metadata():
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def suggest_tag_from_filename(fname):
    name = fname.lower()
    for tag in TAG_SUGGESTIONS:
        if tag.lower().replace(" ", "") in name:
            return tag
    return ""

class SettingsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setGeometry(200, 200, 300, 400)
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

        self.tag_list = QListWidget()
        self.tag_list.setSelectionMode(QListWidget.MultiSelection)
        for tag in TAG_SUGGESTIONS:
            item = QListWidgetItem(tag)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.tag_list.addItem(item)
        layout.addRow("Filter Tags:", self.tag_list)

        self.confidence_input = QLineEdit("90")
        layout.addRow("Confidence Threshold (%):", self.confidence_input)
        self.debug_mode_checkbox = QCheckBox("Enable Debug Mode")
        layout.addRow(self.debug_mode_checkbox)
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_settings)
        layout.addRow(apply_button)
        self.setLayout(layout)

    def apply_settings(self):
        if self.parent():
            self.parent().keep_foil_checked = self.keep_foil_checked.isChecked()
            self.parent().confidence_threshold = float(self.confidence_input.text()) / 100.0
            self.parent().debug_mode = self.debug_mode_checkbox.isChecked()
            selected = []
            for i in range(self.set_list.count()):
                item = self.set_list.item(i)
                if item.checkState() == Qt.Checked:
                    selected.append(item.text())
            self.parent().selected_sets = selected
            tag_selected = []
            for i in range(self.tag_list.count()):
                item = self.tag_list.item(i)
                if item.checkState() == Qt.Checked:
                    tag_selected.append(item.text())
            self.parent().selected_tags = tag_selected
        self.close()

class DatabaseManager(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Database Manager")
        self.setFixedSize(500, 600)
        layout = QVBoxLayout()
        self.card_list = QListWidget(self)
        self.card_list.setSelectionMode(QListWidget.SingleSelection)
        self.metadata = load_metadata()
        for fname in os.listdir(databasePath):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                tag = self.metadata.get(fname, "")
                label = f"{fname}  [Tag: {tag}]" if tag else fname
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, fname)
                self.card_list.addItem(item)
        self.metadata_label = QLabel("Custom Tag:")
        self.metadata_input = QLineEdit(self)
        self.metadata_save = QPushButton("Save Tag")
        self.metadata_save.clicked.connect(self.save_metadata)
        self.remove_button = QPushButton("Remove Selected Card")
        self.remove_button.clicked.connect(self.remove_card)
        self.card_list.currentItemChanged.connect(self.load_metadata)
        layout.addWidget(self.card_list)
        layout.addWidget(self.metadata_label)
        layout.addWidget(self.metadata_input)
        layout.addWidget(self.metadata_save)
        layout.addWidget(self.remove_button)
        self.setLayout(layout)

    def load_metadata(self):
        item = self.card_list.currentItem()
        if item:
            fname = item.data(Qt.UserRole)
            self.metadata_input.setText(self.metadata.get(fname, ""))

    def save_metadata(self):
        item = self.card_list.currentItem()
        if item:
            fname = item.data(Qt.UserRole)
            self.metadata[fname] = self.metadata_input.text()
            with open(METADATA_FILE, "w") as f:
                json.dump(self.metadata, f, indent=2)
            QMessageBox.information(self, "Saved", f"Tag saved for {fname}.")

    def remove_card(self):
        selected = self.card_list.currentItem()
        if selected:
            fname = selected.data(Qt.UserRole)
            confirm = QMessageBox.question(
                self, "Confirm Deletion",
                f"Are you sure you want to delete {fname}?",
                QMessageBox.Yes | QMessageBox.No
            )
            if confirm == QMessageBox.Yes:
                try:
                    os.remove(os.path.join(databasePath, fname))
                    self.card_list.takeItem(self.card_list.row(selected))
                    if fname in self.metadata:
                        del self.metadata[fname]
                        with open(METADATA_FILE, "w") as f:
                            json.dump(self.metadata, f, indent=2)
                    QMessageBox.information(self, "Success", f"Removed {fname}.")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to remove: {e}")
