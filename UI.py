# UI.py
# Writes ONLY to CardList.csv (no Bulk_Add)
# Camera UI:
# - Auto-open camera
# - Stable window (labels ignore pixmap size hints)
# - Crop-to-fill preview (no black bars)
# - Rotate/crop options, saved to ui_settings.json
# - Next/Prev across top matches
# - "Add to card list" enabled after a successful scan; disabled again after adding

import os
import json
import time
import platform
import threading
import cv2
import numpy as np
from typing import List, Tuple, Optional

from PySide6.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QCheckBox,
    QDialog, QFormLayout, QLineEdit, QListWidget, QListWidgetItem, QProgressBar,
    QMessageBox, QComboBox, QSizePolicy, QSplitter
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QTimer

from PhotoMatching import (
    extract_features, find_best_matches, build_feature_database, databasePath,
    update_cardlist, get_available_sets, load_cache, visualize_activation_overlay,
    is_probably_foil
)

SETTINGS_FILE = "ui_settings.json"

class SettingsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(360, 520)

        layout = QFormLayout()

        # Camera index selector
        self.camera_combo = QComboBox()
        self.camera_indices = self.scan_cameras(max_index=8)
        for idx in self.camera_indices:
            self.camera_combo.addItem(f"Camera {idx}", userData=idx)
        current_idx = getattr(parent, "camera_index", 0)
        if current_idx in self.camera_indices:
            self.camera_combo.setCurrentIndex(self.camera_indices.index(current_idx))
        layout.addRow("Camera:", self.camera_combo)

        self.keep_foil_checked = QCheckBox("Keep Foil Checked")
        self.keep_foil_checked.setChecked(getattr(parent, "keep_foil_checked", False))
        layout.addRow(self.keep_foil_checked)

        self.rotate_checkbox = QCheckBox("Rotate preview & capture 180°")
        self.rotate_checkbox.setChecked(getattr(parent, "rotate_display", False))
        layout.addRow(self.rotate_checkbox)

        self.crop_checkbox = QCheckBox("Scan only inside focus box")
        self.crop_checkbox.setChecked(getattr(parent, "crop_to_focus", True))
        layout.addRow(self.crop_checkbox)

        # Confidence & debug
        self.confidence_input = QLineEdit(str(int(100 * getattr(parent, "confidence_threshold", 0.90))))
        layout.addRow("Confidence Threshold (%):", self.confidence_input)

        self.debug_mode_checkbox = QCheckBox("Enable Debug Mode (heatmap overlay)")
        self.debug_mode_checkbox.setChecked(getattr(parent, "debug_mode", False))
        layout.addRow(self.debug_mode_checkbox)

        # Set filter
        self.set_list = QListWidget()
        self.set_list.setSelectionMode(QListWidget.MultiSelection)
        current_selected = set(getattr(parent, "selected_sets", []))
        for set_code in get_available_sets():
            item = QListWidgetItem(set_code)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if set_code in current_selected else Qt.Unchecked)
            self.set_list.addItem(item)
        layout.addRow("Filter Sets:", self.set_list)

        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_settings)
        layout.addRow(apply_button)
        self.setLayout(layout)

    def scan_cameras(self, max_index: int = 8) -> List[int]:
        found = []
        for i in range(max_index):
            if platform.system() == "Windows":
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if not (cap and cap.isOpened()):
                    cap = cv2.VideoCapture(i)
            else:
                cap = cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                found.append(i)
                cap.release()
        return found or [0]

    def apply_settings(self):
        if self.parent():
            self.parent().keep_foil_checked = self.keep_foil_checked.isChecked()
            try:
                pct = float(self.confidence_input.text())
                self.parent().confidence_threshold = max(0.0, min(1.0, pct / 100.0))
            except Exception:
                self.parent().confidence_threshold = 0.90
            self.parent().debug_mode = self.debug_mode_checkbox.isChecked()

            # sets
            selected = []
            for i in range(self.set_list.count()):
                it = self.set_list.item(i)
                if it.checkState() == Qt.Checked:
                    selected.append(it.text())
            self.parent().selected_sets = selected

            # view options
            self.parent().camera_index = self.camera_combo.currentData()
            self.parent().rotate_display = self.rotate_checkbox.isChecked()
            self.parent().crop_to_focus = self.crop_checkbox.isChecked()

            self.parent().save_settings()
        self.close()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Card Matcher — Camera")
        self.resize(1280, 800)

        # State
        self.featureDB = load_cache()  # preload from cache, refresh in background
        self.selected_sets: List[str] = []
        self.keep_foil_checked = False
        self.confidence_threshold = 0.90
        self.debug_mode = False
        self.camera_index = 0
        self.rotate_display = False
        self.crop_to_focus = True

        self.cap: Optional[cv2.VideoCapture] = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._grab_frame)

        self.watchdog = QTimer(self)
        self.watchdog.setInterval(1000)
        self.watchdog.timeout.connect(self._watchdog_tick)
        self.last_frame_time: Optional[float] = None
        self.read_fail_count = 0
        self.max_read_fail = 20

        self.last_frame: Optional[np.ndarray] = None

        # Matches state
        self.last_matches: List[Tuple[str, float]] = []
        self.current_match_idx: int = 0

        self.load_settings()

        # UI ------------------------------------------------------------------
        self.progress = QProgressBar()
        self.progress.setFixedHeight(6)
        self.progress.setTextVisible(True)
        self.progress.setValue(0)

        # Left: live preview (crop-to-fill). Use Ignored policy so pixmap size doesn’t grow window
        self.preview_label = QLabel("Camera stopped")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid #444; background: #111; color: #bbb;")
        self.preview_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.preview_label.setMinimumSize(640, 360)

        # Right: match image (keep aspect). Also use Ignored policy to avoid growth loops
        self.image_label = QLabel("Best match image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #444; background: #111; color: #bbb;")
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setMinimumSize(360, 360)

        self.match_label = QLabel("—")
        self.match_label.setAlignment(Qt.AlignCenter)

        self.prev_btn = QPushButton("◀ Prev")
        self.next_btn = QPushButton("Next ▶")
        self.prev_btn.clicked.connect(self.prev_match)
        self.next_btn.clicked.connect(self.next_match)
        self.match_pos_label = QLabel("")

        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self.open_settings)

        self.capture_btn = QPushButton("Scan Card")
        self.capture_btn.clicked.connect(self.capture_and_match)
        self.capture_btn.setStyleSheet(
            "background-color: #ff9800; color: white; font-weight: bold; "
            "font-size: 18px; padding: 12px 24px; border-radius: 8px;"
        )

        self.foil_check = QCheckBox("Foil")
        self.foil_check.setChecked(self.keep_foil_checked)

        self.count_edit = QLineEdit("1")
        self.count_edit.setFixedWidth(60)

        self.add_csv_btn = QPushButton("Add to card list")
        self.add_csv_btn.clicked.connect(self.add_to_csv)
        self.add_csv_btn.setEnabled(False)  # enabled after a successful scan

        # Inline CSV status
        self.csv_status = QLabel("")
        self.csv_status.setStyleSheet("color: #8bc34a; padding-left: 8px;")

        # --- Top bar
        top = QHBoxLayout()
        top.addWidget(self.start_btn)
        top.addWidget(self.stop_btn)
        top.addWidget(self.settings_btn)
        top.addStretch()

        # --- Split panels
        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.preview_label)
        splitter.addWidget(self.image_label)
        splitter.setSizes([600, 600])  # balanced start
        # --- Bottom: controls
        under = QHBoxLayout()
        under.addWidget(self.capture_btn)
        under.addStretch()
        under.addWidget(self.prev_btn)
        under.addWidget(self.match_pos_label)
        under.addWidget(self.next_btn)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.progress)
        layout.addWidget(splitter, stretch=1)
        layout.addLayout(under)
        layout.addWidget(self.match_label)

        bottom = QHBoxLayout()
        bottom.addWidget(self.foil_check)
        bottom.addWidget(QLabel("Count:"))
        bottom.addWidget(self.count_edit)
        bottom.addWidget(self.add_csv_btn)
        bottom.addWidget(self.csv_status)
        bottom.addStretch()
        layout.addLayout(bottom)

        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.setLayout(layout)

        self.image_label.hide()
        self.start_db_build_in_background()

    # ---- Settings persistence ----
    def load_settings(self):
        if not os.path.exists(SETTINGS_FILE):
            return
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                s = json.load(f)
            self.camera_index = s.get("camera_index", self.camera_index)
            self.keep_foil_checked = s.get("keep_foil_checked", self.keep_foil_checked)
            self.confidence_threshold = s.get("confidence_threshold", self.confidence_threshold)
            self.debug_mode = s.get("debug_mode", self.debug_mode)
            self.selected_sets = s.get("selected_sets", [])
            self.rotate_display = s.get("rotate_display", self.rotate_display)
            self.crop_to_focus = s.get("crop_to_focus", self.crop_to_focus)
        except Exception:
            pass

    def save_settings(self):
        s = {
            "camera_index": self.camera_index,
            "keep_foil_checked": self.keep_foil_checked,
            "confidence_threshold": float(self.confidence_threshold),
            "debug_mode": self.debug_mode,
            "selected_sets": self.selected_sets,
            "rotate_display": self.rotate_display,
            "crop_to_focus": self.crop_to_focus,
        }
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(s, f, indent=2)
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            self.save_settings()
        finally:
            super().closeEvent(event)

    # ---- Background DB build (non-blocking) ----
    def start_db_build_in_background(self):
        self._db_progress_pct = 0
        self._db_progress_timer = QTimer(self)
        self._db_progress_timer.setInterval(100)
        self._db_progress_timer.timeout.connect(self._tick_db_progress)

        def cb(pct, _last):
            self._db_progress_pct = int(pct)

        def worker():
            try:
                db = build_feature_database(cb)
                self.featureDB.update(db)
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()
        self._db_progress_timer.start()

    def _tick_db_progress(self):
        try:
            self.progress.setValue(int(getattr(self, "_db_progress_pct", 0)))
        except Exception:
            pass
        if getattr(self, "_db_progress_pct", 0) >= 100:
            self._db_progress_timer.stop()

    # -------- Camera control --------
    def start_camera(self):
        self.stop_camera()  # just in case

        # Prefer CAP_DSHOW on Windows
        if platform.system() == "Windows":
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not (self.cap and self.cap.isOpened()):
                self.cap = cv2.VideoCapture(self.camera_index)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)

        if not (self.cap and self.cap.isOpened()):
            self._fail_and_stop(f"Failed to open camera index {self.camera_index}.")
            return

        # Optional resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Warm-up
        ok = False
        for _ in range(20):
            ret, frm = self.cap.read()
            if ret and frm is not None:
                ok = True
                break
            time.sleep(0.05)
        if not ok:
            self._fail_and_stop("Camera opened but not delivering frames.")
            return

        self.read_fail_count = 0
        self.last_frame_time = time.monotonic()
        self.timer.start(30)
        self.watchdog.start()
        self.match_label.setText("Camera running …")

    def _fail_and_stop(self, message: str):
        self.stop_camera()
        QMessageBox.warning(self, "Camera", message)

    def stop_camera(self):
        self.timer.stop()
        self.watchdog.stop()
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.preview_label.setText("Camera stopped")

    def _watchdog_tick(self):
        if self.cap is None or self.last_frame_time is None:
            return
        if (time.monotonic() - self.last_frame_time) > 3.0:
            self._fail_and_stop("No frames received for 3 seconds. Camera stopped.")

    def _focus_rect(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """Return (fx, fy, fw, fh) for a 63:88 portrait box occupying ~60% of height, centered."""
        card_ratio = 63 / 88.0
        fh = int(h * 0.6)
        fw = int(fh * card_ratio)
        fw = min(fw, w - 4)
        fh = min(fh, h - 4)
        fx = max((w - fw) // 2, 2)
        fy = max((h - fh) // 2, 2)
        return fx, fy, fw, fh

    def _grab_frame(self):
        if not self.cap:
            return
        try:
            ret, frame = self.cap.read()
        except Exception:
            self._fail_and_stop("Camera backend error while reading frames.")
            return

        if not ret or frame is None:
            self.read_fail_count += 1
            if self.read_fail_count > self.max_read_fail:
                self._fail_and_stop("Camera not delivering frames. Stopping preview.")
            return

        self.read_fail_count = 0
        self.last_frame_time = time.monotonic()

        if self.rotate_display:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        self.last_frame = frame

        # Focus overlay
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        fx, fy, fw, fh = self._focus_rect(h, w)
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(overlay, (fx, fy), (fx + fw, fy + fh), color, thickness)

        # Corners
        cl = 40
        cv2.line(overlay, (fx, fy), (fx + cl, fy), color, thickness)
        cv2.line(overlay, (fx, fy), (fx, fy + cl), color, thickness)
        cv2.line(overlay, (fx + fw, fy), (fx + fw - cl, fy), color, thickness)
        cv2.line(overlay, (fx + fw, fy), (fx + fw, fy + cl), color, thickness)
        cv2.line(overlay, (fx, fy + fh), (fx + cl, fy + fh), color, thickness)
        cv2.line(overlay, (fx, fy + fh), (fx, fy + fh - cl), color, thickness)
        cv2.line(overlay, (fx + fw, fy + fh), (fx + fw - cl, fy + fh), color, thickness)
        cv2.line(overlay, (fx + fw, fy + fh), (fx + fw, fy + fh - cl), color, thickness)

        # Crop-to-fill preview (no black bars)
        disp = overlay if not self.debug_mode else visualize_activation_overlay(overlay)
        self._show_on_label(self.preview_label, disp, fill=True)

    # -------- Matching & navigation --------
    def capture_and_match(self):
        if self.last_frame is None:
            QMessageBox.warning(self, "Scan Card", "Camera is not running or no frame available.")
            return

        img_bgr = self.last_frame.copy()

        if self.crop_to_focus:
            h, w = img_bgr.shape[:2]
            fx, fy, fw, fh = self._focus_rect(h, w)
            fx, fy = max(fx, 0), max(fy, 0)
            fw = min(fw, w - fx)
            fh = min(fh, h - fy)
            if fw > 10 and fh > 10:
                img_bgr = img_bgr[fy:fy+fh, fx:fx+fw].copy()

        self.foil_check.setChecked(self.keep_foil_checked or is_probably_foil(img_bgr))

        features = extract_features(img_bgr)
        if features is None:
            self.match_label.setText("Could not extract features from image.")
            self.image_label.hide()
            self.add_csv_btn.setEnabled(False)
            return

        if not self.featureDB:
            self.match_label.setText("No DB found — building… try again in a moment.")
            self.add_csv_btn.setEnabled(False)
            return

        matches = find_best_matches(features, self.featureDB, threshold=self.confidence_threshold)
        matches = self._filter_matches(matches)
        if not matches:
            self.match_label.setText("No match found.")
            self.image_label.hide()
            self.add_csv_btn.setEnabled(False)
            return

        self.last_matches = matches[:20]
        self.current_match_idx = 0
        self._show_match_at(self.current_match_idx)
        self.add_csv_btn.setEnabled(True)

    def _show_match_at(self, idx: int):
        if not self.last_matches:
            return
        idx = max(0, min(idx, len(self.last_matches)-1))
        self.current_match_idx = idx
        fname, score = self.last_matches[idx]
        self.match_label.setText(f"{score:.3f}  {fname}")
        self.match_pos_label.setText(f"{idx+1} / {len(self.last_matches)}")

        match_path = os.path.join(databasePath, fname)
        img = cv2.imread(match_path, cv2.IMREAD_COLOR)
        if img is not None:
            self._show_on_label(self.image_label, img, fill=False)
            self.image_label.show()
        else:
            self.image_label.hide()

    def prev_match(self):
        if not self.last_matches:
            return
        self._show_match_at(self.current_match_idx - 1)

    def next_match(self):
        if not self.last_matches:
            return
        self._show_match_at(self.current_match_idx + 1)

    def _filter_matches(self, matches: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        if not matches:
            return []
        use_sets = set(self.selected_sets or [])
        if not use_sets:
            return matches
        out = []
        for fname, score in matches:
            ok_set = True
            if use_sets:
                ok_set = (len(fname) >= 3 and fname[:3] in use_sets)
            if ok_set:
                out.append((fname, score))
        return out

    def open_settings(self):
        dlg = SettingsWindow(self)
        dlg.exec()
        self.foil_check.setChecked(self.keep_foil_checked)

    def add_to_csv(self):
        if not self.last_matches:
            self.csv_status.setStyleSheet("color: #f44336; padding-left: 8px;")
            self.csv_status.setText("No match to add.")
            QTimer.singleShot(3500, lambda: self.csv_status.setText(""))
            return
        fname = self.last_matches[self.current_match_idx][0]
        try:
            cnt = int(self.count_edit.text())
        except Exception:
            cnt = 1
        is_foil = self.foil_check.isChecked()

        update_cardlist(fname, is_foil, cnt)

        self.csv_status.setStyleSheet("color: #8bc34a; padding-left: 8px;")
        self.csv_status.setText(f"Added {cnt} of {fname} to CardList.csv")
        self.add_csv_btn.setEnabled(False)  # stays disabled until next successful scan
        QTimer.singleShot(3500, lambda: self.csv_status.setText(""))

    # -------- Helpers --------
    def _crop_to_fill(self, img_bgr, target_w: int, target_h: int):
        """Center-crop img to match target aspect (cover). Always returns a C-contiguous copy."""
        if img_bgr is None or img_bgr.size == 0 or target_w <= 0 or target_h <= 0:
            return img_bgr
        h, w = img_bgr.shape[:2]
        if h == 0 or w == 0:
            return img_bgr
        img_aspect = w / float(h)
        target_aspect = target_w / float(target_h)
        if img_aspect > target_aspect:
            # image too wide -> crop width
            new_w = int(h * target_aspect)
            x0 = max((w - new_w) // 2, 0)
            x1 = min(x0 + new_w, w)
            cropped = img_bgr[:, x0:x1]
        else:
            # image too tall -> crop height
            new_h = int(w / target_aspect)
            y0 = max((h - new_h) // 2, 0)
            y1 = min(y0 + new_h, h)
            cropped = img_bgr[y0:y1, :]

        # return a C-contiguous copy
        return np.ascontiguousarray(cropped)


    def _show_on_label(self, label: QLabel, img_bgr, fill: bool = False):
        if img_bgr is None or img_bgr.size == 0:
            return

        if fill:
            img_bgr = self._crop_to_fill(img_bgr, label.width(), label.height())

        # Ensure 3-channel uint8 BGR and C-contiguous
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[-1] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

        if img_bgr.dtype != np.uint8 or not img_bgr.flags["C_CONTIGUOUS"]:
            img_bgr = np.ascontiguousarray(img_bgr, dtype=np.uint8)

        h, w = img_bgr.shape[:2]
        qimg = QImage(img_bgr.data, w, h, img_bgr.strides[0], QImage.Format_BGR888)
        label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                label.size(),
                Qt.IgnoreAspectRatio if fill else Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


if __name__ == "__main__":
    app = QApplication([])
    w = MainWindow()
    w.show()
    QTimer.singleShot(0, w.start_camera)
    app.exec()
