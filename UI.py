# UI.py
# Writes ONLY to CardList.csv (no Bulk_Add)
# Camera UI with features:
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
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import logging
import logging.handlers

def setup_logging(log_file: str = "card_scanner.log") -> None:
    """
    Configure application-wide logging with both file and console output.
    
    Args:
        log_file: Path to the log file
    """
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir, log_file)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=1024*1024, backupCount=5,  # 1MB per file, keep 5 backups
        encoding='utf-8'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler for warnings and above
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.WARNING)
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress noisy loggers
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('cv2').setLevel(logging.WARNING)
    
    logging.info(f"Logging initialized. Log file: {log_path}")

# Initialize logging
setup_logging()

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
    is_probably_foil, set_database_path, GameType
)

SETTINGS_FILE = "ui_settings.json"


class SettingsWindow(QDialog):
    """
    Settings dialog for camera and UI options.
    
    Handles:
    - Camera selection and configuration
    - UI preferences (foil detection, rotation, cropping)
    - Confidence thresholds for matching
    - Debug mode settings
    - Game and set filtering
    
    Configuration is saved to ui_settings.json on apply.
    """
    logger = logging.getLogger("SettingsWindow")
    
    # Default settings
    DEFAULT_SETTINGS = {
        "camera_index": 0,
        "keep_foil_checked": False,
        "rotate_display": False,
        "crop_to_focus": True,
        "confidence_threshold": 0.90,
        "debug_mode": False,
        "selected_games": {
            "lorcana": True,
            "riftbound": False
        },
        "selected_sets": {
            "lorcana": [],
            "riftbound": []
        }
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(360, 520)
        self.logger.info("Initializing settings window")

        layout = QFormLayout()

        # Simple camera index selector (0-4)
        self.camera_combo = QComboBox()
        for i in range(5):  # Just offer first 5 camera indices
            self.camera_combo.addItem(f"Camera {i}", userData=i)
        current_idx = getattr(parent, "camera_index", 0)
        self.camera_combo.setCurrentIndex(min(current_idx, 4))
        layout.addRow("Camera:", self.camera_combo)
        
        # Add help text
        help_label = QLabel("Note: If camera doesn't work, try a different index\nand restart the application.")
        help_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addRow(help_label)

        # Checkbox for keeping foil checked by default
        self.keep_foil_checked = QCheckBox("Keep Foil Checked")
        self.keep_foil_checked.setChecked(getattr(parent, "keep_foil_checked", False))
        layout.addRow(self.keep_foil_checked)

        # Checkbox for rotating preview/capture
        self.rotate_checkbox = QCheckBox("Rotate preview & capture 180°")
        self.rotate_checkbox.setChecked(getattr(parent, "rotate_display", False))
        layout.addRow(self.rotate_checkbox)

        # Checkbox for cropping to focus box
        self.crop_checkbox = QCheckBox("Scan only inside focus box")
        self.crop_checkbox.setChecked(getattr(parent, "crop_to_focus", True))
        layout.addRow(self.crop_checkbox)

        # Confidence threshold input
        self.confidence_input = QLineEdit(str(int(100 * getattr(parent, "confidence_threshold", 0.90))))
        layout.addRow("Confidence Threshold (%):", self.confidence_input)

        # Debug mode toggle
        self.debug_mode_checkbox = QCheckBox("Enable Debug Mode (heatmap overlay)")
        self.debug_mode_checkbox.setChecked(getattr(parent, "debug_mode", False))
        layout.addRow(self.debug_mode_checkbox)

        # Game type and set filters using QTreeWidget
        from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem
        
        self.set_tree = QTreeWidget()
        self.set_tree.setHeaderHidden(True)
        self.set_tree.setMinimumHeight(200)
        
        # Get current settings
        current_selected = set(getattr(parent, "selected_sets", []))
        
        # Get list of game folders
        card_images_dir = "Card_Images"
        game_folders = []
        if os.path.exists(card_images_dir):
            game_folders = [item for item in os.listdir(card_images_dir) 
                          if os.path.isdir(os.path.join(card_images_dir, item)) and item != "__pycache__"]
        
        # Create tree items for each game folder
        original_path = None
        if hasattr(parent, 'current_db_path'):
            original_path = parent.current_db_path
            
        selected_games = getattr(parent, 'selected_games', {})
        selected_sets = getattr(parent, 'selected_sets', {})
        
        for game_name in game_folders:
            # Create game branch
            game_root = QTreeWidgetItem(self.set_tree, [game_name])
            game_root.setFlags(game_root.flags() | Qt.ItemIsAutoTristate | Qt.ItemIsUserCheckable)
            
            # Set game checkbox state based on saved selection
            is_game_selected = selected_games.get(game_name.lower(), False)
            game_root.setCheckState(0, Qt.Checked if is_game_selected else Qt.Unchecked)
            
            # Get sets for this game
            set_database_path(game_name)
            try:
                game_sets = get_available_sets(GameType.LORCANA if game_name.lower() == "lorcana" else GameType.RIFTBOUND)
                
                # Add sets
                game_selected_sets = selected_sets.get(game_name.lower(), [])
                for set_code in game_sets:
                    set_name = f"Set {set_code}"
                    set_item = QTreeWidgetItem(game_root, [set_name])
                    set_item.setFlags(set_item.flags() | Qt.ItemIsUserCheckable)
                    is_checked = set_code in game_selected_sets
                    set_item.setCheckState(0, Qt.Checked if is_checked else Qt.Unchecked)
                    set_item.setData(0, Qt.UserRole, (game_name.lower(), set_code))
            except Exception as e:
                self.logger.warning(f"Error getting sets for {game_name}: {str(e)}")
                
        # Restore original path if needed
        if original_path:
            set_database_path(original_path)
        
        self.set_tree.expandAll()
        layout.addRow("Filter Sets:", self.set_tree)

        # Apply button
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.apply_settings)
        layout.addRow(apply_button)
        self.setLayout(layout)

    def scan_cameras(self, max_index: int = 8) -> List[int]:
        """Just return list of first few indices."""
        return list(range(5))  # Simply offer first 5 indices

    def apply_settings(self):
        """
        Apply settings to parent MainWindow and save them.
        """
        if self.parent():
            self.parent().keep_foil_checked = self.keep_foil_checked.isChecked()
            try:
                pct = float(self.confidence_input.text())
                self.parent().confidence_threshold = max(0.0, min(1.0, pct / 100.0))
            except Exception:
                self.parent().confidence_threshold = 0.90
            self.parent().debug_mode = self.debug_mode_checkbox.isChecked()

            # Process tree selection
            selected_games = {
                'lorcana': False,
                'riftbound': False
            }
            selected_sets = {
                'lorcana': [],
                'riftbound': []
            }
            
            # Check which game types are selected and collect their sets
            for i in range(self.set_tree.topLevelItemCount()):
                root = self.set_tree.topLevelItem(i)
                game_name = root.text(0).lower()  # Convert to lowercase
                
                # Only process known games
                if game_name in ['lorcana', 'riftbound']:
                    selected_games[game_name] = root.checkState(0) != Qt.Unchecked
                    
                    # Only process sets for selected games
                    if selected_games[game_name]:
                        # Collect selected sets for this game
                        for j in range(root.childCount()):
                            child = root.child(j)
                            if child.checkState(0) == Qt.Checked:
                                set_code = child.text(0).split(" ")[-1]  # Extract set code from "Set XXX"
                                selected_sets[game_name].append(set_code)
            
            self.logger.info(f"Applying settings - Selected games: {selected_games}")
            self.logger.info(f"Selected sets: {selected_sets}")
            
            # Update parent with new selections
            self.parent().selected_games = selected_games
            self.parent().selected_sets = selected_sets
            
            # Update the feature database for the selected game
            for game_name, is_selected in selected_games.items():
                if is_selected:
                    from PhotoMatching import set_database_path, load_cache
                    set_database_path(game_name.capitalize())
                    self.parent().featureDB = load_cache()
                    break

            # view options
            self.parent().camera_index = self.camera_combo.currentData()
            self.parent().rotate_display = self.rotate_checkbox.isChecked()
            self.parent().crop_to_focus = self.crop_checkbox.isChecked()

            self.parent().save_settings()
        self.close()


class MainWindow(QWidget):
    """
    Main application window for camera preview, card matching, and CSV export.
    Handles camera control, UI updates, and user interactions.
    """
    # Add debug logging
    logger = logging.getLogger("MainWindow")
    logger.setLevel(logging.INFO)
    
    @staticmethod
    def _split_filename(filename: str) -> Tuple[str, str]:
        """Split filename into set code and card number parts."""
        from PhotoMatching import _split_filename
        return _split_filename(filename)
        
    def show_error(self, message: str, title: str = "Error", details: Optional[str] = None) -> None:
        """
        Show error message to user and log it.
        
        Args:
            message: Main error message to display
            title: Dialog title
            details: Optional technical details to log
        """
        self.logger.error(message + (f": {details}" if details else ""))
        QMessageBox.critical(self, title, message)
        
    def show_warning(self, message: str, title: str = "Warning") -> None:
        """Show warning message to user and log it."""
        self.logger.warning(message)
        QMessageBox.warning(self, title, message)
        
    def set_status(self, message: str, is_error: bool = False, timeout_ms: int = 3500) -> None:
        """
        Set status message with optional timeout.
        
        Args:
            message: Status message to display
            is_error: True to show as error (red), False for success (green)
            timeout_ms: How long to show message, 0 for no timeout
        """
        self.csv_status.setStyleSheet(
            f"color: {'#f44336' if is_error else '#8bc34a'}; padding-left: 8px;"
        )
        self.csv_status.setText(message)
        if timeout_ms > 0:
            QTimer.singleShot(timeout_ms, lambda: self.csv_status.setText(""))
            
    def get_active_game(self) -> Optional[str]:
        """
        Get the currently active game name based on selection state.
        
        Returns:
            Game name (capitalized) or None if no game is active
        """
        try:
            for game_name, is_selected in self.selected_games.items():
                if is_selected:
                    return game_name.capitalize()
            return None
        except Exception as e:
            self.logger.error(f"Error getting active game: {e}")
            return None
            
    def switch_active_game(self, game_name: str) -> bool:
        """
        Switch to a different game's database.
        
        Args:
            game_name: Name of game to switch to
            
        Returns:
            True if switch was successful
        """
        try:
            game_name = game_name.lower()
            if game_name not in self.selected_games:
                self.logger.error(f"Unknown game: {game_name}")
                return False
                
            # Update selection state
            for name in self.selected_games:
                self.selected_games[name] = (name == game_name)
                
            # Switch database path
            from PhotoMatching import set_database_path, load_cache
            set_database_path(game_name.capitalize())
            
            # Load game's database
            self.featureDB = load_cache()
            if not self.featureDB:
                self.logger.warning(f"No database entries found for {game_name}")
                
            self.logger.info(f"Switched to {game_name} ({len(self.featureDB)} database entries)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error switching game to {game_name}: {e}")
            return False
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Card Matcher — Camera")
        self.resize(1280, 800)

        # State variables
        self.featureDB = load_cache()  # preload from cache, refresh in background
        self.selected_games = {}  # Track selected game types
        self.selected_sets = {}   # Track selected sets for each game
        
        # Initialize supported games
        self.selected_games = {
            'lorcana': False,
            'riftbound': False
        }
        self.selected_sets = {
            'lorcana': [],
            'riftbound': []
        }
        
        # Validate Card_Images directory structure
        card_images_dir = "Card_Images"
        if os.path.exists(card_images_dir):
            for game in ['Lorcana', 'RiftBound']:
                if os.path.isdir(os.path.join(card_images_dir, game)):
                    self.logger.info(f"Found {game} directory")
        
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
        # Progress bar for DB build
        self.progress = QProgressBar()
        self.progress.setFixedHeight(20)  # Make it taller
        self.progress.setTextVisible(True)
        self.progress.setValue(0)
        self.progress.setFormat("")  # Will be set dynamically
        self.progress.hide()  # Hide initially

        # Left: live preview (crop-to-fill)
        self.preview_label = QLabel("Camera stopped")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid #444; background: #111; color: #bbb;")
        self.preview_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.preview_label.setMinimumSize(640, 360)

        # Right: match image (keep aspect)
        self.image_label = QLabel("Best match image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #444; background: #111; color: #bbb;")
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setMinimumSize(360, 360)

        # Label for match info
        self.match_label = QLabel("—")
        self.match_label.setAlignment(Qt.AlignCenter)

        # Navigation buttons for matches
        self.prev_btn = QPushButton("◀ Prev")
        self.next_btn = QPushButton("Next ▶")
        self.prev_btn.clicked.connect(self.prev_match)
        self.next_btn.clicked.connect(self.next_match)
        self.match_pos_label = QLabel("")

        # Camera control buttons
        self.start_btn = QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self.open_settings)

        # Scan/capture button
        self.capture_btn = QPushButton("Scan Card")
        self.capture_btn.clicked.connect(self.capture_and_match)
        self.capture_btn.setStyleSheet(
            "background-color: #ff9800; color: white; font-weight: bold; "
            "font-size: 18px; padding: 12px 24px; border-radius: 8px;"
        )

        # Foil checkbox
        self.foil_check = QCheckBox("Foil")
        self.foil_check.setChecked(self.keep_foil_checked)

        # Count input for CSV
        self.count_edit = QLineEdit("1")
        self.count_edit.setFixedWidth(60)

        # Add to CSV button
        self.add_csv_btn = QPushButton("Add to card list")
        self.add_csv_btn.clicked.connect(self.add_to_csv)
        self.add_csv_btn.setEnabled(False)  # enabled after a successful scan

        # Inline CSV status label
        self.csv_status = QLabel("")
        self.csv_status.setStyleSheet("color: #8bc34a; padding-left: 8px;")

        # --- Top bar layout
        top = QHBoxLayout()
        top.addWidget(self.start_btn)
        top.addWidget(self.stop_btn)
        top.addWidget(self.settings_btn)
        top.addStretch()

        # --- Split panels for preview and match image
        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.preview_label)
        splitter.addWidget(self.image_label)
        splitter.setSizes([600, 600])  # balanced start

        # --- Bottom controls (scan, navigation)
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

        # --- Bottom bar for foil, count, add, status
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

        # After layout setup, add key bindings
        self.setFocusPolicy(Qt.StrongFocus)  # Enable key events
        self._setup_tooltips()  # Move this here, after all widgets are created

        # Prevent widgets from stealing keyboard focus
        for widget in [self.capture_btn, self.start_btn, self.stop_btn, 
                      self.settings_btn, self.add_csv_btn, self.prev_btn, 
                      self.next_btn, self.foil_check]:
            widget.setFocusPolicy(Qt.NoFocus)
        
        # Keep edit box focusable but prevent space from activating checkboxes
        self.count_edit.setFocusPolicy(Qt.StrongFocus)

    # ---- Settings persistence ----
    def load_settings(self):
        """
        Load UI and camera settings from SETTINGS_FILE.
        
        If settings file doesn't exist or is invalid, uses defaults.
        Logs warnings for missing or invalid settings.
        """
        # Default settings
        defaults = {
            "camera_index": 0,
            "keep_foil_checked": False,
            "confidence_threshold": 0.90,
            "debug_mode": False,
            "selected_games": {
                "lorcana": True,
                "riftbound": False
            },
            "selected_sets": {
                "lorcana": [],
                "riftbound": []
            },
            "rotate_display": False,
            "crop_to_focus": True
        }
        
        if not os.path.exists(SETTINGS_FILE):
            self.logger.info("No settings file found, using defaults")
            for key, value in defaults.items():
                setattr(self, key, value)
            return
            
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)
                
            # Validate and apply each setting
            for key, default in defaults.items():
                try:
                    value = settings.get(key, default)
                    
                    # Type validation
                    if isinstance(default, bool) and not isinstance(value, bool):
                        self.logger.warning(f"Invalid type for {key}, using default")
                        value = default
                    elif isinstance(default, (int, float)) and not isinstance(value, (int, float)):
                        self.logger.warning(f"Invalid type for {key}, using default")
                        value = default
                    elif isinstance(default, dict) and not isinstance(value, dict):
                        self.logger.warning(f"Invalid type for {key}, using default")
                        value = default
                        
                    # Special validation
                    if key == "confidence_threshold":
                        value = max(0.0, min(1.0, float(value)))
                    elif key == "camera_index":
                        value = max(0, int(value))
                        
                    setattr(self, key, value)
                    
                except Exception as e:
                    self.logger.error(f"Error loading setting {key}: {e}")
                    setattr(self, key, default)
                    
            self.logger.info("Settings loaded successfully")
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid settings file format: {e}")
            # Use defaults on error
            for key, value in defaults.items():
                setattr(self, key, value)
                
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
            # Use defaults on error
            for key, value in defaults.items():
                setattr(self, key, value)
            
            # Only apply set selection if settings window exists
            if hasattr(self, 'settings_dlg') and self.settings_dlg is not None:
                # Restore tree state
                def restore_tree_state(root):
                    for i in range(root.childCount()):
                        child = root.child(i)
                        if child.childCount() == 0:  # Leaf node (actual set)
                            game_type, set_code = child.data(0, Qt.UserRole)
                            if game_type == "lorcana" and set_code in self.selected_sets:
                                child.setCheckState(0, Qt.Checked)
                            else:
                                child.setCheckState(0, Qt.Unchecked)
                        else:  # Non-leaf node
                            restore_tree_state(child)
                        
                # Process each top-level item
                for i in range(self.settings_dlg.set_tree.topLevelItemCount()):
                    root = self.settings_dlg.set_tree.topLevelItem(i)
                    if root.text(0) == "RiftBound":
                        root.setCheckState(0, Qt.Checked if self.show_riftbound else Qt.Unchecked)
                    restore_tree_state(root)
        except Exception:
            pass

    def save_settings(self):
        """
        Save UI and camera settings to SETTINGS_FILE.
        """
        s = {
            "camera_index": self.camera_index,
            "keep_foil_checked": self.keep_foil_checked,
            "confidence_threshold": float(self.confidence_threshold),
            "debug_mode": self.debug_mode,
            "selected_games": getattr(self, 'selected_games', {}),
            "selected_sets": getattr(self, 'selected_sets', {}),
            "rotate_display": self.rotate_display,
            "crop_to_focus": self.crop_to_focus,
        }
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(s, f, indent=2)
        except Exception:
            pass

    def closeEvent(self, event):
        """
        Save settings on window close.
        """
        try:
            self.save_settings()
        finally:
            super().closeEvent(event)

    # ---- Background DB build (non-blocking) ----
    def start_db_build_in_background(self):
        """
        Start building feature databases for all game folders in background.
        
        Features:
        - Parallel processing for better performance
        - Progress tracking per game
        - Error recovery and logging
        - Automatic retry for failed builds
        """
        self._db_progress_pct = 0
        self._db_progress_timer = QTimer(self)
        self._db_progress_timer.setInterval(100)
        self._db_progress_timer.timeout.connect(self._tick_db_progress)
        
        # Database build state
        self._db_build_state = {
            "running": False,
            "current_game": None,
            "total_games": 0,
            "completed": 0,
            "failed": [],
            "retry_count": {},
            "start_time": time.time()
        }

        # Get list of game folders
        self.game_folders = []
        card_images_dir = "Card_Images"
        
        try:
            # Scan for game folders
            if not os.path.exists(card_images_dir):
                os.makedirs(card_images_dir)
                self.logger.warning(f"Created missing {card_images_dir} directory")
            
            self.game_folders = [
                item for item in os.listdir(card_images_dir)
                if os.path.isdir(os.path.join(card_images_dir, item))
                and item not in ("__pycache__", "logs")
            ]
            
            if not self.game_folders:
                self.logger.error("No game folders found")
                self.match_label.setText("No game folders found in Card_Images directory")
                return
                
            # Initialize build state
            self._db_build_state.update({
                "running": True,
                "total_games": len(self.game_folders),
                "start_time": time.time()
            })
            
            # Start with first game
            self.current_game_index = 0
            self.current_game = self.game_folders[0]
            
            # Set initial database path
            set_database_path(self.current_game)
            
            # Show progress
            self.progress.setValue(0)
            self.progress.setFormat(f"Building {self.current_game} database... %p%")
            self.progress.show()
            self.match_label.setText(f"Building {self.current_game} database, please wait...")
            
            # Start the build process
            self._start_current_game_build()
            
        except Exception as e:
            self.logger.error(f"Error starting database build: {e}")
            self.match_label.setText("Error starting database build")
            self.progress.hide()
        if os.path.exists(card_images_dir):
            for item in os.listdir(card_images_dir):
                if os.path.isdir(os.path.join(card_images_dir, item)) and item != "__pycache__":
                    self.game_folders.append(item)
        
        if not self.game_folders:
            self.match_label.setText("No game folders found in Card_Images directory")
            return
            
        # Start with the first game
        self.current_game_index = 0
        self.current_game = self.game_folders[0]
        set_database_path(self.current_game)
        
        # Show progress immediately
        self.progress.setValue(0)
        self.progress.setFormat(f"Building {self.current_game} database... %p%")
        self.progress.show()
        self.match_label.setText(f"Building {self.current_game} database, please wait...")

        def cb(pct, last):
            self._db_progress_pct = int(pct)
            if last:  # Show which file is being processed
                self.match_label.setText(f"Processing {last}...")

        def worker():
            try:
                self.logger.info(f"Starting database build for {self.current_game}")
                game_path = os.path.join("Card_Images", self.current_game)
                
                # Store the database path for this game and ensure it's set correctly
                self.current_db_path = game_path
                set_database_path(self.current_game)
                
                # Double-check that we're using the right path
                if not os.path.normpath(databasePath).endswith(self.current_game):
                    self.logger.warning(f"Database path incorrect, fixing for {self.current_game}")
                    set_database_path(self.current_game)
                
                self.logger.info(f"Database path: {databasePath}")
                self.logger.info(f"Game folder path: {game_path}")
                self.logger.info(f"Folder exists: {os.path.exists(game_path)}")
                
                if os.path.exists(game_path):
                    self.logger.info(f"Files in folder: {os.listdir(game_path)}")
                    
                db = build_feature_database(cb)
                if db:  # Only update if we got results
                    self.logger.info(f"Database built successfully with {len(db)} entries")
                    # Store the database in a dictionary using the game name as key
                    db_attr_name = f"{self.current_game.lower()}_db"
                    setattr(self, db_attr_name, db)
                    
                    # Move to next game if available
                    self.current_game_index += 1
                    if self.current_game_index < len(self.game_folders):
                        self.current_game = self.game_folders[self.current_game_index]
                        self.logger.info(f"Starting {self.current_game} database build")
                        # Set path before starting next build
                        set_database_path(self.current_game)
                        self._db_progress_pct = 0
                        self.progress.setValue(0)
                        self.progress.setFormat(f"Building {self.current_game} database... %p%")
                        self.match_label.setText(f"Building {self.current_game} database...")
                        # Clear any cached path to ensure fresh start
                        if hasattr(self, 'current_db_path'):
                            delattr(self, 'current_db_path')
                        threading.Thread(target=worker, daemon=True).start()
                    else:
                        self.match_label.setText("All databases ready.")
                        self.progress.hide()
                else:
                    self.logger.error(f"No database entries found for {self.current_game}")
                    self.match_label.setText(f"Error building {self.current_game} database.")
            except Exception as e:
                self.logger.exception("Database build failed")
                self.match_label.setText(f"Error: {str(e)}")

        threading.Thread(target=worker, daemon=True).start()
        self._db_progress_timer.start()

    def _tick_db_progress(self):
        """
        Update progress bar for DB build.
        """
        try:
            progress = int(getattr(self, "_db_progress_pct", 0))
            
            # Update progress bar
            if progress != self.progress.value():  # Only update if changed
                self.progress.setValue(progress)
                self.progress.show()  # Make sure it's visible
                
                # Force immediate update
                QApplication.processEvents()
            
            if progress >= 100:
                self._db_progress_timer.stop()
                self.progress.hide()  # Hide when complete
                
                # Clear any "building" message if it wasn't changed by the worker
                if self.match_label.text().startswith("Building") or self.match_label.text().startswith("Processing"):
                    from PhotoMatching import databasePath
                    current_game = os.path.basename(databasePath)
                    self.match_label.setText(f"{current_game} database ready. Scan a card to begin.")
        except Exception as e:
            self.match_label.setText(f"Progress error: {str(e)}")
            pass

    # -------- Camera control --------
    def _start_current_game_build(self):
        """Start the database build for the current game."""
        
        def progress_callback(pct: int, current_file: Optional[str]) -> None:
            self._db_progress_pct = pct
            if current_file:
                self.match_label.setText(f"Processing {current_file}...")
        
        def build_worker():
            try:
                self.logger.info(f"Starting database build for {self.current_game}")
                game_path = os.path.join("Card_Images", self.current_game)
                
                # Ensure correct database path
                self.current_db_path = game_path
                set_database_path(self.current_game)
                
                if not os.path.exists(game_path):
                    raise RuntimeError(f"Game folder {game_path} not found")
                
                # Build database
                db = build_feature_database(progress_callback)
                if not db:
                    raise RuntimeError("Database build returned no entries")
                    
                # Store results
                self.logger.info(f"Database built successfully with {len(db)} entries")
                setattr(self, f"{self.current_game.lower()}_db", db)
                
                # Update state
                self._db_build_state["completed"] += 1
                
                # Move to next game if available
                self.current_game_index += 1
                if self.current_game_index < len(self.game_folders):
                    self.current_game = self.game_folders[self.current_game_index]
                    self.logger.info(f"Starting {self.current_game} database build")
                    set_database_path(self.current_game)
                    self._db_progress_pct = 0
                    self.progress.setValue(0)
                    self.progress.setFormat(f"Building {self.current_game} database... %p%")
                    self.match_label.setText(f"Building {self.current_game} database...")
                    if hasattr(self, 'current_db_path'):
                        delattr(self, 'current_db_path')
                    threading.Thread(target=build_worker, daemon=True).start()
                else:
                    build_time = time.time() - self._db_build_state["start_time"]
                    self.logger.info(f"All databases built in {build_time:.1f}s")
                    self.match_label.setText("All databases ready.")
                    self.progress.hide()
                    self._db_build_state["running"] = False
                    
            except Exception as e:
                self.logger.error(f"Error building database for {self.current_game}: {e}")
                self._db_build_state["failed"].append(self.current_game)
                
                # Handle retry logic
                retries = self._db_build_state["retry_count"].get(self.current_game, 0)
                if retries < 2:  # Allow up to 2 retries
                    self.logger.info(f"Retrying {self.current_game} (attempt {retries + 1})")
                    self._db_build_state["retry_count"][self.current_game] = retries + 1
                    threading.Thread(target=build_worker, daemon=True).start()
                else:
                    self.logger.error(f"Failed to build {self.current_game} after {retries} retries")
                    self.match_label.setText(f"Error building {self.current_game} database")
                    self.progress.hide()
        
        # Start the build process
        threading.Thread(target=build_worker, daemon=True).start()
        
    def start_camera(self):
        """
        Start the camera with error handling and automatic backend selection.
        
        Tries multiple backends in order:
        1. DirectShow (Windows-specific, preferred)
        2. MSMF (Windows Media Foundation)
        3. Generic backend
        
        Also attempts to configure optimal camera settings.
        """
        # Clean up old resources
        self.stop_camera()
        cv2.destroyAllWindows()

        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (0, "Default")  # No extra flag
        ]

        camera_opened = False
        last_error = None
        
        for backend_flag, backend_name in backends:
            try:
                self.logger.info(f"Trying camera {self.camera_index} with {backend_name} backend...")
                
                # Create capture object
                if backend_flag == 0:
                    self.cap = cv2.VideoCapture(self.camera_index)
                else:
                    self.cap = cv2.VideoCapture(self.camera_index + backend_flag)
                
                if not self.cap.isOpened():
                    raise RuntimeError(f"{backend_name} backend failed to open camera")
                
                # Configure camera
                self.logger.info("Setting camera properties...")
                props = [
                    (cv2.CAP_PROP_BUFFERSIZE, 1, "Buffer Size"),
                    (cv2.CAP_PROP_FRAME_WIDTH, 1280, "Width"),
                    (cv2.CAP_PROP_FRAME_HEIGHT, 720, "Height"),
                    (cv2.CAP_PROP_FPS, 30, "FPS"),
                    (cv2.CAP_PROP_AUTOFOCUS, 1, "Autofocus"),
                    (cv2.CAP_PROP_AUTO_EXPOSURE, 1, "Auto Exposure")
                ]
                
                for prop, value, name in props:
                    try:
                        if self.cap.set(prop, value):
                            actual = self.cap.get(prop)
                            self.logger.info(f"Set {name}: requested={value}, actual={actual}")
                        else:
                            self.logger.warning(f"Failed to set {name} to {value}")
                    except Exception as e:
                        self.logger.warning(f"Error setting {name}: {e}")
                
                # Test frame capture
                self.logger.info("Testing frame capture...")
                ret, frame = self.cap.read()
                if not ret or frame is None or frame.size == 0:
                    raise RuntimeError("Camera not providing valid frames")
                
                # Success!
                camera_opened = True
                self.logger.info(f"Successfully initialized camera with {backend_name} backend")
                break
                
            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"Failed to initialize with {backend_name}: {e}")
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                    
        if not camera_opened:
            error_msg = f"Failed to open camera {self.camera_index} with any backend"
            if last_error:
                error_msg += f"\nLast error: {last_error}"
            self._fail_and_stop(error_msg)
            return
        
        # Just call the initialization with logging - it's already well set up now
        pass

        # Warm-up with more attempts and longer timeout
        ok = False
        for _ in range(30):  # More attempts
            try:
                ret, frm = self.cap.read()
                if ret and frm is not None and frm.size > 0:
                    ok = True
                    break
            except Exception:
                pass
            time.sleep(0.1)  # Longer sleep between attempts

        if not ok:
            self._fail_and_stop("Camera opened but not delivering frames.")
            return

        self.read_fail_count = 0
        self.last_frame_time = time.monotonic()
        self.timer.start(30)
        self.watchdog.start()
        self.match_label.setText("Camera running …")

    def _fail_and_stop(self, message: str):
        """Stop camera and show error message."""
        self.stop_camera()
        QMessageBox.warning(self, "Camera Error", 
                          f"{message}\n\nTry a different camera index in Settings.")

    def stop_camera(self):
        """
        Stop camera and timers, thoroughly clean up resources.
        """
        # Stop all timers first
        self.timer.stop()
        self.watchdog.stop()
        
        # Clean up camera resources
        if self.cap is not None:
            try:
                # Clear frame buffer
                for _ in range(5):
                    self.cap.grab()
                # Release camera
                self.cap.release()
            except Exception:
                pass
            finally:
                self.cap = None
        
        # Clear frame buffer references
        self.last_frame = None
        self.last_frame_time = None
        self.read_fail_count = 0
        
        # Update UI
        self.preview_label.setText("Camera stopped")
        
        # Force immediate cleanup
        import gc
        gc.collect()

    def _watchdog_tick(self):
        """
        Watchdog timer to detect camera hangs.
        """
        if self.cap is None or self.last_frame_time is None:
            return
        if (time.monotonic() - self.last_frame_time) > 3.0:
            self._fail_and_stop("No frames received for 3 seconds. Camera stopped.")

    def _focus_rect(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """
        Return (fx, fy, fw, fh) for a 63:88 portrait box occupying ~60% of height, centered.
        """
        card_ratio = 63 / 88.0
        fh = int(h * 0.6)
        fw = int(fh * card_ratio)
        fw = min(fw, w - 4)
        fh = min(fh, h - 4)
        fx = max((w - fw) // 2, 2)
        fy = max((h - fh) // 2, 2)
        return fx, fy, fw, fh

    def _grab_frame(self):
        """
        Grab and process a frame from the camera with error handling.
        
        Handles frame capture, rotation, and overlay drawing. Uses
        configurable focus box and handles preview display.
        
        Note: Camera state (self.cap) must be valid before calling.
        """
        if not self.cap:
            return
            
        try:
            # Frame capture with error handling
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                self.read_fail_count += 1
                if self.read_fail_count > self.max_read_fail:
                    self._fail_and_stop("Camera not delivering frames. Stopping preview.")
                return
            
            self.read_fail_count = 0  # Reset counter on success
            self.last_frame_time = time.monotonic()
            
            # Apply rotation if configured
            if self.rotate_display:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                
            # Store current frame
            self.last_frame = frame.copy()
            
            # Create overlay for focus box
            overlay = frame.copy()
            h, w = overlay.shape[:2]
            
            # Calculate focus box dimensions (63:88 portrait, ~60% height)
            fx, fy, fw, fh = self._focus_rect(h, w)
            
            # Draw focus box
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(overlay, (fx, fy), (fx + fw, fy + fh), color, thickness)
            
            # Draw corner guides
            corner_length = 40
            corners = [
                # Top left
                [(fx, fy), (fx + corner_length, fy)],
                [(fx, fy), (fx, fy + corner_length)],
                # Top right 
                [(fx + fw, fy), (fx + fw - corner_length, fy)],
                [(fx + fw, fy), (fx + fw, fy + corner_length)],
                # Bottom left
                [(fx, fy + fh), (fx + corner_length, fy + fh)],
                [(fx, fy + fh), (fx, fy + fh - corner_length)],
                # Bottom right
                [(fx + fw, fy + fh), (fx + fw - corner_length, fy + fh)],
                [(fx + fw, fy + fh), (fx + fw, fy + fh - corner_length)]
            ]
            
            for start, end in corners:
                cv2.line(overlay, start, end, color, thickness)
                
            # Add activation visualization in debug mode
            preview = visualize_activation_overlay(overlay) if self.debug_mode else overlay
            
            # Show preview (crop to fill)
            self._show_on_label(self.preview_label, preview, fill=True)
            
        except Exception as e:
            self.read_fail_count += 1
            if self.read_fail_count > self.max_read_fail:
                self._fail_and_stop(f"Camera error: {e}")
            return

    # -------- Matching & navigation --------
    def capture_and_match(self):
        """
        Capture current frame, extract features, match against DB, and show results.
        """
        if self.last_frame is None:
            QMessageBox.warning(self, "Scan Card", "Camera is not running or no frame available.")
            return

        img_bgr = self.last_frame.copy()

        # Crop to focus box if enabled
        if self.crop_to_focus:
            h, w = img_bgr.shape[:2]
            fx, fy, fw, fh = self._focus_rect(h, w)
            fx, fy = max(fx, 0), max(fy, 0)
            fw = min(fw, w - fx)
            fh = min(fh, h - fy)
            if fw > 10 and fh > 10:
                img_bgr = img_bgr[fy:fy+fh, fx:fx+fw].copy()

        # Auto-check foil if detected
        self.foil_check.setChecked(self.keep_foil_checked or is_probably_foil(img_bgr))

        # Update feature database based on selected game type
        if hasattr(self, 'selected_games'):
            active_game = None
            for game_name, is_selected in self.selected_games.items():
                if is_selected:
                    active_game = game_name
                    break
            
            if active_game:
                from PhotoMatching import set_database_path, load_cache
                # Capitalize game name for folder lookup
                game_folder = active_game.capitalize()
                self.logger.info(f"Setting active game to: {game_folder}")
                
                # Set the correct database path first
                set_database_path(game_folder)
                
                # Load the cache for the selected game
                self.featureDB = load_cache()
                if self.featureDB:
                    self.logger.info(f"Loaded feature DB for {game_folder} with {len(self.featureDB)} entries")
                    self.logger.info(f"Current database path: {databasePath}")
                    self.logger.info(f"Database keys sample: {list(self.featureDB.keys())[:5]}")
            else:
                self.match_label.setText("Please select a game type in settings.")
                self.image_label.hide()
                self.add_csv_btn.setEnabled(False)
                return

        features = extract_features(img_bgr)
        if features is None:
            self.match_label.setText("Could not extract features from image.")
            self.logger.error("Feature extraction failed")
            self.image_label.hide()
            self.add_csv_btn.setEnabled(False)
            return

        self.logger.info("Features extracted successfully")

        # The feature database should already be set correctly above
        if not self.featureDB:
            self.match_label.setText("Please select a game type in settings.")
            self.logger.error("No feature database loaded")
            self.image_label.hide()
            self.add_csv_btn.setEnabled(False)
            return

        self.logger.info(f"Searching through {len(self.featureDB)} database entries with confidence {self.confidence_threshold}")
        matches = find_best_matches(features, self.featureDB, threshold=self.confidence_threshold)
        self.logger.info(f"Found {len(matches)} initial matches before filtering")
        
        matches = self._filter_matches(matches)
        self.logger.info(f"Found {len(matches)} matches after filtering")
        
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
        """
        Show match at given index in the UI.
        """
        if not self.last_matches:
            return
        idx = max(0, min(idx, len(self.last_matches)-1))
        self.current_match_idx = idx
        fname, score = self.last_matches[idx]
        self.match_label.setText(f"{score:.3f}  {fname}")
        self.match_pos_label.setText(f"{idx+1} / {len(self.last_matches)}")

        # Get active game folder
        active_game = None
        for game_name, is_selected in self.selected_games.items():
            if is_selected:
                active_game = game_name.capitalize()
                break
        
        if active_game:
            # Construct path using the active game folder
            match_path = os.path.join("Card_Images", active_game, fname)
            self.logger.info(f"Loading card image from: {match_path}")
            img = cv2.imread(match_path, cv2.IMREAD_COLOR)
            if img is not None:
                self._show_on_label(self.image_label, img, fill=False)
                self.image_label.show()
            else:
                self.logger.error(f"Failed to load image: {match_path}")
                self.image_label.hide()
        else:
            self.logger.error("No active game selected")
            self.image_label.hide()

    def prev_match(self):
        """
        Show previous match in the list.
        """
        if not self.last_matches:
            return
        self._show_match_at(self.current_match_idx - 1)

    def next_match(self):
        """
        Show next match in the list.
        """
        if not self.last_matches:
            return
        self._show_match_at(self.current_match_idx + 1)

    def _filter_matches(self, matches: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Filter matches by selected game type and sets.
        
        Args:
            matches: List of (filename, score) tuples from feature matching
            
        Returns:
            Filtered list of matches that belong to the active game and selected sets
        """
        # Basic validation
        if not matches:
            return []
            
        try:
            # Get selection state
            selected_games = getattr(self, 'selected_games', {})
            selected_sets = getattr(self, 'selected_sets', {})
            
            # Validate we have game selection state
            if not selected_games:
                self.logger.warning("No game selection state found")
                return matches
                
            # Get active game
            active_game = None
            for game_name, is_selected in selected_games.items():
                if is_selected:
                    active_game = game_name
                    break
            
            if not active_game:
                self.logger.warning("No active game selected")
                return matches
                
            # Filter matches by game type and set
            filtered = []
            for fname, score in matches:
                try:
                    filename = os.path.basename(fname)
                    split_result = self._split_filename(filename)
                    
                    if split_result is None:
                        self.logger.warning(f"Could not parse filename: {filename}")
                        continue
                        
                    set_code = split_result[0]
                    game_selected_sets = selected_sets.get(active_game, [])
                    
                    # Include if no sets filtered or if set is selected
                    if not game_selected_sets or set_code in game_selected_sets:
                        filtered.append((filename, score))
                        
                except Exception as e:
                    self.logger.warning(f"Error filtering match {fname}: {e}")
                    continue
                    
            self.logger.info(f"Filtered {len(matches)} matches to {len(filtered)}")
            return filtered
                    
        except Exception as e:
            self.logger.error(f"Error filtering matches: {e}")
            return matches  # Return unfiltered on error
        
        self.logger.info(f"Filtering matches with selected games: {selected_games}")
        self.logger.info(f"Selected sets: {selected_sets}")
        
        # If we don't have selections yet, enable all games by default
        if not selected_games:
            card_images_dir = "Card_Images"
            if os.path.exists(card_images_dir):
                for item in os.listdir(card_images_dir):
                    if os.path.isdir(os.path.join(card_images_dir, item)) and item != "__pycache__":
                        selected_games[item.lower()] = True
                        selected_sets[item.lower()] = []
        
        out = []
        for fname, score in matches:
            self.logger.info(f"\nAnalyzing match: {fname} (score: {score})")
            
            # Use filename only for matching, not full path
            filename = os.path.basename(fname)
            self.logger.info(f"Processing match filename: {filename}")
            
            # Get active game type
            active_game = None
            for game_name, is_selected in self.selected_games.items():
                if is_selected:
                    active_game = game_name
                    break
            
            self.logger.info(f"Active game: {active_game}")
            game_type = active_game  # Use the active game as the type
            
            self.logger.info(f"Selected games: {selected_games}")
            self.logger.info(f"Found game type: {game_type}")
            
            self.logger.info(f"Detected game type: {game_type}")
                    
            if not game_type:
                self.logger.info(f"Skipping - couldn't determine game type for {fname}")
                continue
                
            # Check if game type is enabled
            if not selected_games.get(game_type, False):
                self.logger.info(f"Skipping - game type {game_type} is not enabled")
                continue
                
            # Check set selection
            from PhotoMatching import _split_filename
            set_code, _ = _split_filename(fname)
            self.logger.info(f"Card set code: {set_code}")
            
            game_selected_sets = selected_sets.get(game_type, [])
            self.logger.info(f"Selected sets for {game_type}: {game_selected_sets}")
            
            # Get set code from filename using current code's method
            split_result = self._split_filename(filename)
            if split_result:
                set_code = split_result[0]  # Use first element (set code)
                self.logger.info(f"Card set code: {set_code}")
            else:
                self.logger.warning(f"Could not extract set code from filename: {filename}")
                set_code = None
                
            if not game_selected_sets:  # If no sets selected for this game, include all its cards
                self.logger.info(f"Including - no specific sets selected for {game_type}")
                out.append((filename, score))  # Use filename only
            elif set_code in game_selected_sets:  # Check if the card's set is selected
                self.logger.info(f"Including - set {set_code} is selected")
                out.append((filename, score))  # Use filename only
            else:
                self.logger.info(f"Skipping - set {set_code} is not in selected sets: {game_selected_sets}")
                
        self.logger.info(f"\nFinal filtered matches count: {len(out)}")
        return out

    def open_settings(self):
        """
        Open the settings dialog.
        """
        self.settings_dlg = SettingsWindow(self)
        self.settings_dlg.exec()
        self.foil_check.setChecked(self.keep_foil_checked)

    def add_to_csv(self):
        """
        Add the currently selected match to the appropriate CSV file.
        Pass the FULL image path so PhotoMatching.get_game_type() works.
        """
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

        # Determine active game (prefer UI selection; fall back to databasePath)
        active_game = None
        try:
            active_game = next((g for g, sel in self.selected_games.items() if sel), None)
        except Exception:
            active_game = None
        if not active_game:
            from PhotoMatching import databasePath as PM_databasePath
            active_game = os.path.basename(os.path.normpath(PM_databasePath)).lower()

        game_folder = (active_game or "Lorcana").capitalize()
        full_match_path = os.path.join("Card_Images", game_folder, fname)

        # For the UI message only (update_cardlist will pick correctly)
        target_file = "RiftboundList.csv" if game_folder.lower() == "riftbound" else "LorcanaList.csv"

        # IMPORTANT: pass the full path so get_game_type() can detect Riftbound vs Lorcana
        update_cardlist(full_match_path, is_foil, cnt)

        self.csv_status.setStyleSheet("color: #8bc34a; padding-left: 8px;")
        self.csv_status.setText(f"Added {cnt} of {fname} to {target_file}")
        self.add_csv_btn.setEnabled(False)
        QTimer.singleShot(3500, lambda: self.csv_status.setText(""))


    # -------- Helpers --------
    def _crop_to_fill(self, img_bgr, target_w: int, target_h: int) -> np.ndarray:
        """
        Center-crop img to match target aspect (cover). Returns a C-contiguous copy.
        """
        if img_bgr is None or img_bgr.size == 0 or target_w <= 0 or target_h <= 0:
            return np.ascontiguousarray(img_bgr) if img_bgr is not None else img_bgr
        h, w = img_bgr.shape[:2]
        if h == 0 or w == 0:
            return np.ascontiguousarray(img_bgr)

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

        return np.ascontiguousarray(cropped)

    def _show_on_label(self, label: QLabel, img_bgr, fill: bool = False):
        """
        Display an image on a QLabel, optionally cropping to fill.
        Ensures format & contiguity for QImage.
        """
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

    def focusInEvent(self, event):
        # Ensure main window keeps focus
        self.setFocus()
        super().focusInEvent(event)

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        key = event.key()
        if key == Qt.Key_C:
            event.accept()
            self.capture_and_match()
        elif key == Qt.Key_S and (event.modifiers() & (Qt.ControlModifier | Qt.AltModifier)):
            if self.add_csv_btn.isEnabled():
                self.add_to_csv()
        elif key == Qt.Key_A:
            self.prev_match()
        elif key == Qt.Key_D:
            self.next_match()
        elif key == Qt.Key_F:  # Added F key to toggle foil
            self.foil_check.setChecked(not self.foil_check.isChecked())
        else:
            super().keyPressEvent(event)

    def _setup_tooltips(self):
        """Setup keyboard shortcut tooltips."""
        self.capture_btn.setToolTip("Scan Card (C)")
        self.add_csv_btn.setToolTip("Add to card list (Ctrl+S or Alt+S)")
        self.prev_btn.setToolTip("Previous match (A)")
        self.next_btn.setToolTip("Next match (D)")
        self.foil_check.setToolTip("Toggle Foil (F)")  # Added foil tooltip


if __name__ == "__main__":
    # Entry point: create and show the main window, start camera automatically
    app = QApplication([])
    w = MainWindow()
    w.show()
    QTimer.singleShot(0, w.start_camera)
    app.exec()
