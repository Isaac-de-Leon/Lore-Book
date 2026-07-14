# styles.py — Application-wide Qt stylesheet (dark collector-app theme).

APP_STYLESHEET = """
/* ── Base ─────────────────────────────────────────────── */
QWidget {
    background-color: #0D1117;
    color: #E6EDF3;
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    font-size: 13px;
}
QDialog { background-color: #0D1117; }

/* ── Buttons ───────────────────────────────────────────── */
QPushButton {
    background-color: #21262D;
    color: #E6EDF3;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 5px 12px;
}
QPushButton:hover  { background-color: #30363D; border-color: #8B949E; }
QPushButton:pressed { background-color: #161B22; }
QPushButton:disabled { color: #484F58; border-color: #21262D; }

/* Scan FAB — size, radius, font and padding are set dynamically by
   MainWindow._apply_scale() so the button scales with the window. */
QPushButton#scanBtn {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #D97706, stop:1 #F97316);
    color: #fff;
    border: none;
    font-weight: bold;
}
QPushButton#scanBtn:hover {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #F59E0B, stop:1 #FB923C);
}
QPushButton#scanBtn:pressed {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #B45309, stop:1 #EA580C);
}

/* Add to Collection */
QPushButton#addBtn {
    background-color: #0D2818;
    color: #3FB950;
    border: 1px solid #238636;
    border-radius: 6px;
    font-weight: bold;
    padding: 5px 14px;
}
QPushButton#addBtn:hover    { background-color: #122D1F; }
QPushButton#addBtn:disabled { color: #23863680; border-color: #23863650; }

/* ── Result panel card ─────────────────────────────────── */
QFrame#resultPanel {
    background-color: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
}

/* ── Labels ────────────────────────────────────────────── */
QLabel { color: #E6EDF3; background: transparent; }
QLabel#appTitle {
    color: #F97316;
    font-size: 17px;
    font-weight: bold;
    letter-spacing: 2px;
}
QLabel#previewLabel {
    background-color: #010409;
    border: 1px solid #21262D;
    border-radius: 8px;
    color: #484F58;
}
QLabel#thumbLabel {
    background-color: #0D1117;
    border: 1px solid #21262D;
    border-radius: 6px;
    color: #484F58;
}
QLabel#matchName {
    color: #E6EDF3;
    font-size: 15px;
    font-weight: bold;
}
QLabel#matchDetail { color: #8B949E; font-size: 12px; }
QLabel#statusLabel { color: #8B949E; font-size: 11px; }

/* ── Inputs ────────────────────────────────────────────── */
QLineEdit {
    background-color: #0D1117;
    color: #E6EDF3;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 4px 8px;
}
QLineEdit:focus { border-color: #F97316; }

QCheckBox { color: #E6EDF3; spacing: 6px; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid #30363D;
    border-radius: 3px;
    background: #0D1117;
}
QCheckBox::indicator:checked {
    background-color: #F97316;
    border-color: #F97316;
}

QComboBox {
    background-color: #21262D;
    color: #E6EDF3;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 4px 8px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background-color: #161B22;
    color: #E6EDF3;
    border: 1px solid #30363D;
    selection-background-color: #F97316;
    selection-color: #fff;
}

/* ── Progress bar (thin accent strip) ─────────────────── */
QProgressBar {
    background-color: #21262D;
    border: none;
    border-radius: 2px;
    max-height: 4px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #D97706, stop:1 #F97316);
    border-radius: 2px;
}

/* ── Tabs (Scanner / Collection) ───────────────────────── */
QTabWidget::pane {
    border: 1px solid #21262D;
    border-radius: 6px;
    top: -1px;
}
QTabBar::tab {
    background: transparent;
    color: #8B949E;
    padding: 7px 18px;
    border: none;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:hover    { color: #E6EDF3; }
QTabBar::tab:selected {
    color: #F97316;
    font-weight: bold;
    border-bottom: 2px solid #F97316;
}

/* ── Collection table ──────────────────────────────────── */
QTableWidget {
    background-color: #0D1117;
    alternate-background-color: #11161D;
    color: #E6EDF3;
    border: 1px solid #30363D;
    border-radius: 6px;
    gridline-color: #21262D;
    outline: none;
}
QTableWidget::item { padding: 2px 8px; }
QTableWidget::item:selected { background-color: #1F3A5F; color: #E6EDF3; }
QHeaderView::section {
    background-color: #161B22;
    color: #8B949E;
    border: none;
    border-bottom: 1px solid #30363D;
    padding: 5px 8px;
    font-weight: bold;
}
QTableCornerButton::section { background-color: #161B22; border: none; }

/* ── Tree widget (Settings) ────────────────────────────── */
QTreeWidget {
    background-color: #0D1117;
    color: #E6EDF3;
    border: 1px solid #30363D;
    border-radius: 6px;
    outline: none;
}
QTreeWidget::item:selected   { background-color: #1F3A5F; }
QTreeWidget::item:hover      { background-color: #21262D; }

/* ── Scroll bars ───────────────────────────────────────── */
QScrollBar:vertical {
    background: #0D1117; width: 6px; border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #30363D; border-radius: 3px; min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""
