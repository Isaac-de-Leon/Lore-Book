# styles.py — Theme system: token palettes (dark/light) + Qt stylesheet generator.

from string import Template

# Each theme is a flat token → color map. build_stylesheet() substitutes them
# into the QSS template below; widgets that need theme colors in Python code
# (e.g. status text) read them via theme_tokens().
THEMES = {
    "dark": {
        "bg": "#0D1117",
        "surface": "#151B23",          # sidebar, result card, headers
        "surface_alt": "#1C232C",      # buttons, inputs resting state
        "deep": "#010409",             # camera viewfinder well
        "border": "#2A313C",
        "border_soft": "#21262D",
        "text": "#E6EDF3",
        "muted": "#8D96A0",
        "faint": "#58616B",
        "disabled": "#484F58",
        "hover_bg": "#232B36",
        "accent": "#F97316",
        "accent_hover": "#FB923C",
        "accent_soft": "rgba(249, 115, 22, 0.16)",
        "grad_a": "#D97706", "grad_b": "#F97316",
        "grad_a_hover": "#F59E0B", "grad_b_hover": "#FB923C",
        "grad_a_press": "#B45309", "grad_b_press": "#EA580C",
        "on_accent": "#FFFFFF",
        "success": "#3FB950",
        "success_bg": "#0D2818",
        "success_bg_hover": "#122D1F",
        "success_border": "#238636",
        "success_fill": "#238636",
        "success_fill_hover": "#2EA043",
        "on_success": "#FFFFFF",
        "danger": "#F85149",
        "danger_soft": "rgba(248, 81, 73, 0.14)",
        "selection_bg": "#1F3A5F",
        "selection_text": "#E6EDF3",
        "table_alt": "#11161D",
        "scroll_handle": "#30363D",
    },
    "light": {
        "bg": "#F6F8FA",
        "surface": "#FFFFFF",
        "surface_alt": "#EFF2F5",
        "deep": "#10151B",             # viewfinder stays dark — camera frames read better
        "border": "#D0D7DE",
        "border_soft": "#DDE2E8",
        "text": "#1F2328",
        "muted": "#59636E",
        "faint": "#8C959F",
        "disabled": "#ABB2BA",
        "hover_bg": "#EAEEF2",
        "accent": "#EA580C",
        "accent_hover": "#F97316",
        "accent_soft": "rgba(234, 88, 12, 0.12)",
        "grad_a": "#EA580C", "grad_b": "#F97316",
        "grad_a_hover": "#F97316", "grad_b_hover": "#FB923C",
        "grad_a_press": "#C2410C", "grad_b_press": "#EA580C",
        "on_accent": "#FFFFFF",
        "success": "#1A7F37",
        "success_bg": "#DAFBE1",
        "success_bg_hover": "#C7F0D2",
        "success_border": "#4AC26B",
        "success_fill": "#1F883D",
        "success_fill_hover": "#1A7F37",
        "on_success": "#FFFFFF",
        "danger": "#CF222E",
        "danger_soft": "rgba(207, 34, 46, 0.10)",
        "selection_bg": "#FFE0CC",
        "selection_text": "#1F2328",
        "table_alt": "#F6F8FA",
        "scroll_handle": "#C4CCD4",
    },
}

DEFAULT_THEME = "dark"


def theme_tokens(theme: str = DEFAULT_THEME) -> dict:
    """Return the token map for theme, falling back to the default theme."""
    return THEMES.get(theme, THEMES[DEFAULT_THEME])


_QSS = Template("""
/* ── Base ─────────────────────────────────────────────── */
QWidget {
    background-color: $bg;
    color: $text;
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    font-size: 13px;
}
QDialog { background-color: $bg; }
QStackedWidget { background: transparent; }
QToolTip {
    background-color: $surface;
    color: $text;
    border: 1px solid $border;
    padding: 4px 8px;
}

/* ── Sidebar ───────────────────────────────────────────── */
QFrame#sidebar {
    background-color: $surface;
    border-right: 1px solid $border_soft;
}
QPushButton#navBtn {
    background: transparent;
    border: none;
    border-radius: 12px;
    min-width: 44px;  max-width: 44px;
    min-height: 44px; max-height: 44px;
}
QPushButton#navBtn:hover    { background-color: $hover_bg; }
QPushButton#navBtn:checked  { background-color: $accent_soft; }

/* ── Buttons ───────────────────────────────────────────── */
QPushButton {
    background-color: $surface_alt;
    color: $text;
    border: 1px solid $border;
    border-radius: 8px;
    padding: 6px 14px;
}
QPushButton:hover  { background-color: $hover_bg; border-color: $muted; }
QPushButton:pressed { background-color: $bg; }
QPushButton:disabled { color: $disabled; border-color: $border_soft; }

/* Scan FAB — size, radius, font and padding are set dynamically by
   MainWindow._apply_scale() so the button scales with the window. */
QPushButton#scanBtn {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 $grad_a, stop:1 $grad_b);
    color: $on_accent;
    border: none;
    font-weight: bold;
    letter-spacing: 1px;
}
QPushButton#scanBtn:hover {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 $grad_a_hover, stop:1 $grad_b_hover);
}
QPushButton#scanBtn:pressed {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 $grad_a_press, stop:1 $grad_b_press);
}

/* Camera smart toggle — the [cam=...] property is set by
   MainWindow._set_camera_state and tracks the camera lifecycle. */
QPushButton#cameraBtn[cam="idle"] {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 $grad_a, stop:1 $grad_b);
    color: $on_accent;
    border: none;
    border-radius: 8px;
    font-weight: bold;
    padding: 6px 16px;
}
QPushButton#cameraBtn[cam="idle"]:hover {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 $grad_a_hover, stop:1 $grad_b_hover);
}
QPushButton#cameraBtn[cam="starting"] {
    background-color: $surface_alt;
    color: $muted;
    border: 1px solid $border_soft;
    border-radius: 8px;
    padding: 6px 16px;
}
QPushButton#cameraBtn[cam="running"] {
    background-color: $danger_soft;
    color: $danger;
    border: 1px solid $danger;
    border-radius: 8px;
    font-weight: bold;
    padding: 6px 16px;
}

/* Add to Collection — the one filled primary action in the result card */
QPushButton#addBtn {
    background-color: $success_fill;
    color: $on_success;
    border: none;
    border-radius: 8px;
    font-weight: bold;
    padding: 7px 18px;
}
QPushButton#addBtn:hover    { background-color: $success_fill_hover; }
QPushButton#addBtn:disabled {
    color: $disabled;
    background-color: $surface_alt;
}

/* Destructive actions (Clear collection) — quiet until hovered */
QPushButton#dangerBtn { color: $danger; }
QPushButton#dangerBtn:hover {
    background-color: $danger_soft;
    border-color: $danger;
}

/* Match pager: ‹ 2 / 8 › grouped in one pill */
QFrame#pagerPill {
    background-color: $surface_alt;
    border: 1px solid $border_soft;
    border-radius: 15px;
}
QFrame#pagerPill QPushButton {
    background: transparent;
    border: none;
    border-radius: 12px;
    padding: 0;
}
QFrame#pagerPill QPushButton:hover { background-color: $hover_bg; }
QFrame#pagerPill QLabel { color: $muted; font-size: 12px; padding: 0 2px; }

/* ── Result panel card ─────────────────────────────────── */
QFrame#resultPanel {
    background-color: $surface;
    border: 1px solid $border;
    border-radius: 14px;
}
QFrame#resultPanel QWidget { background: transparent; }

/* ── Labels ────────────────────────────────────────────── */
QLabel { color: $text; background: transparent; }
QLabel#appTitle {
    color: $accent;
    font-size: 17px;
    font-weight: bold;
    letter-spacing: 3px;
}
QLabel#previewLabel {
    background-color: $deep;
    border: 1px solid $border_soft;
    border-radius: 12px;
    color: $faint;
}
QLabel#thumbLabel {
    background-color: $bg;
    border: 1px solid $border_soft;
    border-radius: 8px;
    color: $faint;
}
QLabel#matchName {
    color: $text;
    font-size: 15px;
    font-weight: bold;
}
QLabel#matchDetail { color: $muted; font-size: 12px; }
QLabel#statusLabel { color: $muted; font-size: 11px; }

/* ── Inputs ────────────────────────────────────────────── */
QLineEdit {
    background-color: $bg;
    color: $text;
    border: 1px solid $border;
    border-radius: 8px;
    padding: 5px 9px;
    selection-background-color: $selection_bg;
    selection-color: $selection_text;
}
QLineEdit:focus { border-color: $accent; }

QCheckBox { color: $text; spacing: 6px; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid $border;
    border-radius: 4px;
    background: $bg;
}
QCheckBox::indicator:hover   { border-color: $muted; }
QCheckBox::indicator:checked {
    background-color: $accent;
    border-color: $accent;
}

QComboBox {
    background-color: $surface_alt;
    color: $text;
    border: 1px solid $border;
    border-radius: 8px;
    padding: 5px 9px;
}
QComboBox:hover { border-color: $muted; }
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background-color: $surface;
    color: $text;
    border: 1px solid $border;
    selection-background-color: $accent;
    selection-color: $on_accent;
}

/* ── Progress bar (DB build dialog) ───────────────────── */
QProgressBar#buildProgressBar {
    background-color: $surface_alt;
    border: 1px solid $border;
    border-radius: 6px;
    min-height: 18px;
    max-height: 18px;
    text-align: center;
    color: $text;
}
QProgressBar#buildProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 $grad_a, stop:1 $grad_b);
    border-radius: 5px;
}

/* ── Collection table ──────────────────────────────────── */
QTableWidget {
    background-color: $bg;
    alternate-background-color: $table_alt;
    color: $text;
    border: 1px solid $border;
    border-radius: 8px;
    gridline-color: $border_soft;
    outline: none;
}
QTableWidget::item { padding: 2px 8px; }
QTableWidget::item:selected { background-color: $selection_bg; color: $selection_text; }
QHeaderView::section {
    background-color: $surface;
    color: $muted;
    border: none;
    border-bottom: 1px solid $border;
    padding: 6px 8px;
    font-weight: bold;
}
QTableCornerButton::section { background-color: $surface; border: none; }

/* ── Tree widget (Settings) ────────────────────────────── */
QTreeWidget {
    background-color: $bg;
    color: $text;
    border: 1px solid $border;
    border-radius: 8px;
    outline: none;
}
QTreeWidget::item:selected   { background-color: $selection_bg; color: $selection_text; }
QTreeWidget::item:hover      { background-color: $hover_bg; }
QTreeWidget::indicator {
    width: 16px; height: 16px;
    border: 1px solid $border;
    border-radius: 4px;
    background: $bg;
}
QTreeWidget::indicator:hover { border-color: $muted; }
QTreeWidget::indicator:checked {
    background-color: $accent;
    border-color: $accent;
}
QTreeWidget::indicator:indeterminate {
    background-color: $accent_soft;
    border-color: $accent;
}

/* ── Scroll bars ───────────────────────────────────────── */
QScrollBar:vertical {
    background: transparent; width: 8px; border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: $scroll_handle; border-radius: 4px; min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: $muted; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: transparent; height: 8px; border-radius: 4px;
}
QScrollBar::handle:horizontal {
    background: $scroll_handle; border-radius: 4px; min-width: 20px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
""")


def build_stylesheet(theme: str = DEFAULT_THEME) -> str:
    """Render the app-wide QSS for the given theme name."""
    return _QSS.substitute(theme_tokens(theme))


# Backward-compat: the original single dark stylesheet constant.
APP_STYLESHEET = build_stylesheet("dark")
