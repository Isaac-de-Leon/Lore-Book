# Lore-Book — Claude Code Guide

## Project Overview

Lore-Book is a desktop trading-card scanner application. It uses a webcam to photograph physical cards, matches them against a local image database using deep-learning feature vectors, and writes the results to a CSV file you can import into [Dreamborn.ink](https://dreamborn.ink).

**Supported games:** Disney Lorcana, Riftbound

---

## Architecture

```
Lore-Book/
├── UI.py                  # Entry point — creates app, shows MainWindow
├── PhotoMatching.py       # Backward-compat re-export shim + CLI entry point
├── pyproject.toml         # Package metadata (pip install -e .)
│
├── lorebook/              # Importable package
│   ├── __init__.py
│   ├── core/              # Game-agnostic logic (no Qt)
│   │   ├── __init__.py
│   │   ├── game_types.py      # GameType enum, get_game_type, shared constants
│   │   ├── image_utils.py     # ensure_valid_image, foil_score, is_probably_foil
│   │   ├── matching.py        # L2-normalize, cosine similarity, find_best_matches
│   │   ├── features.py        # MobileNetV2 feature extraction + heatmap overlay
│   │   ├── card_database.py   # Feature cache build/load, set_database_path
│   │   └── csv_manager.py     # CSV read/write, update_cardlist, get_available_sets
│   └── ui/                # PySide6 widgets
│       ├── __init__.py
│       ├── styles.py          # APP_STYLESHEET dark theme
│       ├── settings_window.py # SettingsWindow dialog
│       └── main_window.py     # MainWindow: camera, scan, match nav, CSV export
│
├── Card_Images/
│   ├── Lorcana/           # Reference card images (.webp / .jpg / .png)
│   └── Riftbound/
├── DBCardCache_Lorcana.db      # Cached feature vectors (SQLite, auto-generated)
├── DBCardCache_Riftbound.db
├── LorcanaList.csv        # Scanned Lorcana collection
├── RiftboundList.csv      # Scanned Riftbound collection
├── ui_settings.json       # Persisted UI preferences
├── requirements.txt
└── tests/
    └── test_photo_matching.py  # Unit tests (no camera / no model weights needed)
```

### Module responsibilities

| File | What it owns |
|------|-------------|
| `lorebook/core/game_types.py` | `GameType` enum, `get_game_type()`, path/CSV constants |
| `lorebook/core/image_utils.py` | `ensure_valid_image`, `foil_score`, `is_probably_foil` |
| `lorebook/core/matching.py` | `_l2_normalize`, `_cosine_score`, `find_best_matches` |
| `lorebook/core/features.py` | MobileNetV2 lazy-load, `extract_features`, `visualize_activation_overlay` |
| `lorebook/core/card_database.py` | `databasePath` global, `set_database_path`, `load_cache`, `build_feature_database` |
| `lorebook/core/csv_manager.py` | `update_cardlist`, `get_available_sets`, CSV read/write helpers |
| `lorebook/ui/styles.py` | `APP_STYLESHEET` dark theme |
| `lorebook/ui/settings_window.py` | `SettingsWindow` PySide6 dialog |
| `lorebook/ui/main_window.py` | `MainWindow`, `setup_logging` |
| `UI.py` | `QApplication` entry point |
| `PhotoMatching.py` | Re-exports all of the above for backward compatibility; also runnable as CLI |

> Internal imports use absolute package paths (e.g. `from lorebook.core.matching import find_best_matches`). Tests and external callers can keep importing from the `PhotoMatching` shim unchanged.

---

## Setup

### Prerequisites
- Python 3.11+
- Webcam (optional for testing — only needed at runtime)
- Card images placed under `Card_Images/Lorcana/` or `Card_Images/Riftbound/`

### Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** `tensorflow_intel` in `requirements.txt` is Windows-specific. On Linux/macOS remove it and keep `tensorflow`.

### Build the feature database (first run)
```bash
python PhotoMatching.py --game Lorcana --build
```
This scans all images in `Card_Images/Lorcana/` and writes `DBCardCache_Lorcana.db`.

### Run the application
```bash
python UI.py
```

---

## Running Tests

Tests live in `tests/test_photo_matching.py`. They cover the pure utility functions in `PhotoMatching.py` and **do not** require a camera, a GPU, or pre-downloaded model weights (no TF model is loaded).

```bash
python -m pytest tests/ -v
```

Or run a single test:
```bash
python -m pytest tests/test_photo_matching.py::TestSplitFilename -v
```

---

## CSV Format

Every game gets its own `<GameName>List.csv` (e.g. `LorcanaList.csv`, `RiftboundList.csv`), all using the same 4-column format:

| Set Number | Card Number | Variant | Count |
|------------|-------------|---------|-------|
| 009        | 041         | normal  | 2     |
| 001        | 007         | foil    | 1     |

The target file comes from `csv_for_game(game)` (`lorebook/core/game_types.py`): callers that know the game (GUI, sorter) pass it explicitly via `update_cardlist(..., game=...)`. When no game is passed, `update_cardlist()` falls back to inferring it from the `Card_Images/<game>/` folder in the matched filename path (Riftbound paths → `RiftboundList.csv`, everything else → `LorcanaList.csv`).

### Filename convention for card images
```
<SetCode>-<CardCode>[.<ext>]
```
Examples: `001-042.webp`, `ONG-23c-alt.jpg`

---

## Settings (`ui_settings.json`)

| Key | Type | Description |
|-----|------|-------------|
| `camera_index` | int | OpenCV camera index (0–4) |
| `confidence_threshold` | float | Minimum cosine similarity to show a match (0.0–1.0) |
| `keep_foil_checked` | bool | Pre-check the Foil checkbox after each scan |
| `rotate_display` | bool | Rotate camera preview 180° |
| `crop_to_focus` | bool | Crop the capture to the on-screen focus box before matching |
| `debug_mode` | bool | Overlay activation heatmap on the matched card image |
| `selected_games` | object | Which games are active (`{"lorcana": true, "riftbound": false}`) |
| `selected_sets` | object | Which set codes to search within each game |

---

## Adding a New Card Game

1. Create `Card_Images/<GameName>/` and add card images.
2. That's it for the common path: the UI's game/set tree widget discovers game folders
   automatically, and the collection CSV (`<GameName>List.csv`) is derived from the folder
   name by `csv_for_game()` — no code changes needed.
3. Optional: add a `GameType` entry in `lorebook/core/game_types.py` and update
   `get_game_type()` if code needs to detect the game from file paths (only the legacy
   path-inference fallback in `update_cardlist()` uses this).

---

## Common Tasks for Claude

- **Add a feature** — edit the relevant module under `lorebook/core/` for logic, `lorebook/ui/` for UI wiring.
- **Fix a CSV parsing bug** — see `_normalize_existing_rows()` and `_write_rows_4col()` in `lorebook/core/csv_manager.py`.
- **Tune foil detection** — adjust `foil_score()` weights or threshold in `is_probably_foil()` (`lorebook/core/image_utils.py`).
- **Change match threshold** — default is `0.70` in `find_best_matches()` (`lorebook/core/matching.py`); UI exposes it as confidence %.
- **Cache issues** — delete `DBCardCache_<game>.db` and rebuild with `--build` flag.

---

## Known Limitations / Gotchas

- `tensorflow_intel` is only for Windows Intel CPUs — remove from `requirements.txt` on other platforms.
- `Card_Images/` is git-ignored (images can be large); the SQLite/JSON caches are also git-ignored.
- The feature model (MobileNetV2 weights) is downloaded from the internet on first run (~14 MB).
- Camera index `0` may not be correct on machines with multiple cameras — adjust in Settings.
- `foil_score` threshold (default `0.08`) was tuned empirically; may need adjustment per lighting setup.
