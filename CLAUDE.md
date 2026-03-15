# Lore-Book — Claude Code Guide

## Project Overview

Lore-Book is a desktop trading-card scanner application. It uses a webcam to photograph physical cards, matches them against a local image database using deep-learning feature vectors, and writes the results to a CSV file you can import into [Dreamborn.ink](https://dreamborn.ink).

**Supported games:** Disney Lorcana, Riftbound

---

## Architecture

```
Lore-Book/
├── PhotoMatching.py      # Core logic: feature extraction, DB build, CSV writer
├── UI.py                 # PySide6 GUI (camera preview, match display, settings)
├── Card_Images/
│   ├── Lorcana/          # Reference card images (.webp / .jpg / .png)
│   └── Riftbound/        # Reference card images
├── DBCardCache_Lorcana.json   # Cached feature vectors (auto-generated)
├── DBCardCache_Riftbound.json # Cached feature vectors (auto-generated)
├── LorcanaList.csv       # Scanned Lorcana card collection
├── RiftboundList.csv     # Scanned Riftbound card collection
├── CardList.csv          # Legacy combined list (older format)
├── ui_settings.json      # Persisted UI preferences
├── requirements.txt      # Python dependencies
└── tests/
    └── test_photo_matching.py  # Unit tests (no camera / no model weights needed)
```

### Key Modules

**`PhotoMatching.py`**
- `extract_features(img_or_path)` — runs image through MobileNetV2, returns 1280-d L2-normalized vector
- `build_feature_database(callback)` — parallel scan of `Card_Images/<game>/`, writes `DBCardCache_<game>.json`
- `find_best_matches(features, db, threshold)` — cosine similarity search against cached vectors
- `update_cardlist(filename, is_foil, count)` — appends/increments a row in the correct CSV
- `foil_score(img_bgr)` / `is_probably_foil(img_bgr)` — heuristic foil detector (bright spots + Laplacian contrast)
- `GameType` enum + `get_game_type(filepath)` — LORCANA / RIFTBOUND / UNKNOWN from folder path

**`UI.py`**
- `MainWindow` — live camera preview, scan-on-click, next/prev match navigation, "Add to card list" button
- `SettingsWindow` — camera index, confidence threshold, foil defaults, game/set filter, debug mode (activation heatmap)

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
This scans all images in `Card_Images/Lorcana/` and writes `DBCardCache_Lorcana.json`.

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

Both `LorcanaList.csv` and `RiftboundList.csv` use the same 4-column format:

| Set Number | Card Number | Variant | Count |
|------------|-------------|---------|-------|
| 009        | 041         | normal  | 2     |
| 001        | 007         | foil    | 1     |

`update_cardlist()` determines which file to write by checking the `Card_Images/<game>/` folder in the matched filename path.

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
2. Add a `GameType` entry in `PhotoMatching.py`'s `GameType` enum.
3. Update `get_game_type()` to detect the new folder name.
4. Add a `<GameName>List.csv` constant and update `update_cardlist()`.
5. The UI's game/set tree widget discovers game folders automatically.

---

## Common Tasks for Claude

- **Add a feature** — edit `PhotoMatching.py` for logic, `UI.py` for UI wiring.
- **Fix a CSV parsing bug** — see `_normalize_existing_rows()` and `_write_rows_4col()` in `PhotoMatching.py`.
- **Tune foil detection** — adjust `foil_score()` weights or threshold in `is_probably_foil()`.
- **Change match threshold** — default is `0.70` in `find_best_matches()`; UI exposes it as confidence %.
- **Cache issues** — delete `DBCardCache_<game>.json` and rebuild with `--build` flag.

---

## Known Limitations / Gotchas

- `tensorflow_intel` is only for Windows Intel CPUs — remove from `requirements.txt` on other platforms.
- `Card_Images/` is git-ignored (images can be large); the JSON cache is also git-ignored.
- The feature model (MobileNetV2 weights) is downloaded from the internet on first run (~14 MB).
- Camera index `0` may not be correct on machines with multiple cameras — adjust in Settings.
- `foil_score` threshold (default `0.08`) was tuned empirically; may need adjustment per lighting setup.
