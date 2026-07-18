# Lore-Book — Claude Code Guide

## Project Overview

Lore-Book is a desktop trading-card scanner application. It uses a webcam to photograph physical cards, matches them against a local image database using deep-learning feature vectors, and writes the results to a CSV file you can import into [Dreamborn.ink](https://dreamborn.ink).

**Supported games:** Disney Lorcana, Riftbound

---

## Architecture

```
Lore-Book/
├── UI.py                  # Entry point — creates app, shows MainWindow
├── PhotoMatching.py       # Backward-compat re-export shim + CLI (--build/--match/--sort)
├── pyproject.toml         # Package metadata (pip install -e .; [sorter] extra for the Pi)
│
├── lorebook/              # Importable package
│   ├── core/              # Game-agnostic logic (no Qt)
│   │   ├── game_types.py      # GameType enum, get_game_type, csv_for_game, constants
│   │   ├── image_utils.py     # ensure_valid_image, foil_score, focus_rect/crop_to_card, MotionGate
│   │   ├── matching.py        # L2-normalize, cosine similarity, find_best_matches, MatchIndex
│   │   ├── features.py        # get_extractor("keras"|"tflite"), extract_features(+_batch), heatmap
│   │   ├── card_database.py   # Feature cache build/load, set_database_path
│   │   ├── card_names.py      # Optional card-name lookup (card_names_<Game>.json)
│   │   ├── card_prices.py     # Optional market-price lookup/format (card_prices_<Game>.json, currency_rates.json)
│   │   ├── price_fetcher.py   # download_card_prices / download_currency_rates — Lorcast + Frankfurter
│   │   ├── image_fetcher.py   # download_new_images — card art from LorcanaJSON / Riot API (Riftcodex fallback)
│   │   └── csv_manager.py     # CSV read/write, update_cardlist(_batch), split_filename
│   ├── hardware/          # Hardware abstraction (no Qt; mocks run on any desktop)
│   │   ├── camera.py          # open_capture, CameraSource, OpenCVCameraSource, MockCameraSource
│   │   └── transport.py       # Transport interface + MockTransport (gantry TBD)
│   ├── sorter/            # Headless card sorter (roadmap: docs/SORTER_ROADMAP.md)
│   │   ├── rules.py           # Rule/SortRules, decide_bin — multi-bin rules engine
│   │   ├── pipeline.py        # SortPipeline: capture→match→decide→route→CSV
│   │   └── __main__.py        # python -m lorebook.sorter CLI
│   └── ui/                # PySide6 widgets
│       ├── app.py             # main() — QApplication bootstrap (lorebook-ui script)
│       ├── styles.py          # Theme system: dark/light token palettes + build_stylesheet()
│       ├── icons.py           # Inline SVG icons → theme-tinted QIcons (get_icon)
│       ├── settings_window.py # SettingsWindow dialog (theme + currency selectors)
│       ├── progress_dialog.py # BuildProgressDialog: DB build status/progress + Cancel
│       ├── collection_view.py # Collection page: CSV table, Export CSV, Clear…
│       └── main_window.py     # MainWindow: sidebar nav, Scanner/Collection pages, camera
│
├── .github/workflows/tests.yml # CI: pytest on every push/PR (no TF needed)
├── configs/
│   └── sort_rules.example.json # Example multi-bin sort rules
├── scripts/
│   ├── convert_to_tflite.py    # Export the feature model → mobilenetv2_features.tflite
│   ├── check_parity.py         # Verify keras vs tflite vectors agree
│   ├── fetch_card_names.py     # Download card names → card_names_<Game>.json (gitignored)
│   ├── fetch_card_prices.py    # CLI over price_fetcher — market prices → card_prices_<Game>.json
│   └── fetch_card_images.py    # CLI over image_fetcher — card art → Card_Images/<Game>/
├── docs/
│   ├── MATCHING.md             # How recognition works (pipeline, thresholds)
│   └── SORTER_ROADMAP.md       # Phased plan for the physical sorter
│
├── Card_Images/
│   ├── Lorcana/           # Reference card images (.webp / .jpg / .png)
│   └── Riftbound/
├── DBCardCache_<Game>.db       # Cached feature vectors (SQLite, auto-generated)
├── <Game>List.csv              # Scanned collection per game (e.g. LorcanaList.csv)
├── mobilenetv2_features.tflite # tflite model (gitignored; generate via scripts/)
├── ui_settings.json       # Persisted UI preferences (gitignored — per-user state)
├── riot.txt               # Riot Games API domain-verification token (leave in place)
├── requirements.txt
└── tests/                 # No camera / GPU / TF needed (stubbed in conftest.py)
    ├── conftest.py
    ├── test_photo_matching.py
    ├── test_matching_index.py
    ├── test_card_database.py
    ├── test_card_names.py
    ├── test_card_prices.py
    ├── test_image_fetcher.py
    ├── test_camera.py
    ├── test_sorter_rules.py
    └── test_sorter_pipeline.py
```

### Module responsibilities

| File | What it owns |
|------|-------------|
| `lorebook/core/game_types.py` | `GameType` enum, `get_game_type()`, `game_type_from_name()`, `csv_for_game()`, constants |
| `lorebook/core/image_utils.py` | `ensure_valid_image`, `foil_score`, `is_probably_foil`, `focus_rect`/`crop_to_card` (shared GUI/sorter card crop), `MotionGate` (auto-scan) |
| `lorebook/core/card_names.py` | `name_for()` — optional display names from `card_names_<Game>.json` (see `scripts/fetch_card_names.py`) |
| `lorebook/core/card_prices.py` | `price_for()`, `format_price()`, `rate_for()`, `prices_stale()`, `clear_price_cache()` — optional market prices from `card_prices_<Game>.json` + `currency_rates.json` (display only) |
| `lorebook/core/price_fetcher.py` | `download_card_prices()` (Lorcast → TCGplayer USD prices; `GAMES` registry), `download_currency_rates()` (Frankfurter/ECB). Auto-run by the GUI when files are >24 h old; CLI: `scripts/fetch_card_prices.py` |
| `lorebook/core/image_fetcher.py` | `download_new_images()` — fetch missing card art into `Card_Images/<Game>/` (Lorcana: LorcanaJSON; Riftbound: Riot content API when `RIOT_API_KEY` is set, else Riftcodex). Auto-run by the GUI's Rebuild Database; CLI: `scripts/fetch_card_images.py` |
| `lorebook/core/matching.py` | `_l2_normalize`, `_cosine_score`, `find_best_matches`, `MatchIndex` (vectorized) |
| `lorebook/core/features.py` | `get_extractor(backend)` (keras/tflite), `extract_features`, `visualize_activation_overlay` |
| `lorebook/core/card_database.py` | `databasePath` global, `set_database_path`, `load_cache`, `build_feature_database` |
| `lorebook/core/csv_manager.py` | `update_cardlist`, `update_cardlist_batch`, `split_filename`, `get_available_sets`, `read_collection_rows` |
| `lorebook/hardware/camera.py` | `open_capture` (shared with GUI), `CameraSource` + OpenCV/Mock implementations |
| `lorebook/hardware/transport.py` | `Transport` interface, `MockTransport` (real gantry driver comes later) |
| `lorebook/sorter/rules.py` | `Rule`, `SortRules`, `load_rules`, `decide_bin` |
| `lorebook/sorter/pipeline.py` | `SortPipeline`, `SortOutcome` — the headless sort loop |
| `lorebook/sorter/__main__.py` | `python -m lorebook.sorter` argument parsing and wiring |
| `lorebook/ui/styles.py` | `THEMES` (dark/light token palettes), `build_stylesheet(theme)`, `theme_tokens()`; `APP_STYLESHEET` kept as the dark sheet for backward compat |
| `lorebook/ui/icons.py` | `get_icon(name, color, checked_color=None)` — inline SVG outline icons rendered to tinted QIcons (high-DPI safe, cached) |
| `lorebook/ui/settings_window.py` | `SettingsWindow` PySide6 dialog (theme + currency selectors; stylesheet inherited from parent) |
| `lorebook/ui/progress_dialog.py` | `BuildProgressDialog` — modeless DB-build progress popup with Cancel |
| `lorebook/ui/collection_view.py` | `CollectionView` — collection table (game dropdown, sortable, Export CSV… save-a-copy, Clear… with confirm) |
| `lorebook/ui/main_window.py` | `MainWindow` (icon sidebar switching Scanner/Collection pages, `apply_theme`, smart camera toggle `_set_camera_state`, responsive `_apply_scale`), `setup_logging` (incl. faulthandler crash log) |
| `lorebook/ui/app.py` | `main()` — QApplication bootstrap (installed as `lorebook-ui`) |
| `UI.py` | Thin wrapper around `lorebook.ui.app.main()` |
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

> **Note:** `tensorflow_intel` in `requirements.txt` carries a `sys_platform == "win32"` marker, so it installs only on Windows — no manual editing needed on Linux/macOS.

### Build the feature database (first run)
```bash
python PhotoMatching.py --game Lorcana --build
```
This scans all images in `Card_Images/Lorcana/` and writes `DBCardCache_Lorcana.db`.

### Run the application
```bash
python UI.py
```

### Run the headless sorter (dry run — motion is mocked until the gantry exists)
```bash
# Replay a folder of images, log the chosen bin per card, no CSV writes:
python -m lorebook.sorter --game Lorcana --source Card_Images/Lorcana \
    --rules configs/sort_rules.example.json --dry-run

# Live camera + CSV logging:
python -m lorebook.sorter --game Lorcana --source camera --no-dry-run
```
Live-camera runs crop each frame to the centered card focus box (same crop as the GUI)
before matching; folder replays don't. Override with `--crop` / `--no-crop`.
`python PhotoMatching.py --sort …` is an equivalent alias. On a Raspberry Pi, install
`pip install .[sorter]` (tflite-runtime instead of full TensorFlow), generate the model
once on a desktop with `python scripts/convert_to_tflite.py`, copy it over, and run with
`--backend tflite`.

---

## Running Tests

Tests live in `tests/` (shared TF/Keras stubbing in `tests/conftest.py`). They cover the pure utility functions, the matching index, and the sorter rules/pipeline with mock hardware, and **do not** require a camera, a GPU, or pre-downloaded model weights (no TF model is loaded).

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
| `auto_scan` | bool | Scan automatically when a card settles in the focus box (`MotionGate`) |
| `currency` | str | Display currency for scanned-card market prices (`USD`/`CAD`/`EUR`/`GBP`; source prices are USD) |
| `theme` | string | UI theme: `"dark"` (default) or `"light"` — set in Settings |
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

- **Add a feature** — edit the relevant module under `lorebook/core/` for logic, `lorebook/ui/` for UI wiring, `lorebook/sorter/`/`lorebook/hardware/` for the headless sorter.
- **Fix a CSV parsing bug** — see `_normalize_existing_rows()` and `_write_rows_4col()` in `lorebook/core/csv_manager.py`.
- **Tune foil detection** — adjust `foil_score()` weights or threshold in `is_probably_foil()` (`lorebook/core/image_utils.py`).
- **Change match threshold** — core default is `0.70` in `find_best_matches()` / `MatchIndex.find()` (`lorebook/core/matching.py`); the GUI ships a stricter `0.90` default, exposed in Settings as confidence %. See `docs/MATCHING.md` for how the whole pipeline fits together.
- **Show card names** — run `python scripts/fetch_card_names.py --game <Game>` once; the GUI/sorter pick up `card_names_<Game>.json` automatically (display-only, never written to the CSV).
- **Card prices on scan** — the GUI refreshes `card_prices_<Game>.json` (Lorcast API, TCGplayer-sourced USD) and `currency_rates.json` (Frankfurter/ECB) in the background whenever they're >24 h old, and shows e.g. `$1.24 · foil $3.80` under the match details; display currency is a Settings dropdown. Only games registered in `GAMES` (`lorebook/core/price_fetcher.py`) get prices — others show nothing. CLI: `python scripts/fetch_card_prices.py --game <Game>`. Display-only, never written to the CSV.
- **Get a new set's images** — the GUI's Rebuild Database button auto-downloads missing card art before rebuilding (offline → warning logged, build continues). Lorcana pulls from LorcanaJSON; Riftbound uses the official Riot content API when the `RIOT_API_KEY` env var is set and falls back to the open Riftcodex API otherwise. CLI: `python scripts/fetch_card_images.py --game <Game> [--set <N>] [--dry-run]`. Registered fetchers live in `GAMES` (`lorebook/core/image_fetcher.py`); unregistered games are skipped silently.
- **Change sort routing** — edit the rules JSON (see `configs/sort_rules.example.json`); the engine is `decide_bin()` in `lorebook/sorter/rules.py`. Unmatched cards always go to `reject_bin`.
- **Implement the real transport** — subclass `Transport` (`lorebook/hardware/transport.py`); the pipeline needs `route_to_bin`, `advance`, `home`.
- **Cache issues** — delete `DBCardCache_<game>.db` and rebuild with `--build` flag.

---

## Known Limitations / Gotchas

- `tensorflow_intel` installs only on Windows (guarded by a `sys_platform == "win32"` marker in `requirements.txt`).
- `Card_Images/` is git-ignored (images can be large); the SQLite/JSON caches and `ui_settings.json` are also git-ignored.
- The feature model (MobileNetV2 weights) is downloaded from the internet on first run (~14 MB).
- Camera index `0` may not be correct on machines with multiple cameras — adjust in Settings.
- `foil_score` threshold (default `0.08`) was tuned empirically; may need adjustment per lighting setup.
- `riot.txt` at the repo root is a Riot Games API domain-verification token — don't delete it.
- Riftbound image fetching prefers the official Riot API (`RIOT_API_KEY` env var; dev keys from developer.riotgames.com expire every 24 h) and silently falls back to Riftcodex without one, so keyless runs still work. The two sources' JSON shapes are both handled by `riftbound_targets()`.
- `PySide6` is pinned to the 6.8 LTS line for NumPy 2 compatibility (6.6's shiboken predates NumPy 2; 6.11.1 fails to bootstrap on Windows/Py3.12) — see the comment in `requirements.txt` before bumping.
