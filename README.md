# Lore-Book

A desktop trading-card scanner. Point a webcam at a physical card, and Lore-Book matches it against a local reference-image database using MobileNetV2 feature vectors, then records it in a CSV you can import into [Dreamborn.ink](https://dreamborn.ink) (Bulk Add).

**Supported games:** Disney Lorcana, Riftbound — and any other game, just by adding a `Card_Images/<GameName>/` folder.

## Features

- Live camera preview with a card-shaped focus box; scan on demand (`C`) or **auto-scan** — a scan fires by itself when a card settles in the focus box (toggle in Settings)
- Cosine-similarity matching against cached MobileNetV2 features (SQLite cache, built automatically), with a **close-match warning** when the top two candidates are nearly tied
- **Card names** ("Elsa — Spirit of Winter · 009-041") from a locally cached names file, and **auto-download of reference card art** from LorcanaJSON (Rebuild Database fetches new sets by itself)
- Foil detection (specular-highlight + contrast heuristic) with a manual override checkbox
- Per-game collection CSVs (`LorcanaList.csv`, `RiftboundList.csv`, …) in Dreamborn's 4-column format, with count validation and single-level **Undo** (`Ctrl+Z`)
- A **Collection tab** to browse the current collection (sortable, per-game) and export a CSV copy
- A headless multi-bin **card sorter** pipeline (`python -m lorebook.sorter`) with a JSON rules engine — motion hardware is mocked until the gantry exists (see `docs/SORTER_ROADMAP.md`)

## Install (Windows)

Grab `LoreBook-Setup-<version>.exe` from the [latest release](https://github.com/Isaac-de-Leon/Lore-Book/releases/latest) and run it — no Python required. It installs per-user (no admin prompt) to `%LOCALAPPDATA%\Programs\Lore Book`; app data (settings, card images, caches, collection CSVs, logs) lives in `%LOCALAPPDATA%\LoreBook` and survives uninstall/upgrade. On first run the app downloads the MobileNetV2 model weights (~14 MB).

## Setup (from source)

Requires Python 3.11+ and a webcam (only at runtime).

```bash
pip install -r requirements.txt
```

Get reference card images into `Card_Images/<Game>/`, named `<SetCode>-<CardCode>.<ext>` (e.g. `001-042.webp`) — for Lorcana they can be downloaded automatically:

```bash
python scripts/fetch_card_images.py --game Lorcana   # card art (also auto-runs on Rebuild Database)
python scripts/fetch_card_names.py --game Lorcana    # display names (optional, once per game)
python scripts/fetch_card_names.py --game Riftbound
```

Then build the feature database once:

```bash
python PhotoMatching.py --game Lorcana --build
```

## Usage

Run the GUI:

```bash
python UI.py
```

Scan a card, review the match (navigate alternatives with `A`/`D`), toggle Foil with `F`, and add it to your collection with **+ Add to Collection** (`Ctrl+S`); undo a mistaken add with `Ctrl+Z`. The collection lives in `<Game>List.csv` and is browsable in the **Collection** tab.

Run the headless sorter (dry run over a folder of images, no CSV writes):

```bash
python -m lorebook.sorter --game Lorcana --source Card_Images/Lorcana \
    --rules configs/sort_rules.example.json --dry-run
```

On a Raspberry Pi, install `pip install .[sorter]` (tflite-runtime instead of full TensorFlow), generate the model once on a desktop with `python scripts/convert_to_tflite.py`, copy it over, and run with `--backend tflite`.

## CSV format

| Set Number | Card Number | Variant | Count |
|------------|-------------|---------|-------|
| 009        | 041         | normal  | 2     |
| 001        | 007         | foil    | 1     |

## Tests

No camera, GPU, or TensorFlow needed (TF/Keras are stubbed in `tests/conftest.py`):

```bash
python -m pytest tests/ -v
```

## Releasing

1. Bump `__version__` in `lorebook/__init__.py` and commit.
2. Tag and push: `git tag v<version> && git push origin v<version>`.
3. The `Release` workflow builds the Windows installer (PyInstaller + Inno Setup) and attaches it to a GitHub Release with auto-generated notes. The build fails fast if the tag doesn't match `__version__`.

## More

Architecture, settings reference, and contributor guidance live in [CLAUDE.md](CLAUDE.md); how recognition works is explained in [docs/MATCHING.md](docs/MATCHING.md); the physical-sorter plan is in [docs/SORTER_ROADMAP.md](docs/SORTER_ROADMAP.md).
