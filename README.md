# Lore-Book

A desktop trading-card scanner. Point a webcam at a physical card, and Lore-Book matches it against a local reference-image database using MobileNetV2 feature vectors, then records it in a CSV you can import into [Dreamborn.ink](https://dreamborn.ink) (Bulk Add).

**Supported games:** Disney Lorcana, Riftbound — and any other game, just by adding a `Card_Images/<GameName>/` folder.

## Features

- Live camera preview with a card-shaped focus box; scan on demand (or press `C`)
- Cosine-similarity matching against cached MobileNetV2 features (SQLite cache, built automatically)
- Foil detection (specular-highlight + contrast heuristic) with a manual override checkbox
- Per-game collection CSVs (`LorcanaList.csv`, `RiftboundList.csv`, …) in Dreamborn's 4-column format
- A headless multi-bin **card sorter** pipeline (`python -m lorebook.sorter`) with a JSON rules engine — motion hardware is mocked until the gantry exists (see `docs/SORTER_ROADMAP.md`)

## Setup

Requires Python 3.11+ and a webcam (only at runtime).

```bash
pip install -r requirements.txt
```

Place reference card images under `Card_Images/<Game>/`, named `<SetCode>-<CardCode>.<ext>` (e.g. `001-042.webp`). Then build the feature database once:

```bash
python PhotoMatching.py --game Lorcana --build
```

## Usage

Run the GUI:

```bash
python UI.py
```

Scan a card, review the match (navigate alternatives with `A`/`D`), toggle Foil with `F`, and add it to your collection with **+ Add to Collection** (`Ctrl+S`). The collection lives in `<Game>List.csv`.

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

## More

Architecture, settings reference, and contributor guidance live in [CLAUDE.md](CLAUDE.md); the physical-sorter plan is in [docs/SORTER_ROADMAP.md](docs/SORTER_ROADMAP.md).
