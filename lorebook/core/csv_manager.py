# csv_manager.py — CSV read/write and card-list management.

import csv
import logging
import os
from typing import List, Optional, Tuple

from lorebook.core.game_types import (
    BASE_DATABASE_PATH,
    LORCANA_CSV,
    RIFTBOUND_CSV,
    GameType,
    csv_for_game,
    get_game_type,
)


def _split_filename(matchedFilename: str) -> Tuple[str, str]:
    """
    Split a card image filename into (set_code, card_code).

    Examples:
        "001-042.webp"      → ("001", "042")
        "ONG-23c-alt.jpg"   → ("ONG", "23c-alt")
        "nosetcode.png"     → ("",    "nosetcode")
    """
    try:
        base = os.path.splitext(os.path.basename(matchedFilename))[0]
        parts = base.split("-", maxsplit=2)
        if len(parts) < 2:
            return "", base
        return parts[0], "-".join(parts[1:])
    except Exception as e:
        logging.error(f"Error splitting filename {matchedFilename}: {e}")
        return "", matchedFilename


def get_available_sets(
    game_type: Optional[GameType] = None,
    db_path: Optional[str] = None,
) -> List[str]:
    """
    Return a sorted list of all set codes found in db_path (or the current databasePath).

    game_type is accepted for API compatibility but no longer restricts results —
    all set codes present in the folder are returned regardless of format.
    """
    from lorebook.core.card_database import _list_image_files, databasePath

    resolved = db_path or databasePath
    sets = set()
    for fname in _list_image_files(resolved):
        set_code, _ = _split_filename(fname)
        if set_code:
            sets.add(set_code)
    return sorted(sets)


def _normalize_existing_rows(csv_path: str) -> List[List[str]]:
    """
    Read a CSV and normalize every row to exactly 4 columns:
    [Set Number, Card Number, Variant, Count].

    Handles 3-col (count defaults to 0), 4-col, and 5-col rows (5th column
    is a legacy "Tag" field that is ignored).
    """
    rows: List[List[str]] = []
    if not os.path.exists(csv_path):
        return rows
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            first = True
            for row_num, row in enumerate(reader, 1):
                try:
                    if first:
                        first = False
                        if row and any("set" in c.lower() for c in row):
                            continue  # skip header
                    if not row:
                        continue
                    if len(row) >= 4:
                        # Count is always col[3]; a 5th legacy "Tag" column is ignored
                        count = row[3] if row[3] != "" else "0"
                        rows.append([row[0], row[1], row[2], count])
                    elif len(row) == 3:
                        rows.append([row[0], row[1], row[2], "0"])
                except Exception as e:
                    logging.warning(f"Error processing row {row_num} in {csv_path}: {e}")
        return rows
    except Exception as e:
        logging.error(f"Error reading CSV file {csv_path}: {e}")
        return rows


def _write_rows_4col(csv_path: str, rows: List[List[str]]) -> None:
    """Write rows to CSV with a 4-column header."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Set Number", "Card Number", "Variant", "Count"])
        for r in rows:
            writer.writerow([r[0], r[1], r[2], r[3]])


def _csv_for_game_type(game_type: GameType) -> str:
    """Map a GameType to its CSV file (legacy behavior: everything non-Riftbound → Lorcana)."""
    return RIFTBOUND_CSV if game_type == GameType.RIFTBOUND else LORCANA_CSV


def update_cardlist_batch(
    cards,
    game: Optional[str] = None,
) -> None:
    """
    Increment (or insert) rows for many cards with one read+write per CSV file.

    cards: iterable of (matchedFilename, is_foil, count) tuples.
    game: explicit game folder name; the target file is csv_for_game(game),
    so any game gets its own <Game>List.csv. When None (legacy behavior),
    each card's game is inferred from its filename path: Riftbound paths →
    RiftboundList.csv, everything else → LorcanaList.csv.
    """
    target = csv_for_game(game) if game is not None else None
    by_file: dict = {}
    for matchedFilename, is_foil, count in cards:
        try:
            count = int(count)
        except Exception:
            count = 1
        if count < 1:
            continue
        target_file = target or _csv_for_game_type(get_game_type(matchedFilename))
        set_code, card_code = _split_filename(matchedFilename)
        variant = "foil" if is_foil else "normal"
        by_file.setdefault(target_file, []).append((set_code, card_code, variant, count))

    for target_file, updates in by_file.items():
        existing = _normalize_existing_rows(target_file)
        for set_code, card_code, variant, count in updates:
            for r in existing:
                if r[0] == set_code and r[1] == card_code and r[2] == variant:
                    try:
                        r[3] = str(int(r[3]) + count)
                    except Exception:
                        r[3] = str(count)
                    break
            else:
                existing.append([set_code, card_code, variant, str(count)])
        _write_rows_4col(target_file, existing)


def update_cardlist(
    matchedFilename: str,
    is_foil: bool,
    count: int = 1,
    game: Optional[str] = None,
) -> None:
    """
    Increment (or insert) a row in the game-appropriate CSV.

    When game is None the target file is chosen by inspecting the path of
    matchedFilename: Riftbound paths → RiftboundList.csv, everything else →
    LorcanaList.csv. Pass the game folder name explicitly to skip the path
    inference and write to csv_for_game(game).
    """
    update_cardlist_batch([(matchedFilename, is_foil, count)], game=game)
