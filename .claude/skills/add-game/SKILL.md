---
name: add-game
description: Add support for a new trading-card game to Lore-Book end to end — image folder, GameType, set selection in Settings, image/name/price fetchers, tests and docs. Use when the user wants to scan a game other than Lorcana or Riftbound.
---

# Add a new card game

Ask for the game's name (this becomes the folder name, e.g. `OnePiece`), and whether there is a
public card-data API with image URLs. Write the plan up front. Each step below is a separate place
that has to know about the game.

## 1. Minimum: scanning + CSV

- `Card_Images/<Game>/` with images named `<SetCode>-<CardCode>.<ext>` (see `split_filename()`
  in `lorebook/core/csv_manager.py`). The folder is gitignored, so never commit images.
- `csv_for_game()` derives `<Game>List.csv` from the folder name automatically.
- **Settings set selection is hardcoded.** `lorebook/ui/settings_window.py` (`apply`, around the
  "Collect game / set selections" block) initialises `selected_games`/`selected_sets` with only
  `lorcana`/`riftbound` and `continue`s past any other folder. The same defaults sit in
  `lorebook/ui/main_window.py` (`__init__` and the default-settings dict). As a result, a new game
  appears in the Settings tree but can never be selected. Fix this generally: build the dicts from
  the tree's top-level items or the `Card_Images/` folders, not a list with the new name added.
  That way the *next* game really does need no code changes. ruff is the only check over this Qt code, so
  re-read the change carefully.

## 2. GameType (only if code must detect the game from paths)

`lorebook/core/game_types.py`: add an enum member (value = lowercase folder name) and a branch in
`get_game_type()`. Only the legacy path-inference fallback in `update_cardlist()` needs it.
If the game has foil-only printings, extend `is_foil_only_card()` and add tests.

## 3. Fetchers (each optional, each has its own `GAMES` registry)

Unregistered games are skipped silently everywhere, so add only what has a real source:

| Data | Registry | Entry shape |
|------|----------|-------------|
| Card art | `lorebook/core/image_fetcher.py` `GAMES` | `(url, fetcher, targets_fn)`, where `targets_fn(data)` yields `(set_code, number, image_url)` |
| Names | `scripts/fetch_card_names.py` `GAMES` | `(url, fetcher, mapper)`, where mapper returns `{"<set>-<number>": name}` |
| Prices | `lorebook/core/price_fetcher.py` `GAMES` | `(base_url, fetcher(url, progress_cb), mapper)`, where mapper returns `{"<set>-<number>": {"usd", "usd_foil"}}` |

Follow the existing patterns: zero-pad numbers with `_pad()`, dedupe keys, tolerate `None`/missing
fields, cap pagination loops, and keep a source comment at the top of the file. Fetch one real
response once to learn the JSON shape, then write tests against a **small inline fixture**. Tests
must never hit the network (see `tests/test_image_fetcher.py`).

## 4. Tests

Add parsing tests next to the existing per-game ones (`test_image_fetcher.py`,
`test_card_prices.py`, `test_card_names.py`) plus a `csv_for_game` case. Use `tmp_path` for any
file output. Run:

```bash
ruff check . && python -m pytest tests/ -v
```

## 5. Docs

Update CLAUDE.md (Supported games, module table entries for the registries, Gotchas for any
API-key quirks), README.md, and add a manual-test section to `docs/TESTING.md`: select the game in
Settings, Rebuild Database, scan a card, and confirm the row lands in `<Game>List.csv`.

## Report

List which of steps 1–3 were done, which data sources are wired up (art/names/prices), and what
the user must verify by hand (DB build with TensorFlow, real scans in the GUI).
