---
name: new-set
description: Onboard a newly released card set for an existing game (Lorcana or Riftbound) — fetch its card art, names and prices, rebuild the feature database, and check that set-specific rules (foil-only numbering, filename codes) still hold. Use when the user says a new set dropped, asks to "add set N", or scans of a new set aren't matching.
---

# Onboard a new set

Arguments: game (`Lorcana` / `Riftbound`) and optionally the set number. Ask if the game is unclear.

## 1. Check the data sources know about the set

The upstream APIs sometimes lag a release by days. Dry-run the image fetch first:

```bash
python scripts/fetch_card_images.py --game <Game> --set <N> --dry-run
```

- **Zero targets** → the source hasn't published the set yet. Stop and tell the user. Don't
  hand-craft URLs or write placeholder images.
- **Riftbound**: without `RIOT_API_KEY` it falls back to Riftcodex, plus the official card gallery
  for whole sets Riftcodex lacks. If one source has the set and the other doesn't, say which one
  you used. The log line names it.

## 2. Fetch art, names, prices

```bash
python scripts/fetch_card_images.py --game <Game> --set <N>
python scripts/fetch_card_names.py  --game <Game>
python scripts/fetch_card_prices.py --game <Game>   # Lorcana only (see price_fetcher.GAMES)
```

Then check the new files in `Card_Images/<Game>/`:
- Filenames follow `<SetCode>-<CardCode>.webp`, zero-padded like the existing sets (`013-041`).
  `split_filename()` / `get_available_sets()` depend on this.
- The count is plausible (Lorcana main sets: 204 regular cards plus Enchanted/Epic/Iconic above that).
- Image files, `card_names_*.json`, `card_prices_*.json` and `DBCardCache_*.db` are all
  gitignored. **Never commit them.**

## 3. Re-check set-specific rules

- **Foil-only cutoff**: `LORCANA_MAX_NORMAL_CARD = 204` in `lorebook/core/game_types.py`.
  Check the new set's highest *non-foil* card number in the LorcanaJSON data
  (`rarity` / `foilTypes` fields). If the set breaks the 204 assumption, make
  `is_foil_only_card()` set-aware, add a test in `tests/test_photo_matching.py`, and
  update the CSV Format section of CLAUDE.md. Don't just bump the constant, because older sets still use 204.
- **Promo collisions**: if new promo printings share a set/number with main-set cards,
  `_is_promo_printing()` in `image_fetcher.py` should still prefer the main printing. Spot-check one.
- **Odd card codes** (letters, `-alt` suffixes): confirm `split_filename()` handles them.

## 4. Rebuild the database

```bash
python PhotoMatching.py --game <Game> --build
```

(In the GUI this is the Rebuild Database button, which also runs step 2's image fetch.) This needs
TensorFlow and the model weights. In a cloud session or CI those usually aren't available, so
say the rebuild is left to the user rather than claiming it ran.

## 5. Verify and report

- `ruff check . && python -m pytest tests/ -v` if any code changed.
- Report: cards downloaded, any skipped/failed downloads, whether the foil rule needed changes,
  and what the user still has to do by hand (rebuild, test scans with real cards).
- **Do not** edit `LorcanaList.csv` / `RiftboundList.csv`. Those hold the user's real collection.
