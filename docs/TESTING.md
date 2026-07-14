# Manual test plan — round 1 + 2 changes

Everything the automated suite can't verify (camera, Qt, live websites), in the
order that catches problems fastest. Each item says what to do and exactly what
you should see. Tick them off; anything that misbehaves, note what happened and
at which step.

## 0. Setup (once)

```bash
git checkout main && git pull
pip install -r requirements.txt
python scripts/fetch_card_names.py --game Lorcana
python scripts/fetch_card_names.py --game Riftbound
python scripts/fetch_card_images.py --game Lorcana   # optional: card art
```

- [ ] Both name-fetch commands print `Wrote N card names to card_names_<Game>.json`
      (N in the low thousands for Lorcana).
- [ ] The image fetch skips files you already have and only downloads what's
      missing; re-running it immediately downloads nothing.

## 1. Startup & database build

- [ ] `python UI.py` — the progress bar runs for **each** game folder in turn
      (watch it reset to 0 between Lorcana and Riftbound) and ends with
      "All databases ready. Scan a card to begin."
      *(Round-1 fix: it used to go dark after the first game.)*
- [ ] Temporarily move one card image out of `Card_Images/Lorcana/`, relaunch:
      `logs/card_scanner.log` shows `Pruned 1 stale cache entries`, and that
      card can no longer be matched. Move the file back; next launch re-adds it.
      *(Stale-cache prune — deleted cards used to match forever.)*

## 2. Basic scan + card names

- [ ] Start the camera, scan a card (`C` or the button): the result panel shows
      **"Card Name · 009-041"**, not just the code. Delete
      `card_names_Lorcana.json` and rescan → falls back to the bare code, no
      errors. Re-run the fetch script to restore it.
- [ ] `A`/`D` navigate alternative matches; the thumbnail updates.
- [ ] First scan right after starting the camera is properly exposed — not a
      dark/washed-out frame. *(Warm-up fix: auto-exposure now actually settles.)*

## 3. Close-match warning

- [ ] Scan a card that has reprints/foil variants in your reference images:
      when the top two scores are within 2% the status line shows
      **"⚠ Close match (…% vs …%) — check alternatives with A/D"**.
- [ ] Scan something unambiguous: no warning.
- [ ] Note whether 2% feels right — too chatty or too quiet is a one-line tune
      (`MainWindow.AMBIGUOUS_GAP`).

## 4. Count validation

- [ ] The count box won't accept letters at all, and with it emptied, **Add**
      shows the red "Invalid count — enter a number from 1 to 999." and the CSV
      is untouched.
- [ ] Count `3` → the CSV row's Count goes up by exactly 3.

## 5. Undo

- [ ] Add a card, click **Undo** (or `Ctrl+Z`): status shows "Removed …", the
      CSV count drops back, and Undo greys out (pressing it again does nothing).
- [ ] Add the same card twice (two separate Adds), Undo once: only the **last**
      add is reversed.

## 6. Game switching (the big round-1 bug)

- [ ] With both games' images present: Settings → check **Riftbound**, uncheck
      Lorcana → Apply → scan a Riftbound card. It must match a **Riftbound**
      card (thumbnail loads from `Card_Images/Riftbound/`) and Add must write
      to `RiftboundList.csv`. Switch back and confirm Lorcana still works.
      *(Before the fix this scanned against the previous game's index.)*

## 7. Auto-scan

Enable "Auto-scan when a card settles in the focus box" in Settings, then:

- [ ] Place a card in the focus box and hold it still ~0.5s → one scan fires
      by itself.
- [ ] Leave the card sitting → **no** repeat scans.
- [ ] Remove the card → no scan of the empty mat.
- [ ] Place the next card → fires again. Rhythm test: scan 10 cards in a row
      only by swapping them.
- [ ] If it *does* fire on the empty mat, your background is textured — note it
      (the fix is raising `min_std` in `MotionGate(min_std=12.0)`,
      `main_window.py`).

## 8. Foil detection

- [ ] Scan a foil: the Foil checkbox self-checks and Add records variant
      `foil`. Scan a matte card: stays unchecked. (Threshold lives in
      Settings if your lighting disagrees.)

## 9. CSV integrity (the point of the app)

- [ ] Open `LorcanaList.csv`: header + 4 columns, one row per card/variant,
      counts merged. No `.tmp` files left in the folder.
- [ ] Import it into Dreamborn.ink bulk add — still accepted.

## 9b. Collection tab & art auto-download

- [ ] Switch to the **Collection** tab: the table shows your `<Game>List.csv`
      rows, sorts by clicking headers, and follows the game dropdown.
- [ ] **Export CSV…** saves a copy that opens cleanly (original untouched).
- [ ] Settings → **Rebuild Database**: with new Lorcana sets available, missing
      card art downloads automatically before the build; offline it logs a
      warning and the build continues with what you have.

## 10. Camera robustness

- [ ] Unplug the webcam mid-preview: within ~3s you get the "Camera stopped"
      warning instead of a frozen frame. Replug, Start again — recovers.

## 11. Headless sorter (dry run, no hardware)

```bash
python -m lorebook.sorter --game Lorcana --source Card_Images/Lorcana \
    --rules configs/sort_rules.example.json --dry-run --max-cards 10
```

- [ ] Log lines show `card=001-042.webp (Card Name) … → bin=…` and a per-bin
      total at the end.
- [ ] Typo test: copy the rules file, misspell a key (e.g. `min_confidnce`),
      run again → it refuses to start and the error names the bad field.

## 12. Build speed (optional)

- [ ] Delete `DBCardCache_Lorcana.db` and time a full rebuild. Should be
      noticeably faster than you remember on a large image set (batched
      inference), with the same match results afterwards.

## Reporting back

Worth mentioning even if everything passes: whether the 2% close-match gap
feels right, and whether auto-scan needed the `min_std` tweak — both were
calibrated without hardware.
