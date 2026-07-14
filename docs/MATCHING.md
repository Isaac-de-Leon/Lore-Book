# How card matching works

A one-page tour of the recognition pipeline for contributors. Code lives under
`lorebook/core/`; nothing here requires the GUI.

## Pipeline

```
camera frame ──► focus-box crop ──► MobileNetV2 ──► 1280-dim vector ──► cosine vs cache ──► ranked matches
                (image_utils)       (features)      (L2-normalized)     (matching)
```

1. **Capture & crop.** The GUI (and the sorter with `--crop`) crops the frame
   to a centered 63:88 portrait box at ~60% of frame height —
   `focus_rect()` / `crop_to_card()` in `lorebook/core/image_utils.py` — so
   the model sees the card, not the table.

2. **Feature extraction.** `lorebook/core/features.py` runs MobileNetV2
   (ImageNet weights, no top, global average pooling): the image is resized
   to 224×224, scaled to [-1, 1], and pooled into a **1280-dim float32
   vector**, which is L2-normalized (`_finalize`). Two interchangeable
   backends — `get_extractor("keras")` (desktop) and `"tflite"` (Raspberry
   Pi) — share identical preprocessing, so their vectors are
   cache-compatible (`scripts/check_parity.py` verifies). DB builds use
   `extract_batch()` to amortize predict overhead.

3. **The reference cache.** `build_feature_database()`
   (`lorebook/core/card_database.py`) extracts a vector for every image in
   `Card_Images/<Game>/` and stores them in `DBCardCache_<Game>.db`
   (SQLite, filename → vector blob). Incremental: only new images are
   processed; entries whose image was deleted are pruned (but a *missing*
   folder is treated as a bad path, never as "everything was deleted").
   Delete the .db to force a full rebuild.

4. **Matching.** Because all vectors are unit-length, cosine similarity is a
   dot product. `MatchIndex` (`lorebook/core/matching.py`) stacks the cache
   into one (N, 1280) matrix; each query is a single matrix-vector product,
   returning `[(filename, score), …]` sorted descending, filtered by
   threshold. `find_best_matches()` is the equivalent per-entry reference
   implementation.

## Thresholds and heuristics

| Knob | Default | Where |
|------|---------|-------|
| Core match threshold | 0.70 | `find_best_matches()` / `MatchIndex.find()` |
| GUI confidence threshold | 0.90 | Settings → "Confidence Threshold (%)" |
| Ambiguous-match gap | 0.02 | `MainWindow.AMBIGUOUS_GAP` — top-2 scores closer than this trigger a "check alternatives" warning (foil variants / reprints / alt arts score nearly identically) |
| Foil score threshold | 0.08 | `is_probably_foil()` — 0.7·bright-spot ratio + 0.3·Laplacian contrast, tuned empirically; sensitive to lighting |
| Auto-scan steadiness | 10 frames / diff 4.0 / min-std 12 | `MotionGate` — motion arms, steady frames fire one scan, near-uniform frames are ignored |

## Why scores behave the way they do

MobileNetV2 features are generic ImageNet embeddings, not card-specific:
same-franchise art styles cluster, so *wrong* matches still score ~0.6–0.8.
That's why the GUI default is a strict 0.90 and why the ambiguous-gap warning
exists. If accuracy ever needs a step change, the leverage points are a
fine-tuned embedding or a reranking pass — not threshold tuning.
