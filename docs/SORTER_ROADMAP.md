# Lore-Book Card Sorter — Roadmap

> Status: planning. This document sequences the work to turn Lore-Book from a desktop
> webcam scanner into an automated physical **card sorter**. No sorter code exists yet;
> this roadmap defines the phases, the reuse seams in the current codebase, and the key
> technical decisions so the pure-software pieces can land and be tested on a desktop
> **before** any hardware is built.

---

## 1. Overview & goals

Today Lore-Book is a PySide6 desktop app: it captures a webcam frame, extracts a 1280-dim
MobileNetV2 feature vector, cosine-matches it against a per-game SQLite cache, and appends the
result to a CSV. The sorter extends this into an automated machine:

> **A Raspberry Pi drives a pick-and-place gantry arm (on rods, with a vacuum/gripper pickup
> and X/Y/Z motion). For each card it: photographs → matches → decides a bin from configurable
> rules → moves the arm → drops the card in that bin → records it to CSV.**

**Goals**

- Reuse the existing, Qt-free matching pipeline (`lorebook/core/*`) verbatim — no reimplementation.
- Run inference on a Pi, which cannot comfortably host full TensorFlow → introduce a
  `get_extractor("tflite")` backend.
- Keep every pure-software phase **testable on a dev laptop with mocked hardware**, so hardware
  (the expensive, slow-to-iterate part) is only brought up once the software loop is proven.
- Configurable sorting: bin assignment is data-driven (a rules file), not hard-coded.

**Non-goals (for now):** specific hardware part numbers, value/price-based sorting (needs a price
data source), and any production enclosure/mechanical design.

---

## 2. Architecture & module layout

Two new packages, both kept importable on a laptop (hardware libraries imported lazily):

```
lorebook/
├── core/            # EXISTING — game-agnostic, Qt-free. Reused as-is.
├── ui/              # EXISTING — PySide6 GUI. Untouched by the sorter.
├── sorter/          # NEW — headless orchestration, no Qt, no top-level hardware imports
│   ├── pipeline.py      # capture → extract → match → decide_bin → place → CSV
│   ├── rules.py         # decide_bin() + rules-file loading (pure, unit-tested)
│   └── config/          # example rules files (by-set, foil, matched-vs-reject)
└── hardware/        # NEW — hardware abstraction: interface + real + mock impls
    ├── camera.py        # CameraSource  (USB/OpenCV today, picamera2 later)
    ├── motion.py        # MotionController (gantry X/Y/Z)
    └── pickup.py        # CardPickup (vacuum / gripper)
```

**Data flow**

```
 feed ─▶ CameraSource.capture()
            │
            ▼
   get_extractor(backend).extract(frame)        # lorebook/core/features.py
            │  1280-dim vector
            ▼
   find_best_matches(feat, featureDB, threshold) # lorebook/core/matching.py
            │  [(filename, score), ...]
            ▼
   is_probably_foil(frame)                        # lorebook/core/image_utils.py
            │
            ▼
   decide_bin(match, is_foil, score, rules) ─▶ bin_id   # lorebook/sorter/rules.py
            │
            ├─▶ MotionController.move_to(bin) ; CardPickup.pick()/place()
            └─▶ update_cardlist(filename, is_foil, count)  # lorebook/core/csv_manager.py
```

### Reuse seams (already exist, Qt-free)

| Capability | Symbol | File |
|---|---|---|
| Feature extraction | `extract_features()` | `lorebook/core/features.py:41` |
| Lazy model load (to be wrapped by `get_extractor`) | `_get_models()` | `lorebook/core/features.py:31` |
| Matching | `find_best_matches(feat, db, threshold=0.70)` | `lorebook/core/matching.py:43` |
| Feature cache load / build / select game | `load_cache()`, `build_feature_database()`, `set_database_path()` | `lorebook/core/card_database.py` |
| Foil detection | `is_probably_foil()` | `lorebook/core/image_utils.py` |
| CSV write | `update_cardlist(filename, is_foil, count)` | `lorebook/core/csv_manager.py:104` |
| Camera open/grab logic (currently Qt-bound — to factor out) | `start_camera()`, `_grab_frame()`, `capture_and_match()` | `lorebook/ui/main_window.py:524 / 655 / 699` |

---

## 3. Phase 1 — tflite extractor seam *(pure software, no hardware)*

The current extractor hard-codes Keras MobileNetV2 (`lorebook/core/features.py`). Full TensorFlow
is too heavy for a Pi, so the first step decouples inference behind a backend.

- **Introduce** `get_extractor(backend: str = "keras")` in `features.py`, returning an object with a
  uniform `extract(img_or_path) -> Optional[np.ndarray]` API (1280-dim, L2-normalized).
- **Refactor** the existing `extract_features()` to delegate to the `"keras"` backend so current
  GUI behavior is byte-for-byte unchanged (the GUI keeps calling `extract_features`).
- **Add** a `"tflite"` backend:
  - A one-time conversion script (`MobileNetV2` → `model.tflite`).
  - Inference via `tflite-runtime`'s `Interpreter` (no full TF on the Pi).
- **Parity check:** cosine similarity between keras-produced and tflite-produced vectors on the same
  images must stay **≥ ~0.99**, so existing `DBCardCache_*.db` caches remain valid regardless of
  which backend built them.
- **Dependencies:** `tflite-runtime` added as an optional extra (e.g. `[sorter]`/`[pi]`) in
  `pyproject.toml`; full `tensorflow` stays a desktop-only dependency.

**Risk:** float32 vs quantized output drift — mitigated by starting with a non-quantized `.tflite`
and gating on the parity check before considering quantization.

**Done when:** `get_extractor("keras")` and `get_extractor("tflite")` both pass the parity test and
the GUI is unaffected.

---

## 4. Phase 2 — headless capture→match loop *(dry run, still no motion)*

Prove the full software path end-to-end with motion stubbed.

- **Factor** the camera open/read logic out of `main_window.py` (`start_camera`/`_grab_frame`) into a
  `CameraSource` in `lorebook/hardware/camera.py`, reusing the existing platform backend-probe order
  (V4L2/DirectShow/MSMF). USB/OpenCV implementation now; a `picamera2`/libcamera implementation later.
- **Build** a headless runner: `lorebook/sorter/pipeline.py` plus a `--sort` subcommand on
  `PhotoMatching.py`. Per card it: `CameraSource.capture()` → `get_extractor(...).extract()` →
  `find_best_matches()` → `is_probably_foil()` → `decide_bin()` → **logs the chosen bin and calls
  `update_cardlist()`**, with `MotionController`/`CardPickup` replaced by logging mocks.

**Done when:** a sequence of card images run through the loop produces correct CSV rows and correct
logged bin decisions on a laptop, with zero hardware present.

---

## 5. Phase 3 — configurable rules engine *(pure software)*

- A rules file (JSON/YAML) plus a **pure** function
  `decide_bin(match_result, is_foil, confidence, rules) -> bin_id` in `lorebook/sorter/rules.py`.
- Rules express ordered conditions on: set code, game (Lorcana/Riftbound), foil flag, and match
  confidence, with an explicit **fallthrough / reject bin** for low-confidence or unmatched cards.
- Ship example configs proving the engine subsumes the obvious modes:
  - **by-set** — bin per set code.
  - **foil-vs-normal** — `is_probably_foil()` splits two bins.
  - **matched-vs-reject** — above-threshold → keep bin, else → reject bin.

**Done when:** `decide_bin()` is fully unit-tested (no hardware, no camera) and the Phase-2 loop reads
its bin choice from a rules file.

---

## 6. Phase 4 — hardware bring-up on the Pi

- **Pi/OS setup**, camera feed validation (`picamera2`/libcamera or USB via OpenCV), and the physical
  capture station: lighting and a fixed card position so framing matches the training images.
- **Gantry arm:** steppers for X/Y/Z on the rods + a vacuum pump (or servo gripper) for pickup.
  **Decision to record:** drive GPIO directly (`gpiozero`/`RPi.GPIO`) **vs** offload motion to a
  microcontroller (Arduino/GRBL) over serial. *Recommendation:* the serial/MCU split — it keeps
  real-time step generation off the Pi and gives smoother motion. Either way both satisfy the same
  `MotionController` / `CardPickup` interfaces from Phase 2, so the pipeline doesn't change.
- **Bring-up scripts:** home axes, jog, pick, place, and **bin-coordinate calibration** (record the
  X/Y/Z for each bin into the rules/config).

**Done when:** the real `MotionController`/`CardPickup` can home, pick a single card, and place it in a
named bin from a script.

---

## 7. Phase 5 — full integration & ops

- Swap the Phase-2 logging mocks for the real `MotionController`/`CardPickup`.
- **Sequencing:** feed singulation → pick → move-to-bin → place → confirm → advance.
- **Error/reject handling:** no match, ambiguous match, failed pickup, jam → reject bin + log/retry.
- **Safety:** e-stop / soft limits; **throughput** tuning; structured logging; optional status UI or a
  `systemd` service for headless operation.

**Done when:** the machine sorts a real stack unattended, writing CSV and routing rejects.

---

## 8. Testing strategy

- **Mock** `CameraSource`, `MotionController`, and `CardPickup` so the whole pipeline runs in CI on a
  laptop with no hardware (dry-run mode).
- **Parity test** for the keras-vs-tflite extractor (Phase 1).
- **Unit tests** for `decide_bin()` covering every example rules config (Phase 3).
- Existing tests (`tests/test_photo_matching.py`) already need no camera/GPU — extend the same style.

---

## 9. Dependencies & platform matrix

Wire hardware/Pi libraries as **optional extras** so the repo still `pip install`s on a laptop.

| Dependency | Desktop dev | Pi runtime | Purpose |
|---|---|---|---|
| `tensorflow` / `keras` | ✅ | ❌ | `keras` extractor backend + `.tflite` conversion |
| `tflite-runtime` | optional | ✅ | `tflite` extractor backend (lightweight inference) |
| `opencv-python` | ✅ | ✅ | image I/O, USB camera capture |
| `picamera2` / libcamera | ❌ | ✅ (optional) | Pi camera capture |
| `gpiozero` / `RPi.GPIO` | ❌ | ✅ (if direct GPIO) | motion/pickup control |
| `pyserial` | optional | ✅ (if MCU split) | talk to Arduino/GRBL controller |

---

## 10. Open questions / decisions to revisit

- **Motion control:** direct GPIO vs MCU-over-serial (leaning MCU/serial).
- **Pickup:** vacuum cup vs mechanical gripper.
- **Feed:** how cards are singulated from the input stack.
- **Bin count & layout:** fixes the gantry's reachable coordinate set.
- **Value-based sorting:** deferred — needs an external price data source.
- **tflite quantization:** only if size/speed demands it and parity holds.
