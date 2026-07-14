# image_fetcher.py — download reference card images into Card_Images/<Game>/.
#
# Core, no Qt. Used by scripts/fetch_card_images.py (CLI) and by the GUI's
# Rebuild Database flow, which calls download_new_images() before each build so
# newly released sets populate automatically. Images are named
# "<setCode>-<number>.<ext>" (zero-padded to match the collection's 009-041
# style) so split_filename()/get_available_sets() pick up the set codes and
# card_names lookups line up. Existing files are skipped, so re-running only
# downloads what's new.
#
# Source (community-maintained; pass url= if the format changes):
#   Lorcana — LorcanaJSON bulk data   https://lorcanajson.org
#
# Note: card art is copyrighted by Ravensburger/Riot. Downloads are local, for
# personal-collection use only (Card_Images/ is gitignored); nothing is
# redistributed.

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional, Tuple

LORCANA_URL = "https://lorcanajson.org/files/current/en/allCards.json"

# A UA header — the Ravensburger image CDN can 403 an empty urllib agent.
_UA = "Lore-Book-image-fetcher/1.0 (+https://github.com/; personal collection tool)"


@dataclass
class FetchStats:
    """Result of a download_new_images() run."""
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0


def _fetch_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def _pad(part: str) -> str:
    """Zero-pad a numeric code to 3 digits to match the 009-041 filename style;
    leave non-numeric codes (promos like P1, Q1) untouched."""
    part = str(part).strip()
    return part.zfill(3) if part.isdigit() else part


def _norm(part: str) -> str:
    """Strip leading zeros for padding-insensitive comparison ('009' == '9')."""
    part = str(part).strip().lstrip("0")
    return part if part else "0"


def _is_promo_printing(card: dict) -> bool:
    """
    True for promo printings, which share their home set's setCode/number with
    the main-set card in LorcanaJSON (e.g. Zeus "18/P3 · EN · 10" collides with
    Scrooge "18/204 · EN · 10"). The fullIdentifier denominator tells them
    apart: main-set cards have a numeric one.
    """
    ident = str(card.get("fullIdentifier", ""))
    denom = ident.split("/", 1)[1].split()[0] if "/" in ident else ""
    return bool(denom) and not denom.isdigit()


def lorcana_targets(data) -> Iterator[Tuple[str, str, str]]:
    """
    Yield (set_code, number, image_url) for every Lorcana card with a full image.

    setCode-number keys are deduplicated with main-set printings preferred over
    promos (a promo is only used when no main printing shares its key), so the
    downloaded image always shows the art the filename claims.
    """
    best: dict = {}  # (set_code, number) -> (is_promo, url)
    order = []
    for card in (data or {}).get("cards", []):
        set_code = str(card.get("setCode", "")).strip()
        number = str(card.get("number", "")).strip()
        url = (card.get("images") or {}).get("full")
        if not (set_code and number and url):
            continue
        key = (set_code, number)
        promo = _is_promo_printing(card)
        if key not in best:
            best[key] = (promo, url)
            order.append(key)
        elif best[key][0] and not promo:  # main printing beats an earlier promo
            best[key] = (promo, url)
    for set_code, number in order:
        yield set_code, number, best[(set_code, number)][1]


# game key (lowercased folder name) → (bulk-data URL, fetcher, targets fn)
GAMES = {
    "lorcana": (LORCANA_URL, _fetch_json, lorcana_targets),
}


def _download(url: str, retries: int = 3, backoff: float = 1.5) -> bytes:
    """Fetch image bytes with a couple of retries on transient failures."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise last


def _to_webp(raw: bytes, quality: int) -> bytes:
    """Re-encode source (JPG) bytes to WebP via OpenCV. Raises if cv2/webp is unavailable."""
    import cv2
    import numpy as np

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode source image")
    ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, quality])
    if not ok:
        raise ValueError("WebP encoding failed (OpenCV build lacks WebP support?)")
    return buf.tobytes()


def download_new_images(
    game: str,
    out_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    url: Optional[str] = None,
    sets: Optional[Iterable[str]] = None,
    fmt: str = "webp",
    quality: int = 95,
    limit: Optional[int] = None,
    force: bool = False,
    delay: float = 0.05,
    dry_run: bool = False,
) -> Optional[FetchStats]:
    """
    Download any missing card images for a game into out_dir.

    game: game folder name (e.g. "Lorcana"). Returns None when no fetcher is
    registered for it (e.g. Riftbound) — callers should skip silently.
    out_dir: defaults to Card_Images/<Game>.
    progress_callback: receives one-line status strings (per-image and summary).
    sets: restrict to these set codes ('9' and '009' both match); None = all.
    fmt: "webp" (re-encode via OpenCV) or "jpg" (keep the source bytes).
    dry_run: report what would download without fetching images or writing.

    Errors fetching the card list propagate to the caller; per-image failures
    are counted in FetchStats.failed and never raise.
    """
    key = game.strip().lower()
    if key not in GAMES:
        return None

    def report(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    default_url, fetcher, target_fn = GAMES[key]
    resolved_url = url or default_url
    resolved_out = out_dir or os.path.join("Card_Images", game.strip().capitalize())
    ext = ".webp" if fmt == "webp" else ".jpg"
    wanted = {_norm(s) for s in sets} if sets else None

    report(f"Fetching card list from {resolved_url} ...")
    data = fetcher(resolved_url)

    targets = [t for t in target_fn(data) if wanted is None or _norm(t[0]) in wanted]
    stats = FetchStats()
    if not targets:
        report("No cards matched." if wanted else
               "No cards found — the source format may have changed.")
        return stats

    if not dry_run:
        os.makedirs(resolved_out, exist_ok=True)

    report(f"{len(targets)} card(s) to consider -> {resolved_out}  (format: {fmt})")
    for set_code, number, img_url in targets:
        fname = f"{_pad(set_code)}-{_pad(number)}{ext}"
        dest = os.path.join(resolved_out, fname)

        if not force and os.path.exists(dest):
            stats.skipped += 1
            continue
        if limit is not None and stats.downloaded >= limit:
            break
        if dry_run:
            report(f"  would fetch {fname}  <- {img_url}")
            stats.downloaded += 1
            continue

        try:
            raw = _download(img_url)
            payload = _to_webp(raw, quality) if fmt == "webp" else raw
            with open(dest, "wb") as f:
                f.write(payload)
            stats.downloaded += 1
            report(f"  {fname}  ({len(payload) // 1024} KB)")
        except Exception as e:
            stats.failed += 1
            logging.warning(f"Failed to fetch card image {fname}: {e}")
            report(f"  FAILED {fname}: {e}")
        if delay:
            time.sleep(delay)

    verb = "would download" if dry_run else "downloaded"
    report(f"Done. {verb} {stats.downloaded}, skipped {stats.skipped} existing, "
           f"{stats.failed} failed.")
    return stats
