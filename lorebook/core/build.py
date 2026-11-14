# build.py — the "refresh and build every game" job, Qt-free.
#
# Per game: download newly released card art, refresh market prices when
# stale, then build/update the feature cache. Shared USD exchange rates are
# refreshed once per run. Network steps never block the build (offline,
# source down or a format change → warning, continue). Progress and status go
# through plain callbacks and cancellation through a threading.Event, so the
# same job runs under the GUI's QThread worker, a CLI, or a test.

import logging
import os
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from lorebook.core.card_database import build_feature_database, list_image_files
from lorebook.core.card_prices import RATES_FILE, prices_stale
from lorebook.core.game_types import BASE_DATABASE_PATH
from lorebook.core.image_fetcher import download_new_images
from lorebook.core.price_fetcher import download_card_prices, download_currency_rates

logger = logging.getLogger(__name__)

# progress(-1) means "indeterminate" (a phase that only reports text lines).
INDETERMINATE = -1


@dataclass
class BuildReport:
    failed: list[str] = field(default_factory=list)
    cancelled: bool = False
    prices_refreshed: bool = False
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.failed and not self.cancelled


def refresh_and_build(
    games: Iterable[str],
    *,
    cancel_event: threading.Event,
    status: Callable[[str], None] = lambda _msg: None,
    progress: Callable[[int], None] = lambda _pct: None,
    base: str = BASE_DATABASE_PATH,
    extractor=None,
) -> BuildReport:
    """Download, refresh prices and build each game's cache; see module docstring.

    Stops between images/batches/games once cancel_event is set; partial
    caches stay valid and the next run resumes from them.
    """
    start = time.time()
    report = BuildReport()

    # Boundary (each network step below): a refresh may fail in any way and
    # must never block the build or abort the job.
    if prices_stale(path=RATES_FILE):
        try:
            download_currency_rates()
            report.prices_refreshed = True
        except Exception as e:  # noqa: BLE001
            logger.warning("Currency-rate refresh failed (continuing): %s", e)

    for game in games:
        if cancel_event.is_set():
            break
        game_path = os.path.join(base, game)
        progress(INDETERMINATE)

        try:
            status(f"Checking for new {game} card images…")
            stats = download_new_images(game, out_dir=game_path, progress_callback=status,
                                        cancel_event=cancel_event)
            if stats and stats.downloaded:
                logger.info("Downloaded %s new %s images (%s failed)", stats.downloaded, game, stats.failed)
                status(f"Downloaded {stats.downloaded} new {game} card images")
        except Exception as e:  # noqa: BLE001 — network boundary, see above
            logger.warning("Image fetch for %s failed (continuing build): %s", game, e)

        if prices_stale(game):
            try:
                status(f"Updating {game} card prices…")
                count = download_card_prices(game)
                if count is not None:
                    logger.info("Refreshed %s %s price entries", count, game)
                    report.prices_refreshed = True
            except Exception as e:  # noqa: BLE001 — network boundary, see above
                logger.warning("Price fetch for %s failed (continuing build): %s", game, e)

        status(f"Building {game} database…")
        progress(0)
        if not os.path.isdir(game_path):
            logger.error("Game folder not found: %s", game_path)
            report.failed.append(game)
            continue
        if not list_image_files(game_path):
            # Nothing to build (no images, and none downloaded) — not a failure.
            logger.info("No images for %s — skipped", game)
            status(f"No images for {game} — skipped")
            continue

        def on_progress(pct: int, current_file: str | None) -> None:
            progress(pct)
            if current_file:
                status(f"Processing {current_file}…")

        logger.info("Building DB for %s at %s", game, game_path)
        try:
            db = build_feature_database(progress_callback=on_progress, db_path=game_path,
                                        cancel_event=cancel_event, extractor=extractor)
        except Exception:  # noqa: BLE001 — fail this game, build the rest
            logger.exception("Database build for %s failed", game)
            report.failed.append(game)
            status(f"Error building {game} database")
            continue
        if cancel_event.is_set():  # a cancelled build legitimately returns few/no entries
            break
        if not db:
            logger.error("Database build for %s produced no entries", game)
            report.failed.append(game)
            status(f"Error building {game} database")
        else:
            logger.info("DB ready for %s: %s entries", game, len(db))

    report.cancelled = cancel_event.is_set()
    report.elapsed = time.time() - start
    logger.info("DB builds finished in %.1fs (%s failed%s)", report.elapsed, len(report.failed),
                ", cancelled" if report.cancelled else "")
    return report
