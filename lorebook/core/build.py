"""The "refresh and build every game" job, Qt-free.

Per game: download newly released card art, refresh market prices when
stale, then build/update the feature cache. Shared USD exchange rates are
refreshed once per run. Network steps never block the build (offline,
source down or a format change → warning, continue). Progress and status go
through plain callbacks and cancellation through a threading.Event, so the
same job runs under the GUI's QThread worker, a CLI, or a test.
"""

import logging
import os
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from lorebook.core.card_prices import RATES_FILE, prices_stale
from lorebook.core.feature_cache import build_feature_database
from lorebook.core.game_types import CARD_IMAGES_DIR
from lorebook.core.games import list_image_files
from lorebook.core.image_fetcher import FetchOptions, download_new_images
from lorebook.core.price_fetcher import download_card_prices, download_currency_rates

logger = logging.getLogger(__name__)

# progress(-1) means "indeterminate" (a phase that only reports text lines).
INDETERMINATE = -1


@dataclass
class BuildReport:
    """How a refresh-and-build run ended: failed games, cancellation, price refresh, duration."""

    failed: list[str] = field(default_factory=list)
    cancelled: bool = False
    prices_refreshed: bool = False
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        """Return True when every game built and the run was not cancelled."""
        return not self.failed and not self.cancelled


class _BuildJob:
    """One refresh-and-build run: the callbacks, cancel flag and report its steps share.

    Each network step is a boundary: a refresh may fail in any way and must
    never block the build or abort the job, so those steps catch everything,
    log a warning and carry on.
    """

    def __init__(
        self,
        cancel_event: threading.Event,
        status: Callable[[str], None],
        progress: Callable[[int], None],
        extractor=None,
    ) -> None:
        """Store what every step needs and start an empty report."""
        self.cancel_event = cancel_event
        self.status = status
        self.progress = progress
        self.extractor = extractor
        self.report = BuildReport()

    def refresh_rates(self) -> None:
        """Download the shared USD exchange rates when the cached file is stale."""
        if not prices_stale(path=RATES_FILE):
            return
        try:
            download_currency_rates()
            self.report.prices_refreshed = True
        except Exception as e:  # noqa: BLE001 — network boundary, see class docstring
            logger.warning("Currency-rate refresh failed (continuing): %s", e)

    def refresh_images(self, game: str, image_dir: str) -> None:
        """Download card art that is not in the game's image folder yet."""
        try:
            self.status(f"Checking for new {game} card images…")
            stats = download_new_images(
                game,
                FetchOptions(out_dir=image_dir),
                progress_callback=self.status,
                cancel_event=self.cancel_event,
            )
            if stats and stats.downloaded:
                logger.info("Downloaded %s new %s images (%s failed)", stats.downloaded, game, stats.failed)
                self.status(f"Downloaded {stats.downloaded} new {game} card images")
        except Exception as e:  # noqa: BLE001 — network boundary, see class docstring
            logger.warning("Image fetch for %s failed (continuing build): %s", game, e)

    def refresh_prices(self, game: str) -> None:
        """Download the game's market prices when its cached file is stale."""
        if not prices_stale(game):
            return
        try:
            self.status(f"Updating {game} card prices…")
            count = download_card_prices(game)
            if count is not None:
                logger.info("Refreshed %s %s price entries", count, game)
                self.report.prices_refreshed = True
        except Exception as e:  # noqa: BLE001 — network boundary, see class docstring
            logger.warning("Price fetch for %s failed (continuing build): %s", game, e)

    def build_cache(self, game: str, image_dir: str) -> None:
        """Build the game's feature cache, recording the game as failed if it can't be built."""
        self.status(f"Building {game} database…")
        self.progress(0)
        if not os.path.isdir(image_dir):
            logger.error("Game folder not found: %s", image_dir)
            self.report.failed.append(game)
            return
        if not list_image_files(image_dir):
            # Nothing to build (no images, and none downloaded) — not a failure.
            logger.info("No images for %s — skipped", game)
            self.status(f"No images for {game} — skipped")
            return

        logger.info("Building DB for %s at %s", game, image_dir)
        try:
            vectors = build_feature_database(
                image_dir,
                progress_callback=self._on_build_progress,
                cancel_event=self.cancel_event,
                extractor=self.extractor,
            )
        except Exception:  # noqa: BLE001 — fail this game, build the rest
            logger.exception("Database build for %s failed", game)
            self._fail(game)
            return
        if self.cancel_event.is_set():  # a cancelled build legitimately returns few/no entries
            return
        if not vectors:
            logger.error("Database build for %s produced no entries", game)
            self._fail(game)
        else:
            logger.info("DB ready for %s: %s entries", game, len(vectors))

    def _on_build_progress(self, pct: int, current_file: str | None) -> None:
        """Forward one batch's progress, naming the file just processed."""
        self.progress(pct)
        if current_file:
            self.status(f"Processing {current_file}…")

    def _fail(self, game: str) -> None:
        """Record a game whose cache could not be built and tell the user."""
        self.report.failed.append(game)
        self.status(f"Error building {game} database")


def refresh_and_build(
    games: Iterable[str],
    *,
    cancel_event: threading.Event,
    status: Callable[[str], None] = lambda _msg: None,
    progress: Callable[[int], None] = lambda _pct: None,
    base: str = CARD_IMAGES_DIR,
    extractor=None,
) -> BuildReport:
    """Download, refresh prices and build each game's cache; see module docstring.

    Stops between images/batches/games once cancel_event is set; partial
    caches stay valid and the next run resumes from them.
    """
    start = time.time()
    job = _BuildJob(cancel_event, status, progress, extractor)
    job.refresh_rates()

    for game in games:
        if cancel_event.is_set():
            break
        image_dir = os.path.join(base, game)
        progress(INDETERMINATE)
        job.refresh_images(game, image_dir)
        job.refresh_prices(game)
        job.build_cache(game, image_dir)

    report = job.report
    report.cancelled = cancel_event.is_set()
    report.elapsed = time.time() - start
    logger.info(
        "DB builds finished in %.1fs (%s failed%s)",
        report.elapsed,
        len(report.failed),
        ", cancelled" if report.cancelled else "",
    )
    return report
