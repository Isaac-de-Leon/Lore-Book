# pipeline.py — headless capture→match→decide→route→CSV loop.
#
# Ties the existing matching stack to the hardware abstraction and the rules
# engine. Every collaborator (camera, transport, extractor) is injected, so the
# whole loop runs against desktop mocks with no camera, motors, or TensorFlow.
# In dry_run mode (the default) it logs the bin decision and never writes the
# CSV; with dry_run=False it records the card via the existing csv_manager,
# batching writes so the CSV is rewritten every csv_flush_interval cards
# (and at end of run) instead of once per card.

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from lorebook.core.csv_manager import split_filename, update_cardlist_batch
from lorebook.core.image_utils import crop_to_card, is_probably_foil
from lorebook.core.matching import MatchIndex
from lorebook.hardware.camera import CameraSource
from lorebook.hardware.transport import Transport
from lorebook.sorter.rules import SortRules, decide_bin

logger = logging.getLogger(__name__)


@dataclass
class SortOutcome:
    """The result of processing a single card."""

    filename: Optional[str]      # matched reference image name, or None if no match
    set_code: str
    card_code: str
    confidence: float
    is_foil: bool
    bin: str
    matched: bool


class SortPipeline:
    """
    Drives one card at a time through capture → match → decide-bin → route.

    Unmatched cards (no match above threshold, or extraction failure) always
    go to rules.reject_bin — they never flow through the rule list, so a
    foil-only or game-only rule can't capture an unidentified card.

    Args:
        camera:      frame source (real or mock).
        transport:   bin router (mock until the gantry exists).
        extractor:   object with ``.extract(img) -> Optional[np.ndarray]``
                     (from ``get_extractor(backend)``).
        feature_db:  {filename: vector} reference cache (from ``load_cache``).
        rules:       SortRules controlling bin assignment.
        game:        active game name (e.g. "Lorcana"); used for rule matching
                     and CSV routing — cards are recorded in the game's own
                     <Game>List.csv (csv_for_game).
        threshold:   minimum cosine match score.
        foil_threshold: optional override for foil detection.
        dry_run:     when True (default), never write the CSV.
        csv_flush_interval: matched cards to accumulate between CSV writes.
        crop_to_focus: crop each frame to the centered 63:88 card box before
                     foil detection and matching — the same crop the GUI
                     applies. Use for live-camera frames where the card sits
                     centered against background; leave off when replaying
                     already-cropped reference images.
    """

    def __init__(
        self,
        *,
        camera: CameraSource,
        transport: Transport,
        extractor,
        feature_db: Dict[str, np.ndarray],
        rules: SortRules,
        game: str,
        threshold: float = 0.70,
        foil_threshold: Optional[float] = None,
        dry_run: bool = True,
        csv_flush_interval: int = 25,
        crop_to_focus: bool = False,
    ):
        self.camera = camera
        self.transport = transport
        self.extractor = extractor
        self.feature_db = feature_db
        self.rules = rules
        self.game = game
        self.threshold = threshold
        self.foil_threshold = foil_threshold
        self.dry_run = dry_run
        self.csv_flush_interval = max(1, csv_flush_interval)
        self.crop_to_focus = crop_to_focus
        self._pending: Counter = Counter()  # (filename, is_foil) -> count
        self._index = MatchIndex(feature_db)  # one matmul per card instead of a dict scan

    # Cap for the foil-detection pass: specular/contrast stats survive
    # downscaling, and the full-res Laplacian is a needless per-card cost on
    # a Pi. Note the Laplacian score shifts slightly with resolution, so
    # --foil-threshold may need retuning versus the GUI's full-res 0.08.
    _FOIL_MAX_HEIGHT = 360

    def _is_foil(self, frame: np.ndarray) -> bool:
        h = frame.shape[0] if frame is not None and frame.ndim >= 2 else 0
        if h > self._FOIL_MAX_HEIGHT:
            scale = self._FOIL_MAX_HEIGHT / h
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if self.foil_threshold is None:
            return is_probably_foil(frame)
        return is_probably_foil(frame, threshold=self.foil_threshold)

    def flush_csv(self) -> None:
        """Write any accumulated matched cards to the CSV in one read+write."""
        if not self._pending:
            return
        update_cardlist_batch(
            [(fname, foil, count) for (fname, foil), count in self._pending.items()],
            game=self.game,
        )
        self._pending.clear()

    def process_one(self, frame: np.ndarray) -> SortOutcome:
        """Match a single frame, decide its bin, route it, and (optionally) queue it for CSV."""
        if self.crop_to_focus:
            frame = crop_to_card(frame)

        is_foil = self._is_foil(frame)

        feat = self.extractor.extract(frame)
        matches = self._index.find(feat, threshold=self.threshold) if feat is not None else []

        if matches:
            filename, confidence = matches[0]
            set_code, card_code = split_filename(filename)
            matched = True
            bin_id = decide_bin(
                set_code=set_code,
                card_code=card_code,
                game=self.game,
                is_foil=is_foil,
                confidence=confidence,
                rules=self.rules,
            )
        else:
            filename, confidence = None, 0.0
            set_code, card_code = "", ""
            matched = False
            bin_id = self.rules.reject_bin

        self.transport.route_to_bin(bin_id)
        self.transport.advance()

        if matched and not self.dry_run:
            self._pending[(filename, is_foil)] += 1
            if sum(self._pending.values()) >= self.csv_flush_interval:
                self.flush_csv()

        logger.info(
            "card=%s conf=%.3f foil=%s → bin=%s%s",
            filename or "<no match>",
            confidence,
            is_foil,
            bin_id,
            " (dry-run)" if self.dry_run else "",
        )

        return SortOutcome(
            filename=filename,
            set_code=set_code,
            card_code=card_code,
            confidence=confidence,
            is_foil=is_foil,
            bin=bin_id,
            matched=matched,
        )

    def run(self, max_cards: Optional[int] = None) -> List[SortOutcome]:
        """
        Process frames until the camera feed is exhausted or max_cards is hit.
        Flushes pending CSV rows, releases the camera, and homes the transport
        on completion.
        """
        outcomes: List[SortOutcome] = []
        try:
            while max_cards is None or len(outcomes) < max_cards:
                frame = self.camera.read()
                if frame is None:
                    break
                outcomes.append(self.process_one(frame))
        finally:
            self.flush_csv()
            self.camera.release()
            self.transport.home()

        logger.info("Sorted %d card(s).", len(outcomes))
        return outcomes
