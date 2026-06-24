# pipeline.py — headless capture→match→decide→route→CSV loop.
#
# Ties the existing matching stack to the hardware abstraction and the rules
# engine. Every collaborator (camera, transport, extractor) is injected, so the
# whole loop runs against desktop mocks with no camera, motors, or TensorFlow.
# In dry_run mode (the default) it logs the bin decision and never writes the
# CSV; with dry_run=False it records the card via the existing update_cardlist.

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from lorebook.core.csv_manager import _split_filename, update_cardlist
from lorebook.core.image_utils import is_probably_foil
from lorebook.core.matching import find_best_matches
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

    Args:
        camera:      frame source (real or mock).
        transport:   bin router (mock until the gantry exists).
        extractor:   object with ``.extract(img) -> Optional[np.ndarray]``
                     (from ``get_extractor(backend)``).
        feature_db:  {filename: vector} reference cache (from ``load_cache``).
        rules:       SortRules controlling bin assignment.
        game:        active game name (e.g. "Lorcana"); used for rule matching
                     and to build the CSV path so update_cardlist picks the
                     right file.
        threshold:   minimum cosine match score.
        foil_threshold: optional override for foil detection.
        dry_run:     when True (default), never write the CSV.
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

    def _is_foil(self, frame: np.ndarray) -> bool:
        if self.foil_threshold is None:
            return is_probably_foil(frame)
        return is_probably_foil(frame, threshold=self.foil_threshold)

    def process_one(self, frame: np.ndarray) -> SortOutcome:
        """Match a single frame, decide its bin, route it, and (optionally) log to CSV."""
        is_foil = self._is_foil(frame)

        feat = self.extractor.extract(frame)
        matches = find_best_matches(feat, self.feature_db, threshold=self.threshold) if feat is not None else []

        if matches:
            filename, confidence = matches[0]
            set_code, card_code = _split_filename(filename)
            matched = True
        else:
            filename, confidence = None, 0.0
            set_code, card_code = "", ""
            matched = False

        bin_id = decide_bin(
            set_code=set_code,
            card_code=card_code,
            game=self.game,
            is_foil=is_foil,
            confidence=confidence,
            rules=self.rules,
        )

        self.transport.route_to_bin(bin_id)
        self.transport.advance()

        if matched and not self.dry_run:
            # Build a game-scoped path so update_cardlist writes the right CSV.
            csv_path = os.path.join("Card_Images", self.game, filename)
            update_cardlist(csv_path, is_foil)

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
        Releases the camera and homes the transport on completion.
        """
        outcomes: List[SortOutcome] = []
        try:
            while max_cards is None or len(outcomes) < max_cards:
                frame = self.camera.read()
                if frame is None:
                    break
                outcomes.append(self.process_one(frame))
        finally:
            self.camera.release()
            self.transport.home()

        logger.info("Sorted %d card(s).", len(outcomes))
        return outcomes
