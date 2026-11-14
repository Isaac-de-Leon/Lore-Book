"""Headless capture→match→decide→route→CSV loop.

Ties the existing matching stack to the hardware abstraction and the rules
engine. Every collaborator (camera, transport, extractor) is injected, so the
whole loop runs against desktop mocks with no camera, motors, or TensorFlow.
In dry_run mode (the default) it logs the bin decision and never writes the
CSV; with dry_run=False it records the card via the existing csv_manager,
batching writes so the CSV is rewritten every csv_flush_interval cards
(and at end of run) instead of once per card.
"""

import logging
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from lorebook.core.card_names import name_for
from lorebook.core.card_prices import format_price, price_for
from lorebook.core.csv_manager import update_cardlist_batch
from lorebook.core.game_types import game_type_from_name, is_foil_only_card
from lorebook.core.games import split_filename
from lorebook.core.image_utils import MotionGate, crop_to_card, is_probably_foil, motion_sample
from lorebook.core.matching import MatchIndex
from lorebook.hardware.camera import CameraSource
from lorebook.hardware.transport import Transport
from lorebook.sorter.rules import SortRules, decide_bin

logger = logging.getLogger(__name__)


@dataclass
class SortOutcome:
    """The result of processing a single card."""

    filename: str | None      # matched reference image name, or None if no match
    set_code: str
    card_code: str
    confidence: float
    is_foil: bool
    bin: str
    matched: bool


@dataclass(frozen=True)
class SortConfig:
    """What a sort run should do with each card.

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

    rules: SortRules
    game: str
    threshold: float = 0.70
    foil_threshold: float | None = None
    dry_run: bool = True
    csv_flush_interval: int = 25
    crop_to_focus: bool = False


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
        config:      SortConfig — rules, game, thresholds, CSV and crop behavior.
        gate:        optional MotionGate for live cameras. Without it every
                     frame is a new card (right for folder replays, where each
                     image is one card). With it, a frame is only processed
                     once a card has been placed and settled, so a card left
                     under the camera is recorded once — not once per frame.
    """

    def __init__(
        self,
        *,
        camera: CameraSource,
        transport: Transport,
        extractor,
        feature_db: dict[str, np.ndarray],
        config: SortConfig,
        gate: MotionGate | None = None,
    ):
        """Wire the collaborators together and index the reference vectors."""
        self.camera = camera
        self.transport = transport
        self.extractor = extractor
        self.feature_db = feature_db
        self.config = config
        self.gate = gate
        self._flush_every = max(1, config.csv_flush_interval)
        self._pending: Counter = Counter()  # (filename, is_foil) -> count
        self._index = MatchIndex(feature_db)  # one matmul per card instead of a dict scan

    # Cap for the foil-detection pass: specular/contrast stats survive
    # downscaling, and the full-res Laplacian is a needless per-card cost on
    # a Pi. Note the Laplacian score shifts slightly with resolution, so
    # --foil-threshold may need retuning versus the GUI's full-res 0.08.
    _FOIL_MAX_HEIGHT = 360

    def _is_foil(self, frame: np.ndarray) -> bool:
        """Return True when the (downscaled) frame looks like a foil card."""
        h = frame.shape[0] if frame is not None and frame.ndim >= 2 else 0
        if h > self._FOIL_MAX_HEIGHT:
            scale = self._FOIL_MAX_HEIGHT / h
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if self.config.foil_threshold is None:
            return is_probably_foil(frame)
        return is_probably_foil(frame, threshold=self.config.foil_threshold)

    def flush_csv(self) -> None:
        """Write any accumulated matched cards to the CSV in one read+write.

        On failure the pending cards are kept (and the CSV left untouched),
        so a later flush can retry.
        """
        if not self._pending:
            return
        update_cardlist_batch(
            [(fname, foil, count) for (fname, foil), count in self._pending.items()],
            game=self.config.game,
        )
        self._pending.clear()

    def _identify(self, frame: np.ndarray) -> SortOutcome:
        """Match a frame and decide its bin; unmatched cards go to the reject bin."""
        config = self.config
        is_foil = self._is_foil(frame)
        features = self.extractor.extract(frame)
        matches = self._index.find(features, threshold=config.threshold) if features is not None else []
        if not matches:
            return SortOutcome(filename=None, set_code="", card_code="", confidence=0.0,
                               is_foil=is_foil, bin=config.rules.reject_bin, matched=False)

        filename, confidence = matches[0]
        set_code, card_code = split_filename(filename)
        # Foil-only rarities (e.g. Lorcana Enchanted) have no normal
        # printing: route them as foil, matching what the CSV records.
        is_foil = is_foil or is_foil_only_card(set_code, card_code, game_type_from_name(config.game))
        bin_id = decide_bin(
            set_code=set_code,
            card_code=card_code,
            game=config.game,
            is_foil=is_foil,
            confidence=confidence,
            rules=config.rules,
        )
        return SortOutcome(filename=filename, set_code=set_code, card_code=card_code,
                           confidence=confidence, is_foil=is_foil, bin=bin_id, matched=True)

    def _route(self, bin_id: str) -> None:
        """Send the card to its bin and get ready for the next one."""
        self.transport.route_to_bin(bin_id)
        self.transport.advance()
        # Frames buffered while the card was moving show the previous card;
        # drop them so the next read sees the newly placed one.
        self.camera.flush()

    def _record(self, outcome: SortOutcome) -> None:
        """Queue a matched card for the CSV, flushing every csv_flush_interval cards."""
        if not outcome.matched or self.config.dry_run:
            return
        self._pending[(outcome.filename, outcome.is_foil)] += 1
        if sum(self._pending.values()) >= self._flush_every:
            self.flush_csv()

    def _log(self, outcome: SortOutcome) -> None:
        """Log one line per card: what it matched, how confidently, and its bin."""
        game = self.config.game
        card_name = name_for(outcome.set_code, outcome.card_code, game) if outcome.matched else None
        price = format_price(price_for(outcome.set_code, outcome.card_code, game)) if outcome.matched else ""
        logger.info(
            "card=%s%s%s conf=%.3f foil=%s → bin=%s%s",
            outcome.filename or "<no match>",
            f" ({card_name})" if card_name else "",
            f" [{price}]" if price else "",
            outcome.confidence,
            outcome.is_foil,
            outcome.bin,
            " (dry-run)" if self.config.dry_run else "",
        )

    def process_one(self, frame: np.ndarray) -> SortOutcome:
        """Match a single frame, decide its bin, route it, and (optionally) queue it for CSV."""
        if self.config.crop_to_focus:
            frame = crop_to_card(frame)
        outcome = self._identify(frame)
        self._route(outcome.bin)
        self._record(outcome)
        self._log(outcome)
        return outcome

    def run(self, max_cards: int | None = None) -> list[SortOutcome]:
        """
        Process cards until the camera feed is exhausted or max_cards is hit.
        With a gate, only frames where a new card has settled are processed.
        Flushes pending CSV rows, releases the camera, and homes the transport
        on completion.
        """
        outcomes: list[SortOutcome] = []
        try:
            while max_cards is None or len(outcomes) < max_cards:
                frame = self.camera.read()
                if frame is None:
                    break
                # Live feeds: wait for a freshly settled card. Skipped frames
                # don't count toward max_cards.
                if self.gate is not None and not self.gate.update(motion_sample(frame)):
                    continue
                outcomes.append(self.process_one(frame))
        finally:
            # Release/home even if the final CSV write fails (e.g. the file
            # is locked); the write error still propagates afterwards.
            try:
                self.flush_csv()
            finally:
                self.camera.release()
                self.transport.home()

        logger.info("Sorted %d card(s).", len(outcomes))
        return outcomes
