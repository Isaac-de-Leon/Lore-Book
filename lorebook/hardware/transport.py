# transport.py — card-transport / motion abstraction for the sorter.
#
# The physical sorter (a Pi-driven gantry that picks each card and drops it
# into one of N bins) does not exist yet, so this module only defines the
# interface plus a MockTransport that logs what it *would* do. When the real
# mechanism is built, a concrete Transport (driving steppers/servos directly
# or via a microcontroller over serial — see docs/SORTER_ROADMAP.md) satisfies
# the same three methods and drops straight into the pipeline.

import logging
from abc import ABC, abstractmethod
from collections import Counter
from typing import List

logger = logging.getLogger(__name__)


class Transport(ABC):
    """Moves a singulated card into a target bin."""

    @abstractmethod
    def route_to_bin(self, bin_id: str) -> None:
        """Deliver the current card to ``bin_id``."""

    @abstractmethod
    def advance(self) -> None:
        """Singulate/feed the next card into the capture position."""

    def home(self) -> None:
        """Return the mechanism to its home/rest position."""


class MockTransport(Transport):
    """
    No-op transport that records routing decisions instead of moving anything.

    Lets the full capture→match→decide loop run end-to-end on a desktop (or on
    the Pi before the gantry is built). Inspect ``routed`` for a per-bin tally
    and ``history`` for the ordered list of bins after a run.
    """

    def __init__(self):
        self.routed: Counter = Counter()
        self.history: List[str] = []
        self.cards_advanced = 0
        self.homed = False

    def route_to_bin(self, bin_id: str) -> None:
        self.routed[bin_id] += 1
        self.history.append(bin_id)
        logger.info("would route card → bin '%s'", bin_id)

    def advance(self) -> None:
        self.cards_advanced += 1
        logger.debug("would advance to next card (#%d)", self.cards_advanced)

    def home(self) -> None:
        self.homed = True
        logger.debug("would home transport")
