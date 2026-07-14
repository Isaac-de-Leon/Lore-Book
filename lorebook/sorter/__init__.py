# lorebook.sorter — headless capture→match→decide→route pipeline.
#
# Pure-software half of the card sorter: it reuses the existing matching stack
# (lorebook.core) and drives a hardware abstraction (lorebook.hardware) that is
# mocked until the physical mechanism exists. The bin-decision logic is a
# standalone, fully testable rules engine.

from lorebook.sorter.pipeline import SortOutcome, SortPipeline
from lorebook.sorter.rules import Rule, SortRules, decide_bin, load_rules

__all__ = [
    "Rule",
    "SortRules",
    "load_rules",
    "decide_bin",
    "SortPipeline",
    "SortOutcome",
]
