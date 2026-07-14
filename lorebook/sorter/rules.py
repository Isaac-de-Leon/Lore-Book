# rules.py — configurable multi-bin sort-decision engine (pure, no hardware).
#
# A SortRules config is an ordered list of Rule conditions plus a fallthrough
# reject_bin. decide_bin() returns the bin for the first rule whose every
# *present* condition matches the card; if none match, the reject_bin. This is
# multi-bin by construction — a two-way keep/reject sorter or a foil/normal
# split are just smaller rule sets of the same engine.

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """
    One routing rule. Every field that is not None is a condition that must
    match; fields left None are ignored. ``bin`` is the destination if all
    present conditions match.

    Conditions:
        game           — game name, case-insensitive (e.g. "Lorcana").
        set_code       — exact set code from the matched filename (e.g. "009").
        foil           — True/False to require a foil / non-foil card.
        min_confidence — minimum cosine match score (inclusive).
    """

    bin: str
    game: Optional[str] = None
    set_code: Optional[str] = None
    foil: Optional[bool] = None
    min_confidence: Optional[float] = None

    def matches(self, *, game: str, set_code: str, is_foil: bool, confidence: float) -> bool:
        if self.game is not None and self.game.lower() != (game or "").lower():
            return False
        if self.set_code is not None and self.set_code != set_code:
            return False
        if self.foil is not None and self.foil != is_foil:
            return False
        if self.min_confidence is not None and confidence < self.min_confidence:
            return False
        return True


@dataclass
class SortRules:
    """An ordered rule set with a fallthrough bin for anything unmatched."""

    rules: List[Rule] = field(default_factory=list)
    reject_bin: str = "reject"


_RULE_FIELDS = {"bin", "game", "set_code", "foil", "min_confidence"}


def _rule_from_dict(d: dict) -> Rule:
    if "bin" not in d:
        raise ValueError(f"rule is missing required 'bin' field: {d!r}")

    # A typo'd condition key ("min_confidnce") would otherwise be silently
    # dropped, leaving a rule that matches far more than intended.
    unknown = set(d) - _RULE_FIELDS
    if unknown:
        raise ValueError(
            f"rule has unknown field(s) {sorted(unknown)} (valid: {sorted(_RULE_FIELDS)}): {d!r}"
        )

    foil = d.get("foil")
    if foil is not None and not isinstance(foil, bool):
        raise ValueError(f"rule 'foil' must be true/false, got {foil!r}: {d!r}")

    min_confidence = d.get("min_confidence")
    if min_confidence is not None and not isinstance(min_confidence, (int, float)):
        raise ValueError(f"rule 'min_confidence' must be a number, got {min_confidence!r}: {d!r}")

    for key in ("game", "set_code"):
        value = d.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"rule {key!r} must be a string, got {value!r}: {d!r}")

    return Rule(
        bin=str(d["bin"]),
        game=d.get("game"),
        set_code=d.get("set_code"),
        foil=foil,
        min_confidence=min_confidence,
    )


def load_rules(path: str) -> SortRules:
    """
    Load a SortRules config from JSON.

    Expected shape:
        {
          "reject_bin": "reject",
          "rules": [
            {"set_code": "009", "bin": "bin-1"},
            {"foil": true, "bin": "foils"},
            ...
          ]
        }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rules = [_rule_from_dict(r) for r in data.get("rules", [])]
    reject_bin = data.get("reject_bin", "reject")
    if not isinstance(reject_bin, str) or not reject_bin.strip():
        raise ValueError(f"'reject_bin' must be a non-empty string, got {reject_bin!r}")
    return SortRules(rules=rules, reject_bin=reject_bin)


def decide_bin(
    *,
    set_code: str,
    card_code: str,
    game: str,
    is_foil: bool,
    confidence: float,
    rules: SortRules,
) -> str:
    """
    Return the destination bin for a card. First matching rule wins; if no rule
    matches, the configured reject_bin is returned. ``card_code`` is accepted
    for future per-card rules and logging symmetry even though no built-in
    condition uses it yet.
    """
    for rule in rules.rules:
        if rule.matches(game=game, set_code=set_code, is_foil=is_foil, confidence=confidence):
            return rule.bin
    return rules.reject_bin
