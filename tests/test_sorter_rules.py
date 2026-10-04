# tests/test_sorter_rules.py
# Unit tests for the multi-bin sort-decision engine. Pure logic — no hardware,
# no camera, no TensorFlow.

import json
import os

import pytest

from lorebook.sorter.rules import Rule, SortRules, decide_bin, load_rules


def _decide(rules, *, set_code="001", card_code="042", game="Lorcana", is_foil=False, confidence=0.9):
    return decide_bin(
        set_code=set_code,
        card_code=card_code,
        game=game,
        is_foil=is_foil,
        confidence=confidence,
        rules=rules,
    )


def test_decide_bin():
    assert SortRules().reject_bin == "reject"
    assert _decide(SortRules(rules=[], reject_bin="trash")) == "trash"

    rules = SortRules(rules=[
        Rule(bin="first", set_code="001"),
        Rule(bin="shadowed", set_code="001"),       # also matches: first match wins
        Rule(bin="bin-8", set_code="008"),
        Rule(bin="hit", set_code="009", foil=True),  # all conditions must match
        Rule(bin="lor", game="lorcana", min_confidence=0.8),
    ])
    assert _decide(rules, set_code="001") == "first"
    assert _decide(rules, set_code="008") == "bin-8"
    assert _decide(rules, set_code="009", is_foil=True) == "hit"
    assert _decide(rules, set_code="009", is_foil=False) == "lor"   # game is case-insensitive
    assert _decide(rules, set_code="005", confidence=0.79) == "reject"
    assert _decide(rules, set_code="005", game="Riftbound") == "reject"

    foil_rules = SortRules(rules=[Rule(bin="foils", foil=True)], reject_bin="normal")
    assert _decide(foil_rules, is_foil=True) == "foils"
    assert _decide(foil_rules, is_foil=False) == "normal"


def test_load_rules(tmp_path):
    p = tmp_path / "rules.json"
    p.write_text(json.dumps({
        "reject_bin": "trash",
        "rules": [
            {"foil": True, "bin": "foils"},
            {"set_code": "001", "min_confidence": 0.7, "bin": "bin-1"},
        ],
    }), encoding="utf-8")
    rules = load_rules(str(p))
    assert rules.reject_bin == "trash"
    assert [(r.bin, r.foil, r.set_code, r.min_confidence) for r in rules.rules] == [
        ("foils", True, None, None),
        ("bin-1", None, "001", 0.7),
    ]

    # Typo'd or wrong-typed fields must fail loudly, not silently match everything.
    for cfg, field in [
        ({"rules": [{"set_code": "001"}]}, "bin"),
        ({"rules": [{"bin": "b", "min_confidnce": 0.8}]}, "min_confidnce"),
        ({"rules": [{"bin": "b", "foil": "yes"}]}, "foil"),
        ({"rules": [{"bin": "b", "min_confidence": "high"}]}, "min_confidence"),
        ({"rules": [{"bin": "b", "set_code": 1}]}, "set_code"),
        ({"reject_bin": "", "rules": []}, "reject_bin"),
    ]:
        p.write_text(json.dumps(cfg), encoding="utf-8")
        with pytest.raises(ValueError, match=field):
            load_rules(str(p))

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example = load_rules(os.path.join(repo_root, "configs", "sort_rules.example.json"))
    assert example.reject_bin == "reject" and len(example.rules) >= 1
