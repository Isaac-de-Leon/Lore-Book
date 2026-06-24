# tests/test_sorter_rules.py
# Unit tests for the multi-bin sort-decision engine. Pure logic — no hardware,
# no camera, no TensorFlow.

import json

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


class TestDecideBin:
    def test_first_match_wins(self):
        rules = SortRules(
            rules=[
                Rule(bin="a", set_code="001"),
                Rule(bin="b", set_code="001"),  # also matches but is shadowed
            ],
            reject_bin="reject",
        )
        assert _decide(rules, set_code="001") == "a"

    def test_multi_bin_routing_by_set(self):
        rules = SortRules(
            rules=[
                Rule(bin="bin-1", set_code="001"),
                Rule(bin="bin-8", set_code="008"),
                Rule(bin="bin-9", set_code="009"),
            ]
        )
        assert _decide(rules, set_code="001") == "bin-1"
        assert _decide(rules, set_code="008") == "bin-8"
        assert _decide(rules, set_code="009") == "bin-9"

    def test_foil_rule(self):
        rules = SortRules(rules=[Rule(bin="foils", foil=True)], reject_bin="normal")
        assert _decide(rules, is_foil=True) == "foils"
        assert _decide(rules, is_foil=False) == "normal"

    def test_min_confidence_gating(self):
        rules = SortRules(rules=[Rule(bin="matched", min_confidence=0.8)], reject_bin="reject")
        assert _decide(rules, confidence=0.85) == "matched"
        assert _decide(rules, confidence=0.79) == "reject"

    def test_game_condition_case_insensitive(self):
        rules = SortRules(rules=[Rule(bin="lor", game="lorcana")], reject_bin="reject")
        assert _decide(rules, game="Lorcana") == "lor"
        assert _decide(rules, game="Riftbound") == "reject"

    def test_combined_conditions_all_must_match(self):
        rules = SortRules(rules=[Rule(bin="hit", set_code="009", foil=True)], reject_bin="reject")
        assert _decide(rules, set_code="009", is_foil=True) == "hit"
        assert _decide(rules, set_code="009", is_foil=False) == "reject"
        assert _decide(rules, set_code="001", is_foil=True) == "reject"

    def test_empty_rules_falls_through_to_reject(self):
        rules = SortRules(rules=[], reject_bin="reject")
        assert _decide(rules) == "reject"

    def test_default_reject_bin(self):
        assert SortRules().reject_bin == "reject"


class TestLoadRules:
    def test_load_from_json(self, tmp_path):
        cfg = {
            "reject_bin": "trash",
            "rules": [
                {"foil": True, "bin": "foils"},
                {"set_code": "001", "min_confidence": 0.7, "bin": "bin-1"},
            ],
        }
        p = tmp_path / "rules.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")

        rules = load_rules(str(p))
        assert rules.reject_bin == "trash"
        assert len(rules.rules) == 2
        assert rules.rules[0].foil is True and rules.rules[0].bin == "foils"
        assert rules.rules[1].set_code == "001" and rules.rules[1].min_confidence == 0.7

    def test_missing_bin_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"rules": [{"set_code": "001"}]}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_rules(str(p))

    def test_example_config_loads(self):
        import os

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, "configs", "sort_rules.example.json")
        rules = load_rules(path)
        assert rules.reject_bin == "reject"
        assert len(rules.rules) >= 1
