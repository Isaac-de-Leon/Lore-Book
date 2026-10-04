# tests/test_matching_index.py
# MatchIndex must agree with the reference find_best_matches implementation.

import numpy as np

from lorebook.core.matching import MatchIndex, l2_normalize, find_best_matches


def _random_db(n=50, dim=64, seed=7):
    rng = np.random.default_rng(seed)
    return {
        f"{i:03d}-{i:03d}.webp": l2_normalize(rng.normal(size=dim).astype(np.float32))
        for i in range(n)
    }


def test_matches_reference_implementation():
    # Threshold filtering and descending sort are covered by parity.
    db = _random_db()
    rng = np.random.default_rng(11)
    for _ in range(5):
        q = l2_normalize(rng.normal(size=64).astype(np.float32))
        expected = find_best_matches(q, db, threshold=0.1)
        got = MatchIndex(db).find(q, threshold=0.1)
        assert [n for n, _ in got] == [n for n, _ in expected]
        assert np.allclose([s for _, s in got], [s for _, s in expected], atol=1e-5)


def test_invalid_vectors_and_queries():
    assert MatchIndex({}).find(np.ones(64, np.float32)) == []
    assert len(MatchIndex({})) == 0

    db = _random_db(n=5)
    db["none.webp"] = None
    db["empty.webp"] = np.array([], np.float32)
    db["wrongdim.webp"] = l2_normalize(np.ones(32, np.float32))
    db["zero.webp"] = np.zeros(64, np.float32)
    idx = MatchIndex(db)
    assert len(idx) == 5
    assert idx.find(None) == []
    assert idx.find(np.ones(128, np.float32)) == []

    # Unnormalized vectors are normalized: the scaled copy scores ~1.0, not ~10.
    base = l2_normalize(np.ones(64, np.float32))
    hits = MatchIndex({"unit.webp": base, "scaled.webp": base * 10.0}).find(base, threshold=0.99)
    assert {n for n, _ in hits} == {"unit.webp", "scaled.webp"}
    assert all(abs(s - 1.0) < 1e-5 for _, s in hits)
