# tests/test_matching_index.py
# MatchIndex must agree with the reference find_best_matches implementation.

import numpy as np

from lorebook.core.matching import MatchIndex, _l2_normalize, find_best_matches


def _random_db(n=50, dim=64, seed=7):
    rng = np.random.default_rng(seed)
    return {
        f"{i:03d}-{i:03d}.webp": _l2_normalize(rng.normal(size=dim).astype(np.float32))
        for i in range(n)
    }


class TestMatchIndex:
    def test_matches_reference_implementation(self):
        db = _random_db()
        rng = np.random.default_rng(11)
        for _ in range(5):
            q = _l2_normalize(rng.normal(size=64).astype(np.float32))
            expected = find_best_matches(q, db, threshold=0.1)
            got = MatchIndex(db).find(q, threshold=0.1)
            assert [n for n, _ in got] == [n for n, _ in expected]
            assert np.allclose([s for _, s in got], [s for _, s in expected], atol=1e-5)

    def test_threshold_filters(self):
        db = _random_db(n=20)
        q = next(iter(db.values()))
        hits = MatchIndex(db).find(q, threshold=0.999)
        assert len(hits) == 1  # only the identical vector survives

    def test_sorted_descending(self):
        db = _random_db(n=30)
        q = _l2_normalize(np.ones(64, np.float32))
        scores = [s for _, s in MatchIndex(db).find(q, threshold=-1.0)]
        assert scores == sorted(scores, reverse=True)

    def test_empty_db(self):
        assert MatchIndex({}).find(np.ones(64, np.float32)) == []
        assert len(MatchIndex({})) == 0

    def test_none_and_invalid_vectors_skipped(self):
        db = _random_db(n=5)
        db["none.webp"] = None
        db["empty.webp"] = np.array([], np.float32)
        db["wrongdim.webp"] = _l2_normalize(np.ones(32, np.float32))
        idx = MatchIndex(db)
        assert len(idx) == 5

    def test_unnormalized_vectors_are_normalized(self):
        base = _l2_normalize(np.ones(64, np.float32))
        db = {"unit.webp": base, "scaled.webp": base * 10.0}
        hits = MatchIndex(db).find(base, threshold=0.99)
        # The scaled copy must score ~1.0, not ~10.0 or below-threshold.
        assert {n for n, _ in hits} == {"unit.webp", "scaled.webp"}
        assert all(abs(s - 1.0) < 1e-5 for _, s in hits)

    def test_zero_norm_vector_skipped(self):
        db = _random_db(n=4)
        db["zero.webp"] = np.zeros(64, np.float32)
        assert len(MatchIndex(db)) == 4

    def test_none_query_returns_empty(self):
        idx = MatchIndex(_random_db(n=3))
        assert idx.find(None) == []

    def test_wrong_dim_query_returns_empty(self):
        idx = MatchIndex(_random_db(n=3, dim=64))
        assert idx.find(np.ones(128, np.float32)) == []
