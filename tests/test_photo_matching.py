# tests/test_photo_matching.py
# Unit tests for pure utility functions in PhotoMatching.py.
# Does NOT load TensorFlow models or require a camera.

import csv
import os
import sys
import unittest.mock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub out TensorFlow / Keras before importing PhotoMatching so tests run
# without requiring those heavy packages to be installed.
# ---------------------------------------------------------------------------
_tf_mock = unittest.mock.MagicMock()
for _mod in [
    "tensorflow", "tf",
    "keras", "keras.applications", "keras.applications.mobilenet_v2", "keras.models",
]:
    sys.modules.setdefault(_mod, _tf_mock)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PhotoMatching import (
    GameType,
    _cosine_score,
    _l2_normalize,
    _normalize_existing_rows,
    _split_filename,
    _write_rows_4col,
    ensure_valid_image,
    find_best_matches,
    foil_score,
    get_extractor,
    get_game_type,
    is_probably_foil,
)


# ===========================================================================
# GameType / get_game_type
# ===========================================================================

class TestGetGameType:
    def test_lorcana_folder(self, tmp_path):
        p = tmp_path / "Card_Images" / "Lorcana" / "001-001.webp"
        p.parent.mkdir(parents=True)
        p.touch()
        assert get_game_type(str(p)) == GameType.LORCANA

    def test_riftbound_folder(self, tmp_path):
        p = tmp_path / "Card_Images" / "Riftbound" / "ONG-01.webp"
        p.parent.mkdir(parents=True)
        p.touch()
        assert get_game_type(str(p)) == GameType.RIFTBOUND

    def test_unknown_folder(self, tmp_path):
        p = tmp_path / "SomeOtherGame" / "card.jpg"
        p.parent.mkdir(parents=True)
        p.touch()
        assert get_game_type(str(p)) == GameType.UNKNOWN

    def test_case_insensitive_lorcana(self, tmp_path):
        # Folder names are lowercased before comparison
        p = tmp_path / "lorcana" / "card.jpg"
        p.parent.mkdir(parents=True)
        p.touch()
        assert get_game_type(str(p)) == GameType.LORCANA

    def test_empty_string_returns_unknown(self):
        result = get_game_type("")
        assert result == GameType.UNKNOWN


# ===========================================================================
# _split_filename
# ===========================================================================

class TestSplitFilename:
    def test_numeric_set_card(self):
        assert _split_filename("001-042.webp") == ("001", "042")

    def test_alphanumeric(self):
        assert _split_filename("ONG-23c.jpg") == ("ONG", "23c")

    def test_with_alt_suffix(self):
        # Third hyphen-separated part should stay with card code
        assert _split_filename("ONG-23c-alt.jpg") == ("ONG", "23c-alt")

    def test_with_full_path(self):
        assert _split_filename("/some/path/Card_Images/Lorcana/009-041.webp") == ("009", "041")

    def test_no_hyphen(self):
        set_code, card_code = _split_filename("nosetcode.png")
        assert set_code == ""
        assert card_code == "nosetcode"

    def test_no_extension(self):
        assert _split_filename("005-010") == ("005", "010")

    def test_single_hyphen_only(self):
        # "X-" → card_code is empty string
        set_code, card_code = _split_filename("X-.png")
        assert set_code == "X"
        assert card_code == ""


# ===========================================================================
# _l2_normalize
# ===========================================================================

class TestL2Normalize:
    def test_unit_vector_unchanged(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = _l2_normalize(v)
        np.testing.assert_allclose(result, v, atol=1e-6)

    def test_norm_is_one(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        result = _l2_normalize(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_zero_vector(self):
        v = np.zeros(5, dtype=np.float32)
        result = _l2_normalize(v)
        # Should not raise; result is zero vector
        assert result.shape == (5,)

    def test_negative_values(self):
        v = np.array([-3.0, 4.0], dtype=np.float32)
        result = _l2_normalize(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6


# ===========================================================================
# _cosine_score
# ===========================================================================

class TestCosineScore:
    def _unit(self, *vals):
        v = np.array(vals, dtype=np.float32)
        return v / np.linalg.norm(v)

    def test_identical_vectors(self):
        v = self._unit(1.0, 2.0, 3.0)
        assert abs(_cosine_score(v, v) - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        a = self._unit(1.0, 0.0)
        b = self._unit(0.0, 1.0)
        assert abs(_cosine_score(a, b)) < 1e-5

    def test_opposite_vectors(self):
        v = self._unit(1.0, 0.0)
        assert abs(_cosine_score(v, -v) - (-1.0)) < 1e-5

    def test_raises_if_not_normalized(self):
        a = np.array([3.0, 4.0], dtype=np.float32)  # norm = 5
        b = self._unit(1.0, 0.0)
        with pytest.raises(ValueError):
            _cosine_score(a, b)

    def test_score_in_range(self):
        a = self._unit(1.0, 2.0, 3.0)
        b = self._unit(4.0, 5.0, 6.0)
        score = _cosine_score(a, b)
        assert -1.0 <= score <= 1.0


# ===========================================================================
# find_best_matches
# ===========================================================================

class TestFindBestMatches:
    def _make_db(self):
        """Build a small fake feature DB."""
        a = _l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        b = _l2_normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        c = _l2_normalize(np.array([1.0, 1.0, 0.0], dtype=np.float32))
        return {"card_a.jpg": a, "card_b.jpg": b, "card_c.jpg": c}

    def test_exact_match_first(self):
        db = self._make_db()
        query = db["card_a.jpg"].copy()
        matches = find_best_matches(query, db, threshold=0.5)
        assert matches[0][0] == "card_a.jpg"
        assert abs(matches[0][1] - 1.0) < 1e-5

    def test_threshold_filters(self):
        db = self._make_db()
        query = db["card_a.jpg"].copy()
        # card_b is orthogonal → score ≈ 0, should be filtered
        matches = find_best_matches(query, db, threshold=0.99)
        names = [m[0] for m in matches]
        assert "card_b.jpg" not in names

    def test_sorted_descending(self):
        db = self._make_db()
        query = db["card_c.jpg"].copy()
        matches = find_best_matches(query, db, threshold=0.0)
        scores = [m[1] for m in matches]
        assert scores == sorted(scores, reverse=True)

    def test_empty_db(self):
        q = _l2_normalize(np.ones(3, dtype=np.float32))
        assert find_best_matches(q, {}, threshold=0.5) == []

    def test_none_input(self):
        assert find_best_matches(None, {"x": np.ones(3)}, threshold=0.5) == []


# ===========================================================================
# ensure_valid_image
# ===========================================================================

class TestEnsureValidImage:
    def test_none_returns_none(self):
        assert ensure_valid_image(None) is None

    def test_empty_array_returns_none(self):
        assert ensure_valid_image(np.array([])) is None

    def test_bgr_passthrough(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = ensure_valid_image(img)
        assert result.shape == (10, 10, 3)

    def test_grayscale_converted_to_bgr(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        result = ensure_valid_image(img)
        assert result.shape == (10, 10, 3)

    def test_rgba_stripped_to_bgr(self):
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        result = ensure_valid_image(img)
        assert result.shape == (10, 10, 3)

    def test_weird_channel_count_returns_none(self):
        img = np.zeros((10, 10, 2), dtype=np.uint8)
        assert ensure_valid_image(img) is None


# ===========================================================================
# foil_score / is_probably_foil
# ===========================================================================

class TestFoilScore:
    def test_black_image_is_not_foil(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        assert foil_score(img) < 0.08

    def test_white_image_is_foil(self):
        img = np.full((50, 50, 3), 255, dtype=np.uint8)
        assert foil_score(img) >= 0.08

    def test_score_in_range(self):
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        score = foil_score(img)
        assert 0.0 <= score <= 1.0

    def test_none_returns_zero(self):
        assert foil_score(None) == 0.0

    def test_is_probably_foil_white(self):
        img = np.full((50, 50, 3), 255, dtype=np.uint8)
        assert is_probably_foil(img)

    def test_is_probably_foil_black(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        assert not is_probably_foil(img)


# ===========================================================================
# get_extractor backend dispatch
# ===========================================================================

class TestGetExtractor:
    def test_keras_backend(self):
        ex = get_extractor("keras")
        assert type(ex).__name__ == "_KerasExtractor"
        assert hasattr(ex, "extract")

    def test_tflite_backend(self):
        # Constructing the tflite extractor must NOT load the interpreter,
        # so this works even without a .tflite model present.
        ex = get_extractor("tflite")
        assert type(ex).__name__ == "_TFLiteExtractor"
        assert hasattr(ex, "extract")

    def test_default_is_keras(self):
        assert type(get_extractor()).__name__ == "_KerasExtractor"

    def test_case_insensitive(self):
        assert type(get_extractor("TFLite")).__name__ == "_TFLiteExtractor"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            get_extractor("onnx")

    def test_extractors_are_cached(self):
        assert get_extractor("keras") is get_extractor("keras")


# ===========================================================================
# CSV helpers: _normalize_existing_rows, _write_rows_4col
# ===========================================================================

class TestNormalizeExistingRows:
    def _write_csv(self, path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for r in rows:
                writer.writerow(r)

    def test_missing_file_returns_empty(self, tmp_path):
        result = _normalize_existing_rows(str(tmp_path / "nonexistent.csv"))
        assert result == []

    def test_skips_header_row(self, tmp_path):
        p = tmp_path / "test.csv"
        self._write_csv(p, [["Set Number", "Card Number", "Variant", "Count"], ["001", "001", "normal", "2"]])
        rows = _normalize_existing_rows(str(p))
        assert len(rows) == 1
        assert rows[0] == ["001", "001", "normal", "2"]

    def test_4col_passthrough(self, tmp_path):
        p = tmp_path / "test.csv"
        self._write_csv(p, [["009", "041", "normal", "3"]])
        rows = _normalize_existing_rows(str(p))
        assert rows == [["009", "041", "normal", "3"]]

    def test_3col_adds_zero_count(self, tmp_path):
        p = tmp_path / "test.csv"
        self._write_csv(p, [["009", "041", "normal"]])
        rows = _normalize_existing_rows(str(p))
        assert rows == [["009", "041", "normal", "0"]]

    def test_5col_ignores_fifth_column(self, tmp_path):
        p = tmp_path / "test.csv"
        # 5th column is a legacy "Tag" field and must be ignored; count is col[3]
        self._write_csv(p, [["009", "041", "normal", "2", "Tag"]])
        rows = _normalize_existing_rows(str(p))
        assert rows[0] == ["009", "041", "normal", "2"]

    def test_empty_count_becomes_zero(self, tmp_path):
        p = tmp_path / "test.csv"
        self._write_csv(p, [["009", "041", "normal", ""]])
        rows = _normalize_existing_rows(str(p))
        assert rows[0][3] == "0"

    def test_multiple_rows(self, tmp_path):
        p = tmp_path / "test.csv"
        self._write_csv(p, [
            ["001", "001", "normal", "1"],
            ["001", "002", "foil", "3"],
        ])
        rows = _normalize_existing_rows(str(p))
        assert len(rows) == 2


class TestWriteRows4Col:
    def test_writes_header_and_rows(self, tmp_path):
        p = tmp_path / "out.csv"
        rows = [["001", "001", "normal", "5"], ["002", "010", "foil", "1"]]
        _write_rows_4col(str(p), rows)

        with open(p, newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))

        assert reader[0] == ["Set Number", "Card Number", "Variant", "Count"]
        assert reader[1] == ["001", "001", "normal", "5"]
        assert reader[2] == ["002", "010", "foil", "1"]

    def test_empty_rows_writes_header_only(self, tmp_path):
        p = tmp_path / "out.csv"
        _write_rows_4col(str(p), [])

        with open(p, newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))

        assert len(reader) == 1
        assert reader[0] == ["Set Number", "Card Number", "Variant", "Count"]
