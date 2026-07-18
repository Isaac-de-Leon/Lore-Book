# tests/test_photo_matching.py
# Unit tests for pure utility functions in PhotoMatching.py.
# Does NOT load TensorFlow models or require a camera.

import csv
import os

import numpy as np
import pytest

# TF/Keras stubbing and sys.path setup happen in tests/conftest.py.

from PhotoMatching import (
    GameType,
    clear_collection,
    csv_for_game,
    update_cardlist,
    _cosine_score,
    _l2_normalize,
    _normalize_existing_rows,
    _split_filename,
    _write_rows_4col,
    ensure_valid_image,
    find_best_matches,
    foil_score,
    game_type_from_name,
    get_extractor,
    get_game_type,
    is_probably_foil,
)
from lorebook.core.card_database import _cache_path
from lorebook.core.features import _check_tflite_input_dtype
from lorebook.core.image_utils import CARD_ASPECT, MotionGate, crop_to_card, focus_rect


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
    def test_norm_is_one(self):
        v = np.array([-3.0, 4.0], dtype=np.float32)  # negatives included
        result = _l2_normalize(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_zero_vector(self):
        v = np.zeros(5, dtype=np.float32)
        result = _l2_normalize(v)
        # Should not raise; result is zero vector
        assert result.shape == (5,)

    def test_always_returns_float32(self):
        # float32 is a cache contract: blobs are read back with
        # np.frombuffer(dtype=np.float32), so a float64 result here (sklearn's
        # normalize upcasts) silently corrupts every stored vector into NaN.
        for dtype in (np.float32, np.float64):
            result = _l2_normalize(np.array([3.0, 4.0], dtype=dtype))
            assert result.dtype == np.float32, f"input {dtype} -> {result.dtype}"


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

    def test_bad_db_vector_is_skipped_not_fatal(self):
        db = self._make_db()
        db["corrupt.jpg"] = np.array([5.0, 5.0, 5.0], dtype=np.float32)  # not unit-norm
        query = db["card_a.jpg"].copy()
        matches = find_best_matches(query, db, threshold=0.5)
        names = [m[0] for m in matches]
        assert "corrupt.jpg" not in names
        assert "card_a.jpg" in names


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

class TestFocusRectAndCrop:
    def test_focus_rect_geometry(self):
        fx, fy, fw, fh = focus_rect(720, 1280)
        assert fh == int(720 * 0.6)
        assert fw == int(fh * CARD_ASPECT)
        assert fx == (1280 - fw) // 2
        assert fy == (720 - fh) // 2

    def test_crop_to_card_matches_focus_rect(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        _, _, fw, fh = focus_rect(720, 1280)
        cropped = crop_to_card(frame)
        assert cropped.shape == (fh, fw, 3)

    def test_crop_returns_copy_not_view(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        cropped = crop_to_card(frame)
        cropped[:] = 255
        assert frame.max() == 0

    def test_tiny_frame_returned_unchanged(self):
        tiny = np.zeros((8, 8, 3), np.uint8)
        assert crop_to_card(tiny) is tiny

    def test_none_passthrough(self):
        assert crop_to_card(None) is None


class TestMotionGate:
    # High-variance "card" frame vs uniform "empty mat" frame.
    _card = (np.arange(64, dtype=np.float32).reshape(8, 8) * 3)
    _empty = np.zeros((8, 8), np.float32)

    def _settle(self, gate, frame, n):
        return [gate.update(frame) for _ in range(n)]

    def test_fires_exactly_once_when_card_settles(self):
        gate = MotionGate(steady_frames=3, diff_threshold=5.0)
        self._settle(gate, self._empty, 2)              # prime; steady but unarmed
        assert gate.update(self._card) is False        # motion (card placed) arms
        assert self._settle(gate, self._card, 3) == [False, False, True]
        assert not any(self._settle(gate, self._card, 10))  # sitting card never re-fires

    def test_rearms_for_the_next_card(self):
        gate = MotionGate(steady_frames=2, diff_threshold=5.0)
        gate.update(self._empty)
        gate.update(self._card)                         # place card 1
        assert self._settle(gate, self._card, 2)[-1] is True
        gate.update(self._empty)                        # remove (motion re-arms)
        gate.update(self._card)                         # place card 2
        assert self._settle(gate, self._card, 2)[-1] is True

    def test_min_std_suppresses_empty_scene_trigger(self):
        gate = MotionGate(steady_frames=2, diff_threshold=5.0, min_std=10.0)
        gate.update(self._card)
        gate.update(self._empty)                        # card removed → motion arms
        assert not any(self._settle(gate, self._empty, 6))  # empty mat: no trigger

    def test_constant_motion_never_fires(self):
        gate = MotionGate(steady_frames=2, diff_threshold=5.0)
        frames = [np.full((8, 8), v, np.float32) for v in (0, 50, 100, 150, 200)]
        assert not any(gate.update(f) for f in frames)

    def test_shape_change_and_none_are_safe(self):
        gate = MotionGate(steady_frames=1, diff_threshold=5.0)
        assert gate.update(None) is False
        gate.update(self._empty)
        assert gate.update(np.zeros((4, 4), np.float32)) is False  # re-primes


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

    def test_is_probably_foil_is_thresholded_foil_score(self):
        for img in (np.full((50, 50, 3), 255, np.uint8), np.zeros((50, 50, 3), np.uint8)):
            assert is_probably_foil(img) == (foil_score(img) >= 0.08)


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

    def test_case_insensitive(self):
        assert type(get_extractor("TFLite")).__name__ == "_TFLiteExtractor"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            get_extractor("onnx")

    def test_extractors_are_cached(self):
        assert get_extractor("keras") is get_extractor("keras")


# ===========================================================================
# game_type_from_name / _cache_path / tflite dtype guard
# ===========================================================================

class TestGameTypeFromName:
    def test_known_games(self):
        assert game_type_from_name("Lorcana") == GameType.LORCANA
        assert game_type_from_name("riftbound") == GameType.RIFTBOUND
        assert game_type_from_name("  LORCANA  ") == GameType.LORCANA

    def test_unknown_game(self):
        assert game_type_from_name("Pokemon") == GameType.UNKNOWN
        assert game_type_from_name("") == GameType.UNKNOWN
        assert game_type_from_name(None) == GameType.UNKNOWN


class TestCsvForGame:
    def test_legacy_games_keep_their_files(self):
        assert csv_for_game("Lorcana") == "LorcanaList.csv"
        assert csv_for_game("lorcana") == "LorcanaList.csv"
        assert csv_for_game("RIFTBOUND") == "RiftboundList.csv"

    def test_new_game_derives_from_folder_name(self):
        assert csv_for_game("Pokemon") == "PokemonList.csv"
        assert csv_for_game("MTG") == "MTGList.csv"

    def test_trailing_separators_stripped(self):
        assert csv_for_game("Lorcana/") == "LorcanaList.csv"
        assert csv_for_game("Pokemon\\") == "PokemonList.csv"

    def test_blank_raises(self):
        with pytest.raises(ValueError):
            csv_for_game("")
        with pytest.raises(ValueError):
            csv_for_game(None)


class TestUpdateCardlistGameRouting:
    def test_explicit_game_writes_named_csv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        update_cardlist("001-042.webp", is_foil=False, game="Pokemon")
        rows = list(csv.reader((tmp_path / "PokemonList.csv").open(encoding="utf-8")))
        assert ["001", "042", "normal", "1"] in rows
        assert not os.path.exists(tmp_path / "LorcanaList.csv")

    def test_legacy_path_inference_still_works(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        update_cardlist(os.path.join("Card_Images", "Riftbound", "001-042.webp"), is_foil=True)
        rows = list(csv.reader((tmp_path / "RiftboundList.csv").open(encoding="utf-8")))
        assert ["001", "042", "foil", "1"] in rows


class TestClearCollection:
    def test_clear_leaves_header_only(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        update_cardlist("001-042.webp", is_foil=False, count=3, game="Lorcana")
        update_cardlist("002-007.webp", is_foil=True, count=1, game="Lorcana")
        clear_collection("Lorcana")
        rows = list(csv.reader((tmp_path / "LorcanaList.csv").open(encoding="utf-8")))
        assert rows == [["Set Number", "Card Number", "Variant", "Count"]]

    def test_clear_missing_file_creates_header_only(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        clear_collection("Lorcana")
        rows = list(csv.reader((tmp_path / "LorcanaList.csv").open(encoding="utf-8")))
        assert rows == [["Set Number", "Card Number", "Variant", "Count"]]

    def test_clear_only_touches_named_game(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        update_cardlist("001-042.webp", is_foil=False, game="Lorcana")
        update_cardlist("001-001.webp", is_foil=False, game="Riftbound")
        clear_collection("Riftbound")
        lorcana = list(csv.reader((tmp_path / "LorcanaList.csv").open(encoding="utf-8")))
        assert ["001", "042", "normal", "1"] in lorcana


class TestNegativeCounts:
    def _rows(self, tmp_path):
        return list(csv.reader((tmp_path / "LorcanaList.csv").open(encoding="utf-8")))

    def test_negative_decrements_with_allow_negative(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        update_cardlist("001-042.webp", is_foil=False, count=3, game="Lorcana")
        update_cardlist("001-042.webp", is_foil=False, count=-2, game="Lorcana", allow_negative=True)
        assert ["001", "042", "normal", "1"] in self._rows(tmp_path)

    def test_decrement_clamps_at_zero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        update_cardlist("001-042.webp", is_foil=False, count=1, game="Lorcana")
        update_cardlist("001-042.webp", is_foil=False, count=-5, game="Lorcana", allow_negative=True)
        assert ["001", "042", "normal", "0"] in self._rows(tmp_path)

    def test_negative_skipped_without_allow_negative(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        update_cardlist("001-042.webp", is_foil=False, count=2, game="Lorcana")
        update_cardlist("001-042.webp", is_foil=False, count=-1, game="Lorcana")  # default: ignored
        assert ["001", "042", "normal", "2"] in self._rows(tmp_path)

    def test_decrement_of_absent_card_is_noop(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        update_cardlist("001-001.webp", is_foil=False, count=1, game="Lorcana")
        update_cardlist("009-999.webp", is_foil=False, count=-1, game="Lorcana", allow_negative=True)
        rows = self._rows(tmp_path)
        assert not any(r[0] == "009" for r in rows)  # no phantom row created


class TestCachePath:
    def test_normal_path(self):
        assert _cache_path(os.path.join("Card_Images", "Lorcana")) == "DBCardCache_Lorcana.db"

    def test_trailing_slash_does_not_collapse_to_default(self):
        assert _cache_path("Card_Images/Lorcana/") == "DBCardCache_Lorcana.db"
        assert _cache_path("Card_Images/Lorcana//") == "DBCardCache_Lorcana.db"

    def test_empty_falls_back_to_default(self):
        assert _cache_path("") == "DBCardCache_default.db"


class TestTfliteInputDtypeGuard:
    def test_float32_accepted(self):
        _check_tflite_input_dtype(np.float32, "model.tflite")  # no raise

    def test_quantized_dtypes_rejected(self):
        for dtype in (np.int8, np.uint8):
            with pytest.raises(ValueError, match="float32"):
                _check_tflite_input_dtype(dtype, "model.tflite")


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


class TestReadCollectionRows:
    def test_reads_games_csv_normalized(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with open("LorcanaList.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Set Number", "Card Number", "Variant", "Count"])
            writer.writerow(["009", "041", "normal", "2"])
            writer.writerow(["001", "007", "foil"])  # legacy 3-col row
        from lorebook.core.csv_manager import read_collection_rows
        assert read_collection_rows("Lorcana") == [
            ["009", "041", "normal", "2"],
            ["001", "007", "foil", "0"],
        ]

    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from lorebook.core.csv_manager import read_collection_rows
        assert read_collection_rows("Riftbound") == []


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

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
    def test_preserves_existing_file_mode(self, tmp_path):
        p = tmp_path / "out.csv"
        _write_rows_4col(str(p), [["001", "001", "normal", "1"]])
        os.chmod(p, 0o640)
        _write_rows_4col(str(p), [["001", "001", "normal", "2"]])
        assert (os.stat(p).st_mode & 0o777) == 0o640

    def test_leaves_no_temp_files_behind(self, tmp_path):
        p = tmp_path / "out.csv"
        _write_rows_4col(str(p), [["001", "001", "normal", "1"]])
        assert sorted(os.listdir(tmp_path)) == ["out.csv"]

    def test_failed_write_keeps_existing_file_and_cleans_temp(self, tmp_path):
        p = tmp_path / "out.csv"
        _write_rows_4col(str(p), [["001", "001", "normal", "1"]])

        # A non-iterable row blows up mid-write; the original must survive.
        with pytest.raises(TypeError):
            _write_rows_4col(str(p), [None])

        with open(p, newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))
        assert reader[1] == ["001", "001", "normal", "1"]
        assert sorted(os.listdir(tmp_path)) == ["out.csv"]
