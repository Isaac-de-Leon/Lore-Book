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
    is_foil_only_card,
    is_probably_foil,
)
from lorebook.core.card_database import _cache_path
from lorebook.core.csv_manager import read_collection_rows
from lorebook.core.features import _check_tflite_input_dtype
from lorebook.core.image_utils import CARD_ASPECT, MotionGate, crop_to_card, focus_rect

HEADER = ["Set Number", "Card Number", "Variant", "Count"]


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def test_game_type_and_file_naming(tmp_path):
    for folder, expected in (
        (tmp_path / "Card_Images" / "Lorcana", GameType.LORCANA),
        (tmp_path / "Card_Images" / "Riftbound", GameType.RIFTBOUND),
        (tmp_path / "SomeOtherGame", GameType.UNKNOWN),
    ):
        folder.mkdir(parents=True)
        (folder / "001-001.webp").touch()
        assert get_game_type(str(folder / "001-001.webp")) == expected
    assert get_game_type("") == GameType.UNKNOWN

    assert game_type_from_name("  LORCANA  ") == GameType.LORCANA
    assert game_type_from_name("riftbound") == GameType.RIFTBOUND
    for name in ("Pokemon", "", None):
        assert game_type_from_name(name) == GameType.UNKNOWN

    # csv_for_game: legacy files kept, new games derived, separators stripped
    assert csv_for_game("lorcana") == "LorcanaList.csv"
    assert csv_for_game("RIFTBOUND") == "RiftboundList.csv"
    assert csv_for_game("Pokemon\\") == "PokemonList.csv"
    assert csv_for_game("Lorcana/") == "LorcanaList.csv"
    for blank in ("", None):
        with pytest.raises(ValueError):
            csv_for_game(blank)

    # _cache_path: trailing slashes must not collapse to the default cache
    assert _cache_path(os.path.join("Card_Images", "Lorcana")) == "DBCardCache_Lorcana.db"
    assert _cache_path("Card_Images/Lorcana//") == "DBCardCache_Lorcana.db"
    assert _cache_path("") == "DBCardCache_default.db"


def test_split_filename():
    assert _split_filename("001-042.webp") == ("001", "042")
    assert _split_filename("ONG-23c-alt.jpg") == ("ONG", "23c-alt")  # suffix stays with card
    assert _split_filename("/some/path/Card_Images/Lorcana/009-041.webp") == ("009", "041")
    assert _split_filename("005-010") == ("005", "010")
    assert _split_filename("nosetcode.png") == ("", "nosetcode")
    assert _split_filename("X-.png") == ("X", "")


def test_vector_math():
    v = _l2_normalize(np.array([-3.0, 4.0], dtype=np.float32))
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6
    assert _l2_normalize(np.zeros(5, np.float32)).shape == (5,)  # zero vector: no raise
    # float32 is a cache contract: blobs are read back with np.frombuffer(float32),
    # so a float64 result here silently corrupts every stored vector into NaN.
    for dtype in (np.float32, np.float64):
        assert _l2_normalize(np.array([3.0, 4.0], dtype=dtype)).dtype == np.float32

    a = np.array([1.0, 0.0], np.float32)
    b = np.array([0.0, 1.0], np.float32)
    assert abs(_cosine_score(a, a) - 1.0) < 1e-5
    assert abs(_cosine_score(a, b)) < 1e-5
    assert abs(_cosine_score(a, -a) + 1.0) < 1e-5
    with pytest.raises(ValueError):
        _cosine_score(np.array([3.0, 4.0], np.float32), a)  # not normalized


def test_find_best_matches():
    db = {
        "card_a.jpg": _l2_normalize(np.array([1.0, 0.0, 0.0], np.float32)),
        "card_b.jpg": _l2_normalize(np.array([0.0, 1.0, 0.0], np.float32)),
        "card_c.jpg": _l2_normalize(np.array([1.0, 1.0, 0.0], np.float32)),
        "corrupt.jpg": np.array([5.0, 5.0, 5.0], np.float32),  # not unit-norm: skipped
    }
    matches = find_best_matches(db["card_a.jpg"].copy(), db, threshold=0.5)
    assert matches[0][0] == "card_a.jpg" and abs(matches[0][1] - 1.0) < 1e-5
    names = [m[0] for m in matches]
    assert "card_b.jpg" not in names and "corrupt.jpg" not in names

    scores = [s for _, s in find_best_matches(db["card_c.jpg"], db, threshold=0.0)]
    assert scores == sorted(scores, reverse=True)

    assert find_best_matches(db["card_a.jpg"], {}, threshold=0.5) == []
    assert find_best_matches(None, {"x": np.ones(3)}, threshold=0.5) == []


def test_image_helpers():
    assert ensure_valid_image(None) is None
    assert ensure_valid_image(np.array([])) is None
    assert ensure_valid_image(np.zeros((10, 10, 2), np.uint8)) is None
    for shape in ((10, 10, 3), (10, 10), (10, 10, 4)):  # BGR, gray, BGRA → BGR
        assert ensure_valid_image(np.zeros(shape, np.uint8)).shape == (10, 10, 3)

    fx, fy, fw, fh = focus_rect(720, 1280)
    assert (fh, fw) == (int(720 * 0.6), int(int(720 * 0.6) * CARD_ASPECT))
    assert (fx, fy) == ((1280 - fw) // 2, (720 - fh) // 2)
    frame = np.zeros((720, 1280, 3), np.uint8)
    cropped = crop_to_card(frame)
    assert cropped.shape == (fh, fw, 3)
    cropped[:] = 255
    assert frame.max() == 0  # a copy, not a view
    tiny = np.zeros((8, 8, 3), np.uint8)
    assert crop_to_card(tiny) is tiny
    assert crop_to_card(None) is None

    black = np.zeros((50, 50, 3), np.uint8)
    white = np.full((50, 50, 3), 255, np.uint8)
    assert foil_score(black) < 0.08 <= foil_score(white)
    assert 0.0 <= foil_score(np.random.randint(0, 255, (50, 50, 3), np.uint8)) <= 1.0
    assert foil_score(None) == 0.0
    for img in (black, white):
        assert is_probably_foil(img) == (foil_score(img) >= 0.08)


def test_motion_gate():
    card = np.arange(64, dtype=np.float32).reshape(8, 8) * 3  # high variance
    empty = np.zeros((8, 8), np.float32)

    gate = MotionGate(steady_frames=3, diff_threshold=5.0)
    gate.update(empty), gate.update(empty)            # steady but unarmed
    assert gate.update(card) is False                 # motion (card placed) arms
    assert [gate.update(card) for _ in range(3)] == [False, False, True]
    assert not any(gate.update(card) for _ in range(10))  # sitting card never re-fires
    gate.update(empty)                                # removed: motion re-arms
    gate.update(card)                                 # next card
    assert [gate.update(card) for _ in range(3)][-1] is True

    gate = MotionGate(steady_frames=2, diff_threshold=5.0, min_std=10.0)
    gate.update(card), gate.update(empty)             # removal arms...
    assert not any(gate.update(empty) for _ in range(6))  # ...but empty mat never fires

    gate = MotionGate(steady_frames=2, diff_threshold=5.0)
    assert not any(gate.update(np.full((8, 8), v, np.float32)) for v in (0, 50, 100, 150))

    gate = MotionGate(steady_frames=1, diff_threshold=5.0)
    assert gate.update(None) is False
    gate.update(empty)
    assert gate.update(np.zeros((4, 4), np.float32)) is False  # shape change re-primes


def test_get_extractor():
    # Constructing the tflite extractor must NOT load the interpreter.
    assert type(get_extractor("keras")).__name__ == "_KerasExtractor"
    assert type(get_extractor("TFLite")).__name__ == "_TFLiteExtractor"
    assert get_extractor("keras") is get_extractor("keras")
    with pytest.raises(ValueError):
        get_extractor("onnx")

    _check_tflite_input_dtype(np.float32, "model.tflite")  # no raise
    for dtype in (np.int8, np.uint8):
        with pytest.raises(ValueError, match="float32"):
            _check_tflite_input_dtype(dtype, "model.tflite")


def test_update_cardlist_routing_and_foil_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    update_cardlist("001-042.webp", is_foil=False, game="Pokemon")
    assert ["001", "042", "normal", "1"] in _rows("PokemonList.csv")
    update_cardlist(os.path.join("Card_Images", "Riftbound", "001-042.webp"), is_foil=True)
    assert ["001", "042", "foil", "1"] in _rows("RiftboundList.csv")
    assert not os.path.exists("LorcanaList.csv")

    # Lorcana cards above 204 only exist as foil — Dreamborn rejects "normal" rows.
    assert is_foil_only_card("013", "205", GameType.LORCANA)
    assert not is_foil_only_card("013", "204", GameType.LORCANA)
    assert not is_foil_only_card("ONG", "23c-alt", GameType.LORCANA)
    assert not is_foil_only_card("001", "218", GameType.RIFTBOUND)

    update_cardlist("013-218.webp", is_foil=False, count=2, game="Lorcana")
    update_cardlist(os.path.join("Card_Images", "Lorcana", "013-219.webp"), is_foil=False)
    update_cardlist("013-218.webp", is_foil=False, count=-1, game="Lorcana", allow_negative=True)
    rows = _rows("LorcanaList.csv")
    assert ["013", "218", "foil", "1"] in rows   # undo hit the coerced foil row
    assert ["013", "219", "foil", "1"] in rows   # coercion applies to inferred game too
    assert not any(r[2] == "normal" for r in rows[1:])


def test_negative_counts_and_clear(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def add(name, n, **kw):
        update_cardlist(name, is_foil=False, count=n, game="Lorcana", **kw)

    add("001-042.webp", 3)
    add("001-042.webp", -2, allow_negative=True)
    add("002-001.webp", 1)
    add("002-001.webp", -5, allow_negative=True)   # clamps at zero
    add("003-001.webp", 2)
    add("003-001.webp", -1)                        # ignored without allow_negative
    add("009-999.webp", -1, allow_negative=True)   # absent card: no phantom row
    rows = _rows("LorcanaList.csv")
    assert ["001", "042", "normal", "1"] in rows
    assert ["002", "001", "normal", "0"] in rows
    assert ["003", "001", "normal", "2"] in rows
    assert not any(r[0] == "009" for r in rows)

    update_cardlist("001-001.webp", is_foil=False, game="Riftbound")
    clear_collection("Riftbound")
    assert _rows("RiftboundList.csv") == [HEADER]
    assert ["001", "042", "normal", "1"] in _rows("LorcanaList.csv")  # other game untouched
    clear_collection("Lorcana")
    assert _rows("LorcanaList.csv") == [HEADER]
    os.remove("LorcanaList.csv")
    clear_collection("Lorcana")                    # missing file → header only
    assert _rows("LorcanaList.csv") == [HEADER]


def test_csv_reading_normalizes_legacy_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _normalize_existing_rows("nonexistent.csv") == []
    assert read_collection_rows("Riftbound") == []

    with open("LorcanaList.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([
            HEADER,
            ["009", "041", "normal", "3"],
            ["001", "007", "foil"],                  # legacy 3-col → count 0
            ["009", "042", "normal", "2", "Tag"],    # legacy 5th column ignored
            ["009", "043", "normal", ""],            # empty count → 0
        ])
    expected = [
        ["009", "041", "normal", "3"],
        ["001", "007", "foil", "0"],
        ["009", "042", "normal", "2"],
        ["009", "043", "normal", "0"],
    ]
    assert _normalize_existing_rows("LorcanaList.csv") == expected
    assert read_collection_rows("Lorcana") == expected


def test_write_rows_4col_is_atomic(tmp_path):
    p = tmp_path / "out.csv"
    _write_rows_4col(str(p), [])
    assert _rows(p) == [HEADER]

    rows = [["001", "001", "normal", "5"], ["002", "010", "foil", "1"]]
    _write_rows_4col(str(p), rows)
    assert _rows(p) == [HEADER] + rows
    assert sorted(os.listdir(tmp_path)) == ["out.csv"]  # no temp files left behind

    if os.name != "nt":  # POSIX file modes survive the atomic replace
        os.chmod(p, 0o640)
        _write_rows_4col(str(p), rows)
        assert (os.stat(p).st_mode & 0o777) == 0o640

    # A non-iterable row blows up mid-write; the original must survive.
    with pytest.raises(TypeError):
        _write_rows_4col(str(p), [None])
    assert _rows(p) == [HEADER] + rows
    assert sorted(os.listdir(tmp_path)) == ["out.csv"]
