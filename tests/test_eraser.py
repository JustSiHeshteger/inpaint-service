"""Логика затирания: разбор рамок, маска, результат и проба готовности."""

from __future__ import annotations

import json
import struct

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from inpaint_service.config import Settings
from inpaint_service.eraser import (
    Box,
    EncodeFailed,
    ImageTooLarge,
    InvalidBoxes,
    InvalidImage,
    build_mask,
    decode_image,
    erase,
    parse_boxes,
    probe_size,
    readiness,
)

from .conftest import BACKGROUND, BLOCK, decode, encode_png, page_array

BOX = Box(x=BLOCK[0], y=BLOCK[1], w=BLOCK[2], h=BLOCK[3])


def settings(**overrides: object) -> Settings:
    """Настройки по умолчанию: дефолты прода, без чтения реального env."""
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


# ── разбор рамок ─────────────────────────────────────────────


def test_parse_boxes_keeps_order_and_coordinates() -> None:
    raw = json.dumps([{"x": 40, "y": 60, "w": 200, "h": 80}, {"x": 0, "y": 0, "w": 10, "h": 10}])

    assert parse_boxes(raw) == [Box(40, 60, 200, 80), Box(0, 0, 10, 10)]


def test_parse_boxes_drops_degenerate_boxes() -> None:
    raw = json.dumps([{"x": 1, "y": 2, "w": 0, "h": 30}, {"x": 1, "y": 2, "w": -5, "h": 30}])

    assert parse_boxes(raw) == [], "вырожденные рамки нечего затирать"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json",
        "{}",
        '[{"x": 1, "y": 2, "w": 3}]',
        '["box"]',
        '[{"x": "a", "y": 2, "w": 3, "h": 4}]',
    ],
)
def test_parse_boxes_rejects_broken_payload(raw: str) -> None:
    with pytest.raises(InvalidBoxes):
        parse_boxes(raw)


def test_parse_boxes_rejects_too_many_boxes() -> None:
    raw = json.dumps([{"x": 0, "y": 0, "w": 5, "h": 5}] * 3)

    with pytest.raises(InvalidBoxes, match="рамок"):
        parse_boxes(raw, max_boxes=2)


# ── декодирование и маска ────────────────────────────────────


def test_decode_image_returns_bgr_matrix(page: np.ndarray) -> None:
    decoded = decode_image(encode_png(page))

    assert decoded.shape == page.shape


def test_decode_image_rejects_garbage() -> None:
    with pytest.raises(InvalidImage):
        decode_image(b"not an image")


def test_build_mask_covers_box_and_nothing_else(page: np.ndarray) -> None:
    mask = build_mask(page.shape, [BOX])

    assert mask.shape == page.shape[:2]
    assert mask[BLOCK[1] + 5, BLOCK[0] + 5] == 255
    assert mask[0, 0] == 0, "вне рамки маска пустая"


def test_build_mask_dilate_grows_coverage(page: np.ndarray) -> None:
    plain = build_mask(page.shape, [BOX])
    grown = build_mask(page.shape, [BOX], dilate=4)

    assert int(grown.sum()) > int(plain.sum())
    assert grown[BLOCK[1] - 2, BLOCK[0] + 5] == 255, "обводка букв вне рамки тоже закрашивается"


def test_build_mask_clips_boxes_outside_page(page: np.ndarray) -> None:
    mask = build_mask(page.shape, [Box(-50, -50, 100, 100), Box(380, 280, 100, 100)])

    assert mask[0, 0] == 255, "часть рамки внутри страницы закрашивается"
    assert mask[-1, -1] == 255
    assert mask[10, 100] == 0


# ── затирание ────────────────────────────────────────────────


def test_erase_removes_text_inside_box(page_png: bytes) -> None:
    result = decode(erase(page_png, [BOX], settings()))

    x, y, w, h = BLOCK
    inner = result[y + 20 : y + h - 20, x + 20 : x + w - 20]
    assert float(inner.mean()) > 180.0, "тёмный текст закрашен цветом фона"
    assert np.abs(inner.astype(int) - BACKGROUND).max() < 60


def test_erase_keeps_pixels_outside_boxes(page_png: bytes, page: np.ndarray) -> None:
    result = decode(erase(page_png, [BOX], settings()))

    # полоса вне рамки: сравнение вне маски (±допуск на перепаковку PNG)
    assert np.abs(result[250:280, 20:60].astype(int) - page[250:280, 20:60]).max() <= 4


def test_erase_preserves_size(page_png: bytes, page: np.ndarray) -> None:
    result = decode(erase(page_png, [BOX], settings()))

    assert result.shape == page.shape


def test_erase_without_boxes_returns_same_page(page_png: bytes, page: np.ndarray) -> None:
    result = decode(erase(page_png, [], settings()))

    assert np.abs(result.astype(int) - page.astype(int)).max() <= 2


def test_erase_works_with_navier_stokes(page_png: bytes) -> None:
    result = decode(erase(page_png, [BOX], settings(method="ns")))

    x, y, w, h = BLOCK
    assert float(result[y + 20 : y + h - 20, x + 20 : x + w - 20].mean()) > 180.0


def test_erase_rejects_oversized_page(big_page_png: bytes) -> None:
    with pytest.raises(ImageTooLarge):
        erase(big_page_png, [BOX], settings(max_pixels=1_000_000))


def test_erase_returns_png(page_png: bytes) -> None:
    result = erase(page_png, [BOX], settings())

    assert result[:8] == b"\x89PNG\r\n\x1a\n", "контракт обещает image/png"


def test_erase_rejects_broken_image() -> None:
    with pytest.raises(InvalidImage):
        erase(b"broken", [BOX], settings())


# ── готовность и настройки ──────────────────────────────────


def test_readiness_is_ok_on_default_settings() -> None:
    assert readiness() == (True, None)


def test_readiness_reports_missing_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    import inpaint_service.eraser as module

    monkeypatch.setattr(module, "cv2", None)

    ok, reason = readiness()
    assert ok is False
    assert reason and "opencv" in reason.lower()


@pytest.mark.parametrize(
    "overrides",
    [{"radius": 0}, {"radius": 26}, {"dilate": -1}, {"dilate": 33}, {"method": "bilinear"}],
)
def test_settings_reject_out_of_range(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        settings(**overrides)


def test_settings_reads_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INPAINT_METHOD", "ns")
    monkeypatch.setenv("INPAINT_RADIUS", "7")

    cfg = Settings(_env_file=None)  # type: ignore[call-arg]

    assert (cfg.method, cfg.radius) == ("ns", 7)


def test_page_fixture_is_light_outside_block() -> None:
    page = page_array()
    assert page[250, 30].tolist() == [BACKGROUND] * 3

# ── лимиты до декодирования и коды ошибок ───────────────────


def _png_header(width: int, height: int) -> bytes:
    """PNG-заголовок с заданными размерами: тело не нужно, размер читается из IHDR."""
    import zlib

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    chunk = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr
    chunk += struct.pack(">I", zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk


def test_probe_size_reads_png_header() -> None:
    assert probe_size(_png_header(7000, 5700)) == (7000, 5700)


def test_probe_size_reads_jpeg_header() -> None:
    # SOI + SOF0 (baseline) с высотой 5700 и шириной 7000
    jpeg = b"\xff\xd8" + b"\xff\xc0" + struct.pack(">H", 17) + b"\x08\x16\x44\x1b\x58"
    jpeg += b"\x03" + b"\x01\x11\x00" * 3 + b"\xff\xd9"

    assert probe_size(jpeg) == (7000, 5700)


def test_probe_size_reads_webp_vp8x_header() -> None:
    header = b"RIFF" + struct.pack("<I", 30) + b"WEBPVP8X" + struct.pack("<I", 10)
    header += b"\x00\x00\x00\x00" + (6999).to_bytes(3, "little") + (5699).to_bytes(3, "little")

    assert probe_size(header) == (7000, 5700)


def test_probe_size_returns_none_for_unknown_format() -> None:
    assert probe_size(b"not an image") is None


def test_erase_rejects_huge_page_before_decoding() -> None:
    """PNG-бомба: десятки КБ заголовка разворачиваются в гигабайты памяти.

    Поэтому лимит проверяется по заголовку, до `imdecode`: иначе под падает по
    OOM раньше, чем сработает 413.
    """
    bomb = _png_header(30_000, 30_000)

    with pytest.raises(ImageTooLarge):
        erase(bomb, [BOX], settings())


def test_probe_size_failure_does_not_swallow_broken_image() -> None:
    """Неизвестный контейнер: размер неизвестен, решает декодер."""
    with pytest.raises(InvalidImage):
        erase(b"broken", [BOX], settings())


def test_erase_reports_encode_failure_separately(
    monkeypatch: pytest.MonkeyPatch, page_png: bytes
) -> None:
    import inpaint_service.eraser as module

    monkeypatch.setattr(module.cv2, "imencode", lambda *_: (False, None))

    with pytest.raises(EncodeFailed):
        erase(page_png, [BOX], settings())


def test_readiness_reports_broken_inpaint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Битая сборка OpenCV ловится пробой, а не первым живым запросом."""

    def boom(*_: object, **__: object) -> None:
        raise cv2.error("inpaint недоступен")

    monkeypatch.setattr(cv2, "inpaint", boom)

    ok, reason = readiness()
    assert ok is False
    assert reason and "inpaint" in reason.lower()


def test_settings_expose_workers_and_body_limit() -> None:
    cfg = settings()

    assert cfg.workers >= 1, "параллелизм ограничен: каждая страница держит матрицы"
    assert cfg.max_bytes >= 1_048_576, "страница реального комикса больше мегабайта"