"""HTTP-контракт `/erase`, `/health`, `/ready` по inpaint-service.yaml."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from inpaint_service import eraser
from inpaint_service.app import create_app

from .conftest import BLOCK, decode, encode_png, page_array
from .test_eraser import settings

BOXES = json.dumps([{"x": BLOCK[0], "y": BLOCK[1], "w": BLOCK[2], "h": BLOCK[3]}])


def client(**overrides: object) -> TestClient:
    """Клиент приложения; настройки — явные, без чтения реального env."""
    return TestClient(create_app(settings(**overrides)))


def test_health_reports_ok() -> None:
    response = client().get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "reason": None}


def test_ready_is_ok_by_default() -> None:
    response = client().get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_reports_degraded_with_reason() -> None:
    app = create_app(settings(), probe=lambda: (False, "opencv не импортируется"))

    response = TestClient(app).get("/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "reason": "opencv не импортируется"}


def test_erase_returns_png_of_same_size(page_png: bytes, page: object) -> None:
    response = client().post(
        "/erase",
        files={"image": ("page", page_png, "application/octet-stream")},
        data={"boxes": BOXES},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert decode(response.content).shape == page.shape


def test_erase_fills_the_block_with_background(page_png: bytes) -> None:
    response = client().post(
        "/erase",
        files={"image": ("page", page_png, "application/octet-stream")},
        data={"boxes": BOXES},
    )

    x, y, w, h = BLOCK
    inner = decode(response.content)[y + 20 : y + h - 20, x + 20 : x + w - 20]
    assert float(inner.mean()) > 180.0


def test_erase_without_boxes_keeps_page(page_png: bytes) -> None:
    response = client().post(
        "/erase",
        files={"image": ("page", page_png, "application/octet-stream")},
        data={"boxes": "[]"},
    )

    assert response.status_code == 200


def test_erase_rejects_non_image_body() -> None:
    response = client().post(
        "/erase",
        files={"image": ("page", b"broken", "application/octet-stream")},
        data={"boxes": BOXES},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_IMAGE"


def test_erase_rejects_broken_boxes(page_png: bytes) -> None:
    response = client().post(
        "/erase",
        files={"image": ("page", page_png, "application/octet-stream")},
        data={"boxes": "not json"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_BOXES"


def test_erase_rejects_oversized_page(big_page_png: bytes) -> None:
    response = client(max_pixels=1_000_000).post(
        "/erase",
        files={"image": ("page", big_page_png, "application/octet-stream")},
        data={"boxes": BOXES},
    )

    assert response.status_code == 413
    assert response.json()["code"] == "IMAGE_TOO_LARGE"


def test_erase_requires_both_form_fields(page_png: bytes) -> None:
    response = client().post(
        "/erase", files={"image": ("page", page_png, "application/octet-stream")}
    )

    assert response.status_code == 422, "без boxes задание не определено"


def test_erase_requires_image_field() -> None:
    response = client().post("/erase", data={"boxes": "[]"})

    assert response.status_code == 422


@pytest.mark.parametrize("missing", ["image", "boxes"])
def test_missing_form_field_is_a_contract_error(missing: str, page_png: bytes) -> None:
    """Пропущенное поле — ошибка контракта, а не «сырой» ответ FastAPI.

    Клиент генерируется по inpaint-service.yaml: там 422 объявлен как
    `InpaintError`, поэтому `{"detail": [...]}` сломал бы его разбор.
    """
    files = {"image": ("page", page_png, "application/octet-stream")}
    data = {"boxes": BOXES}
    if missing == "image":
        files = {}
    else:
        data = {}

    response = client().post("/erase", files=files or None, data=data or None)

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "INVALID_REQUEST"
    assert "message" in body


def test_body_over_the_limit_is_rejected_before_decoding() -> None:
    """Лимит тела: страницу больше `max_bytes` не читаем в память целиком."""
    noisy = page_array(1600, 1200).astype(np.int16)
    noisy += np.random.default_rng(1).integers(0, 60, noisy.shape, dtype=np.int16)
    oversized = encode_png(np.clip(noisy, 0, 255).astype(np.uint8))
    assert len(oversized) > 1_048_576

    response = client(max_bytes=1_048_576).post(
        "/erase",
        files={"image": ("page", oversized, "application/octet-stream")},
        data={"boxes": BOXES},
    )

    assert response.status_code == 413
    assert response.json()["code"] == "IMAGE_TOO_LARGE"


def test_read_limited_stops_at_the_limit() -> None:
    """Чтение по чанкам обрывается на лимите, а не после полного тела."""
    from inpaint_service.app import _read_limited

    class _Upload:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload
            self.reads = 0

        async def read(self, size: int) -> bytes:
            self.reads += 1
            return self._payload[:size]

    upload = _Upload(b"x" * 10)

    with pytest.raises(eraser.ImageTooLarge):
        asyncio.run(_read_limited(upload, 4))  # type: ignore[arg-type]
    assert upload.reads <= 2, "чтение обязано прерваться, а не дочитать файл"


def test_page_above_the_spooling_limit_is_accepted() -> None:
    """Страница тяжелее 1 МиБ проходит: спулинг мультипарта — не ошибка.

    Starlette пишет части больше 1 МиБ во временный файл; в контейнере с
    read-only rootfs это работает только при writable `/tmp` (см. манифест).
    """
    noisy = page_array(1200, 900).astype(np.int16)
    noisy += np.random.default_rng(0).integers(0, 40, noisy.shape, dtype=np.int16)
    big = encode_png(np.clip(noisy, 0, 255).astype(np.uint8))
    assert len(big) > 1_048_576, "фикстура обязана превышать порог спулинга"

    response = client().post(
        "/erase",
        files={"image": ("page", big, "application/octet-stream")},
        data={"boxes": BOXES},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_encode_failure_is_a_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Сбой PNG-кодирования — не `INVALID_IMAGE`: картинка была валидной."""
    from inpaint_service import eraser

    def fail(*_: object, **__: object) -> bytes:
        raise eraser.EncodeFailed("не удалось закодировать страницу в PNG")

    monkeypatch.setattr(eraser, "erase", fail)
    response = client().post(
        "/erase",
        files={"image": ("page", b"whatever", "application/octet-stream")},
        data={"boxes": "[]"},
    )

    assert response.status_code == 500
    assert response.json()["code"] == "ENCODE_FAILED"


def test_unexpected_failure_is_reported_as_inpaint_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Неожидаемое падение обязано быть машинно-читаемым, а не трейсбеком."""
    from inpaint_service import eraser

    def boom(*_: object, **__: object) -> bytes:
        raise RuntimeError("OpenCV упал")

    monkeypatch.setattr(eraser, "erase", boom)
    response = client().post(
        "/erase",
        files={"image": ("page", b"whatever", "application/octet-stream")},
        data={"boxes": "[]"},
    )

    assert response.status_code == 500
    assert response.json()["code"] == "INPAINT_FAILED"


def test_error_body_matches_contract() -> None:
    response = client().post(
        "/erase",
        files={"image": ("page", b"broken", "application/octet-stream")},
        data={"boxes": BOXES},
    )

    body = response.json()
    assert set(body) == {"code", "message"}
    assert isinstance(body["message"], str) and body["message"]


def test_app_uses_the_settings_it_was_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INPAINT_RADIUS", "9")

    app = create_app(settings(radius=5))

    assert app.state.settings.radius == 5, "приложение не перечитывает env само"
