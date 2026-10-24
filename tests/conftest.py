"""Общие фикстуры: страница-картинка и рамка с «текстом».

Сервис не знает, где текст: ему дают рамки. Поэтому «текст» в фикстурах —
тёмный прямоугольник с текстурой, который затирание обязано убрать, а фон
страницы — светлое поле, которое обязано остаться нетронутым.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

BACKGROUND = 245
TEXT_INK = 30
#: Рамка блока в тестовых страницах: (x, y, w, h).
BLOCK = (40, 60, 200, 80)


def encode_png(image: np.ndarray) -> bytes:
    """Матрица BGR → байты PNG (формат контракта)."""
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def decode(data: bytes) -> np.ndarray:
    """Байты картинки → матрица BGR (для проверок в тестах)."""
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return image


def page_array(width: int = 400, height: int = 300) -> np.ndarray:
    """Светлая страница с «текстом» внутри `BLOCK` и рамкой панели."""
    image = np.full((height, width, 3), BACKGROUND, np.uint8)
    # граница панели: затирание не должно её сдвигать
    cv2.rectangle(image, (5, 5), (width - 6, height - 6), (200, 200, 200), 2)
    x, y, w, h = BLOCK
    cv2.rectangle(image, (x + 8, y + 12), (x + w - 8, y + h - 12), (TEXT_INK,) * 3, -1)
    return image


@pytest.fixture
def page_png() -> bytes:
    """Готовая страница в PNG — тело запроса `/erase`."""
    return encode_png(page_array())


@pytest.fixture
def page() -> np.ndarray:
    """Страница матрицей — для проверок пикселей после затирания."""
    return page_array()


@pytest.fixture
def big_page_png() -> bytes:
    """Страница больше 1 Мп — для проверки лимита `INPAINT_MAX_PIXELS`."""
    return encode_png(page_array(width=1100, height=1000))
@pytest.fixture(autouse=True)
def _fresh_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Сбросить кэш самопроверки `/ready`: иначе первый успешный прогон
    залипает и тесты неготовности проверяют кэш, а не логику."""
    import inpaint_service.eraser as module

    monkeypatch.setattr(module, "_PROBE_RESULT", None)