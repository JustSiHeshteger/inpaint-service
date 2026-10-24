"""Закрашивание рамок реплик: чистые функции без зависимостей от HTTP.

Модуль держит всю логику контракта `/erase`: разбор рамок, декодирование
страницы, маску и собственно закрашивание. Тесты вызывают функции напрямую,
поэтому API-слой остаётся тонкой обвязкой статусов.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from inpaint_service.config import DEFAULT_MAX_BOXES, Settings

#: Метки OpenCV для двух поддерживаемых методов закрашивания.
_METHODS = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}


@dataclass(frozen=True)
class Box:
    """Рамка реплики в пикселях исходной страницы."""

    x: int
    y: int
    w: int
    h: int


class InpaintError(ValueError):
    """Ошибка задания: код уходит в тело ответа и в логи воркера."""

    code = "INPAINT_FAILED"


class InvalidBoxes(InpaintError):
    """Поле `boxes` не разбирается как массив рамок."""

    code = "INVALID_BOXES"


class InvalidImage(InpaintError):
    """Тело `image` не читается как картинка."""

    code = "INVALID_IMAGE"


class ImageTooLarge(InpaintError):
    """Страница больше лимита `INPAINT_MAX_PIXELS`."""

    code = "IMAGE_TOO_LARGE"


class EncodeFailed(InpaintError):
    """Страницу не удалось закодировать в PNG (картинка при этом валидна)."""

    code = "ENCODE_FAILED"


def probe_size(data: bytes) -> tuple[int, int] | None:
    """Размер страницы из заголовка контейнера: `(ширина, высота)` или `None`.

    Нужен, чтобы отклонить «PNG-бомбу» ДО декодирования: файл на десятки
    килобайт разворачивается в гигабайты памяти, и проверка после `imdecode`
    уже бесполезна — под падает по OOM. Поэтому читаем только заголовок
    (PNG IHDR, JPEG SOFn, WebP VP8X/VP8/VP8L) и возвращаем `None`, если формат
    незнаком: тогда решает декодер и обычная проверка площади.
    """
    return _png_size(data) or _jpeg_size(data) or _webp_size(data)


def _png_size(data: bytes) -> tuple[int, int] | None:
    """IHDR идёт сразу после сигнатуры: ширина и высота — big-endian uint32."""
    if len(data) < 24 or not data.startswith(b"\x89PNG\r\n\x1a\n") or data[12:16] != b"IHDR":
        return None
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return (width, height) if width and height else None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    """Идём по маркерам до SOF0..SOF15 (кроме DHT/JPG/DAC), где лежат размеры."""
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        # маркеры без длины сегмента
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        length = int.from_bytes(data[index + 2 : index + 4], "big")
        is_sof = 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC)
        if is_sof:
            height = int.from_bytes(data[index + 5 : index + 7], "big")
            width = int.from_bytes(data[index + 7 : index + 9], "big")
            return (width, height) if width and height else None
        if length < 2:
            return None
        index += 2 + length
    return None


def _webp_size(data: bytes) -> tuple[int, int] | None:
    """`VP8X` хранит размеры минус единицу в 24 битах, `VP8`/`VP8L` — иначе."""
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    kind = data[12:16]
    if kind == b"VP8X":
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return (width, height)
    if kind == b"VP8 ":
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return (width, height)
    if kind == b"VP8L":
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    return None


def parse_boxes(raw: str, *, max_boxes: int = DEFAULT_MAX_BOXES) -> list[Box]:
    """Разобрать JSON-массив рамок; вырожденные рамки отбрасываются.

    Вырожденная рамка (`w`/`h` ≤ 0) — не ошибка задания, а пустая работа: её
    нечего закрашивать, и она не должна валить страницу целиком.
    """
    try:
        data: Any = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise InvalidBoxes(f"boxes не JSON: {exc}") from exc
    if not isinstance(data, list):
        raise InvalidBoxes("boxes обязан быть JSON-массивом рамок")
    if len(data) > max_boxes:
        raise InvalidBoxes(f"слишком много рамок: {len(data)} > {max_boxes}")
    boxes: list[Box] = []
    for item in data:
        if not isinstance(item, dict):
            raise InvalidBoxes(f"рамка обязана быть объектом, получено {type(item).__name__}")
        try:
            values = [item[key] for key in ("x", "y", "w", "h")]
        except KeyError as exc:
            raise InvalidBoxes(f"рамка обязана иметь целые x/y/w/h: нет поля {exc}") from exc
        # bool — подкласс int, а «10.9» молча усеклось бы до 10: принимаем
        # только настоящие целые, иначе координаты вёрстки разъезжаются молча
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise InvalidBoxes(f"координаты рамки обязаны быть целыми: {values}")
        box = Box(*values)
        if box.w > 0 and box.h > 0:
            boxes.append(box)
    return boxes


def decode_image(data: bytes) -> np.ndarray:
    """Декодировать картинку в BGR-матрицу OpenCV."""
    if not data:
        raise InvalidImage("пустое тело image")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImage("image не читается как картинка (PNG/WEBP/JPEG)")
    return image


def build_mask(shape: tuple[int, ...], boxes: list[Box], *, dilate: int = 0) -> np.ndarray:
    """Маска закрашивания 0/255 по рамкам, обрезанным по границе страницы.

    `dilate` расширяет маску: обводка букв и «хвосты» антиалиасинга выходят за
    рамку блока, и без расширения они остались бы висеть поверх новой вёрстки.
    """
    height, width = shape[0], shape[1]
    mask = np.zeros((height, width), np.uint8)
    for box in boxes:
        left, top = max(0, box.x), max(0, box.y)
        right, bottom = min(width, box.x + box.w), min(height, box.y + box.h)
        if right <= left or bottom <= top:
            continue
        mask[top:bottom, left:right] = 255
    if dilate > 0:
        kernel = np.ones((dilate * 2 + 1, dilate * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, kernel)
    return mask


def erase(image: bytes, boxes: list[Box], settings: Settings) -> bytes:
    """Закрасить рамки и вернуть PNG того же размера.

    Размер не меняется никогда: воркер сверяет его и считает расхождение
    сбоем задания, а вёрстка перевода рассчитывается в координатах исходной
    страницы — сдвиг на пиксель сломал бы всю страницу.
    """
    # лимит проверяется по заголовку, а не по результату декодирования:
    # «PNG-бомба» успевает съесть память раньше, чем матрица станет доступна
    size = probe_size(image)
    if size is not None and size[0] * size[1] > settings.max_pixels:
        raise ImageTooLarge(
            f"страница {size[0]}×{size[1]} больше лимита {settings.max_pixels} px"
        )

    page = decode_image(image)
    height, width = page.shape[:2]
    if width * height > settings.max_pixels:
        raise ImageTooLarge(f"страница {width}×{height} больше лимита {settings.max_pixels} px")

    mask = build_mask(page.shape, boxes, dilate=settings.dilate)
    result = page
    if mask.any():
        result = cv2.inpaint(page, mask, float(settings.radius), _METHODS[settings.method])

    ok, buffer = cv2.imencode(".png", result)
    if not ok:
        # картинка была валидной: это отказ сервиса, а не ошибка задания
        raise EncodeFailed("не удалось закодировать страницу в PNG")
    return buffer.tobytes()


#: Результат самопроверки: считается один раз, дальше отдаётся из памяти.
_PROBE_RESULT: tuple[bool, str | None] | None = None


def readiness() -> tuple[bool, str | None]:
    """Готовность сервиса: `(ok, причина)` для пробы `/ready`.

    Проба не спрашивает «импортировался ли OpenCV» — при сломанной сборке
    импорт уронил бы сам процесс. Она делает настоящий мини-прогон: страница
    8×8 с одной рамкой. Это ровно тот путь, которым пойдёт живое задание, но
    стоит микросекунды, поэтому результат кэшируется, а не повторяется на
    каждый опрос kubelet.
    """
    global _PROBE_RESULT
    if _PROBE_RESULT is None:
        _PROBE_RESULT = _self_check()
    return _PROBE_RESULT


def _self_check() -> tuple[bool, str | None]:
    """Мини-прогон закрашивания: ловит битый OpenCV до первого задания."""
    page = np.full((8, 8, 3), 255, np.uint8)
    mask = np.zeros((8, 8), np.uint8)
    mask[2:6, 2:6] = 255
    try:
        cv2.inpaint(page, mask, 1.0, cv2.INPAINT_TELEA)
        ok, _ = cv2.imencode(".png", page)
    except Exception as exc:  # noqa: BLE001 - любая причина значит «не готов»
        return False, f"opencv не может inpaint: {exc}"
    if not ok:
        return False, "opencv не может отдать PNG"
    return True, None
