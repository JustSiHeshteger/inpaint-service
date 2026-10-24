"""Настройки сервиса: переменные окружения с префиксом `INPAINT_`."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Метод закрашивания OpenCV: `telea` (быстрый) или `ns` (Навье-Стокс).
InpaintMethod = Literal["telea", "ns"]

#: Лимит пикселей страницы по умолчанию: затирание идёт за O(пиксели), и
#: гигантский апскейл занял бы воркер на минуты.
DEFAULT_MAX_PIXELS = 40_000_000

#: Лимит числа рамок в задании по умолчанию: один источник правды для настроек
#: и для функции разбора (она вызывается и напрямую, и из HTTP-слоя).
DEFAULT_MAX_BOXES = 512


class Settings(BaseSettings):
    """Настройки процесса: всё через `INPAINT_*`, дефолты — для compose/k8s."""

    model_config = SettingsConfigDict(env_prefix="INPAINT_", extra="ignore")

    host: str = "0.0.0.0"
    port: int = Field(8094, ge=1, le=65535)
    log_level: str = "INFO"
    #: Метод закрашивания: свойство развёртывания, а не задания.
    method: InpaintMethod = "telea"
    #: Радиус окрестности закрашивания (пиксели).
    radius: int = Field(3, ge=1, le=25)
    #: Расширение маски: закрывает обводку букв, оставшуюся за рамкой.
    dilate: int = Field(2, ge=0, le=32)
    #: Лимит площади страницы: больше — 413, а не минуты CPU.
    max_pixels: int = Field(DEFAULT_MAX_PIXELS, ge=1_000_000)
    #: Лимит числа рамок в одном запросе: защита от «страницы из 100k боксов».
    max_boxes: int = Field(DEFAULT_MAX_BOXES, ge=1)
    #: Лимит размера загружаемой страницы: больше — 413, не читая тело целиком.
    max_bytes: int = Field(32_000_000, ge=1_048_576)
    #: Сколько страниц затирать одновременно: каждая держит матрицы страницы и маски.
    workers: int = Field(2, ge=1, le=8)
