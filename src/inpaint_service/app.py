"""HTTP-слой сервиса: FastAPI-приложение по контракту inpaint-service.yaml.

Все статусы объявлены контрактом: 200 — PNG страницы, 400 `INVALID_IMAGE`,
413 `IMAGE_TOO_LARGE`, 422 `INVALID_BOXES`/`INVALID_REQUEST`, 500 — отказ
сервиса (`ENCODE_FAILED`, `INPAINT_FAILED`). Тело любой ошибки — `InpaintError`,
включая ошибки разбора запроса: клиент генерируется по контракту и ждёт
одинаковую форму, а не `{"detail": [...]}` от FastAPI.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Annotated

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from inpaint_service import eraser
from inpaint_service.config import Settings

_LOG = logging.getLogger(__name__)

#: Проба готовности: `(ok, причина)`. Вынесена в параметр, чтобы тесты могли
#: проверить 503 без поломки реального окружения.
ReadinessProbe = Callable[[], tuple[bool, str | None]]

#: Статус ответа по классу ошибки: контракт фиксирует их явно.
_STATUS = {
    eraser.InvalidImage: 400,
    eraser.ImageTooLarge: 413,
    eraser.InvalidBoxes: 422,
    eraser.EncodeFailed: 500,
}

#: Код ошибки для запроса, не прошедшего собственную валидацию FastAPI.
INVALID_REQUEST = "INVALID_REQUEST"


def create_app(settings: Settings | None = None, *, probe: ReadinessProbe | None = None) -> FastAPI:
    """Собрать приложение; `probe` подменяется в тестах неготовности."""
    cfg = settings or Settings()
    readiness_probe = probe or eraser.readiness
    app = FastAPI(
        title="Inpaint Service",
        version="1.0.0",
        description="Закрашивание текста на страницах комиксов (арт-слой автоперевода)",
    )
    # настройки доступны и тестам, и __main__: один источник правды
    app.state.settings = cfg
    # затирание — CPU-работа: параллелизм ограничен, иначе несколько страниц
    # одновременно выбирают лимит памяти пода и под падает по OOM
    app.state.slots = asyncio.Semaphore(cfg.workers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        """Ошибка разбора запроса — в форме контракта, а не в форме FastAPI.

        422 для отсутствующего `image`/`boxes` — тот же код ответа, что у
        `INVALID_BOXES`, но свой код ошибки: клиенту важно различать «сервис не
        понял запрос» и «рамки не разбираются».
        """
        fields = ", ".join(".".join(str(part) for part in item["loc"][1:]) for item in exc.errors())
        message = f"запрос не соответствует контракту: {fields or 'нет полей'}"
        _LOG.warning("задание отклонено: %s (%s)", INVALID_REQUEST, message)
        return JSONResponse({"code": INVALID_REQUEST, "message": message}, status_code=422)

    @app.get("/health")
    async def health() -> dict[str, object]:
        """Liveness: процесс жив. Зависимостей нет — статус всегда `ok`."""
        return {"status": "ok", "reason": None}

    @app.get("/ready")
    async def ready() -> Response:
        """Readiness: 200 только если сервис реально может затирать."""
        ok, reason = readiness_probe()
        body = {"status": "ok" if ok else "degraded", "reason": None if ok else reason}
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.post("/erase", response_class=Response)
    async def erase_text(
        boxes: Annotated[str, Form()], image: Annotated[UploadFile, File()]
    ) -> Response:
        """Закрасить рамки реплик; ответ — PNG исходного размера."""
        try:
            payload = await _read_limited(image, cfg.max_bytes)
            parsed = eraser.parse_boxes(boxes, max_boxes=cfg.max_boxes)
            async with app.state.slots:
                # затирание — CPU-работа на сотни миллисекунд: не блокируем loop
                result = await asyncio.to_thread(eraser.erase, payload, parsed, cfg)
        except eraser.InpaintError as exc:
            status = next(
                (code for kind, code in _STATUS.items() if isinstance(exc, kind)), 500
            )
            _LOG.warning("задание отклонено: %s (%s)", exc.code, exc)
            return JSONResponse({"code": exc.code, "message": str(exc)}, status_code=status)
        except Exception as exc:  # noqa: BLE001 - отказ сервиса обязан быть читаемым
            _LOG.exception("затирание упало: %s", exc)
            failed = eraser.InpaintError(f"затирание упало: {exc}")
            return JSONResponse(
                {"code": failed.code, "message": str(failed)}, status_code=500
            )
        return Response(content=result, media_type="image/png")

    return app


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    """Прочитать тело страницы, не превышая лимит.

    `await upload.read()` без размера читает файл целиком: гигабайтная страница
    съела бы память пода ещё до проверки размеров. Поэтому читаем по чанкам и
    обрываем чтение на лимите.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1_048_576)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise eraser.ImageTooLarge(f"страница больше лимита {limit} байт")
        chunks.append(chunk)
    return b"".join(chunks)
