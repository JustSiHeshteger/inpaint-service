# inpaint-service

Сервис затирания текста на страницах комиксов — поставщик стадии RENDER в
конвейере автоперевода (карточки 55–58). Принимает страницу и рамки реплик,
возвращает страницу того же размера с закрашенными областями; поверх воркер
рисует перевод.

Контракт: `books-platform-openapi/openapi/inpaint-service.yaml`.

## Что делает

| Метод | Путь | Назначение |
|---|---|---|
| POST | `/erase` | закрасить рамки (multipart: `image` + JSON `boxes`) |
| GET | `/health` | liveness: процесс жив (зависимостей нет) |
| GET | `/ready` | readiness: мини-прогон `inpaint` + PNG на странице 8×8 |

Закрашивание — OpenCV `INPAINT_TELEA` (или `INPAINT_NS`), CPU, без ML-моделей:
сервис не тянет torch/onnx, старт мгновенный.

Гарантии контракта:

- **размер страницы не меняется** — воркер сверяет его и считает расхождение
  сбоем задания, а координаты вёрстки считаются в исходной системе;
- **ответ всегда PNG** (`image/png`) — воркер сам перекодирует в WEBP;
- **пустой список рамок допустим** — страница возвращается без изменений;
- **вырожденные рамки** (`w`/`h` ≤ 0) игнорируются, а не валят страницу;
- **ошибки машинно-читаемы**: `INVALID_IMAGE` (400), `INVALID_BOXES` и
  `INVALID_REQUEST` (422), `IMAGE_TOO_LARGE` (413), `ENCODE_FAILED` и
  `INPAINT_FAILED` (500). Тело любой ошибки — `InpaintError`, включая ошибки
  разбора запроса: клиент генерируется по контракту и ждёт одну форму.

Лимиты — часть контракта с платформой:

- **площадь страницы** проверяется по заголовку контейнера (PNG IHDR, JPEG
  SOFn, WebP VP8X) до декодирования: «PNG-бомба» иначе съедает память раньше,
  чем сработает отказ;
- **размер тела** читается по чанкам и обрывается на `INPAINT_MAX_BYTES`;
- **параллелизм** ограничен `INPAINT_WORKERS`: каждая страница держит матрицы
  страницы, маски и результата, поэтому лимит памяти пода и лимит параллелизма
  задаются вместе.

## Настройки

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `INPAINT_HOST` / `INPAINT_PORT` | `0.0.0.0` / `8094` | адрес и порт |
| `INPAINT_METHOD` | `telea` | метод OpenCV: `telea` (быстрый) или `ns` |
| `INPAINT_RADIUS` | `3` | радиус окрестности закрашивания, 1–25 |
| `INPAINT_DILATE` | `2` | расширение маски, px: закрывает обводку букв за рамкой |
| `INPAINT_MAX_PIXELS` | `40000000` | лимит площади страницы (иначе 413) |
| `INPAINT_MAX_BOXES` | `512` | лимит числа рамок в запросе |
| `INPAINT_MAX_BYTES` | `32000000` | лимит размера тела запроса (байты) |
| `INPAINT_WORKERS` | `2` | сколько страниц затирать одновременно |
| `INPAINT_LOG_LEVEL` | `INFO` | уровень логов |

Метод и радиус — свойство развёртывания, а не задания: они не приходят в
запросе, иначе страницы одной книги рисовались бы по-разному.

## Запуск

```bash
# локально (нужен python 3.12+)
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
$env:PYTHONPATH='src'; .venv\Scripts\python.exe -m inpaint_service

# образ
docker build -t bookplatform/inpaint-service:dev .

# в контейнерах `readOnlyRootFilesystem` обязателен writable `/tmp`:
# Starlette спуливает части мультипарта больше 1 МиБ во временный файл
docker run --rm -p 8094:8094 --read-only --tmpfs /tmp bookplatform/inpaint-service:dev

# в составе стенда
cd docker; docker compose --env-file .env up -d inpaint-service translation-worker
```

Пример запроса:

```bash
curl -sS http://localhost:8094/erase \
  -F 'image=@page.png;type=application/octet-stream' \
  -F 'boxes=[{"x":40,"y":60,"w":520,"h":140}]' \
  -o page-cleaned.png
```

## Тесты

```bash
.venv\Scripts\python.exe -m pytest    # 65 тестов
.venv\Scripts\python.exe -m ruff check .
```

`tests/test_deployment.py` сверяет сервис с файлом контракта (пути, коды,
`required`, имена схем) и с k8s-манифестом (runAsNonRoot, uid, writable `/tmp`
под спулинг мультипарта, пробы, `preStop`). В опубликованном зеркале сервиса
этих файлов нет — тесты там пропускаются, в монорепо обязательны.

Проверяется не «вызвался ли OpenCV», а контракт: размер сохраняется, текст
внутри рамки закрашивается фоном, пиксели вне рамок не меняются, вырожденные
и выходящие за границу рамки не ломают страницу, лимит площади и лимит тела
дают 413, битые `boxes` — 422 `INVALID_BOXES`, пропущенное поле — 422
`INVALID_REQUEST`, битая картинка — 400, сбой кодирования — 500
`ENCODE_FAILED`, `/ready` делает мини-прогон и умеет отвечать 503.

## Место в конвейере

```
ComicTranslationRequested
      ↓
translation-worker: LAYOUT (layer.json, фаза 1) → RENDER
      ↓ POST /erase (page + рамки)
inpaint-service  →  страница без текста
      ↓
translation-worker: вёрстка перевода (Pillow) → web.webp в MinIO
      ↓
ComicTranslationCompleted.pages[].imageKey → book-service → imageUrl у TextLayer
```

Если сервис недоступен, воркер с `TRANSLATION_STRICT_RENDER=false` пишет
страницу с оверлеем фазы 1 и warning `page N: art layer skipped` — книга
дочитывается. Со `STRICT_RENDER=true` сбой арт-слоя становится
`LAYOUT_FAILED` (ретраибельно).
