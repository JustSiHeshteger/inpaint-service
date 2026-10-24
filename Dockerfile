# syntax=docker/dockerfile:1
# Сервис затирания страниц для арт-слоя автоперевода (карточка 58).
# Контекст сборки — каталог inpaint-service (контракт сервису не нужен: он
# принимает картинку и рамки, а не события).
FROM python:3.12-slim AS runtime

# OpenCV-headless не тянет GUI, но его бинарник зависит от libGL-сборки GLib
# и OpenMP: без них `import cv2` падает на старте, а не в запросе.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY src/ ./src/
# Пакет не ставим: src-layout подключается через PYTHONPATH. Так образ не
# зависит от сборки wheel и остаётся работоспособным при readOnlyRootFilesystem.
ENV PYTHONPATH=/app/src

# Непривилегированный пользователь: затирание не требует прав в системе,
# а k8s-манифест требует runAsNonRoot (uid фиксирован — см. k8s/inpaint-service.yaml).
RUN useradd --system --uid 10001 --create-home --shell /usr/sbin/nologin inpaint
USER 10001:10001

# Лимит страницы: 40000000 пикселей ≈ 7000×5700 — с запасом на двухстраничные
# развороты сканов; всё, что больше, сервис отклоняет 413 (контракт).
ENV INPAINT_HOST=0.0.0.0 \
    INPAINT_PORT=8094 \
    INPAINT_METHOD=telea \
    INPAINT_RADIUS=3 \
    INPAINT_DILATE=2 \
    INPAINT_MAX_PIXELS=40000000 \
    INPAINT_MAX_BYTES=32000000 \
    INPAINT_WORKERS=2

EXPOSE 8094

# HEALTHCHECK смотрит на готовность, как compose и k8s: контейнер, который жив,
# но не может затирать, не должен считаться здоровым
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8094/ready || exit 1

CMD ["python", "-m", "inpaint_service"]
