"""Проверки, которые нельзя выразить через HTTP: контракт и манифест.

Оба файла живут в монорепо, но не входят в само зеркало сервиса, поэтому при
отсутствии каталога тесты пропускаются: в опубликованном репозитории сервиса
проверять нечего, в монорепо они обязательны.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parent
CONTRACT = REPO_ROOT / "books-platform-openapi" / "openapi" / "inpaint-service.yaml"
MANIFEST = REPO_ROOT / "k8s" / "inpaint-service.yaml"

needs_contract = pytest.mark.skipif(
    not CONTRACT.is_file(), reason="контракта нет в этом репозитории"
)
needs_manifest = pytest.mark.skipif(
    not MANIFEST.is_file(), reason="манифеста нет в этом репозитории"
)


@needs_contract
def test_contract_matches_the_service() -> None:
    """Контракт обязан совпадать с кодом, а не с самогенерируемой схемой FastAPI.

    Проверка `app.openapi()` доказывает лишь то, что FastAPI помнит свои же
    декораторы. Здесь читается именно файл контракта платформы: переименование
    поля или смена кода ответа в коде должны ронять этот тест.
    """
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    paths = contract["paths"]

    assert set(paths) == {"/erase", "/health", "/ready"}
    erase = paths["/erase"]["post"]
    assert erase["operationId"] == "eraseText"
    assert erase["requestBody"]["required"] is True
    content = erase["requestBody"]["content"]["multipart/form-data"]["schema"]
    multipart = contract["components"]["schemas"][content["$ref"].rsplit("/", 1)[-1]]
    assert set(multipart["required"]) == {"image", "boxes"}

    ok = erase["responses"]["200"]
    assert list(ok["content"]) == ["image/png"]

    statuses = {str(code) for code in erase["responses"]}
    assert {"200", "400", "413", "422", "500"} <= statuses
    for code in ("400", "413", "422", "500"):
        ref = erase["responses"][code]["content"]["application/json"]["schema"]["$ref"]
        assert ref.endswith("/InpaintError"), f"{code} обязан отдавать InpaintError"

    error = contract["components"]["schemas"]["InpaintError"]["properties"]["code"]
    error_codes = set(error["enum"])
    assert {
        "INVALID_BOXES",
        "INVALID_IMAGE",
        "INVALID_REQUEST",
        "IMAGE_TOO_LARGE",
        "ENCODE_FAILED",
    } <= error_codes

    health = contract["components"]["schemas"]["HealthStatus"]
    assert set(health["required"]) == {"status", "reason"}


@needs_contract
def test_contract_declares_readiness_and_liveness_semantics() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    ready = contract["paths"]["/ready"]["get"]
    assert set(ready["responses"]) == {"200", "503"}
    health = contract["paths"]["/health"]["get"]
    assert set(health["responses"]) == {"200"}


@needs_manifest
def test_manifest_runs_pod_without_root_and_drops_capabilities() -> None:
    """k8s-манифест — часть сервиса: он обязан соответствовать образу."""
    docs = [doc for doc in yaml.safe_load_all(MANIFEST.read_text(encoding="utf-8")) if doc]
    deployment = next(doc for doc in docs if doc["kind"] == "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    container = pod["containers"][0]

    assert pod["securityContext"]["runAsNonRoot"] is True
    # uid совпадает с `USER 10001:10001` в Dockerfile: иначе pod не стартует
    assert pod["securityContext"]["runAsUser"] == 10001
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]
    assert container["image"] == "bookplatform/inpaint-service:dev"
    assert container["ports"][0]["containerPort"] == 8094
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert container["lifecycle"]["preStop"]["exec"]["command"]


@needs_manifest
def test_manifest_gives_writable_tmp_for_multipart_spooling() -> None:
    """Голая `readOnlyRootFilesystem` ломает `/erase` на страницах > 1 МиБ.

    Starlette спуливает загружаемый файл на диск, когда он больше 1 МиБ
    (`SpooledTemporaryFile`), а реальная страница комикса весит больше. Без
    записываемого `/tmp` сервис отвечал бы 400 на каждой книге — при этом
    пробы и метрики оставались бы зелёными.
    """
    docs = [doc for doc in yaml.safe_load_all(MANIFEST.read_text(encoding="utf-8")) if doc]
    deployment = next(doc for doc in docs if doc["kind"] == "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    container = pod["containers"][0]

    mounts = {mount["mountPath"]: mount["name"] for mount in container.get("volumeMounts", [])}
    volumes = {volume["name"]: volume for volume in pod.get("volumes", [])}

    assert "/tmp" in mounts, "readOnlyRootFilesystem требует writable /tmp под мультипарт"
    volume = volumes[mounts["/tmp"]]
    assert "emptyDir" in volume
    size_limit = volume["emptyDir"].get("sizeLimit")
    assert size_limit, "у emptyDir обязан быть sizeLimit: диск пода не бесконечен"
    # страница загружается целиком: лимит должен быть заметно больше 1 МиБ
    assert re.fullmatch(r"\d+Mi", size_limit) and int(size_limit.removesuffix("Mi")) >= 64


@needs_manifest
def test_manifest_tmp_is_really_used_by_the_process() -> None:
    """`TMPDIR` обязан указывать в тот же том, что смонтирован."""
    docs = [doc for doc in yaml.safe_load_all(MANIFEST.read_text(encoding="utf-8")) if doc]
    deployment = next(doc for doc in docs if doc["kind"] == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item.get("value") for item in container.get("env", [])}

    assert env.get("TMPDIR") == "/tmp"
    assert env.get("INPAINT_MAX_PIXELS")
    mounts = {mount["mountPath"] for mount in container.get("volumeMounts", [])}
    assert env["TMPDIR"] in mounts
