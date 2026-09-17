"""Integracion Docker sintetica P19: build, up, health, UI, recomendacion,
cero exposicion de api al host, inspeccion sin fugas y limpieza total.

Usa EXCLUSIVAMENTE un snapshot P13 sintetico generado en un directorio
temporal fuera del repositorio (``tmp_path``); nunca el snapshot real,
nunca Parquet/CSV/reports. Un unico test de extremo a extremo (mas
barato que reconstruir el stack por cada aserto) que:

1. construye las dos imagenes con ``docker compose build``;
2. levanta el stack con el snapshot sintetico montado read-only;
3. espera el healthcheck de ``api``;
4. comprueba que la UI responde 200 en ``http://127.0.0.1:8501``;
5. realiza una recomendacion sintetica sin navegador, ejecutando un
   cliente httpx DENTRO del contenedor ``ui`` contra ``http://api:8000``;
6. comprueba que ``127.0.0.1:8000`` (api) NO es alcanzable desde el host;
7. inspecciona ambas imagenes/contenedores sin imprimir contenido
   privado (usuario no-root, ausencia de data/reports/tests/.git,
   ausencia de los identificadores sinteticos y de la ruta del host en
   los logs capturados);
8. hace ``docker compose down`` y borra las dos imagenes construidas
   para este proyecto, verificando cero residuos.

Se omite explicitamente si el binario docker (con el plugin compose)
no esta disponible en PATH.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from datetime import date
from pathlib import Path

import httpx
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DOCKER_AVAILABLE = shutil.which("docker") is not None

pytestmark = pytest.mark.skipif(
    not _DOCKER_AVAILABLE,
    reason="El binario docker (con el plugin compose) no esta disponible.",
)

_PLAYER = "P19DockerPlayerSecretA"
_OPPONENT = "P19DockerRivalSecretB"
_AS_OF_DATE = date(2031, 6, 30)


def _synthetic_snapshot(base: Path) -> Path:
    """Genera un snapshot P13 sintetico minimo fuera del repo (tmp_path)."""
    sys.path.insert(0, str(_ROOT))
    try:
        import src.recommender.persisted_tactical_recommendation_provider as p13
        from tests.test_tactical_recommendation_service import _prioritization
    finally:
        sys.path.remove(str(_ROOT))
    result = _prioritization(
        "available", 0, _AS_OF_DATE, False, player=_PLAYER, opponent=_OPPONENT
    )
    snapshot = p13.build_persisted_tactical_recommendation_snapshot([result])
    path = base / "p19-synthetic-snapshot.json"
    p13.persist_persisted_tactical_recommendation_snapshot(snapshot, path)
    assert path.is_file()
    return path


def _run(
    args: list[str], *, env: dict, timeout: int, check: bool = False
) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=str(_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=check,
    )


def _wait_for_ui(timeout_seconds: float = 90.0) -> int:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(
                "http://127.0.0.1:8501", timeout=2.0, follow_redirects=False
            )
            return response.status_code
        except httpx.HTTPError as error:
            last_error = error
            time.sleep(1.0)
    raise AssertionError(f"UI no respondio en 127.0.0.1:8501: {last_error!r}")


def _wait_for_api_health(compose_env: dict, project: str, timeout_seconds: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        completed = _run(
            ["docker", "compose", "-p", project, "ps", "--format", "json"],
            env=compose_env,
            timeout=30,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            rows = [
                json.loads(line)
                for line in completed.stdout.strip().splitlines()
                if line.strip()
            ]
            api_rows = [row for row in rows if row.get("Service") == "api"]
            if api_rows and api_rows[0].get("Health") == "healthy":
                return
        time.sleep(2.0)
    raise AssertionError("api no alcanzo estado healthy dentro del plazo sintetico")


def _host_port_unreachable(port: int, timeout_seconds: float = 2.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(timeout_seconds)
        result = client.connect_ex(("127.0.0.1", port))
        return result != 0


@pytest.fixture()
def compose_project(tmp_path):
    project = f"tennis-p19-{uuid.uuid4().hex[:12]}"
    snapshot_path = _synthetic_snapshot(tmp_path)
    env = dict(os.environ)
    env["TENNIS_TACTICAL_SNAPSHOT_HOST_PATH"] = str(snapshot_path)
    yield project, env, snapshot_path
    # Limpieza obligatoria: down + eliminar SOLO las imagenes de este
    # proyecto. Las imagenes se localizan por el patron de nombre que
    # Compose asigna (<project>-<service>), no por el estado de los
    # contenedores: sigue funcionando aunque el propio test ya haya
    # hecho ``down`` (idempotente) o si no llego a construir nada.
    _run(
        ["docker", "compose", "-p", project, "down", "--volumes", "--remove-orphans"],
        env=env,
        timeout=120,
    )
    images = _run(
        ["docker", "images", "--filter", f"reference={project}-*", "-q"],
        env=env,
        timeout=30,
    )
    for image_id in {line.strip() for line in images.stdout.splitlines() if line.strip()}:
        _run(["docker", "image", "rm", "-f", image_id], env=env, timeout=60)


def test_p19_docker_compose_end_to_end(compose_project) -> None:
    project, env, snapshot_path = compose_project

    # 1-2. build + up con el snapshot sintetico montado read-only.
    build = _run(
        ["docker", "compose", "-p", project, "build"], env=env, timeout=600
    )
    assert build.returncode == 0, build.stderr

    up = _run(
        ["docker", "compose", "-p", project, "up", "-d"], env=env, timeout=180
    )
    assert up.returncode == 0, up.stderr

    # 3. healthcheck de api.
    _wait_for_api_health(env, project)

    # 4. UI responde 200 en 127.0.0.1:8501.
    ui_status = _wait_for_ui()
    assert ui_status == 200

    # 4b. Regresion del ModuleNotFoundError reportado en produccion: un
    # HTTP 200 del servidor Streamlit certifica solo que el proceso
    # escucha, NUNCA que el script de la aplicacion importe/ejecute sin
    # error (Streamlit lo ejecuta de forma perezosa, por sesion de
    # navegador -- ver tests/test_ui_docker_import_regression.py). Se
    # reproduce aqui, contra el contenedor ui REAL ya en ejecucion (no
    # una imagen aislada), la misma semantica de sys.path que usa
    # ``streamlit run``.
    import_check_script = (
        "import sys\n"
        "sys.path = [entry for entry in sys.path if entry != '']\n"
        "from src.ui.streamlit_app import run_recommendation_experience\n"
        "print('UI_IMPORT_OK')\n"
    )
    import_check = _run(
        [
            "docker", "compose", "-p", project, "exec", "-T", "ui",
            "python", "-c", import_check_script,
        ],
        env=env,
        timeout=20,
    )
    assert import_check.returncode == 0, import_check.stderr
    assert "UI_IMPORT_OK" in import_check.stdout

    # 6. api NUNCA publicado al host.
    assert _host_port_unreachable(8000)

    # 5. recomendacion sintetica sin navegador: httpx DENTRO de ui.
    recommend_script = (
        "import httpx, json\n"
        "body = {\n"
        f"    'player_id': {_PLAYER!r},\n"
        f"    'opponent_id': {_OPPONENT!r},\n"
        f"    'as_of_date': {_AS_OF_DATE.isoformat()!r},\n"
        "}\n"
        "response = httpx.post(\n"
        "    'http://api:8000/api/v1/recommendations', json=body, timeout=10.0\n"
        ")\n"
        "print(response.status_code)\n"
    )
    exec_result = _run(
        [
            "docker", "compose", "-p", project, "exec", "-T", "ui",
            "python", "-c", recommend_script,
        ],
        env=env,
        timeout=30,
    )
    assert exec_result.returncode == 0, exec_result.stderr
    assert exec_result.stdout.strip() == "200"

    # 7. Inspeccion sin fuga de contenido privado.
    uid_check = _run(
        [
            "docker", "compose", "-p", project, "exec", "-T", "api",
            "python", "-c", "import os; print(os.getuid())",
        ],
        env=env,
        timeout=20,
    )
    assert uid_check.returncode == 0
    assert uid_check.stdout.strip() not in ("0", "")

    forbidden_paths_script = (
        "import pathlib, sys\n"
        "forbidden = ['/app/data', '/app/reports', '/app/tests', "
        "'/app/notebooks', '/app/.git', '/app/.env']\n"
        "found = [p for p in forbidden if pathlib.Path(p).exists()]\n"
        "sys.exit(1 if found else 0)\n"
    )
    for service in ("api", "ui"):
        forbidden_check = _run(
            [
                "docker", "compose", "-p", project, "exec", "-T", service,
                "python", "-c", forbidden_paths_script,
            ],
            env=env,
            timeout=20,
        )
        assert forbidden_check.returncode == 0, (
            f"{service} contiene rutas prohibidas"
        )

    images_json = _run(
        ["docker", "compose", "-p", project, "images", "--format", "json"],
        env=env,
        timeout=30,
    )
    parsed = json.loads(images_json.stdout)
    image_rows = parsed if isinstance(parsed, list) else [parsed]
    for row in image_rows:
        inspected = _run(
            ["docker", "image", "inspect", row["ID"]], env=env, timeout=20
        )
        assert inspected.returncode == 0
        config = json.loads(inspected.stdout)[0]["Config"]
        assert config["User"] not in ("", "0", "root")

    logs = _run(
        ["docker", "compose", "-p", project, "logs", "--no-color"],
        env=env,
        timeout=30,
    )
    combined_logs = logs.stdout + logs.stderr
    assert _PLAYER not in combined_logs
    assert _OPPONENT not in combined_logs
    assert str(snapshot_path) not in combined_logs
    assert "TENNIS_TACTICAL_SNAPSHOT_HOST_PATH" not in combined_logs
    # Regresion del ModuleNotFoundError reportado en produccion: ningun
    # servicio debe haber registrado un traceback de importacion.
    assert "ModuleNotFoundError" not in combined_logs
    assert "No module named 'src'" not in combined_logs
    assert "Traceback (most recent call last)" not in combined_logs

    ps = _run(
        ["docker", "compose", "-p", project, "ps", "--format", "json"],
        env=env,
        timeout=30,
    )
    running_services = {
        json.loads(line)["Service"]
        for line in ps.stdout.strip().splitlines()
        if line.strip()
    }
    assert running_services == {"api", "ui"}

    # 8. Limpieza dentro del propio test (no solo en el fixture) para
    # verificar, en el mismo proceso, cero residuos de contenedores,
    # redes e imagenes de este proyecto identificado unicamente.
    down = _run(
        ["docker", "compose", "-p", project, "down", "--volumes", "--remove-orphans"],
        env=env,
        timeout=120,
    )
    assert down.returncode == 0, down.stderr
    images = _run(
        ["docker", "images", "--filter", f"reference={project}-*", "-q"],
        env=env,
        timeout=30,
    )
    for image_id in {line.strip() for line in images.stdout.splitlines() if line.strip()}:
        _run(["docker", "image", "rm", "-f", image_id], env=env, timeout=60)

    ps_after = _run(
        ["docker", "compose", "-p", project, "ps", "-a", "--format", "json"],
        env=env,
        timeout=30,
    )
    assert ps_after.stdout.strip() == ""
    networks_after = _run(
        ["docker", "network", "ls", "--filter", f"name={project}", "--format", "json"],
        env=env,
        timeout=30,
    )
    assert networks_after.stdout.strip() == ""
    images_after = _run(
        ["docker", "images", "--filter", f"reference={project}-*", "-q"],
        env=env,
        timeout=30,
    )
    assert images_after.stdout.strip() == ""
