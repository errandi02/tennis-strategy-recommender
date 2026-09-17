"""Tests estaticos P19: estructura Compose, Dockerfiles y artefactos.

Verifican, SIN construir imagenes ni arrancar contenedores (eso vive en
``test_p19_docker_integration.py``): la configuracion resuelta por
``docker compose config`` (autoridad independiente del binario Docker,
no una relectura del propio YAML) para api (sin puerto publicado,
snapshot montado read-only, cap_drop, security_opt, limites, red
dedicada, sin privileged/host network/docker socket, restart
desactivado, logging acotado) y ui (unico puerto publicado
127.0.0.1:8501, sin volumenes ni variable de snapshot, healthcheck de
api como dependencia); el contenido de los Dockerfile (usuario no-root
UID/GID explicito, WORKDIR fijo, variables PYTHON*/PIP_NO_CACHE_DIR,
HEALTHCHECK del servicio api); ``.dockerignore`` excluye datos
privados; ``.env.example`` solo contiene un placeholder generico; y los
requirements de runtime de api/ui son minimos y no actualizan las
versiones ya fijadas en requirements.txt.

Requiere el binario ``docker`` (con el plugin compose) disponible en
PATH; si no lo esta, el modulo completo se omite explicitamente.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_DOCKER_AVAILABLE = shutil.which("docker") is not None

pytestmark = pytest.mark.skipif(
    not _DOCKER_AVAILABLE,
    reason="El binario docker (con el plugin compose) no esta disponible.",
)


def _compose_config() -> dict:
    env = {"TENNIS_TACTICAL_SNAPSHOT_HOST_PATH": str(_ROOT / "does-not-need-to-exist.json")}
    import os

    full_env = dict(os.environ)
    full_env.update(env)
    completed = subprocess.run(
        ["docker", "compose", "config", "--format", "json"],
        cwd=str(_ROOT),
        env=full_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


@pytest.fixture(scope="module")
def compose_config() -> dict:
    return _compose_config()


@pytest.fixture(scope="module")
def api_service(compose_config) -> dict:
    return compose_config["services"]["api"]


@pytest.fixture(scope="module")
def ui_service(compose_config) -> dict:
    return compose_config["services"]["ui"]


# --------------------------------------------------------------------- #
# 1-2. api no publica puerto al host; solo expose 8000                   #
# --------------------------------------------------------------------- #


def test_api_no_tiene_ports_publicados(api_service) -> None:
    assert "ports" not in api_service or api_service["ports"] in (None, [])


def test_api_solo_expone_8000_internamente(api_service) -> None:
    assert api_service["expose"] == ["8000"]


# --------------------------------------------------------------------- #
# 3-4. ui publicada solo en 127.0.0.1:8501                               #
# --------------------------------------------------------------------- #


def test_ui_publica_solo_127_0_0_1_8501(ui_service) -> None:
    ports = ui_service["ports"]
    assert len(ports) == 1
    port = ports[0]
    assert port["host_ip"] == "127.0.0.1"
    assert str(port["published"]) == "8501"
    assert port["target"] == 8501
    assert port["protocol"] == "tcp"


def test_ui_no_expone_ningun_otro_puerto(ui_service) -> None:
    assert "expose" not in ui_service or ui_service["expose"] in (None, [])


# --------------------------------------------------------------------- #
# 5-6. Snapshot montado read-only solo en api; ui sin volumen/variable   #
# --------------------------------------------------------------------- #


def test_snapshot_montado_read_only_solo_en_api(api_service) -> None:
    volumes = api_service["volumes"]
    assert len(volumes) == 1
    mount = volumes[0]
    assert mount["type"] == "bind"
    assert mount["target"] == "/snapshot/tactical-recommendation-snapshot.json"
    assert mount["read_only"] is True


def test_api_recibe_solo_la_ruta_interna_fija(api_service) -> None:
    env = api_service["environment"]
    assert env["TENNIS_TACTICAL_SNAPSHOT_PATH"] == (
        "/snapshot/tactical-recommendation-snapshot.json"
    )
    # La variable de host nunca se propaga al proceso api.
    assert "TENNIS_TACTICAL_SNAPSHOT_HOST_PATH" not in env


def test_ui_sin_volumenes_ni_variable_de_snapshot(ui_service) -> None:
    assert "volumes" not in ui_service or ui_service["volumes"] in (None, [])
    env = ui_service.get("environment", {})
    assert "TENNIS_TACTICAL_SNAPSHOT_PATH" not in env
    assert "TENNIS_TACTICAL_SNAPSHOT_HOST_PATH" not in env


# --------------------------------------------------------------------- #
# 8-9. Capabilities, no privileged/host network/docker socket            #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("service_name", ["api", "ui"])
def test_cap_drop_all(compose_config, service_name) -> None:
    service = compose_config["services"][service_name]
    assert service["cap_drop"] == ["ALL"]


@pytest.mark.parametrize("service_name", ["api", "ui"])
def test_no_privileged_ni_host_network(compose_config, service_name) -> None:
    service = compose_config["services"][service_name]
    assert not service.get("privileged", False)
    assert service.get("network_mode") != "host"


@pytest.mark.parametrize("service_name", ["api", "ui"])
def test_sin_montaje_del_docker_socket(compose_config, service_name) -> None:
    service = compose_config["services"][service_name]
    for mount in service.get("volumes", []) or []:
        source = str(mount.get("source", ""))
        assert "docker.sock" not in source


@pytest.mark.parametrize("service_name", ["api", "ui"])
def test_no_new_privileges(compose_config, service_name) -> None:
    service = compose_config["services"][service_name]
    assert "no-new-privileges:true" in service["security_opt"]


# --------------------------------------------------------------------- #
# Restart, red dedicada, limites, logging                                #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("service_name", ["api", "ui"])
def test_restart_desactivado(compose_config, service_name) -> None:
    service = compose_config["services"][service_name]
    assert service["restart"] == "no"


@pytest.mark.parametrize("service_name", ["api", "ui"])
def test_read_only_root_filesystem(compose_config, service_name) -> None:
    service = compose_config["services"][service_name]
    assert service["read_only"] is True


def test_red_interna_dedicada_no_default(compose_config) -> None:
    networks = compose_config["networks"]
    assert "tennis_internal" in networks
    for service in compose_config["services"].values():
        assert set(service["networks"].keys()) == {"tennis_internal"}


@pytest.mark.parametrize("service_name", ["api", "ui"])
def test_limites_de_recursos_razonables(compose_config, service_name) -> None:
    service = compose_config["services"][service_name]
    assert int(service["mem_limit"]) > 0
    assert float(service["cpus"]) > 0
    assert int(service["pids_limit"]) > 0


@pytest.mark.parametrize("service_name", ["api", "ui"])
def test_logging_acotado_con_rotacion(compose_config, service_name) -> None:
    service = compose_config["services"][service_name]
    logging_cfg = service["logging"]
    assert logging_cfg["driver"] == "json-file"
    assert logging_cfg["options"]["max-file"] not in (None, "")
    assert logging_cfg["options"]["max-size"] not in (None, "")


def test_ui_depende_del_healthcheck_de_api(ui_service) -> None:
    depends = ui_service["depends_on"]
    assert depends["api"]["condition"] == "service_healthy"


# --------------------------------------------------------------------- #
# Dockerfiles: usuario no-root, WORKDIR, env cerradas, healthcheck       #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "dockerfile_path",
    ["docker/api/Dockerfile", "docker/ui/Dockerfile"],
)
def test_dockerfile_usuario_no_root_uid_gid_explicito(dockerfile_path) -> None:
    content = (_ROOT / dockerfile_path).read_text(encoding="utf-8")
    match = re.search(r"^USER\s+(\d+):(\d+)\s*$", content, re.MULTILINE)
    assert match is not None, f"{dockerfile_path} no fija USER uid:gid"
    uid, gid = int(match.group(1)), int(match.group(2))
    assert uid != 0
    assert gid != 0


@pytest.mark.parametrize(
    "dockerfile_path",
    ["docker/api/Dockerfile", "docker/ui/Dockerfile"],
)
def test_dockerfile_workdir_fijo(dockerfile_path) -> None:
    content = (_ROOT / dockerfile_path).read_text(encoding="utf-8")
    assert re.search(r"^WORKDIR\s+/\S+\s*$", content, re.MULTILINE)


@pytest.mark.parametrize(
    "dockerfile_path",
    ["docker/api/Dockerfile", "docker/ui/Dockerfile"],
)
def test_dockerfile_variables_cerradas(dockerfile_path) -> None:
    content = (_ROOT / dockerfile_path).read_text(encoding="utf-8")
    for token in (
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONUNBUFFERED=1",
        "PIP_NO_CACHE_DIR=1",
    ):
        assert token in content, f"{dockerfile_path} falta {token}"


@pytest.mark.parametrize(
    "dockerfile_path",
    ["docker/api/Dockerfile", "docker/ui/Dockerfile"],
)
def test_dockerfile_no_copia_snapshot_ni_datos(dockerfile_path) -> None:
    content = (_ROOT / dockerfile_path).read_text(encoding="utf-8")
    forbidden = ("data/", "reports/", "models/", "tests/", "notebooks/", ".env")
    copy_lines = [
        line for line in content.splitlines() if line.strip().startswith("COPY")
    ]
    for line in copy_lines:
        for token in forbidden:
            assert token not in line, f"{dockerfile_path}: {line!r} referencia {token}"


def test_dockerfile_ui_declara_pythonpath_app() -> None:
    """Regresion del ModuleNotFoundError reportado en produccion:
    ``streamlit run <script>`` inserta el directorio del propio script
    en ``sys.path`` (igual que ``python script.py``), NUNCA el CWD/
    WORKDIR -- a diferencia de ``python -c``/``python -m``, que si
    tienen el CWD implicito. Sin ``PYTHONPATH=/app`` explicito, ``/app``
    (padre de ``src/``) no esta en ``sys.path`` y
    ``from src.ui.streamlit_app import ...`` falla dentro del
    contenedor aunque WORKDIR sea ``/app``."""
    content = (_ROOT / "docker/ui/Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^\s*PYTHONPATH=/app\s*$", content, re.MULTILINE)


def test_dockerfile_ui_declara_comprobacion_de_importacion_previa() -> None:
    """El build falla (no solo el arranque del contenedor) si el import
    real que hace ``streamlit_container_app.py`` no resuelve dentro de
    la imagen."""
    content = (_ROOT / "docker/ui/Dockerfile").read_text(encoding="utf-8")
    assert "from src.ui.streamlit_app import run_recommendation_experience" in content


def test_dockerfile_ui_import_check_no_depende_del_cwd_implicito_de_python_c() -> None:
    """``python -c`` (a diferencia de ``streamlit run``) SI anade el CWD
    a ``sys.path``: sin retirarlo explicitamente, la comprobacion de
    importacion pasaria incluso sin ``PYTHONPATH=/app`` y no habria
    detectado el defecto real reportado en produccion (verificado
    manualmente: la misma comprobacion sin este filtro pasa incluso
    contra una imagen sin PYTHONPATH)."""
    content = (_ROOT / "docker/ui/Dockerfile").read_text(encoding="utf-8")
    assert "sys.path = [entry for entry in sys.path if entry != '']" in content


def test_dockerfile_api_declara_healthcheck() -> None:
    content = (_ROOT / "docker/api/Dockerfile").read_text(encoding="utf-8")
    assert "HEALTHCHECK" in content
    assert "/healthz" not in content  # el path vive en healthcheck.py, no aqui
    healthcheck_script = (_ROOT / "docker/api/healthcheck.py").read_text(
        encoding="utf-8"
    )
    assert "/healthz" in healthcheck_script
    assert "127.0.0.1" in healthcheck_script


# --------------------------------------------------------------------- #
# .dockerignore, .env.example, requirements minimas                      #
# --------------------------------------------------------------------- #


def test_dockerignore_excluye_datos_privados() -> None:
    content = (_ROOT / ".dockerignore").read_text(encoding="utf-8")
    for token in ("data", "reports", "models", "tests", ".git", ".env"):
        assert re.search(rf"^{re.escape(token)}\b", content, re.MULTILINE), (
            f".dockerignore no excluye {token}"
        )


def test_env_example_solo_placeholder_generico() -> None:
    content = (_ROOT / ".env.example").read_text(encoding="utf-8")
    lines = [
        line for line in content.splitlines()
        if line.strip().startswith("TENNIS_TACTICAL_SNAPSHOT_HOST_PATH=")
    ]
    assert len(lines) == 1
    value = lines[0].split("=", 1)[1].strip()
    assert value in ("/ruta/externa/snapshot-p13.json",)
    # Ninguna ruta absoluta real de este equipo/usuario.
    assert "Errandi" not in content
    assert str(_ROOT) not in content
    assert "C:\\Users" not in content
    assert "C:/Users" not in content


def test_env_esta_ignorado_por_git() -> None:
    gitignore = (_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert re.search(r"^\.env$", gitignore, re.MULTILINE)
    assert re.search(r"^!\.env\.example$", gitignore, re.MULTILINE)


@pytest.mark.parametrize(
    "requirements_path", ["docker/api/requirements.txt", "docker/ui/requirements.txt"]
)
def test_requirements_runtime_sin_dependencias_pesadas(requirements_path) -> None:
    content = (_ROOT / requirements_path).read_text(encoding="utf-8")
    lines = [
        line.split("==")[0].strip().lower()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for heavy in ("pandas", "pyarrow", "scipy", "statsmodels", "numpy"):
        assert heavy not in lines, f"{requirements_path} incluye {heavy}"


def test_requirements_runtime_no_actualizan_versiones_fijadas() -> None:
    """Autoridad independiente: compara contra requirements.txt raiz, no
    contra un valor hardcodeado que pueda quedar desincronizado."""
    root_pins = {}
    for line in (_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "==" not in line:
            continue
        name, version = line.split("==", 1)
        root_pins[name.strip().lower()] = version.strip()

    for requirements_path in ("docker/api/requirements.txt", "docker/ui/requirements.txt"):
        content = (_ROOT / requirements_path).read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "==" not in line:
                continue
            name, version = line.split("==", 1)
            name = name.strip().lower()
            version = version.strip()
            assert name in root_pins, f"{name} no esta fijado en requirements.txt"
            assert version == root_pins[name], (
                f"{requirements_path} actualiza {name} a {version} "
                f"(raiz fija {root_pins[name]})"
            )
