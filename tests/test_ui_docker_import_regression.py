"""Regresion del ``ModuleNotFoundError: No module named 'src'`` reportado
en produccion (Windows/Docker Desktop, clon remoto limpio): el contenedor
``ui`` quedaba ``healthy``/``Running`` y el host respondia HTTP 200 en
``127.0.0.1:8501``, pero abrir la aplicacion en el navegador fallaba
porque ``streamlit run src/ui/streamlit_container_app.py`` inserta el
directorio del propio script (``/app/src/ui``) en ``sys.path`` -- nunca
el CWD/WORKDIR (``/app``) -- y ``from src.ui.streamlit_app import ...``
no podia resolver el paquete ``src``.

La causa raiz (confirmada leyendo ``streamlit/web/bootstrap.py`` de la
version 1.63.0 instalada: ``sys.path.insert(0,
os.path.dirname(main_script_path))``) es que un HTTP 200 del servidor
Streamlit certifica solo que el *proceso servidor* escucha; Streamlit
ejecuta el script de la aplicacion de forma perezosa, por sesion, asi
que un error de import ahi no afecta ``/_stcore/health`` ni al estado
del contenedor. Este archivo reproduce el escenario exacto (sin
navegador, sin depender de esa ejecucion perezosa) invocando el mismo
entrypoint de script que usa el contenedor real.

Requiere el binario ``docker`` disponible en PATH; si no lo esta, el
modulo completo se omite explicitamente.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DOCKER_AVAILABLE = shutil.which("docker") is not None

pytestmark = pytest.mark.skipif(
    not _DOCKER_AVAILABLE,
    reason="El binario docker no esta disponible.",
)


def _run(args: list[str], *, timeout: int, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=check,
    )


@pytest.fixture(scope="module")
def ui_image_tag() -> str:
    tag = f"tennis-ui-import-regression-{uuid.uuid4().hex[:12]}"
    build = _run(
        ["docker", "build", "-f", "docker/ui/Dockerfile", "-t", tag, "."],
        timeout=600,
    )
    assert build.returncode == 0, build.stderr
    yield tag
    _run(["docker", "image", "rm", "-f", tag], timeout=60)


def test_ui_image_builds_with_the_preflight_import_check(ui_image_tag: str) -> None:
    """El propio ``docker build`` ya ejecuto y verifico la comprobacion
    de importacion (``docker/ui/Dockerfile``); si este fixture no lanzo
    ``AssertionError``, el build -- y por tanto esa comprobacion --
    tuvo exito."""
    inspect = _run(["docker", "image", "inspect", ui_image_tag], timeout=30)
    assert inspect.returncode == 0


def test_ui_image_reproduces_exact_reported_entrypoint_without_traceback(
    ui_image_tag: str,
) -> None:
    """Invoca el mismo modulo que arranca el contenedor real
    (``src/ui/streamlit_container_app.py``) como script -- identica
    semantica de ``sys.path`` a la que usa ``streamlit run`` -- sin
    pasar por el bucle de servidor completo (evitado con
    ``--entrypoint python``, que ademas hace la reproduccion
    determinista y rapida, sin esperar a una sesion de navegador)."""
    result = _run(
        [
            "docker", "run", "--rm", "--entrypoint", "python", ui_image_tag,
            "src/ui/streamlit_container_app.py",
        ],
        timeout=30,
    )
    combined = result.stdout + result.stderr
    assert "ModuleNotFoundError" not in combined, combined
    assert "No module named 'src'" not in combined, combined
    assert "Traceback (most recent call last)" not in combined, combined
    assert result.returncode == 0, combined


def test_ui_image_effective_syspath_includes_app_via_pythonpath(ui_image_tag: str) -> None:
    """Prueba equivalente explicita al WORKDIR/PYTHONPATH efectivo
    dentro de la imagen (no solo al texto del Dockerfile, verificado
    aparte en ``test_p19_docker_infrastructure.py``)."""
    workdir = _run(
        ["docker", "run", "--rm", "--entrypoint", "python", ui_image_tag,
         "-c", "import os; print(os.getcwd())"],
        timeout=20,
    )
    assert workdir.stdout.strip() == "/app"

    pythonpath = _run(
        ["docker", "run", "--rm", "--entrypoint", "python", ui_image_tag,
         "-c", "import os; print(os.environ.get('PYTHONPATH'))"],
        timeout=20,
    )
    assert pythonpath.stdout.strip() == "/app"

    import_check = _run(
        [
            "docker", "run", "--rm", "--entrypoint", "python", ui_image_tag, "-c",
            "import sys; "
            "sys.path = [entry for entry in sys.path if entry != '']; "
            "from src.ui.streamlit_app import run_recommendation_experience; "
            "print('OK')",
        ],
        timeout=20,
    )
    assert import_check.returncode == 0, import_check.stderr
    assert "OK" in import_check.stdout


def test_ui_image_without_pythonpath_reproduces_the_original_defect(tmp_path) -> None:
    """Demuestra que la correccion es la causa, no una coincidencia:
    la MISMA imagen sin ``PYTHONPATH=/app`` reproduce el
    ``ModuleNotFoundError`` original -- ni siquiera llega a construirse,
    porque la comprobacion de importacion del propio Dockerfile ya lo
    impide."""
    dockerfile_text = (_ROOT / "docker/ui/Dockerfile").read_text(encoding="utf-8")
    without_pythonpath = dockerfile_text.replace(
        "    STREAMLIT_SERVER_HEADLESS=true \\\n    PYTHONPATH=/app\n",
        "    STREAMLIT_SERVER_HEADLESS=true\n",
    )
    assert without_pythonpath != dockerfile_text, "no se pudo aislar la linea PYTHONPATH"

    variant_path = tmp_path / "Dockerfile.ui.no_pythonpath"
    variant_path.write_text(without_pythonpath, encoding="utf-8")

    tag = f"tennis-ui-rootcause-check-{uuid.uuid4().hex[:12]}"
    build = _run(
        ["docker", "build", "-f", str(variant_path), "-t", tag, "."],
        timeout=600,
    )
    try:
        assert build.returncode != 0
        combined = build.stdout + build.stderr
        assert "ModuleNotFoundError" in combined
        assert "No module named 'src'" in combined
    finally:
        _run(["docker", "image", "rm", "-f", tag], timeout=60)
