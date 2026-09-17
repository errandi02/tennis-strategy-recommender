"""Prueba estructural P27: el flujo Docker documentado para la
demostración (apuntar ``TENNIS_TACTICAL_SNAPSHOT_HOST_PATH`` a la
snapshot demo real) resuelve correctamente con ``docker compose
config`` -- misma autoridad independiente que ``test_p19_docker_
infrastructure.py``, sin construir imagenes ni arrancar contenedores.

Requiere el binario ``docker`` (con el plugin compose); si no está
disponible, el módulo se omite explícitamente.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.generate_demo_snapshot import DEMO_SNAPSHOT_PATH

_ROOT = Path(__file__).resolve().parents[1]
_DOCKER_AVAILABLE = shutil.which("docker") is not None

pytestmark = pytest.mark.skipif(
    not _DOCKER_AVAILABLE,
    reason="El binario docker (con el plugin compose) no esta disponible.",
)


def test_compose_config_resolves_with_the_real_demo_snapshot_path() -> None:
    assert DEMO_SNAPSHOT_PATH.is_file(), (
        "Ejecuta scripts/generate_demo_snapshot.py antes de este test."
    )
    full_env = dict(os.environ)
    full_env["TENNIS_TACTICAL_SNAPSHOT_HOST_PATH"] = str(DEMO_SNAPSHOT_PATH)
    completed = subprocess.run(
        ["docker", "compose", "config", "--format", "json"],
        cwd=str(_ROOT), env=full_env, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    config = json.loads(completed.stdout)
    api_volumes = config["services"]["api"]["volumes"]
    assert len(api_volumes) == 1
    assert api_volumes[0]["source"] == str(DEMO_SNAPSHOT_PATH)
    assert api_volumes[0]["target"] == "/snapshot/tactical-recommendation-snapshot.json"
    assert api_volumes[0]["read_only"] is True
    # La demostracion no cambia NADA de la configuracion ya auditada en
    # P19: sin puerto publicado en api, sin variable de snapshot en ui.
    assert "ports" not in config["services"]["api"]
    assert "TENNIS_TACTICAL_SNAPSHOT_PATH" not in config["services"]["ui"].get(
        "environment", {}
    )
