"""Pruebas estructurales P27 de los scripts de arranque de la
demostración (``scripts/run_demo.sh`` y ``scripts/run_demo.ps1``).

Ningún test de este archivo EJECUTA los scripts (arrancar un servidor
real desde una suite de pytest es fragil y pesado): se verifican
propiedades estaticas del texto -- resolucion de rutas relativa al
propio script, ausencia de rutas absolutas o de usuario fijas, gestion
de Ctrl+C, y que jamas invocan un analisis real ni tocan el snapshot
privado o los gates P20-P24.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SH = _ROOT / "scripts" / "run_demo.sh"
_PS1 = _ROOT / "scripts" / "run_demo.ps1"

_FORBIDDEN_TOKENS = (
    "TENNIS_TACTICAL_SNAPSHOT_PATH=/",  # nunca una ruta absoluta fija
    "points_enriched",
    "REAL_TEST_EVALUATION_AUTHORIZED",
    "final_sealed_evaluation",
    "/home/",
    "C:\\Users\\",
)


def test_both_scripts_exist() -> None:
    assert _SH.is_file()
    assert _PS1.is_file()


def test_sh_script_has_bash_shebang_and_strict_mode() -> None:
    text = _SH.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text


def test_sh_script_resolves_root_from_its_own_location() -> None:
    text = _SH.read_text(encoding="utf-8")
    assert 'ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"' in text


def test_ps1_script_resolves_root_from_its_own_location() -> None:
    text = _PS1.read_text(encoding="utf-8")
    assert "$RootDir = Split-Path -Parent $PSScriptRoot" in text


def test_sh_script_configures_demo_snapshot_env_var() -> None:
    text = _SH.read_text(encoding="utf-8")
    assert 'export TENNIS_TACTICAL_SNAPSHOT_PATH="$RUNTIME_SNAPSHOT"' in text
    assert (
        'DEMO_SNAPSHOT_SOURCE="$ROOT_DIR/demo/tactical-recommendations-demo-v1.json"'
        in text
    )


def test_sh_script_copies_snapshot_outside_the_repository_before_use() -> None:
    """P17 exige (contrato ya existente, no modificado) que la ruta de
    la snapshot viva fuera del repositorio: el script debe copiar el
    archivo versionado a un directorio temporal del sistema antes de
    apuntar la variable de entorno ahi, nunca al archivo dentro de
    ``demo/`` directamente."""
    text = _SH.read_text(encoding="utf-8")
    assert 'RUNTIME_DIR="$(mktemp -d)"' in text
    assert 'cp "$DEMO_SNAPSHOT_SOURCE" "$RUNTIME_SNAPSHOT"' in text
    assert 'rm -rf "$RUNTIME_DIR"' in text


def test_ps1_script_configures_demo_snapshot_env_var() -> None:
    text = _PS1.read_text(encoding="utf-8")
    assert "$env:TENNIS_TACTICAL_SNAPSHOT_PATH = $RuntimeSnapshot" in text
    assert (
        '$DemoSnapshotSource = Join-Path $RootDir "demo\\tactical-recommendations-demo-v1.json"'
        in text
    )


def test_ps1_script_copies_snapshot_outside_the_repository_before_use() -> None:
    """Mismo contrato P17 que la version POSIX: la ruta debe vivir
    fuera del repositorio, nunca dentro de ``demo/`` directamente."""
    text = _PS1.read_text(encoding="utf-8")
    assert "[System.IO.Path]::GetTempPath()" in text
    assert "Copy-Item -Path $DemoSnapshotSource -Destination $RuntimeSnapshot" in text
    assert "Remove-Item -Path $RuntimeDir -Recurse -Force" in text


def test_sh_script_preseeds_streamlit_credentials_to_avoid_interactive_prompt() -> None:
    """Streamlit pide un correo de onboarding interactivamente en la
    primera ejecucion en una maquina; en un script no interactivo esto
    bloquearia el arranque. Se crea un credentials.toml vacio de
    antemano, solo si no existe ya."""
    text = _SH.read_text(encoding="utf-8")
    assert 'STREAMLIT_CREDENTIALS="$STREAMLIT_CONFIG_DIR/credentials.toml"' in text
    assert 'if [ ! -f "$STREAMLIT_CREDENTIALS" ]; then' in text


def test_ps1_script_preseeds_streamlit_credentials_to_avoid_interactive_prompt() -> None:
    text = _PS1.read_text(encoding="utf-8")
    assert "$StreamlitCredentials = Join-Path $StreamlitConfigDir \"credentials.toml\"" in text
    assert "if (-not (Test-Path $StreamlitCredentials)) {" in text


def test_sh_script_starts_api_and_ui() -> None:
    text = _SH.read_text(encoding="utf-8")
    assert "src.api.runtime" in text
    assert "streamlit run" in text


def test_ps1_script_starts_api_and_ui() -> None:
    text = _PS1.read_text(encoding="utf-8")
    assert "src.api.runtime" in text
    assert "streamlit run" in text


def test_sh_script_handles_sigint_and_cleans_up() -> None:
    text = _SH.read_text(encoding="utf-8")
    assert "trap cleanup EXIT INT TERM" in text
    assert "kill -TERM \"$API_PID\"" in text


def test_ps1_script_handles_ctrl_c_and_cleans_up() -> None:
    text = _PS1.read_text(encoding="utf-8")
    assert "finally" in text
    assert "Stop-Process -Id $apiProcess.Id -Force" in text


def test_sh_script_waits_for_healthz_before_starting_ui() -> None:
    text = _SH.read_text(encoding="utf-8")
    assert "/healthz" in text


def test_ps1_script_waits_for_healthz_before_starting_ui() -> None:
    text = _PS1.read_text(encoding="utf-8")
    assert "/healthz" in text


def test_neither_script_hardcodes_a_username_or_fixed_home() -> None:
    """``$HOME`` (bash) es la variable automatica del sistema resuelta
    en tiempo de ejecucion para quien sea que ejecute el script -- se
    usa deliberadamente para ubicar ``~/.streamlit/credentials.toml``
    sin fijar una ruta concreta. Lo que esta prohibido es un valor de
    usuario o de HOME *fijo* incrustado en el script."""
    for path in (_SH, _PS1):
        text = path.read_text(encoding="utf-8")
        assert "Errandi" not in text
        assert "%USERPROFILE%" not in text
        assert "$env:USERPROFILE" not in text
        assert "/home/" not in text
        assert "C:\\Users\\" not in text


def test_neither_script_contains_forbidden_tokens() -> None:
    for path in (_SH, _PS1):
        text = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_TOKENS:
            assert token not in text, f"{path.name}: token prohibido {token!r}"


def test_neither_script_invokes_the_sealed_evaluator() -> None:
    for path in (_SH, _PS1):
        text = path.read_text(encoding="utf-8")
        assert "final_sealed_evaluation_runner" not in text
        assert "run_real_test_evaluation" not in text


def test_sh_script_generates_snapshot_only_if_missing() -> None:
    text = _SH.read_text(encoding="utf-8")
    assert 'if [ ! -f "$DEMO_SNAPSHOT_SOURCE" ]; then' in text


def test_ps1_script_generates_snapshot_only_if_missing() -> None:
    text = _PS1.read_text(encoding="utf-8")
    assert "if (-not (Test-Path $DemoSnapshotSource)) {" in text
