#!/usr/bin/env bash
# P27: arranque local de la demostracion reproducible (API P17 + UI
# P18) sobre la snapshot PUBLICA de demostracion -- nunca la snapshot
# privada real. Pensado para macOS/Linux (POSIX); ver run_demo.ps1
# para Windows/PowerShell.
#
# - Resuelve TODAS las rutas desde la ubicacion de este propio
#   script (raiz del repositorio), nunca desde HOME, el directorio de
#   trabajo actual ni una ruta absoluta fija: funciona igual sin
#   importar el nombre de usuario o donde se haya clonado el repo.
# - Regenera la snapshot demo si aun no existe (nunca toca la
#   snapshot privada real ni ``TENNIS_TACTICAL_SNAPSHOT_PATH`` fuera
#   de este proceso).
# - P17 exige, por contrato de seguridad YA EXISTENTE (no modificado
#   aqui), que la ruta de la snapshot viva FUERA del repositorio
#   (``src/api/runtime.py::_reject_route_inside_repository``): este
#   script copia el archivo demo VERSIONADO a un directorio temporal
#   del sistema antes de arrancar la API, y lo borra al terminar.
#   Nunca apunta ``TENNIS_TACTICAL_SNAPSHOT_PATH`` directamente al
#   archivo dentro de ``demo/``.
# - Arranca la API P17 en segundo plano, espera a /healthz, y luego
#   la UI P18 en primer plano.
# - Ctrl+C (SIGINT) o cualquier salida de este script detiene
#   ordenadamente la API tambien -- nunca deja un proceso residual.
# - No ejecuta ningun analisis real, no lee Parquet/CSV/reports ni
#   el snapshot privado, y no cambia ninguna autorizacion P20-P24.
# - En la primera ejecucion de Streamlit en una maquina, Streamlit
#   pide interactivamente un correo de "onboarding" antes de arrancar;
#   en un script no interactivo esto bloquearia el arranque. Este
#   script crea de antemano (solo si no existe ya) un
#   ``~/.streamlit/credentials.toml`` vacio para omitir esa pregunta,
#   sin tocar ninguna credencial que el usuario ya tuviera.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_SNAPSHOT_SOURCE="$ROOT_DIR/demo/tactical-recommendations-demo-v1.json"
API_HOST="127.0.0.1"
API_PORT="8000"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

if [ ! -f "$DEMO_SNAPSHOT_SOURCE" ]; then
  echo "Generando snapshot de demostración (no existe todavía)..."
  "$PYTHON_BIN" "$ROOT_DIR/scripts/generate_demo_snapshot.py"
fi

# Copia fuera del repositorio (ver nota de contrato arriba): mktemp -d
# crea un directorio en la ubicacion temporal del sistema, nunca
# dentro de $ROOT_DIR.
RUNTIME_DIR="$(mktemp -d)"
RUNTIME_SNAPSHOT="$RUNTIME_DIR/tactical-recommendations-demo-v1.json"
cp "$DEMO_SNAPSHOT_SOURCE" "$RUNTIME_SNAPSHOT"
export TENNIS_TACTICAL_SNAPSHOT_PATH="$RUNTIME_SNAPSHOT"

cd "$ROOT_DIR"

# Evita el prompt interactivo de "onboarding email" de Streamlit en la
# primera ejecucion en esta maquina (ver nota de contrato arriba).
# Nunca sobrescribe unas credenciales ya existentes del usuario.
STREAMLIT_CONFIG_DIR="$HOME/.streamlit"
STREAMLIT_CREDENTIALS="$STREAMLIT_CONFIG_DIR/credentials.toml"
if [ ! -f "$STREAMLIT_CREDENTIALS" ]; then
  mkdir -p "$STREAMLIT_CONFIG_DIR"
  printf '[general]\nemail = ""\n' > "$STREAMLIT_CREDENTIALS"
fi

API_PID=""

cleanup() {
  trap - EXIT INT TERM
  if [ -n "$API_PID" ] && kill -0 "$API_PID" 2>/dev/null; then
    echo ""
    echo "Deteniendo la API de demostración (PID $API_PID)..."
    kill -TERM "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
  fi
  unset TENNIS_TACTICAL_SNAPSHOT_PATH
  if [ -n "${RUNTIME_DIR:-}" ] && [ -d "$RUNTIME_DIR" ]; then
    rm -rf "$RUNTIME_DIR"
  fi
}
trap cleanup EXIT INT TERM

echo "Arrancando la API de demostración en http://$API_HOST:$API_PORT ..."
"$PYTHON_BIN" -m src.api.runtime --host "$API_HOST" --port "$API_PORT" &
API_PID=$!

READY=0
for _ in $(seq 1 40); do
  if "$PYTHON_BIN" -c "
import sys, urllib.request
try:
    with urllib.request.urlopen('http://$API_HOST:$API_PORT/healthz', timeout=1) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "La API de demostración terminó antes de estar lista." >&2
    exit 1
  fi
  sleep 0.5
done

if [ "$READY" -ne 1 ]; then
  echo "La API de demostración no respondió a tiempo en /healthz." >&2
  exit 1
fi

echo "API lista. Arrancando la interfaz Streamlit..."
"$PYTHON_BIN" -m streamlit run "$ROOT_DIR/src/ui/streamlit_app.py" \
  --browser.gatherUsageStats false
