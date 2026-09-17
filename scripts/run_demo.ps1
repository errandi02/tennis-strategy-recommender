# P27: arranque local de la demostracion reproducible (API P17 + UI
# P18) sobre la snapshot PUBLICA de demostracion -- nunca la privada
# real. Ver run_demo.sh para macOS/Linux.
#
# - Resuelve TODAS las rutas desde la ubicacion de este propio script
#   (raiz del repositorio), nunca desde un nombre de usuario ni una
#   ruta absoluta fija.
# - Regenera la snapshot demo si aun no existe.
# - P17 exige, por contrato de seguridad YA EXISTENTE (no modificado
#   aqui), que la ruta de la snapshot viva FUERA del repositorio
#   (``src/api/runtime.py::_reject_route_inside_repository``): este
#   script copia el archivo demo VERSIONADO a un directorio temporal
#   del sistema antes de arrancar la API, y lo borra al terminar.
# - Arranca la API P17 en segundo plano, espera a /healthz, y luego la
#   UI P18 en primer plano.
# - Ctrl+C, o cualquier salida del script, detiene ordenadamente la
#   API (bloque `finally`) -- nunca deja un proceso residual.
# - No ejecuta ningun analisis real, no lee Parquet/CSV/reports ni el
#   snapshot privado, y no cambia ninguna autorizacion P20-P24.
# - En la primera ejecucion de Streamlit en una maquina, Streamlit
#   pide interactivamente un correo de "onboarding" antes de arrancar;
#   este script crea de antemano (solo si no existe ya) un
#   credentials.toml vacio para omitir esa pregunta, sin tocar ninguna
#   credencial que el usuario ya tuviera.

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$DemoSnapshotSource = Join-Path $RootDir "demo\tactical-recommendations-demo-v1.json"
$ApiHostName = "127.0.0.1"
$ApiPort = "8000"

if (-not (Test-Path $DemoSnapshotSource)) {
    Write-Host "Generando snapshot de demostracion (no existe todavia)..."
    python (Join-Path $RootDir "scripts\generate_demo_snapshot.py")
    if ($LASTEXITCODE -ne 0) {
        Write-Error "No se pudo generar la snapshot de demostracion."
        exit 1
    }
}

# Copia fuera del repositorio (ver nota de contrato arriba): un
# directorio nuevo bajo la ruta temporal del sistema, nunca dentro de
# $RootDir.
$RuntimeDir = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
$RuntimeSnapshot = Join-Path $RuntimeDir "tactical-recommendations-demo-v1.json"
Copy-Item -Path $DemoSnapshotSource -Destination $RuntimeSnapshot -Force
$env:TENNIS_TACTICAL_SNAPSHOT_PATH = $RuntimeSnapshot

Set-Location $RootDir

# Evita el prompt interactivo de "onboarding email" de Streamlit en la
# primera ejecucion en esta maquina (ver nota de contrato arriba).
# Nunca sobrescribe unas credenciales ya existentes del usuario.
$StreamlitConfigDir = Join-Path $HOME ".streamlit"
$StreamlitCredentials = Join-Path $StreamlitConfigDir "credentials.toml"
if (-not (Test-Path $StreamlitCredentials)) {
    New-Item -ItemType Directory -Path $StreamlitConfigDir -Force | Out-Null
    Set-Content -Path $StreamlitCredentials -Value "[general]`nemail = `"`"" -Encoding utf8
}

Write-Host "Arrancando la API de demostracion en http://${ApiHostName}:${ApiPort} ..."
$apiProcess = Start-Process -FilePath "python" `
    -ArgumentList @("-m", "src.api.runtime", "--host", $ApiHostName, "--port", $ApiPort) `
    -PassThru -NoNewWindow

$ready = $false
try {
    for ($i = 0; $i -lt 40; $i++) {
        if ($apiProcess.HasExited) {
            Write-Error "La API de demostracion termino antes de estar lista."
            exit 1
        }
        try {
            $response = Invoke-WebRequest -Uri "http://${ApiHostName}:${ApiPort}/healthz" `
                -UseBasicParsing -TimeoutSec 1
            if ($response.StatusCode -eq 200) {
                $ready = $true
                break
            }
        } catch {
            # Todavia no esta lista; se reintenta tras la espera.
        }
        Start-Sleep -Milliseconds 500
    }

    if (-not $ready) {
        Write-Error "La API de demostracion no respondio a tiempo en /healthz."
        exit 1
    }

    Write-Host "API lista. Arrancando la interfaz Streamlit..."
    python -m streamlit run (Join-Path $RootDir "src\ui\streamlit_app.py") `
        --browser.gatherUsageStats false
}
finally {
    if (-not $apiProcess.HasExited) {
        Write-Host ""
        Write-Host "Deteniendo la API de demostracion (PID $($apiProcess.Id))..."
        Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Remove-Item Env:\TENNIS_TACTICAL_SNAPSHOT_PATH -ErrorAction SilentlyContinue
    if (Test-Path $RuntimeDir) {
        Remove-Item -Path $RuntimeDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
