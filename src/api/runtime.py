"""Arranque local productivo P17: API HTTP P12 con el provider persistido P13.

Frontera de arranque explícita para servir ``create_app(provider)`` de P12
sobre el snapshot privado P13 ya generado por el flujo manual P15/P16:

    TENNIS_TACTICAL_SNAPSHOT_PATH=<ruta externa obligatoria>
      -> validacion de ruta cerrada (pre-check sin I/O)
      -> P13 carga y verifica el snapshot EXACTAMENTE UNA VEZ
      -> create_app(provider) de P12
      -> uvicorn loopback, un worker, sin reload

Contrato cerrado del modulo:

- La ruta del snapshot es configuracion OBLIGATORIA mediante la unica
  variable dedicada ``TENNIS_TACTICAL_SNAPSHOT_PATH``: no existe ruta
  por defecto ni flag que la copie a ``argv``; no se imprime, no se
  registra en logs, no se expone en ``app.state``, OpenAPI ni
  respuestas, y no se hardcodea ninguna ruta real.
- Sin I/O al importar el modulo: no se lee ningun snapshot, no se
  conecta ninguna red y no se instancia ningun provider hasta que una
  llamada explícita lo haga.
- La carga P13 es exactamente una vez por arranque; luego el provider
  inmutable se reutiliza para todas las requests (cero carga por
  request).
- No ejecuta P10, P14 ni P15 y no lee Parquet/CSV: la unica fuente de
  datos es el snapshot P13 ya persistido.
- Fallos cerrados: ruta fuera de contrato, snapshot dentro del
  repositorio, snapshot ausente/invalido/incompatible o servidor que
  no puede iniciarse producen salida 1 con mensajes del catalogo
  cerrado, sin traceback, sin exception chaining y sin rutas,
  identidades ni contenido del snapshot. Los errores de uso del CLI
  producen salida 2 con un mensaje cerrado, sin repetir argumentos, y
  sin cargar nada. SIGTERM/SIGINT recibidos mientras Uvicorn sirve se
  normalizan despues de su apagado ordenado a salida 0 sin traceback;
  un ``KeyboardInterrupt`` fuera de esa frontera devuelve 130.
- Uvicorn: host fijo ``127.0.0.1``, puerto decimal estricto
  ``1..65535``, ``workers=1`` inmutable (no existe flag ``--workers``),
  ``reload=False`` inmutable (no existe flag ``--reload``), logging
  cerrado y sanitizado.
- Solo se arranca el servidor despues de que la carga P13 completo con
  exito: ``/healthz`` solo existe cuando el provider ya se cargo.
- Ejecucion local nativa (Windows/macOS/Linux); sin Docker.

Modo contenedor P19 (opcional, server-side, cerrado): la variable
dedicada ``TENNIS_TACTICAL_RUNTIME_MODE`` decide EXCLUSIVAMENTE si el
proceso escucha en ``0.0.0.0`` (el valor literal nunca proviene del
entorno, solo la decision de activarlo). Sin la variable, el
comportamiento es identico al local (host siempre ``127.0.0.1``, sin
excepciones). Con el valor exacto ``container`` el host pasa a
``0.0.0.0`` dentro de la red privada de Compose; ningun flag ``--host``
puede producir ese valor (sigue rechazado por el CLI). Cualquier otro
valor de la variable (vacio, manipulado o desconocido) falla cerrado
antes de cargar el snapshot, con salida 1 y el mismo mensaje cerrado
del catalogo, sin revelar la variable ni su valor.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
from contextlib import contextmanager
from collections.abc import Sequence
from pathlib import Path
from types import FrameType
from typing import Final, Iterator

from fastapi import FastAPI

from src.api.app import create_app
from src.recommender import persisted_tactical_recommendation_provider as p13


RUNTIME_NAME: Final = "tactical_recommendation_api_runtime"
RUNTIME_DEFAULT_HOST: Final = "127.0.0.1"
RUNTIME_DEFAULT_PORT: Final = 8000
RUNTIME_MIN_PORT: Final = 1
RUNTIME_MAX_PORT: Final = 65535
RUNTIME_WORKERS: Final = 1
RUNTIME_RELOAD: Final = False
RUNTIME_SNAPSHOT_PATH_ENV: Final = "TENNIS_TACTICAL_SNAPSHOT_PATH"
RUNTIME_MODE_ENV: Final = "TENNIS_TACTICAL_RUNTIME_MODE"
RUNTIME_CONTAINER_MODE_VALUE: Final = "container"
RUNTIME_CONTAINER_HOST: Final = "0.0.0.0"

# Mismo criterio de ruta absoluta que el contrato P13 (POSIX, unidad
# Windows, UNC); sin divergencia.
_ABSOLUTE_PATH: Final = re.compile(
    r"^(/|[A-Za-z]:[\\/]|\\\\)"
)

_RUNTIME_MESSAGES: Final = {
    "route_rejected": (
        "Arranque P17 rechazado: ruta de snapshot fuera del contrato cerrado."
    ),
    "route_in_repository": (
        "Arranque P17 rechazado: el snapshot debe vivir fuera del repositorio."
    ),
    "snapshot_load_failed": (
        "Arranque P17 fallido: snapshot ausente, invalido o incompatible "
        "(cierre cerrado, sin detalles)."
    ),
    "server_unavailable": (
        "Arranque P17 fallido: el servidor local no pudo iniciarse "
        "(cierre cerrado, sin detalles)."
    ),
    "cli_usage_error": (
        "Uso P17 invalido: argumentos rechazados por el contrato cerrado."
    ),
    "runtime_mode_rejected": (
        "Arranque P17 rechazado: configuracion de entorno fuera del "
        "contrato cerrado (cierre cerrado, sin detalles)."
    ),
}


class RuntimePathContractError(RuntimeError):
    """Rechazo cerrado del pre-check de ruta (mensaje del catalogo)."""

    __slots__ = ()

    def __init__(self, message: str) -> None:
        super().__init__(message)


class RuntimeModeContractError(RuntimeError):
    """Rechazo cerrado del modo de entorno P19 (mensaje del catalogo)."""

    __slots__ = ()

    def __init__(self, message: str) -> None:
        super().__init__(message)


class _ManagedServerSignal(BaseException):
    """Senal de apagado ya gestionada por Uvicorn, sin datos sensibles."""

    __slots__ = ()


class _ClosedArgumentParser(argparse.ArgumentParser):
    """Argparse con salida 2 fija que nunca repite argumentos privados."""

    def error(self, message: str) -> None:
        del message
        self.exit(2, f'{_RUNTIME_MESSAGES["cli_usage_error"]}\n')


def _repository_root() -> Path:
    """Resuelve la raiz solo durante una llamada explicita, nunca al importar."""
    return Path(__file__).resolve().parents[2]


def _closed_logging_config() -> dict[str, object]:
    """Crea una configuracion nueva sin access log ni estado global mutable."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "closed": {"format": "%(levelname)s %(name)s: %(message)s"},
        },
        "handlers": {
            "closed": {
                "class": "logging.StreamHandler",
                "formatter": "closed",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            # Ninguna URL, query ni path solicitado debe llegar al log.
            "uvicorn.access": {
                "handlers": [],
                "level": "CRITICAL",
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": [],
                "level": "CRITICAL",
                "propagate": False,
            },
            "uvicorn": {
                "handlers": [],
                "level": "CRITICAL",
                "propagate": False,
            },
            # Los service events sanitizados de P12 pasan por el canal
            # cerrado sin incluir inputs individuales.
            "tactical_recommendation_api": {
                "handlers": ["closed"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }


def _raise_managed_server_signal(
    signum: int,
    frame: FrameType | None,
) -> None:
    """Convierte la reemision de Uvicorn en una salida normal cerrada."""
    del signum, frame
    raise _ManagedServerSignal


@contextmanager
def _normalize_server_signals() -> Iterator[None]:
    """Normaliza SIGTERM/SIGINT que Uvicorn reemite tras cerrar.

    Uvicorn conserva estos handlers, instala los suyos mientras sirve,
    restaura los nuestros tras el apagado ordenado y reemite las
    senales capturadas. La excepcion privada impide que el handler por
    defecto termine el proceso con 128+senal; ``main`` la transforma
    en salida 0. Los handlers originales siempre se restauran.
    """
    managed = (signal.SIGTERM, signal.SIGINT)
    originals: dict[signal.Signals, object] = {}
    try:
        for managed_signal in managed:
            originals[managed_signal] = signal.getsignal(managed_signal)
            signal.signal(managed_signal, _raise_managed_server_signal)
        yield
    finally:
        for managed_signal, original in originals.items():
            signal.signal(managed_signal, original)


def _serve_app(app: FastAPI, host: str, port: int) -> None:
    """Importa Uvicorn solo al servir y mantiene su contrato cerrado."""
    import uvicorn

    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=RUNTIME_WORKERS,
        reload=RUNTIME_RELOAD,
        access_log=False,
        log_config=_closed_logging_config(),
    )


def _route_raw_text(value: object) -> str:
    """Normaliza el tipo de entrada sin ejecutar PathLike arbitrarios."""
    if type(value) is str:
        return value
    if isinstance(value, Path):
        return str(value)
    raise RuntimePathContractError(_RUNTIME_MESSAGES["route_rejected"])


def _reject_route_outside_contract(raw: str) -> None:
    """Pre-check textu SIN I/O; la verificacion completa la hace P13.

    Mismo contrato que la validacion de ruta P13 (sin divergencia):
    texto no vacio, sin espacios extremos, absoluto, sin NUL, sin URI
    esquemas, sin tilde, sin ``//`` inicial y sin segmentos ``..``.
    Los demas rechazos (no regular, symlink/junction, tamano, JSON
    hostil, integridad, upstream) los impone P13 durante la carga,
    con sus errores cerrados propios.
    """
    normalized = raw.replace("\\", "/")
    if (
        not raw
        or raw != raw.strip()
        or "\x00" in raw
        or "://" in raw
        or raw.startswith("~")
        or raw.startswith("//")
        or _ABSOLUTE_PATH.match(raw) is None
        or any(part == ".." for part in normalized.split("/"))
    ):
        raise RuntimePathContractError(_RUNTIME_MESSAGES["route_rejected"])


def _reject_route_inside_repository(raw: str) -> None:
    """El snapshot debe vivir fuera del repositorio (texto o resuelto)."""
    textual = Path(raw)
    repository_root = _repository_root()
    if textual.is_relative_to(repository_root):
        raise RuntimePathContractError(_RUNTIME_MESSAGES["route_in_repository"])
    try:
        resolved = textual.resolve()
    except (OSError, RuntimeError, ValueError):
        raise RuntimePathContractError(_RUNTIME_MESSAGES["route_rejected"]) from None
    if resolved.is_relative_to(repository_root):
        raise RuntimePathContractError(_RUNTIME_MESSAGES["route_in_repository"])


def _strict_host(value: str) -> str:
    if value != RUNTIME_DEFAULT_HOST:
        raise argparse.ArgumentTypeError("host debe ser 127.0.0.1 (loopback).")
    return value


def _strict_port(value: str) -> int:
    closed = (
        f"port debe ser un entero decimal estricto "
        f"{RUNTIME_MIN_PORT}-{RUNTIME_MAX_PORT}."
    )
    if not value or not value.isdigit() or value.startswith("0") or len(value) > 5:
        raise argparse.ArgumentTypeError(closed)
    port = int(value)
    if not RUNTIME_MIN_PORT <= port <= RUNTIME_MAX_PORT:
        raise argparse.ArgumentTypeError(closed)
    return port


def _parse_cli(argv: Sequence[str] | None) -> tuple[str, str, int]:
    """Valida el uso del CLI (salida 2 argparse) sin cargar nada."""
    parser = _ClosedArgumentParser(
        prog="tactical-recommendation-api-runtime",
        description=(
            "P17: arranque local productivo de la API P12 sobre el "
            "snapshot privado P13 (loopback 127.0.0.1, un worker, sin "
            "reload, una sola carga del snapshot)."
        ),
    )
    parser.add_argument(
        "--host",
        default=RUNTIME_DEFAULT_HOST,
        type=_strict_host,
        help="Solo 127.0.0.1 (loopback).",
    )
    parser.add_argument(
        "--port",
        default=str(RUNTIME_DEFAULT_PORT),
        type=_strict_port,
        help=f"Entero decimal estricto {RUNTIME_MIN_PORT}-{RUNTIME_MAX_PORT}.",
    )
    arguments = parser.parse_args(argv)
    raw = os.environ.get(RUNTIME_SNAPSHOT_PATH_ENV)
    if type(raw) is not str or not raw:
        parser.error("snapshot environment missing")
    return raw, arguments.host, arguments.port


def _resolve_serve_host(cli_host: str) -> str:
    """Selecciona el host de escucha real segun el modo de entorno cerrado.

    Sin la variable dedicada, el host es exactamente el validado por CLI
    (siempre ``127.0.0.1``): comportamiento local identico al previo a
    P19. Con el valor exacto del modo contenedor, el host pasa a
    ``0.0.0.0`` SOLO dentro de esta funcion -- el literal nunca proviene
    del entorno, solo la decision binaria de activarlo, por lo que el
    usuario no puede introducir un host arbitrario via configuracion.
    Cualquier otro valor (vacio, manipulado o desconocido) falla cerrado.
    """
    raw_mode = os.environ.get(RUNTIME_MODE_ENV)
    if raw_mode is None:
        return cli_host
    if raw_mode == RUNTIME_CONTAINER_MODE_VALUE:
        return RUNTIME_CONTAINER_HOST
    raise RuntimeModeContractError(_RUNTIME_MESSAGES["runtime_mode_rejected"])


def build_app_from_snapshot(snapshot_path: object) -> FastAPI:
    """Construye la app P12 cargando el snapshot P13 exactamente una vez.

    Funcion pura sobre la ruta explicita: valida el texto de la ruta
    (sin I/O), rechaza rutas dentro del repositorio, delega la carga y
    verificacion completa a la frontera publica P13
    (``create_persisted_tactical_recommendation_provider``) y
    devuelve ``create_app(provider)`` de P12. Ningun objeto de estado
    global: la ruta solo existe dentro de esta llamada y no se
    conserva en ``app.state`` ni en el OpenAPI.
    """
    raw = _route_raw_text(snapshot_path)
    _reject_route_outside_contract(raw)
    _reject_route_inside_repository(raw)
    provider = p13.create_persisted_tactical_recommendation_provider(raw)
    return create_app(provider)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI de arranque: valida, carga una vez y sirve; salidas cerradas.

    2 = error de uso (argparse), sin cargar nada; 1 = snapshot fuera
    de contrato o servidor no iniciable (mensaje cerrado del
    catalogo, sin traceback ni chaining); 0 = servidor terminado de
    forma normal.
    """
    snapshot_raw, host, port = _parse_cli(argv)
    try:
        serve_host = _resolve_serve_host(host)
    except RuntimeModeContractError:
        raise SystemExit(_RUNTIME_MESSAGES["runtime_mode_rejected"]) from None
    try:
        app = build_app_from_snapshot(snapshot_raw)
    except KeyboardInterrupt:
        return 130
    except Exception:
        raise SystemExit(_RUNTIME_MESSAGES["snapshot_load_failed"]) from None
    try:
        with _normalize_server_signals():
            _serve_app(app, serve_host, port)
    except _ManagedServerSignal:
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception:
        raise SystemExit(_RUNTIME_MESSAGES["server_unavailable"]) from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main()) from None


__all__ = (
    "RUNTIME_CONTAINER_HOST",
    "RUNTIME_CONTAINER_MODE_VALUE",
    "RUNTIME_DEFAULT_HOST",
    "RUNTIME_DEFAULT_PORT",
    "RUNTIME_MAX_PORT",
    "RUNTIME_MIN_PORT",
    "RUNTIME_MODE_ENV",
    "RUNTIME_NAME",
    "RUNTIME_RELOAD",
    "RUNTIME_SNAPSHOT_PATH_ENV",
    "RUNTIME_WORKERS",
    "RuntimeModeContractError",
    "RuntimePathContractError",
    "build_app_from_snapshot",
    "main",
)
