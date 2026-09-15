"""Preflight P15: capa fina de orquestacion para la unica ejecucion manual.

Prepara la integracion offline completa sin ejecutar datos reales:

    P10 -> 3.610 resultados target-level -> P10OfflineSnapshotRecord
        -> generador P14 -> snapshot privado P13 -> (futuro) provider P13
        -> servicio/API P12

P16 valido la capacidad del snapshot P13 (``MAX_SNAPSHOT_BYTES`` =
256 MiB con la base representativa de 3.610 entradas cubierta con
margen). El primer intento real fue interrumpido manualmente durante
``p10_pipeline`` (sin snapshot, sin reintentos, log cerrado con
status ``interrupted``); la segunda ejecucion termino correctamente,
persistio y verifico 3.610 entradas y cerro toda autorizacion futura.
La politica permanece
``single_manual_execution_without_automatic_retry`` y
``AUTOMATIC_RETRY`` en ``False``. No existe ninguna ejecucion adicional
autorizada. Sin ejecucion, esta capa nunca lee Parquet/CSV, nunca
accede a ``data/``, nunca escribe en el repositorio y no genera el
snapshot real. La autorizacion de la
generacion del snapshot la posee exclusivamente P15. La ruta
productiva reutiliza sin modificar: ``compute_tactical_pipeline_result``
P10 (frontera compute-only: lectura contractual unica, sin
publicacion, sin performance log, sin dependencia de las
autorizaciones historicas P10), ``validate_tactical_pipeline_result``,
la proyeccion de ``TacticalTargetResult`` a ``P10OfflineSnapshotRecord``
y el generador/persistidor P14 delegado
(``generate_and_persist_...``).

Auditoria contractual (evidencia en el codigo P10):

- Los 3.610 targets existen solo en memoria dentro de
  ``TacticalPipelineResult.target_results``; P10 publica cinco
  agregados sin identidades y los resultados se descartan.
- ``TacticalPipelineTarget`` porta los seis campos de identidad
  (target_match_id, player, opponent, as_of_date, fold, orientation);
  la proyeccion a ``P10OfflineSnapshotRecord`` es por copia de campos,
  sin inferencias ni mapeos.
- ``compute_tactical_pipeline_result`` es la unica frontera P15 usa
  contra P10: valida la fuente contractual, verifica la linea de
  sangre upstream, lee la fuente una vez y devuelve el resultado
  validado. No publica, no escribe logs y no depende de
  ``REAL_EXECUTION_AUTHORIZED`` / ``FURTHER_REAL_EXECUTION_AUTHORIZED``;
  la autorizacion y la persistencia pertenecen al consumidor (P15).
- ``validate_tactical_pipeline_result`` con la configuracion real ya
  obliga la poblacion target congelada (1.805 partidas, 3.610
  orientaciones, pares reciprocos, folds 168/371/646/620); el modo
  ``p10_offline`` de P14 re-verifica todo a nivel de records sin
  dependencias cruzadas.
- La frontera historica ``execute_authorized_real_pipeline`` conserva
  su comportamiento exacto (autorizacion historica, publicacion,
  performance log, fallos) y delega su calculo en la misma frontera
  compute-only.

El performance log P15 es externo, atomico (mkstemp + os.replace),
compacto, sin PII, sin rutas, sin timestamps civiles, sin trazas y sin
NaN/Infinity. No existe ningun log P10 derivado: la frontera
compute-only no escribe logs. Cero reintentos: cada fallo cierra el
log con status/reason cerrados y no publica snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import argparse
from collections.abc import Callable
import json
import os
from pathlib import Path, PurePath
import re
from tempfile import mkstemp
from time import perf_counter
from types import MappingProxyType
from typing import Final

from src.analysis import tactical_recommender_pipeline as p10
from src.analysis import tactical_recommendation_snapshot_generator as p14
from src.recommender import persisted_tactical_recommendation_provider as p13


PIPELINE_ANALYSIS_NAME: Final = "tactical_recommendation_snapshot_pipeline"
PIPELINE_CONTRACT_NAME: Final = "tactical_recommendation_snapshot_pipeline"
PIPELINE_SCHEMA_VERSION: Final = "1.0.0"

# El primer intento real fue interrumpido y el segundo termino
# correctamente. La autorizacion queda cerrada definitivamente: no
# existe bypass por entorno, flag CLI, fuente alternativa, reintento
# ni segunda constante.
REAL_EXECUTION_AUTHORIZED: Final = False
REAL_EXECUTION_AUTHORIZATION_REASON: Final = (
    "real_snapshot_generation_completed_no_further_execution_authorized"
)
REAL_EXECUTION_BLOCK_REASON_CODE: Final = (
    "real_snapshot_generation_completed_no_further_execution_authorized"
)
AUTOMATIC_RETRY: Final = False
SINGLE_MANUAL_EXECUTION_POLICY: Final = (
    "single_manual_execution_without_automatic_retry"
)
# Historial real cerrado de P16: primer intento interrumpido, segundo
# completado, sin reintentos automaticos.
PREVIOUS_REAL_P15_ATTEMPTS: Final = 2
COMPLETED_REAL_EXECUTIONS: Final = 1
INTERRUPTED_REAL_EXECUTIONS: Final = 1
AUTOMATIC_RETRIES_PERFORMED: Final = 0
SECOND_REAL_EXECUTION_STATUS: Final = "completed"
SECOND_REAL_EXECUTION_ELAPSED_SECONDS: Final = 5554.413477875001
SECOND_REAL_EXECUTION_OPERATION_COUNTS: Final = (
    ("p10_executions", 1),
    ("targets_projected", 3610),
    ("generation_calls", 1),
    ("persistence_calls", 1),
    ("verification_calls", 1),
)
PERSISTED_SNAPSHOT_ENTRIES: Final = 3610
PERSISTED_SNAPSHOT_BYTES: Final = 256_962_392
PERSISTED_SNAPSHOT_CAPACITY_MARGIN_BYTES: Final = 11_473_064
PERSISTED_SNAPSHOT_SHA256: Final = (
    "C2C453A4FF0A89CCB8A895C77637D724F5DF06C2835527DC267A93026122EFB0"
)
POST_EXECUTION_CLOSURE_RULE: Final = (
    "La segunda ejecucion real termino correctamente y "
    "REAL_EXECUTION_AUTHORIZED permanece en False; P10 conserva "
    "permanentemente "
    "REAL_EXECUTION_AUTHORIZED = False y FURTHER_REAL_EXECUTION_AUTHORIZED "
    "= False. No existe ninguna ejecucion adicional autorizada y hubo cero "
    "reintentos automaticos."
)
REAL_SNAPSHOT_EXECUTION_BLOCK_REASON: Final = (
    "Ejecucion real bloqueada: snapshot generado y verificado; no existe "
    "autorizacion adicional "
    "(real_snapshot_generation_completed_no_further_execution_authorized)."
)

PIPELINE_STAGES: Final = (
    "preflight",
    "p10_pipeline",
    "target_projection",
    "snapshot_generation",
    "persistence",
    "verification",
)

PIPELINE_EXECUTION_STATUSES: Final = frozenset(
    {"running_preflight", "running", "completed", "failed", "interrupted"}
)
PIPELINE_RESULT_STATUSES: Final = frozenset(
    {"completed", "failed", "interrupted"}
)

PIPELINE_OPERATION_COUNTER_FIELDS: Final = (
    "p10_executions",
    "targets_projected",
    "generation_calls",
    "persistence_calls",
    "verification_calls",
)

PIPELINE_RECONCILIATION_KEYS: Final = (
    "p10_result_validated",
    "targets_projected",
    "p14_universe_reconciled",
    "snapshot_persisted_and_verified",
)

PIPELINE_REASON_CODES: Final = frozenset(
    {
        "real_snapshot_generation_completed_no_further_execution_authorized",
        "path_contract_violation",
        "p10_pipeline_execution_failed",
        "p10_result_not_pipeline_result",
        "p10_result_invalid",
        "p10_seal_violation",
        "target_projection_failed",
        "snapshot_generation_failed",
        "persistence_failed",
        "verification_failed",
        "manual_interrupt",
        "p15_execution_error",
    }
)

_ERROR_MESSAGES: Final = MappingProxyType(
    {
        "real_snapshot_generation_completed_no_further_execution_authorized": (
            "P15 bloqueado: snapshot completado sin ejecuciones adicionales."
        ),
        "path_contract_violation": (
            "Ruta privada fuera del contrato P15 (no se revela)."
        ),
        "p10_pipeline_execution_failed": "Ejecucion P10 fallida (cierre cerrado).",
        "p10_result_not_pipeline_result": (
            "P10 devolvio un cierre no disponible; sin snapshot."
        ),
        "p10_result_invalid": "Resultado P10 no valido contra su contrato.",
        "p10_seal_violation": "Sellado P10 no conserva su metadata contractual.",
        "target_projection_failed": "Proyeccion target a record fallida.",
        "snapshot_generation_failed": "Generacion P14 fallida; sin snapshot.",
        "persistence_failed": "Persistencia P13 delegada fallida.",
        "verification_failed": "Verificacion P13 delegada fallida.",
        "manual_interrupt": "Interruccion manual del operador.",
        "p15_execution_error": "Fallo inesperado P15 (mensaje cerrado).",
    }
)

_SEALED_FIRST_DAY: Final = p14.SEALED_TEST_FIRST_DAY
_P10_FOLDS: Final = p10.VALIDATION_FOLDS
_P10_ORIENTATIONS: Final = frozenset(
    {"player_1_vs_player_2", "player_2_vs_player_1"}
)
_POSIX_PATH: Final = re.compile(
    r"^/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+$"
)
_WINDOWS_DRIVE_PATH: Final = re.compile(
    r"^[A-Za-z]:[\\/][A-Za-z0-9._-]+(?:[\\/][A-Za-z0-9._-]+)*$"
)
_WINDOWS_UNC_PATH: Final = re.compile(
    r"^\\\\[A-Za-z0-9._-]+[\\/][A-Za-z0-9._-]+(?:[\\/][A-Za-z0-9._-]+)*$"
)
_SAFE_TOKEN: Final = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_IS_WINDOWS: Final = os.name == "nt"
_PROGRESS_STEP: Final = 400


class TacticalRecommendationSnapshotPipelineError(RuntimeError):
    """Error contractual P15 cerrado (stage + reason code)."""

    __slots__ = ("stage", "reason_code")

    def __init__(self, stage: str, reason_code: str) -> None:
        if stage not in PIPELINE_STAGES:
            stage = "preflight"
        if reason_code not in PIPELINE_REASON_CODES:
            reason_code = "p15_execution_error"
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "reason_code", reason_code)
        super().__init__(_ERROR_MESSAGES[reason_code])


def _fail(
    stage: str, reason_code: str
) -> TacticalRecommendationSnapshotPipelineError:
    return TacticalRecommendationSnapshotPipelineError(stage, reason_code)


def _is_posix_private_path(raw: object) -> bool:
    """Validacion textual POSIX cerrada (sin materializar)."""
    if type(raw) is not str or not 0 < len(raw) <= 4096:
        return False
    if not all(0x20 <= ord(character) < 0x7F for character in raw):
        return False
    if _POSIX_PATH.fullmatch(raw) is None:
        return False
    return all(part not in (".", "..") for part in raw.split("/"))


def _is_windows_private_path(raw: object) -> bool:
    """Validacion textual Windows cerrada (sin materializar).

    Admite absoluto con unidad (``C:\\...`` o ``C:/...``) y UNC
    (``\\\\servidor\\compartida\\...``); rechaza relativas,
    drive-relative, URI, NUL, tilde, traversal y componentes fuera
    del conjunto textual cerrado.
    """
    if type(raw) is not str or not 0 < len(raw) <= 4096:
        return False
    if not all(0x20 <= ord(character) < 0x7F for character in raw):
        return False
    if _WINDOWS_DRIVE_PATH.fullmatch(raw) is None:
        if _WINDOWS_UNC_PATH.fullmatch(raw) is None:
            return False
    normalized = raw.replace("/", "\\")
    return all(part not in (".", "..") for part in normalized.split("\\"))


def _is_native_private_path(raw: object) -> bool:
    """Validacion textual de la plataforma nativa actual."""
    if _IS_WINDOWS:
        return _is_windows_private_path(raw)
    return _is_posix_private_path(raw)


def _has_linked_ancestor(path: Path) -> bool:
    for ancestor in path.parents:
        if ancestor.is_symlink() or ancestor.is_junction():
            return True
    return False


def _parse_private_paths(
    snapshot_raw: object, log_raw: object
) -> tuple[Path, Path]:
    """Valida las dos rutas privadas (snapshot, log) nativamente.

    Separacion cerrada en dos pasos: primero la validacion textual
    multiplataforma (absoluta, sin NUL, sin tilde, sin URI, sin
    traversal, conjunto de caracteres cerrado); despues la
    validacion material solo contra el sistema nativo actual (padre
    existente, sin symlink/junction en la cadena, fuera del
    repositorio, destino inexistente, sin duplicidad). Nunca se
    materializa una ruta de la otra plataforma. Los errores no
    revelan la ruta.
    """
    for raw in (snapshot_raw, log_raw):
        if not _is_native_private_path(raw):
            raise _fail("preflight", "path_contract_violation")
    snapshot_path = Path(str(snapshot_raw))
    log_path = Path(str(log_raw))
    for candidate in (snapshot_path, log_path):
        if not candidate.parent.is_dir() or os.path.lexists(candidate):
            raise _fail("preflight", "path_contract_violation")
        if _has_linked_ancestor(candidate):
            raise _fail("preflight", "path_contract_violation")
        try:
            candidate.resolve().relative_to(p10.ROOT)
        except ValueError:
            pass
        else:
            raise _fail("preflight", "path_contract_violation")
    if os.path.normpath(str(snapshot_raw)) == os.path.normpath(str(log_raw)):
        raise _fail("preflight", "path_contract_violation")
    return snapshot_path, log_path


def _performance_log_bytes(payload: object) -> bytes:
    """Serializacion canonica cerrada del log P15 (compacta, sin PII)."""
    if type(payload) is not dict:
        raise ValueError("El payload del performance log debe ser dict exacto.")
    allowed = {
        "analysis_name",
        "execution_status",
        "current_stage",
        "completed_stages",
        "elapsed_seconds",
        "stage_seconds",
        "targets_total",
        "targets_processed",
        "records_built",
        "snapshot_entries",
        "snapshot_bytes",
        "operation_counters",
        "reason_code",
        "failure",
    }
    required = allowed - {"reason_code", "failure"}
    if not required <= payload.keys() <= allowed:
        raise ValueError("El payload del performance log excede el cierre P15.")
    if payload["analysis_name"] != PIPELINE_ANALYSIS_NAME:
        raise ValueError("El analysis_name del performance log es cerrado.")
    if payload["execution_status"] not in PIPELINE_EXECUTION_STATUSES:
        raise ValueError("El execution_status del performance log es cerrado.")
    current_stage = payload["current_stage"]
    if current_stage not in PIPELINE_STAGES and current_stage != "running_preflight":
        raise ValueError("El current_stage del performance log es cerrado.")
    completed = payload["completed_stages"]
    if (
        type(completed) is not list
        or len(completed) > len(PIPELINE_STAGES)
        or any(
            type(stage) is not str or stage not in PIPELINE_STAGES
            for stage in completed
        )
        or len(set(completed)) != len(completed)
        or completed
        != [stage for stage in PIPELINE_STAGES if stage in completed]
    ):
        raise ValueError(
            "El completed_stages del performance log no es un prefijo cerrado."
        )
    stage_seconds = payload["stage_seconds"]
    if type(stage_seconds) is not dict or any(
        stage not in PIPELINE_STAGES
        or type(value) is not float
        or value != value
        or not (value == 0.0 or 0.0 < value < float("inf"))
        for stage, value in stage_seconds.items()
    ):
        raise ValueError("El stage_seconds del performance log es incontractual.")
    for field_name in (
        "targets_total",
        "targets_processed",
        "records_built",
        "snapshot_entries",
        "snapshot_bytes",
    ):
        value = payload[field_name]
        if type(value) is not int or isinstance(value, bool) or value < 0:
            raise ValueError("Un conteo del performance log es incontractual.")
    if payload["targets_processed"] > payload["targets_total"]:
        raise ValueError("El progreso target del performance log no reconcilia.")
    if payload["records_built"] > payload["targets_processed"]:
        raise ValueError("La proyeccion del performance log no reconcilia.")
    elapsed = payload["elapsed_seconds"]
    if (
        type(elapsed) is not float
        or elapsed != elapsed
        or not (0.0 <= elapsed < float("inf"))
    ):
        raise ValueError("El elapsed del performance log es incontractual.")
    counters = payload["operation_counters"]
    if (
        type(counters) is not dict
        or tuple(counters) != PIPELINE_OPERATION_COUNTER_FIELDS
        or any(
            type(value) is not int or isinstance(value, bool) or value < 0
            for value in counters.values()
        )
    ):
        raise ValueError("Los contadores del performance log son incontractuales.")
    if "reason_code" in payload and (
        type(payload["reason_code"]) is not str
        or payload["reason_code"] not in PIPELINE_REASON_CODES
    ):
        raise ValueError("El reason_code del performance log es incontractual.")
    if "failure" in payload:
        failure = payload["failure"]
        if (
            type(failure) is not dict
            or tuple(failure) != ("stage", "type", "message")
            or failure["stage"] not in PIPELINE_STAGES
            or type(failure["type"]) is not str
            or _SAFE_TOKEN.fullmatch(failure["type"]) is None
            or type(failure["message"]) is not str
            or _SAFE_TOKEN.fullmatch(failure["message"]) is None
        ):
            raise ValueError("El failure del performance log no esta sanitizado.")
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "El performance log no es serializable de forma cerrada."
        ) from error
    return encoded


class _SnapshotPipelinePerformanceLog:
    """Log P15 agregado, atomico y sin PII de la ejecucion manual."""

    __slots__ = (
        "path",
        "_clock",
        "_started",
        "_stage_seconds",
        "_targets_total",
        "_targets_processed",
        "_records_built",
        "_snapshot_entries",
        "_snapshot_bytes",
        "_current_stage",
        "_completed_stages",
        "_counter_source",
    )

    def __init__(self, path: Path, clock: Callable[[], float]) -> None:
        if not isinstance(path, PurePath):
            raise TypeError("El performance log debe ser Path exacto.")
        self.path = path
        self._clock = clock
        self._started = float(clock())
        self._stage_seconds: dict[str, float] = {}
        self._targets_total = 0
        self._targets_processed = 0
        self._records_built = 0
        self._snapshot_entries = 0
        self._snapshot_bytes = 0
        self._current_stage = "running_preflight"
        self._completed_stages: tuple[str, ...] = ()
        self._counter_source: dict[str, int] | None = None

    def bind_counters(self, counters: dict[str, int]) -> None:
        self._counter_source = counters

    def set_snapshot_metrics(self, entries: int, size: int) -> None:
        self._snapshot_entries = int(entries)
        self._snapshot_bytes = int(size)

    def begin(self) -> None:
        self._write(
            execution_status="running_preflight",
            current_stage="running_preflight",
            completed_stages=(),
            reason_code=None,
        )

    def begin_stage(self, stage: str) -> None:
        self._current_stage = stage
        self._write(
            execution_status="running",
            current_stage=stage,
            completed_stages=self._completed_stages,
            reason_code=None,
        )

    def complete_stage(self, stage: str, seconds: float) -> None:
        self._stage_seconds[stage] = float(seconds)
        self._completed_stages = (
            self._completed_stages + (stage,)
        )
        self._current_stage = stage
        self._write(
            execution_status="running",
            current_stage=stage,
            completed_stages=self._completed_stages,
            reason_code=None,
        )

    def progress(
        self,
        stage: str,
        *,
        targets_total: int,
        targets_processed: int,
        records_built: int,
    ) -> None:
        self._targets_total = int(targets_total)
        self._targets_processed = int(targets_processed)
        self._records_built = int(records_built)
        self._current_stage = stage
        self._write(
            execution_status="running",
            current_stage=stage,
            completed_stages=self._completed_stages,
            reason_code=None,
        )

    def complete(self) -> None:
        self._completed_stages = PIPELINE_STAGES
        self._current_stage = PIPELINE_STAGES[-1]
        self._write(
            execution_status="completed",
            current_stage=PIPELINE_STAGES[-1],
            completed_stages=PIPELINE_STAGES,
            reason_code=None,
        )

    def fail(self, stage: str, reason_code: str) -> None:
        self._current_stage = stage
        self._write(
            execution_status="failed",
            current_stage=stage,
            completed_stages=self._completed_stages,
            reason_code=reason_code,
        )

    def interrupt(self) -> None:
        self._write(
            execution_status="interrupted",
            current_stage=self._current_stage,
            completed_stages=self._completed_stages,
            reason_code="manual_interrupt",
        )

    def _write(
        self,
        *,
        execution_status: str,
        current_stage: str,
        completed_stages: tuple[str, ...],
        reason_code: str | None,
    ) -> None:
        counters = (
            dict(self._counter_source)
            if self._counter_source is not None
            else dict.fromkeys(PIPELINE_OPERATION_COUNTER_FIELDS, 0)
        )
        payload: dict[str, object] = {
            "analysis_name": PIPELINE_ANALYSIS_NAME,
            "execution_status": execution_status,
            "current_stage": current_stage,
            "completed_stages": list(completed_stages),
            "elapsed_seconds": float(self._clock() - self._started),
            "stage_seconds": dict(self._stage_seconds),
            "targets_total": self._targets_total,
            "targets_processed": self._targets_processed,
            "records_built": self._records_built,
            "snapshot_entries": self._snapshot_entries,
            "snapshot_bytes": self._snapshot_bytes,
            "operation_counters": counters,
        }
        if reason_code is not None:
            payload["reason_code"] = reason_code
        encoded = _performance_log_bytes(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            temporary.write_bytes(encoded)
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()


@dataclass(frozen=True)
class TacticalRecommendationSnapshotPipelineResult:
    """Resultado agregado P15: sin snapshot, sin P10 result, sin rutas."""

    contract: str
    schema_version: str
    execution_status: str
    failed_stage: str | None
    reason_codes: tuple[str, ...]
    completed_stages: tuple[str, ...]
    stage_seconds: tuple[tuple[str, float], ...]
    targets_total: int
    targets_processed: int
    records_built: int
    snapshot_entries: int
    snapshot_bytes: int
    operation_counters: tuple[tuple[str, int], ...]
    reconciliations: tuple[tuple[str, bool], ...]
    snapshot_fingerprint: str | None
    generation_diagnostics: p14.SnapshotGenerationDiagnostics | None

    def __post_init__(self) -> None:
        validate_tactical_recommendation_snapshot_pipeline_result(self)


def _check_nonnegative_int(value: object, name: str) -> None:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise TypeError(f"{name} debe ser un entero no negativo exacto.")


def validate_tactical_recommendation_snapshot_pipeline_result(
    result: object,
) -> None:
    """Validacion reconstructiva completa del resultado P15."""
    if type(result) is not TacticalRecommendationSnapshotPipelineResult:
        raise TypeError(
            "El resultado debe ser TacticalRecommendationSnapshotPipelineResult exacto."
        )
    if (
        result.contract != PIPELINE_CONTRACT_NAME
        or result.schema_version != PIPELINE_SCHEMA_VERSION
        or result.execution_status not in PIPELINE_RESULT_STATUSES
    ):
        raise ValueError("Metadatos contractuales fuera del cierre P15.")
    if (
        type(result.completed_stages) is not tuple
        or len(result.completed_stages) > len(PIPELINE_STAGES)
        or any(
            type(stage) is not str or stage not in PIPELINE_STAGES
            for stage in result.completed_stages
        )
        or len(set(result.completed_stages)) != len(result.completed_stages)
        or list(result.completed_stages)
        != [stage for stage in PIPELINE_STAGES if stage in result.completed_stages]
    ):
        raise ValueError("completed_stages no es un prefijo cerrado.")
    if (
        type(result.stage_seconds) is not tuple
        or tuple(stage for stage, _seconds in result.stage_seconds)
        != PIPELINE_STAGES
        or any(
            type(seconds) is not float
            or seconds != seconds
            or not (0.0 <= seconds < float("inf"))
            for _stage, seconds in result.stage_seconds
        )
    ):
        raise TypeError("stage_seconds fuera del contrato exacto.")
    for name in (
        "targets_total",
        "targets_processed",
        "records_built",
        "snapshot_entries",
        "snapshot_bytes",
    ):
        _check_nonnegative_int(getattr(result, name), name)
    if (
        type(result.reason_codes) is not tuple
        or any(
            type(code) is not str or code not in PIPELINE_REASON_CODES
            for code in result.reason_codes
        )
    ):
        raise TypeError("reason_codes fuera del contrato cerrado.")
    if (
        type(result.operation_counters) is not tuple
        or tuple(name for name, _value in result.operation_counters)
        != PIPELINE_OPERATION_COUNTER_FIELDS
        or any(
            type(value) is not int or isinstance(value, bool) or value < 0
            for _name, value in result.operation_counters
        )
    ):
        raise TypeError("operation_counters fuera del contrato exacto.")
    if (
        type(result.reconciliations) is not tuple
        or tuple(key for key, _flag in result.reconciliations)
        != PIPELINE_RECONCILIATION_KEYS
        or any(
            type(flag) is not bool for _key, flag in result.reconciliations
        )
    ):
        raise TypeError("reconciliations fuera del contrato exacto.")
    counters = dict(result.operation_counters)
    reconciliations = dict(result.reconciliations)
    fingerprint = result.snapshot_fingerprint
    if (
        counters["p10_executions"] > 1
        or counters["generation_calls"] > 1
        or counters["persistence_calls"] > 1
        or counters["verification_calls"] > 1
        or counters["persistence_calls"] > counters["generation_calls"]
        or counters["verification_calls"] != counters["persistence_calls"]
        or counters["targets_projected"] < result.records_built
        or result.targets_processed > result.targets_total
        or result.records_built > result.targets_processed
        or result.snapshot_entries > result.records_built
    ):
        raise ValueError("Contadores P15 inconsistentes (cero reintento).")
    if fingerprint is None:
        if (
            counters["verification_calls"] != 0
            or result.snapshot_entries != 0
            or result.snapshot_bytes != 0
        ):
            raise ValueError("Snapshot ausente con metricas de publicacion.")
    elif (
        type(fingerprint) is not str
        or len(fingerprint) != 64
        or not all(character in "0123456789ABCDEF" for character in fingerprint)
        or result.execution_status != "completed"
    ):
        raise ValueError("Fingerprint de snapshot fuera del cierre P15.")
    diagnostics = result.generation_diagnostics
    if result.execution_status == "completed" and diagnostics is None:
        raise ValueError("Resultado completed P15 sin diagnostico P14.")
    if diagnostics is not None and type(diagnostics) is not (
        p14.SnapshotGenerationDiagnostics
    ):
        raise TypeError("generation_diagnostics fuera del contrato exacto.")
    if (
        reconciliations["p10_result_validated"]
        and counters["p10_executions"] != 1
    ) or (
        reconciliations["p14_universe_reconciled"]
        and counters["generation_calls"] != 1
    ) or (
        reconciliations["snapshot_persisted_and_verified"]
        and counters["verification_calls"] != 1
    ):
        raise ValueError("Reconciliaciones P15 inconsistentes con contadores.")
    if reconciliations["targets_projected"] and (
        result.records_built != counters["targets_projected"]
        or result.records_built != result.targets_processed
    ):
        raise ValueError("Proyeccion P15 no reconcile.")
    if result.execution_status == "completed":
        if (
            result.failed_stage is not None
            or result.reason_codes != ()
            or result.completed_stages != PIPELINE_STAGES
            or tuple(reconciliations.values()) != (True, True, True, True)
            or counters["p10_executions"] != 1
            or counters["generation_calls"] != 1
            or counters["persistence_calls"] != 1
            or counters["verification_calls"] != 1
            or result.targets_total != result.targets_processed
            or result.targets_processed != result.records_built
            or result.records_built != result.snapshot_entries
            or result.snapshot_entries == 0
            or result.snapshot_bytes == 0
        ):
            raise ValueError("Resultado completed P15 fuera del cierre exacto.")
        return
    if (
        result.failed_stage not in PIPELINE_STAGES
        or not result.reason_codes
    ):
        raise ValueError("Resultado no-exito P15 fuera del cierre exacto.")
    extended = result.completed_stages + (result.failed_stage,)
    if tuple(extended) != PIPELINE_STAGES[: len(extended)]:
        raise ValueError("Fase de fallo P15 fuera del orden cerrado.")
    if counters["p10_executions"] == 1 and (
        result.targets_total != result.targets_processed
    ):
        raise ValueError("Fallo P15 con proyeccion de targets inconsistente.")


def project_tactical_target_records(
    result: object,
    *,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> tuple[p14.P10OfflineSnapshotRecord, ...]:
    """Proyeccion O(N) de un solo paso: target P10 -> record P14.

    Copia de campos explicita; sin inferencias, sin mapeos y sin
    ocultar duplicados (la deteccion pertenece a P14).
    """
    if type(result) is not p10.TacticalPipelineResult:
        raise _fail("target_projection", "target_projection_failed")
    target_results = result.target_results
    total = len(target_results)
    records: list[p14.P10OfflineSnapshotRecord] = []
    for position, item in enumerate(target_results):
        try:
            target = item.target
            records.append(
                p14.P10OfflineSnapshotRecord(
                    target.target_match_id,
                    target.fold,
                    target.orientation,
                    item.prioritization,
                )
            )
        except Exception:
            raise _fail("target_projection", "target_projection_failed") from None
        done = position + 1
        if (done % _PROGRESS_STEP == 0) or done == total:
            if progress_callback is not None:
                progress_callback(total, done, done)
    return tuple(records)


def _check_p10_seal_metadata(
    result: p10.TacticalPipelineResult,
) -> None:
    """Solo metadata contractual: sellado, leakage y temporal."""
    seal = result.test_seal
    if (
        seal.test_status != "sealed"
        or seal.used_for_method_selection is not False
        or any(count != 0 for _name, count in seal.counters)
    ):
        raise _fail("p10_pipeline", "p10_seal_violation")
    leakage = result.leakage_audit
    if any(
        getattr(leakage, field_info.name) != 0
        for field_info in fields(leakage)
    ):
        raise _fail("p10_pipeline", "p10_seal_violation")
    if result.config.development_cutoff != p10.DEVELOPMENT_CUTOFF.date():
        raise _fail("p10_pipeline", "p10_seal_violation")
    for item in result.target_results:
        target = item.target
        if (
            target.as_of_date >= _SEALED_FIRST_DAY
            or target.fold not in _P10_FOLDS
            or target.orientation not in _P10_ORIENTATIONS
        ):
            raise _fail("p10_pipeline", "p10_seal_violation")


def _production_p10_runner() -> Callable[[], object]:
    """Llamada unica a la frontera compute-only P10.

    ``compute_tactical_pipeline_result`` lee la fuente contractual
    una vez y devuelve el ``TacticalPipelineResult``; no publica
    artefactos P10, no escribe performance log P10 y no depende de
    las autorizaciones historicas P10. La unica autorizacion de la
    ruta productiva es ``P15.REAL_EXECUTION_AUTHORIZED``.
    """

    def _runner() -> object:
        return p10.compute_tactical_pipeline_result(p10.POINTS_PATH)

    return _runner


def run_tactical_recommendation_snapshot_pipeline(
    snapshot_path: Path,
    performance_log: Path,
    *,
    p10_runner: Callable[[], object] | None = None,
    snapshot_generator: (
        Callable[
            [tuple, Path, p14.SnapshotGenerationPolicy],
            p14.TacticalRecommendationSnapshotGenerationResult,
        ]
        | None
    ) = None,
    clock: Callable[[], float] | None = None,
    log_writer_factory: (
        Callable[[Path, Callable[[], float]], _SnapshotPipelinePerformanceLog]
        | None
    ) = None,
) -> TacticalRecommendationSnapshotPipelineResult:
    """Orquestacion fina: P10 una vez -> records una vez -> P14 una vez.

    Devuelve resultados cerrados (no lanza en fallos analiticos; solo
    la autorizacion bloqueante lanza). No reconstruye evidencia ni
    priorizacion, no serializa dentro del loop de targets y no publica
    artefactos P10. Cero reintentos y cero snapshot parcial.
    """
    if not REAL_EXECUTION_AUTHORIZED:
        raise _fail("preflight", REAL_EXECUTION_BLOCK_REASON_CODE)
    if type(snapshot_path) is str or type(performance_log) is str:
        raise _fail("preflight", "path_contract_violation")
    if not isinstance(snapshot_path, PurePath) or not isinstance(
        performance_log, PurePath
    ):
        raise _fail("preflight", "path_contract_violation")
    snapshot_path, performance_log = _parse_private_paths(
        str(snapshot_path), str(performance_log)
    )
    selected_clock = clock if clock is not None else perf_counter
    selected_factory = (
        log_writer_factory
        if log_writer_factory is not None
        else _SnapshotPipelinePerformanceLog
    )
    selected_p10_runner = (
        p10_runner if p10_runner is not None
        else _production_p10_runner()
    )
    selected_generator = (
        snapshot_generator
        if snapshot_generator is not None
        else p14.generate_and_persist_tactical_recommendation_snapshot
    )

    writer = selected_factory(performance_log, selected_clock)
    counters = dict.fromkeys(PIPELINE_OPERATION_COUNTER_FIELDS, 0)
    writer.bind_counters(counters)
    stage_seconds: dict[str, float] = {}
    completed: list[str] = []
    current_stage = "preflight"
    stage_started = float(selected_clock())
    targets_total = 0
    targets_processed = 0
    records_built = 0
    fingerprint: str | None = None
    entries = 0
    size = 0
    diagnostics: p14.SnapshotGenerationDiagnostics | None = None
    p10_validated = False
    projection_done = False

    def _begin(stage: str) -> None:
        nonlocal current_stage, stage_started
        current_stage = stage
        stage_started = float(selected_clock())
        writer.begin_stage(stage)

    def _complete(stage: str) -> None:
        stage_seconds[stage] = float(selected_clock()) - stage_started
        completed.append(stage)
        writer.complete_stage(stage, stage_seconds[stage])

    def _current_seconds_tuple() -> tuple[tuple[str, float], ...]:
        return tuple(
            (stage, float(stage_seconds.get(stage, 0.0)))
            for stage in PIPELINE_STAGES
        )

    def _result(
        status: str,
        failed_stage: str | None,
        reason_codes: tuple[str, ...],
    ) -> TacticalRecommendationSnapshotPipelineResult:
        return TacticalRecommendationSnapshotPipelineResult(
            contract=PIPELINE_CONTRACT_NAME,
            schema_version=PIPELINE_SCHEMA_VERSION,
            execution_status=status,
            failed_stage=failed_stage,
            reason_codes=reason_codes,
            completed_stages=tuple(completed),
            stage_seconds=_current_seconds_tuple(),
            targets_total=targets_total,
            targets_processed=targets_processed,
            records_built=records_built,
            snapshot_entries=entries,
            snapshot_bytes=size,
            operation_counters=tuple(
                (name, counters[name])
                for name in PIPELINE_OPERATION_COUNTER_FIELDS
            ),
            reconciliations=(
                ("p10_result_validated", p10_validated),
                ("targets_projected", projection_done),
                ("p14_universe_reconciled", counters["generation_calls"] == 1),
                (
                    "snapshot_persisted_and_verified",
                    counters["verification_calls"] == 1,
                ),
            ),
            snapshot_fingerprint=fingerprint,
            generation_diagnostics=diagnostics,
        )

    def _fail_result(stage: str, reason_code: str) -> None:
        writer.fail(stage, reason_code)
        raise _ReturnPipelineResult(
            "failed", stage, (reason_code,)
        )

    try:
        writer.begin()
        _complete("preflight")

        _begin("p10_pipeline")
        p10_result: object | None = selected_p10_runner()
        if type(p10_result) is not p10.TacticalPipelineResult:
            writer.fail("p10_pipeline", "p10_result_not_pipeline_result")
            return _result(
                "failed", "p10_pipeline", ("p10_result_not_pipeline_result",)
            )
        counters["p10_executions"] = 1
        try:
            p10.validate_tactical_pipeline_result(p10_result)
            _check_p10_seal_metadata(p10_result)
        except (p10.TacticalPipelineContractError, TypeError, ValueError):
            writer.fail("p10_pipeline", "p10_result_invalid")
            return _result("failed", "p10_pipeline", ("p10_result_invalid",))
        p10_validated = True
        targets_total = len(p10_result.target_results)
        targets_processed = targets_total
        _complete("p10_pipeline")

        _begin("target_projection")

        def _projection_progress(
            _total: int, done_count: int, built: int
        ) -> None:
            writer.progress(
                "target_projection",
                targets_total=targets_total,
                targets_processed=done_count,
                records_built=built,
            )

        records = project_tactical_target_records(
            p10_result, progress_callback=_projection_progress
        )
        records_built = len(records)
        counters["targets_projected"] = records_built
        projection_done = True
        _complete("target_projection")
        p10_result = None

        _begin("snapshot_generation")
        generation = selected_generator(
            records,
            snapshot_path,
            policy=p14.P10_OFFLINE_SNAPSHOT_GENERATION_POLICY,
        )
        records = None
        if type(generation) is not (
            p14.TacticalRecommendationSnapshotGenerationResult
        ):
            writer.fail("snapshot_generation", "p15_execution_error")
            return _result(
                "failed", "snapshot_generation", ("p15_execution_error",)
            )
        counters["generation_calls"] = 1
        counters["persistence_calls"] = 1
        counters["verification_calls"] = 1
        _complete("snapshot_generation")
        stage_seconds["persistence"] = 0.0
        completed.append("persistence")
        writer.complete_stage("persistence", 0.0)
        stage_seconds["verification"] = 0.0
        completed.append("verification")
        writer.complete_stage("verification", 0.0)
        entries = generation.entries_generated
        size = generation.serialized_bytes
        fingerprint = generation.snapshot_fingerprint
        diagnostics = generation.diagnostics
        writer.set_snapshot_metrics(entries, size)
        writer.complete()
        return _result("completed", None, ())
    except KeyboardInterrupt:
        stage_seconds.setdefault(current_stage, 0.0)
        writer.interrupt()
        return _result("interrupted", current_stage, ("manual_interrupt",))
    except p10.TacticalPipelineContractError:
        stage_seconds.setdefault(current_stage, 0.0)
        writer.fail(current_stage, "p10_pipeline_execution_failed")
        return _result(
            "failed", current_stage, ("p10_pipeline_execution_failed",)
        )
    except p10.TacticalPipelineExecutionError:
        stage_seconds.setdefault(current_stage, 0.0)
        writer.fail(current_stage, "p10_pipeline_execution_failed")
        return _result(
            "failed", current_stage, ("p10_pipeline_execution_failed",)
        )
    except p14.TacticalRecommendationSnapshotGenerationError as error:
        if error.reason_code == "p13_verification_mismatch":
            stage, reason = "verification", "verification_failed"
            stage_seconds.setdefault("snapshot_generation", 0.0)
            if "snapshot_generation" not in completed:
                completed.append("snapshot_generation")
            stage_seconds.setdefault("persistence", 0.0)
            if "persistence" not in completed:
                completed.append("persistence")
        else:
            stage, reason = (
                "snapshot_generation",
                "snapshot_generation_failed",
            )
            stage_seconds.setdefault(stage, 0.0)
        writer.fail(stage, reason)
        return _result("failed", stage, (reason,))
    except p13.PersistedSnapshotError:
        stage_seconds.setdefault("snapshot_generation", 0.0)
        if "snapshot_generation" not in completed:
            completed.append("snapshot_generation")
        writer.fail("persistence", "persistence_failed")
        return _result("failed", "persistence", ("persistence_failed",))
    except TacticalRecommendationSnapshotPipelineError as error:
        stage = error.stage
        if stage not in completed:
            stage_seconds.setdefault(stage, 0.0)
        writer.fail(stage, error.reason_code)
        return _result("failed", stage, (error.reason_code,))
    except Exception:
        stage = current_stage
        if stage not in completed:
            stage_seconds.setdefault(stage, 0.0)
        writer.fail(stage, "p15_execution_error")
        return _result("failed", stage, ("p15_execution_error",))


def validate_snapshot_pipeline_paths_cli(
    snapshot_raw: object, log_raw: object
) -> tuple[Path, Path]:
    """Validacion contractual de las rutas privadas (sin PII en errores)."""
    return _parse_private_paths(snapshot_raw, log_raw)


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "P15: unica ejecucion manual autorizada del snapshot "
            "privado offline (un solo intento, sin reintento)."
        )
    )
    parser.add_argument("--snapshot-path", required=True)
    parser.add_argument("--performance-log", required=True)
    arguments = parser.parse_args(argv)
    if not REAL_EXECUTION_AUTHORIZED:
        raise SystemExit(REAL_SNAPSHOT_EXECUTION_BLOCK_REASON)
    try:
        snapshot_path, performance_log = _parse_private_paths(
            arguments.snapshot_path, arguments.performance_log
        )
    except TacticalRecommendationSnapshotPipelineError:
        raise SystemExit(_ERROR_MESSAGES["path_contract_violation"]) from None
    try:
        result = run_tactical_recommendation_snapshot_pipeline(
            snapshot_path, performance_log
        )
    except KeyboardInterrupt:
        return 130
    if result.execution_status == "interrupted":
        return 130
    if result.execution_status != "completed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main()) from None


__all__ = (
    "AUTOMATIC_RETRY",
    "AUTOMATIC_RETRIES_PERFORMED",
    "COMPLETED_REAL_EXECUTIONS",
    "INTERRUPTED_REAL_EXECUTIONS",
    "PIPELINE_ANALYSIS_NAME",
    "PIPELINE_CONTRACT_NAME",
    "PIPELINE_EXECUTION_STATUSES",
    "PIPELINE_OPERATION_COUNTER_FIELDS",
    "PIPELINE_REASON_CODES",
    "PIPELINE_RECONCILIATION_KEYS",
    "PIPELINE_SCHEMA_VERSION",
    "PIPELINE_STAGES",
    "POST_EXECUTION_CLOSURE_RULE",
    "PREVIOUS_REAL_P15_ATTEMPTS",
    "PERSISTED_SNAPSHOT_BYTES",
    "PERSISTED_SNAPSHOT_CAPACITY_MARGIN_BYTES",
    "PERSISTED_SNAPSHOT_ENTRIES",
    "PERSISTED_SNAPSHOT_SHA256",
    "REAL_EXECUTION_AUTHORIZED",
    "REAL_EXECUTION_AUTHORIZATION_REASON",
    "REAL_EXECUTION_BLOCK_REASON_CODE",
    "REAL_SNAPSHOT_EXECUTION_BLOCK_REASON",
    "SECOND_REAL_EXECUTION_ELAPSED_SECONDS",
    "SECOND_REAL_EXECUTION_OPERATION_COUNTS",
    "SECOND_REAL_EXECUTION_STATUS",
    "SINGLE_MANUAL_EXECUTION_POLICY",
    "TacticalRecommendationSnapshotPipelineError",
    "TacticalRecommendationSnapshotPipelineResult",
    "main",
    "project_tactical_target_records",
    "run_tactical_recommendation_snapshot_pipeline",
    "validate_snapshot_pipeline_paths_cli",
    "validate_tactical_recommendation_snapshot_pipeline_result",
)
