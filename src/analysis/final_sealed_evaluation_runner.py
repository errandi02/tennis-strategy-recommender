"""P22: runner productivo completo, y serializacion/publicacion
atomica del bundle de artefactos finales de la evaluacion sellada.
Gobernado por la UNICA puerta
``final_sealed_evaluation.REAL_TEST_EVALUATION_AUTHORIZED`` (P20, sin
reasignar aqui: solo se importa y se lee). Con la puerta en ``False``,
``execute_real_sealed_test_evaluation`` aborta en su primer paso,
antes de cualquier ``open``, ``stat`` de la fuente, pandas/pyarrow,
lectura, construccion o escritura. Autorizacion puntual (P23): una
UNICA ejecucion manual y desacoplada queda autorizada -- ver
``final_sealed_evaluation.py`` para el contrato exacto de cierre
(un commit posterior debe devolver la puerta a ``False`` y
actualizar los contadores tras completar/fallar/interrumpir).

Bundle atomico (revision tras hallazgo de diseno): cinco ``os.replace``
independientes con rollback NO son una transaccion -- durante la
publicacion pueden verse subconjuntos parciales, y un SIGKILL/caida de
proceso/perdida de energia entre dos de ellos deja el rollback sin
ejecutar. Los artefactos ahora viven en un UNICO directorio
contractual (``reports/final_evaluation/``); la publicacion construye
un directorio temporal HERMANO bajo ``reports/`` (mismo sistema de
archivos, para que el rename final sea atomico), escribe y verifica
TODO el contenido ahi dentro, y ejecuta EXACTAMENTE UN
``os.replace(directorio_temporal, directorio_final)``. Verificado en
este entorno (Windows) que ``os.replace`` renombra un directorio de
forma atomica cuando el destino NO existe (que es la unica situacion
permitida: un bundle preexistente bloquea antes de leer la fuente). Si
esa garantia no se pudiese ofrecer en alguna plataforma, la funcion
debe fallar de forma cerrada -- nunca regresar silenciosamente a
publicaciones de archivo por archivo.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import signal
import stat
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Final

import pandas as pd

from src.analysis.chronological_validation import POINTS_PATH
from src.analysis.final_sealed_evaluation import (
    REAL_TEST_EVALUATION_AUTHORIZED,
    REAL_TEST_EVALUATION_BLOCK_REASON,
)
from src.analysis.final_sealed_evaluation_adapter import adapt_real_points_dataframe
from src.analysis.final_sealed_evaluation_orchestrator import (
    FinalSealedEvaluationOutcome,
    ProtocolMetricsSummary,
    evaluate_final_sealed_test,
)


FINAL_SEALED_EVALUATION_RUNNER_CONTRACT_VERSION: Final = "1.0.0"

_ROOT: Final = Path(__file__).resolve().parents[2]
_REPORTS_DIR: Final = _ROOT / "reports"

# Directorio contractual UNICO del bundle final (P22, tras revision).
FINAL_EVALUATION_BUNDLE_DIR: Final = _REPORTS_DIR / "final_evaluation"

# Rutas relativas fijas DENTRO del bundle (orden canonico de escritura).
BUNDLE_SUMMARY_RELATIVE_PATH: Final = "summary.json"
BUNDLE_PERFORMANCE_RELATIVE_PATH: Final = "performance.json"
BUNDLE_COVERAGE_RELATIVE_PATH: Final = "tables/coverage.csv"
BUNDLE_P02_PERFORMANCE_RELATIVE_PATH: Final = "tables/p02_performance.csv"
BUNDLE_SECONDARY_METRICS_RELATIVE_PATH: Final = "tables/secondary_metrics.csv"

BUNDLE_RELATIVE_PATHS: Final = (
    BUNDLE_SUMMARY_RELATIVE_PATH,
    BUNDLE_PERFORMANCE_RELATIVE_PATH,
    BUNDLE_COVERAGE_RELATIVE_PATH,
    BUNDLE_P02_PERFORMANCE_RELATIVE_PATH,
    BUNDLE_SECONDARY_METRICS_RELATIVE_PATH,
)

# Preflight P20 nunca debe sobrescribirse desde P22 (sigue siendo un
# archivo suelto en reports/, fuera del bundle).
_PROTECTED_PATHS: Final = (_REPORTS_DIR / "final_evaluation_preflight.json",)

# Fuente contractual real UNICA (misma constante que chronological_
# validation.py); sin variable de entorno ni flag CLI alternativos.
FINAL_SEALED_EVALUATION_SOURCE_PATH: Final = POINTS_PATH

MAX_ARTIFACT_BYTES: Final = 10 * 1024 * 1024

_COVERAGE_FIELDNAMES: Final = (
    "protocol", "dimension", "key", "count", "denominator", "proportion",
)
_P02_FIELDNAMES: Final = ("protocol", "metric", "value", "denominator")
_SECONDARY_FIELDNAMES: Final = ("protocol", "metric", "key", "value")


class FinalSealedEvaluationRunnerError(RuntimeError):
    """Rechazo cerrado del runner P22 (mensaje del catalogo, sin fugas)."""

    __slots__ = ("reason_code",)

    def __init__(self, message: str, *, reason_code: str) -> None:
        object.__setattr__(self, "reason_code", reason_code)
        super().__init__(message)


_ERROR_MESSAGES: Final = {
    "not_authorized": REAL_TEST_EVALUATION_BLOCK_REASON,
    "destination_exists": (
        "Publicacion bloqueada: el bundle final ya existe (cierre cerrado)."
    ),
    "source_path_contract_violation": (
        "Fuente contractual fuera de contrato (no se revela)."
    ),
    "artifact_too_large": "Artefacto final fuera del limite de tamano permitido.",
    "publish_failed": "Publicacion atomica del bundle fallida; ningun bundle parcial queda.",
    "verification_failed": "Verificacion del bundle fallida.",
}


def _fail(reason_code: str) -> FinalSealedEvaluationRunnerError:
    return FinalSealedEvaluationRunnerError(
        _ERROR_MESSAGES[reason_code], reason_code=reason_code
    )


# --------------------------------------------------------------------- #
# SIGTERM/SIGINT gestionados (revision P23): sin esto, una terminacion   #
# por senal durante la publicacion mata el proceso sin ejecutar ningun   #
# bloque except/finally de Python, dejando el directorio temporal del    #
# bundle huerfano bajo reports/. Mismo patron que P17
# (src/api/runtime.py): instala manejadores que convierten la senal en  #
# una excepcion Python capturable SOLO durante la ejecucion real, y      #
# restaura los manejadores originales al salir, con o sin excepcion.     #
# --------------------------------------------------------------------- #


class _ManagedTerminationSignal(Exception):
    """SIGTERM/SIGINT capturados de forma gestionada durante la
    ejecucion real. Hereda de ``Exception`` (no ``BaseException``)
    deliberadamente: debe fluir a traves de los mismos bloques
    ``except Exception`` de limpieza de ``publish_final_evaluation_
    bundle`` que cualquier otro fallo, para que el temporal se borre
    igual; esos bloques la vuelven a lanzar SIN convertirla en
    ``FinalSealedEvaluationRunnerError`` para que ``main`` distinga una
    terminacion (salida 130) de un fallo ordinario (salida 1)."""

    __slots__ = ()


def _raise_managed_termination(signum: int, frame: FrameType | None) -> None:
    del signum, frame
    raise _ManagedTerminationSignal


@contextmanager
def _normalize_execution_signals() -> Iterator[None]:
    """Instala manejadores de SIGTERM/SIGINT solo durante la ejecucion
    real y los restaura siempre al salir (con o sin excepcion)."""
    managed = (signal.SIGTERM, signal.SIGINT)
    originals: dict[int, object] = {}
    try:
        for managed_signal in managed:
            originals[managed_signal] = signal.getsignal(managed_signal)
            signal.signal(managed_signal, _raise_managed_termination)
        yield
    finally:
        for managed_signal, original in originals.items():
            signal.signal(managed_signal, original)


# --------------------------------------------------------------------- #
# Serializacion pura (sin I/O)                                           #
# --------------------------------------------------------------------- #


def _protocol_summary_json(summary: ProtocolMetricsSummary) -> dict[str, object]:
    return {
        "protocol": summary.protocol,
        "targets_total": summary.targets_total,
        "coverage_by_status": dict(summary.coverage_by_status),
        "pattern_coverage": {
            pattern: dict(values) for pattern, values in summary.pattern_coverage.items()
        },
        "reconciliation_by_year": {
            year: dict(values) for year, values in summary.reconciliation_by_year.items()
        },
        "wilson_width_distribution": dict(summary.wilson_width_distribution),
        "rank_and_tie_distribution": {
            "rank_position_counts": dict(summary.rank_and_tie_distribution["rank_position_counts"]),
            "tie_group_size_counts": dict(summary.rank_and_tie_distribution["tie_group_size_counts"]),
            "boundary_tie_expanded_rankings": summary.rank_and_tie_distribution[
                "boundary_tie_expanded_rankings"
            ],
        },
        "abstention_reason_codes": dict(summary.abstention_reason_codes),
        "coverage_by_surface_and_period": {
            key: dict(values) for key, values in summary.coverage_by_surface_and_period.items()
        },
        "error_bucket": dict(summary.error_bucket),
        "p02_performance": dict(summary.p02_performance),
        "p02_population_baseline_comparison": dict(summary.p02_population_baseline_comparison),
    }


def build_final_evaluation_summary(outcome: FinalSealedEvaluationOutcome) -> dict[str, object]:
    """Manifiesto agregado: sin identidades, sin rutas, sin resultados
    target-level. ``2026`` siempre aparece como bucket ``2026_partial``
    (heredado de los agregadores P21), nunca agrupado con anos completos.
    Incluye numerador/denominador/tasa/regla de la baseline poblacional
    P02 (decision humana congelada, P22) y su fingerprint independiente."""
    if type(outcome) is not FinalSealedEvaluationOutcome:
        raise _fail("verification_failed")
    return {
        "contract_name": "final_evaluation_summary",
        "contract_version": outcome.contract_version,
        "fingerprints": {
            "configuration_fingerprint": outcome.configuration_fingerprint,
            "specification_fingerprint": outcome.specification_fingerprint,
            "p02_population_baseline_fingerprint": outcome.p02_population_baseline_fingerprint,
        },
        "p02_population_baseline": {
            "rule": outcome.p02_population_baseline.rule,
            "numerator": outcome.p02_population_baseline.numerator,
            "denominator": outcome.p02_population_baseline.denominator,
            "rate": outcome.p02_population_baseline.rate,
        },
        "population_reconciled": outcome.population_reconciled,
        "targets_total": outcome.targets_total,
        "principal_protocol": "rolling_origin",
        "sensitivity_protocol": "frozen",
        "rolling": _protocol_summary_json(outcome.rolling),
        "frozen": _protocol_summary_json(outcome.frozen),
        "p02_only_predictive_metrics": True,
        "p04_p05_p06_coverage_only_not_predictive": True,
    }


def build_final_evaluation_coverage_rows(
    outcome: FinalSealedEvaluationOutcome,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for summary in (outcome.rolling, outcome.frozen):
        for status, count in sorted(summary.coverage_by_status["by_status"].items()):
            denominator = summary.coverage_by_status["denominator"]
            rows.append(
                {
                    "protocol": summary.protocol,
                    "dimension": "orientation_status",
                    "key": status,
                    "count": count,
                    "denominator": denominator,
                    "proportion": (None if denominator == 0 else count / denominator),
                }
            )
        for pattern, values in sorted(summary.pattern_coverage.items()):
            for key in ("scored", "abstained"):
                denominator = values["total_candidates"]
                rows.append(
                    {
                        "protocol": summary.protocol,
                        "dimension": f"pattern_{pattern}",
                        "key": key,
                        "count": values[key],
                        "denominator": denominator,
                        "proportion": (None if denominator == 0 else values[key] / denominator),
                    }
                )
    return tuple(rows)


def build_final_evaluation_p02_performance_rows(
    outcome: FinalSealedEvaluationOutcome,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for summary in (outcome.rolling, outcome.frozen):
        performance = summary.p02_performance
        baseline = summary.p02_population_baseline_comparison
        denominator = performance["denominator"]
        for metric in ("brier_score", "log_loss"):
            rows.append(
                {
                    "protocol": summary.protocol,
                    "metric": metric,
                    "value": performance[metric],
                    "denominator": denominator,
                }
            )
        for metric in ("delta_brier_vs_population", "delta_log_loss_vs_population"):
            rows.append(
                {
                    "protocol": summary.protocol,
                    "metric": metric,
                    "value": baseline[metric],
                    "denominator": baseline["denominator"],
                }
            )
        rows.append(
            {
                "protocol": summary.protocol,
                "metric": "population_baseline_rate",
                "value": baseline["population_baseline_rate"],
                "denominator": baseline["population_baseline_denominator"],
            }
        )
    return tuple(rows)


def build_final_evaluation_secondary_metrics_rows(
    outcome: FinalSealedEvaluationOutcome,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for summary in (outcome.rolling, outcome.frozen):
        for key in ("min", "max", "mean", "denominator"):
            rows.append(
                {
                    "protocol": summary.protocol, "metric": "wilson_width",
                    "key": key, "value": summary.wilson_width_distribution[key],
                }
            )
        for key, value in sorted(summary.rank_and_tie_distribution["rank_position_counts"].items()):
            rows.append(
                {"protocol": summary.protocol, "metric": "rank_position_count", "key": str(key), "value": value}
            )
        for key, value in sorted(summary.rank_and_tie_distribution["tie_group_size_counts"].items()):
            rows.append(
                {"protocol": summary.protocol, "metric": "tie_group_size_count", "key": str(key), "value": value}
            )
        rows.append(
            {
                "protocol": summary.protocol, "metric": "boundary_tie_expanded_rankings",
                "key": "total", "value": summary.rank_and_tie_distribution["boundary_tie_expanded_rankings"],
            }
        )
        for key, value in sorted(summary.abstention_reason_codes.items()):
            rows.append(
                {"protocol": summary.protocol, "metric": "abstention_reason_code", "key": key, "value": value}
            )
        for key, values in sorted(summary.coverage_by_surface_and_period.items()):
            rows.append(
                {
                    "protocol": summary.protocol, "metric": "coverage_by_surface_and_period",
                    "key": f"{key}:orientations", "value": values["orientations"],
                }
            )
        for year, values in sorted(summary.reconciliation_by_year.items()):
            rows.append(
                {
                    "protocol": summary.protocol, "metric": "reconciliation_by_year",
                    "key": f"{year}:orientations", "value": values["orientations"],
                }
            )
    return tuple(rows)


def build_final_evaluation_performance_manifest(
    stage_seconds: dict[str, float], operation_counters: dict[str, int]
) -> dict[str, object]:
    """Solo tiempos/contadores operativos (mismo espiritu que el log de
    performance P15): sin identidades, sin rutas, sin contenido del
    dataset."""
    return {
        "contract_name": "final_evaluation_performance",
        "contract_version": FINAL_SEALED_EVALUATION_RUNNER_CONTRACT_VERSION,
        "stage_seconds": dict(sorted(stage_seconds.items())),
        "operation_counters": dict(sorted(operation_counters.items())),
    }


def _canonical_json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _csv_bytes(rows: tuple[dict[str, object], ...], fieldnames: tuple[str, ...]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def build_final_evaluation_bundle_payloads(
    outcome: FinalSealedEvaluationOutcome,
    *,
    stage_seconds: dict[str, float] | None = None,
    operation_counters: dict[str, int] | None = None,
) -> dict[str, bytes]:
    """Ensambla TODOS los payloads del bundle en memoria (sin I/O),
    indexados por su ruta RELATIVA fija dentro del bundle
    (``BUNDLE_RELATIVE_PATHS``). Serializacion determinista: JSON
    canonico (``sort_keys``, sin espacios, sin NaN/Infinity) y CSV con
    cabecera y orden de filas fijos."""
    payloads: dict[str, bytes] = {
        BUNDLE_SUMMARY_RELATIVE_PATH: _canonical_json_bytes(
            build_final_evaluation_summary(outcome)
        ),
        BUNDLE_COVERAGE_RELATIVE_PATH: _csv_bytes(
            build_final_evaluation_coverage_rows(outcome), _COVERAGE_FIELDNAMES
        ),
        BUNDLE_P02_PERFORMANCE_RELATIVE_PATH: _csv_bytes(
            build_final_evaluation_p02_performance_rows(outcome), _P02_FIELDNAMES
        ),
        BUNDLE_SECONDARY_METRICS_RELATIVE_PATH: _csv_bytes(
            build_final_evaluation_secondary_metrics_rows(outcome), _SECONDARY_FIELDNAMES
        ),
        BUNDLE_PERFORMANCE_RELATIVE_PATH: _canonical_json_bytes(
            build_final_evaluation_performance_manifest(
                stage_seconds or {}, operation_counters or {}
            )
        ),
    }
    for relative, payload in payloads.items():
        if len(payload) > MAX_ARTIFACT_BYTES:
            raise _fail("artifact_too_large")
    return payloads


# --------------------------------------------------------------------- #
# Validacion del destino del bundle (antes de leer la fuente)            #
# --------------------------------------------------------------------- #


def validate_bundle_destination(bundle_dir: Path = FINAL_EVALUATION_BUNDLE_DIR) -> None:
    """Solo ``stat``/``exists``/``lstat``: el bundle final NO puede
    preexistir (ni como directorio, ni como symlink/junction, ni como
    archivo suelto), y su directorio padre debe existir y no ser
    symlink. Se comprueba ANTES de leer la fuente contractual (paso 2
    del runner)."""
    if bundle_dir in _PROTECTED_PATHS:
        raise _fail("destination_exists")
    if bundle_dir.is_symlink() or bundle_dir.exists():
        raise _fail("destination_exists")
    parent = bundle_dir.parent
    if parent.is_symlink() or not parent.is_dir():
        raise _fail("destination_exists")


# --------------------------------------------------------------------- #
# Publicacion atomica de bundle: UN unico os.replace de directorio       #
# --------------------------------------------------------------------- #


def _fsync_directory_best_effort(directory: Path) -> None:
    """fsync del directorio cuando el SO lo soporta (POSIX); en Windows
    no existe un equivalente fiable via os.open de un directorio, asi
    que se omite silenciosamente ahi (no es un fallo de publicacion)."""
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _fail("verification_failed")
        result[key] = value
    return result


def _verify_bundle_directory(
    bundle_dir: Path, expected_relative: tuple[str, ...] = BUNDLE_RELATIVE_PATHS
) -> None:
    """Verificacion integral del bundle (temporal ANTES del replace, o
    final DESPUES de el): exactamente los archivos esperados, cada uno
    JSON sin claves duplicadas o CSV con cabecera, dentro del limite de
    tamano. No imprime ni registra contenido."""
    found = {
        str(path.relative_to(bundle_dir)).replace(os.sep, "/")
        for path in bundle_dir.rglob("*")
        if path.is_file()
    }
    if found != set(expected_relative):
        raise _fail("verification_failed")
    for relative in expected_relative:
        path = bundle_dir / relative
        if path.is_symlink() or not path.is_file():
            raise _fail("verification_failed")
        content = path.read_bytes()
        if not content or len(content) > MAX_ARTIFACT_BYTES:
            raise _fail("verification_failed")
        if relative.endswith(".json"):
            try:
                json.loads(content, object_pairs_hook=_no_duplicate_keys)
            except ValueError:
                raise _fail("verification_failed") from None
        elif relative.endswith(".csv"):
            reader = csv.reader(io.StringIO(content.decode("utf-8")))
            if next(reader, None) is None:
                raise _fail("verification_failed")


def publish_final_evaluation_bundle(
    outcome: FinalSealedEvaluationOutcome,
    *,
    bundle_dir: Path = FINAL_EVALUATION_BUNDLE_DIR,
    stage_seconds: dict[str, float] | None = None,
    operation_counters: dict[str, int] | None = None,
) -> Path:
    """Publicacion atomica del bundle completo: EXACTAMENTE UN
    ``os.replace`` de directorio. ``bundle_dir`` es el destino final
    contractual (en tests SIEMPRE un ``tmp_path``, nunca
    ``FINAL_EVALUATION_BUNDLE_DIR`` real). El directorio temporal se
    crea HERMANO de ``bundle_dir`` (mismo padre, mismo sistema de
    archivos) para que el rename final sea atomico.

    El rollback ante fallo previo al replace es seguro UNICAMENTE
    porque ``validate_bundle_destination`` ya confirmo que el bundle
    final NO preexistia: por eso basta eliminar el temporal, nunca hay
    nada previo del llamador que preservar."""
    validate_bundle_destination(bundle_dir)
    payloads = build_final_evaluation_bundle_payloads(
        outcome, stage_seconds=stage_seconds, operation_counters=operation_counters
    )
    parent = bundle_dir.parent
    temp_dir = Path(
        tempfile.mkdtemp(dir=str(parent), prefix=f".{bundle_dir.name}-", suffix=".tmp")
    )
    try:
        (temp_dir / "tables").mkdir()
        for relative, blob in payloads.items():
            target = temp_dir / relative
            with open(target, "wb") as handle:
                handle.write(blob)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass
        _fsync_directory_best_effort(temp_dir / "tables")
        try:
            os.chmod(temp_dir / "tables", stat.S_IRWXU)
            os.chmod(temp_dir, stat.S_IRWXU)
        except OSError:
            pass
        _fsync_directory_best_effort(temp_dir)
        _verify_bundle_directory(temp_dir)
    except _ManagedTerminationSignal:
        # SIGTERM/SIGINT durante la escritura/verificacion: limpia el
        # temporal igual que ante cualquier fallo, pero preserva la
        # identidad de la senal (no la convierte en publish_failed)
        # para que main() distinga terminacion (130) de fallo (1).
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise _fail("publish_failed") from None

    try:
        os.replace(temp_dir, bundle_dir)
    except _ManagedTerminationSignal:
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(bundle_dir, ignore_errors=True)
        raise
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(bundle_dir, ignore_errors=True)
        raise _fail("publish_failed") from None
    _fsync_directory_best_effort(parent)
    return bundle_dir


def verify_published_bundle(bundle_dir: Path = FINAL_EVALUATION_BUNDLE_DIR) -> None:
    """Verificacion posterior del bundle YA publicado (misma logica que
    la verificacion previa al replace, aplicada al destino final)."""
    if not bundle_dir.is_dir() or bundle_dir.is_symlink():
        raise _fail("verification_failed")
    _verify_bundle_directory(bundle_dir)


# --------------------------------------------------------------------- #
# Frontera productiva unica: orden obligatorio de 9 pasos                #
# --------------------------------------------------------------------- #


def _default_source_reader(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _validate_source_path_contract(path: Path) -> None:
    """Pre-check cerrado: ruta fija, regular, sin symlink/junction."""
    if path != FINAL_SEALED_EVALUATION_SOURCE_PATH:
        raise _fail("source_path_contract_violation")
    if path.is_symlink():
        raise _fail("source_path_contract_violation")
    if not path.exists():
        raise _fail("source_path_contract_violation")
    mode = path.stat().st_mode
    if not stat.S_ISREG(mode):
        raise _fail("source_path_contract_violation")


def execute_real_sealed_test_evaluation(
    *,
    source_reader: Callable[[Path], pd.DataFrame] = _default_source_reader,
    bundle_dir: Path = FINAL_EVALUATION_BUNDLE_DIR,
) -> Path:
    """Unica frontera productiva real, en el orden obligatorio de 9
    pasos. Con ``REAL_TEST_EVALUATION_AUTHORIZED=False`` (el unico
    valor posible hoy) aborta en el paso 1, antes de ``open``, ``stat``
    de la fuente, pandas/pyarrow, lectura, construccion o escritura.

    Los pasos 2-9 se ejecutan bajo ``_normalize_execution_signals``:
    un SIGTERM/SIGINT en cualquier punto se convierte en
    ``_ManagedTerminationSignal`` (en vez de matar el proceso sin
    limpieza), permitiendo que ``publish_final_evaluation_bundle``
    borre su temporal igual que ante cualquier otro fallo antes de que
    la senal termine de propagarse."""
    # 1. comprobar la unica puerta.
    if not REAL_TEST_EVALUATION_AUTHORIZED:
        raise SystemExit(REAL_TEST_EVALUATION_BLOCK_REASON)
    with _normalize_execution_signals():
        # 2. validar destino del bundle y fuente contractual (sin leerla).
        validate_bundle_destination(bundle_dir)
        _validate_source_path_contract(FINAL_SEALED_EVALUATION_SOURCE_PATH)
        # 3. cargar una sola vez.
        points = source_reader(FINAL_SEALED_EVALUATION_SOURCE_PATH)
        # 4. adaptar.
        adapted = adapt_real_points_dataframe(points)
        # 5-6. evaluar y agregar (el orquestador hace ambos).
        outcome = evaluate_final_sealed_test(adapted)
        # 7-8. serializar y publicar atomicamente (un unico os.replace).
        published_dir = publish_final_evaluation_bundle(outcome, bundle_dir=bundle_dir)
        # 9. verificar el bundle publicado.
        verify_published_bundle(published_dir)
        return published_dir


# --------------------------------------------------------------------- #
# CLI bloqueado (no se ejecuta durante P22)                              #
# --------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """CLI sin ruta alternativa a la fuente, sin --force, sin --retry
    y sin flag de autorizacion: el unico control es la constante del
    modulo. Con la puerta en False, sale con 1 antes de cualquier I/O.

    Codigos de salida cerrados: ``0`` completado y verificado; ``1``
    puerta cerrada o cualquier fallo (fuente, adaptacion, evaluacion,
    publicacion, verificacion); ``2`` uso invalido (cualquier ``argv``,
    sin excepcion, ni siquiera comprobado contra la puerta); ``130``
    interrupcion (SIGINT/``KeyboardInterrupt`` o SIGTERM gestionado via
    ``_normalize_execution_signals``)."""
    if argv:
        return 2
    if not REAL_TEST_EVALUATION_AUTHORIZED:
        return 1
    try:  # pragma: no cover - deliberadamente sin ejercitar en tests:
        # con la puerta en True este bloque SI es alcanzable, pero
        # invocarlo sin mockear ``source_reader`` dispara una lectura
        # real de la fuente contractual (prohibida fuera de una
        # ejecucion manual autorizada); los tests cubren cada rama de
        # fallo/interrupcion llamando a ``execute_real_sealed_test_
        # evaluation`` directamente con un ``source_reader`` sintetico,
        # nunca a traves de ``main()``.
        execute_real_sealed_test_evaluation()
    except (_ManagedTerminationSignal, KeyboardInterrupt):
        return 130
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    # Bug corregido en revision P23: llamar a main() sin argumentos
    # dejaba el rechazo "if argv: return 2" como codigo muerto en
    # ejecucion real (python -m ...) -- cualquier flag que el usuario
    # escribiese en la linea de comandos (--force, --retry, una ruta)
    # se ignoraba en silencio en vez de rechazarse con salida 2.
    raise SystemExit(main(sys.argv[1:])) from None


__all__ = (
    "BUNDLE_COVERAGE_RELATIVE_PATH",
    "BUNDLE_P02_PERFORMANCE_RELATIVE_PATH",
    "BUNDLE_PERFORMANCE_RELATIVE_PATH",
    "BUNDLE_RELATIVE_PATHS",
    "BUNDLE_SECONDARY_METRICS_RELATIVE_PATH",
    "BUNDLE_SUMMARY_RELATIVE_PATH",
    "FINAL_EVALUATION_BUNDLE_DIR",
    "FINAL_SEALED_EVALUATION_RUNNER_CONTRACT_VERSION",
    "FINAL_SEALED_EVALUATION_SOURCE_PATH",
    "MAX_ARTIFACT_BYTES",
    "FinalSealedEvaluationRunnerError",
    "build_final_evaluation_bundle_payloads",
    "build_final_evaluation_coverage_rows",
    "build_final_evaluation_p02_performance_rows",
    "build_final_evaluation_performance_manifest",
    "build_final_evaluation_secondary_metrics_rows",
    "build_final_evaluation_summary",
    "execute_real_sealed_test_evaluation",
    "main",
    "publish_final_evaluation_bundle",
    "validate_bundle_destination",
    "verify_published_bundle",
)
