"""P27: genera la snapshot de demostración pública, pequeña y determinista
``demo/tactical-recommendations-demo-v1.json``.

Este script NO lee ``points_enriched.parquet``, NO abre el snapshot
privado real, NO ejecuta el pipeline offline P14/P15/P16 sobre datos
reales y NO toca ``reports/`` ni ``data/``. Construye, exclusivamente en
memoria, un puñado de intentos sintéticos (nombres de fantasía, fechas
futuras claramente ficticias) y los hace pasar por la MISMA tubería de
producción que ya usan P10-P13 (extracción de señales → encoder →
evidencia de enfrentamiento → priorización → snapshot persistido), de
modo que el archivo resultante es estructuralmente indistinguible de uno
real: mismo ``contract``/``schema_version``/``format_version``, mismos
fingerprints SHA-256, mismas reglas de abstención y mínimos de evidencia
(50 activaciones etiquetadas / 5 partidos distintos).

Determinismo: ninguna entrada depende de reloj, aleatoriedad ni orden de
iteración externo -- ejecutar este script dos veces produce EXACTAMENTE
los mismos bytes (ver ``tests/test_generate_demo_snapshot.py``).

Ejecución:
    python scripts/generate_demo_snapshot.py

Contenido garantizado (ver ``_DEMO_MATCHUPS`` más abajo):
- 3 jugadores sintéticos (papel ``player``): Ana Ibarra, Marco Rossi,
  Priya Verma.
- Varios rivales y fechas de corte válidas por jugador.
- Los cuatro patrones P02/P04/P05/P06 con evidencia suficiente
  (estado ``ranked``) en varias parejas.
- Al menos una pareja con evidencia deliberadamente insuficiente
  (``abstained_insufficient_evidence``), no ausencia total de datos.
- Cero rutas, identidades de partido/punto o secuencias privadas: los
  ``match_id`` son literales sintéticos con prefijo ``demo-``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    # Permite ejecutar "python scripts/generate_demo_snapshot.py" desde
    # cualquier directorio de trabajo, sin depender de PYTHONPATH ni de
    # una ruta absoluta fija: se resuelve siempre desde la ubicacion de
    # este propio archivo, relativa a la raiz del repositorio.
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))

from src.recommender.persisted_tactical_recommendation_provider import (
    build_persisted_tactical_recommendation_snapshot,
    persist_persisted_tactical_recommendation_snapshot,
    serialize_persisted_tactical_recommendation_snapshot,
)
from src.recommender.tactical_feature_encoder import (
    TacticalEncodingPolicy,
    build_tactical_feature_schema,
    encode_tactical_attempt,
    feature_schema_fingerprint,
)
from src.recommender.tactical_history_profiles import (
    HISTORY_CONTRACT_VERSION,
    OBSERVATION_PROVENANCE,
    TacticalHistoricalObservation,
    TacticalLabelAvailability,
    TacticalOutcomeLabel,
    TacticalPlayerRole,
    validate_tactical_historical_observation,
)
from src.recommender.tactical_matchup_evidence import (
    MATCHUP_EVIDENCE_CONTRACT_VERSION,
    TacticalEvidenceScopeStrategy,
    TacticalMatchupQuery,
    build_tactical_matchup_evidence,
    validate_tactical_matchup_evidence,
)
from src.recommender.tactical_prioritization import (
    TacticalPrioritizationResult,
    prioritize_tactical_matchup,
    validate_tactical_prioritization_result,
)
from src.recommender.tactical_signal_orchestrator import (
    ORCHESTRATOR_CONTRACT_VERSION,
    AttemptSignalRequest,
    extract_tactical_signals_for_attempt,
)


ROOT: Path = Path(__file__).resolve().parents[1]
DEMO_SNAPSHOT_PATH: Path = ROOT / "demo" / "tactical-recommendations-demo-v1.json"

SCORING_PATTERNS: tuple[str, ...] = ("P02", "P04", "P05", "P06")
SERVE_DIRECTIONS: tuple[tuple[str, str], ...] = (("4", "wide"), ("5", "body"), ("6", "T"))
SHOT_TYPES: str = "fbrsvzopuylmhijkt"

# Umbral de evidencia RICA (por encima del minimo real de 50/5, con
# margen) vs. DELIBERADAMENTE INSUFICIENTE (por debajo de ambos
# minimos: menos de 50 activaciones y menos de 5 partidos distintos).
RICH_ATTEMPTS_PER_CATEGORY: int = 60
THIN_ATTEMPTS_PER_CATEGORY: int = 3


@dataclass(frozen=True, slots=True)
class _SyntheticAttempt:
    match_id: str
    point_number: int
    effective_date: date
    server: str
    returner: str
    sequence: str
    serve_number: int
    previous_attempt_was_fault: bool
    server_won_point: bool


def _attempt(
    *,
    match_id: str,
    point_number: int,
    effective_date: date,
    server: str,
    returner: str,
    sequence: str,
    serve_number: int = 1,
    previous_fault: bool = False,
    success: bool = True,
) -> _SyntheticAttempt:
    return _SyntheticAttempt(
        match_id, point_number, effective_date, server, returner,
        sequence, serve_number, previous_fault, success,
    )


def _synthetic_history(
    *,
    namespace: str,
    letter: str,
    base_date: date,
    player: str,
    opponent: str,
    attempts_per_category: int,
) -> tuple[_SyntheticAttempt, ...]:
    """Historial 100% sintético que cubre P02 (dirección del saque) y,
    en la misma secuencia MCP, P04+P05+P06 (dirección/profundidad/tipo
    del resto). ``attempts_per_category`` controla si la evidencia
    resultante queda por ENCIMA (``RICH_...``) o por DEBAJO
    (``THIN_...``) de los mínimos contractuales -- nunca en cero,
    para que el resultado sea una abstención real, no ausencia total
    de datos."""
    attempts: list[_SyntheticAttempt] = []
    point = 1
    for perspective in ("executor", "opponent"):
        for serve_code, category in SERVE_DIRECTIONS:
            for index in range(attempts_per_category):
                server = player if perspective == "executor" else f"{namespace}P02{category}"
                returner = (
                    f"{namespace}P02{category}" if perspective == "executor" else opponent
                )
                attempts.append(
                    _attempt(
                        match_id=f"demo-{namespace}-p02-{perspective}-{category}-{index // 3}",
                        point_number=point,
                        effective_date=base_date + timedelta(days=index % 20),
                        server=server,
                        returner=returner,
                        sequence=f"{letter}{serve_code}#",
                        success=index % 2 == 0,
                    )
                )
                point += 1
    for perspective in ("executor", "opponent"):
        for shot_index, shot_type in enumerate(SHOT_TYPES):
            lateral = str(shot_index % 3 + 1)
            depth = str(shot_index % 3 + 7)
            for index in range(attempts_per_category):
                server = (
                    f"{namespace}P06{shot_type}" if perspective == "executor" else opponent
                )
                returner = (
                    player if perspective == "executor" else f"{namespace}P06{shot_type}"
                )
                second_serve = index % 2 == 1
                attempts.append(
                    _attempt(
                        match_id=f"demo-{namespace}-p06-{perspective}-{shot_type}-{index // 3}",
                        point_number=point,
                        effective_date=base_date + timedelta(days=index % 20),
                        server=server,
                        returner=returner,
                        sequence=f"{letter}4{shot_type}{lateral}{depth}",
                        serve_number=2 if second_serve else 1,
                        previous_fault=second_serve and index % 4 == 1,
                        success=index % 2 != 0,
                    )
                )
                point += 1
    return tuple(attempts)


def _observations_from_attempts(
    attempts: tuple[_SyntheticAttempt, ...], schema
) -> tuple[TacticalHistoricalObservation, ...]:
    observations: list[TacticalHistoricalObservation] = []
    for attempt in attempts:
        request = AttemptSignalRequest(
            ORCHESTRATOR_CONTRACT_VERSION,
            attempt.sequence,
            attempt.serve_number,
            attempt.previous_attempt_was_fault,
        )
        extraction = extract_tactical_signals_for_attempt(request)
        vector = encode_tactical_attempt(extraction, schema)
        observation = TacticalHistoricalObservation(
            HISTORY_CONTRACT_VERSION,
            attempt.match_id,
            attempt.point_number,
            attempt.serve_number,
            attempt.effective_date,
            attempt.server,
            attempt.returner,
            TacticalPlayerRole.SERVER,
            vector,
            (
                TacticalOutcomeLabel.SERVER_WON_POINT
                if attempt.server_won_point
                else TacticalOutcomeLabel.RETURNER_WON_POINT
            ),
            TacticalLabelAvailability.AVAILABLE,
            OBSERVATION_PROVENANCE,
        )
        validate_tactical_historical_observation(observation)
        observations.append(observation)
    return tuple(observations)


def _matchup_query(
    player: str, opponent: str, as_of: date, schema
) -> TacticalMatchupQuery:
    return TacticalMatchupQuery(
        MATCHUP_EVIDENCE_CONTRACT_VERSION,
        player,
        opponent,
        as_of,
        TacticalEncodingPolicy.COMPONENT_ONLY,
        feature_schema_fingerprint(schema),
        (1, 2),
        None,
        50,
        5,
        SCORING_PATTERNS,
        TacticalEvidenceScopeStrategy.GLOBAL_ONLY,
    )


def _prioritization_for(
    *,
    namespace: str,
    letter: str,
    base_date: date,
    player: str,
    opponent: str,
    as_of: date,
    attempts_per_category: int,
) -> TacticalPrioritizationResult:
    schema = build_tactical_feature_schema(TacticalEncodingPolicy.COMPONENT_ONLY)
    attempts = _synthetic_history(
        namespace=namespace, letter=letter, base_date=base_date,
        player=player, opponent=opponent,
        attempts_per_category=attempts_per_category,
    )
    observations = _observations_from_attempts(attempts, schema)
    query = _matchup_query(player, opponent, as_of, schema)
    evidence = build_tactical_matchup_evidence(observations, query, schema)
    validate_tactical_matchup_evidence(evidence)
    result = prioritize_tactical_matchup(evidence, top_k=3)
    validate_tactical_prioritization_result(result)
    return result


# --------------------------------------------------------------------- #
# Universo de demostracion: 3 jugadores, varios rivales y fechas, las   #
# 4 oportunidades tacticas, y al menos una abstencion real (evidencia   #
# insuficiente, nunca ausencia total). Nombres de fantasia, fechas      #
# futuras claramente ficticias (2031) -- ninguna coincide con datos     #
# reales ni con el test sellado 2024-2026.                              #
# --------------------------------------------------------------------- #

_DEMO_MATCHUPS: tuple[dict[str, object], ...] = (
    {
        "namespace": "AnaMarco1", "letter": "",
        "base_date": date(2030, 12, 20),
        "player": "Ana Ibarra", "opponent": "Marco Rossi",
        "as_of": date(2031, 1, 15),
        "attempts_per_category": RICH_ATTEMPTS_PER_CATEGORY,
    },
    {
        "namespace": "AnaMarco2", "letter": "c",
        "base_date": date(2031, 2, 20),
        "player": "Ana Ibarra", "opponent": "Marco Rossi",
        "as_of": date(2031, 3, 20),
        "attempts_per_category": THIN_ATTEMPTS_PER_CATEGORY,
    },
    {
        "namespace": "AnaPriya", "letter": "",
        "base_date": date(2031, 1, 20),
        "player": "Ana Ibarra", "opponent": "Priya Verma",
        "as_of": date(2031, 2, 10),
        "attempts_per_category": RICH_ATTEMPTS_PER_CATEGORY,
    },
    {
        "namespace": "MarcoPriya", "letter": "",
        "base_date": date(2031, 3, 10),
        "player": "Marco Rossi", "opponent": "Priya Verma",
        "as_of": date(2031, 4, 5),
        "attempts_per_category": RICH_ATTEMPTS_PER_CATEGORY,
    },
    {
        "namespace": "MarcoAna", "letter": "c",
        "base_date": date(2031, 4, 10),
        "player": "Marco Rossi", "opponent": "Ana Ibarra",
        "as_of": date(2031, 5, 1),
        "attempts_per_category": RICH_ATTEMPTS_PER_CATEGORY,
    },
    {
        "namespace": "PriyaAna", "letter": "",
        "base_date": date(2031, 5, 10),
        "player": "Priya Verma", "opponent": "Ana Ibarra",
        "as_of": date(2031, 6, 12),
        "attempts_per_category": THIN_ATTEMPTS_PER_CATEGORY,
    },
)


def build_demo_results() -> tuple[TacticalPrioritizationResult, ...]:
    """Construye, en orden determinista, todos los resultados de
    demostración a partir de ``_DEMO_MATCHUPS``."""
    return tuple(
        _prioritization_for(
            namespace=matchup["namespace"],
            letter=matchup["letter"],
            base_date=matchup["base_date"],
            player=matchup["player"],
            opponent=matchup["opponent"],
            as_of=matchup["as_of"],
            attempts_per_category=matchup["attempts_per_category"],
        )
        for matchup in _DEMO_MATCHUPS
    )


def build_demo_snapshot_bytes() -> bytes:
    """Bytes canónicos deterministas del snapshot de demostración
    completo, sin escribir nada a disco (para verificación de
    determinismo sin tocar el archivo versionado)."""
    results = build_demo_results()
    snapshot = build_persisted_tactical_recommendation_snapshot(results)
    return serialize_persisted_tactical_recommendation_snapshot(snapshot)


def main() -> None:
    results = build_demo_results()
    snapshot = build_persisted_tactical_recommendation_snapshot(results)
    DEMO_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    persist_persisted_tactical_recommendation_snapshot(snapshot, DEMO_SNAPSHOT_PATH)
    print(f"Snapshot de demostración escrita en: {DEMO_SNAPSHOT_PATH}")
    print(f"Entradas: {snapshot.entry_count}")


if __name__ == "__main__":
    main()


__all__ = (
    "DEMO_SNAPSHOT_PATH",
    "build_demo_results",
    "build_demo_snapshot_bytes",
    "main",
)
