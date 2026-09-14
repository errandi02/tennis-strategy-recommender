"""Provider persistido P13: catalogo privado, versionado y de solo lectura.

Conexa la API P12 con recomendaciones precomputadas sin recalcular P10
por request. El snapshot privado persiste, por clave de consulta
``(player, opponent, as_of_date)``, un ``TacticalPrioritizationResult``
completo mediante un schema JSON explicito y reconstructivamente
validable. No ejecuta P10, no lee datos, no usa pandas/NumPy/PyArrow/
sklearn, no usa red, no usa pickle y no instancia nada al importar.

Diseno del payload (schema cerrado, version 1.0.0, formato 1):

- Un unico objeto JSON UTF-8 compacto (``sort_keys=True``,
  ``ensure_ascii=False``, ``allow_nan=False``, sin newline final) que
  contiene metadata contractual, la politica congelada permitida y una
  lista de entradas en orden canonico ``(player, opponent, as_of_date)``.
- Cada entrada persiste la clave privada completa, el resultado
  upstream representado por fields explicitos (enumeraciones por su
  valor textual y tuplas de pares como matrices ordenadas
  ``[clave, valor]`` para preservar el orden contractual que los
  validadores P10/P11 verifican), el fingerprint upstream publico del
  resultado y un fingerprint de entrada propio.
- Los fingerprints son SHA-256 mayuscula (64 hex) con separacion de
  dominio exclusiva de P13; el fingerprint del snapshot cubre toda la
  semantica contractual excluyendo su propio campo y no depende de
  rutas, timestamps, orden de insercion ni plataforma.

Frontera de I/O: el unico acceso a disco ocurre en
``load_persisted_tactical_recommendation_snapshot``,
``persist_persisted_tactical_recommendation_snapshot`` y
``verify_persisted_tactical_recommendation_snapshot``. La persistencia
es atomica (temporal en el mismo directorio, flush, fsync,
``os.replace``). Ante un fallo anterior al replace se limpian temporales
y se preserva el destino previo; un fallo posterior no implica rollback.
El provider inmutable no toca disco tras construirse.

``load`` levanta los errores internos P13 finos;
``create_persisted_tactical_recommendation_provider`` es el unico punto
que los traduce a errores de servicio P12. Mapeo cerrado (buenas
practicas: los errores publicos son instancias nuevas sin ``__cause__``
ni ``__context__``):

==============================  ================================  =====
Caso interno P13                  Error de servicio P12           HTTP
==============================  ================================  =====
clave ausente                     RecommendationNotFoundError     404
snapshot ausente                  ProviderUnavailableError        503
snapshot ilegible (I/O, directorio,
symlink, tamano 0 o abusivo)     ProviderUnavailableError        503
schema/contract/version
incompatible                      ProviderUnavailableError        503
JSON corrupto (no UTF-8, invalido,
claves duplicadas, NaN/Infinity) UpstreamContractViolationError  502
integridad (contador, orden,
fingerprint, identidad, claves)  UpstreamContractViolationError  502
resultado upstream invalido       UpstreamContractViolationError  502
fallo inesperado                  InternalServiceError            500
query invalida (uso directo)      InvalidRequestError             422
==============================  ================================  =====

La funcion ``persist_persisted_tactical_recommendation_snapshot`` es
herramienteria offline y eleva los errores internos P13 directamente.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date
from hashlib import sha256
from itertools import islice
import json
import math
import os
import re
import stat
import tempfile
from types import MappingProxyType
from typing import Final
from pathlib import Path

from src.recommender.tactical_feature_encoder import TacticalEncodingPolicy
from src.recommender.tactical_prioritization import (
    PRIORITIZATION_CONTRACT_VERSION,
    TacticalCandidateState,
    TacticalEvidenceSummary,
    TacticalMatchupQuerySummary,
    TacticalPatternRanking,
    TacticalPrioritizationResult,
    TacticalPrioritizationState,
    TacticalRankingState,
    TacticalScoredCandidate,
    TacticalScoringPolicy,
    default_tactical_scoring_policy,
    tactical_prioritization_result_fingerprint,
    validate_tactical_prioritization_result,
)
from src.recommender.tactical_recommendation_service import (
    MAX_IDENTIFIER_LENGTH,
    InternalServiceError,
    InvalidRequestError,
    ProviderUnavailableError,
    RecommendationNotFoundError,
    TacticalRecommendationQuery,
    UpstreamContractViolationError,
    validate_tactical_recommendation_query,
)


SNAPSHOT_CONTRACT_NAME: Final = "persisted_tactical_recommendation_snapshot"
SNAPSHOT_SCHEMA_VERSION: Final = "1.0.0"
SNAPSHOT_FORMAT_VERSION: Final = 1
SNAPSHOT_UPSTREAM_RESULT_CONTRACT: Final = "tactical_prioritization_result"
SNAPSHOT_ENTRY_DOMAIN: Final = b"tennis-persisted-tactical-recommendation-entry\x00"
SNAPSHOT_SNAPSHOT_DOMAIN: Final = b"tennis-persisted-tactical-recommendation-snapshot\x00"

# Limite exacto del snapshot (P16, 256 MiB). Proposito exclusivo:
# snapshots privados offline de generacion unica manual, no hosting.
# Dimensionamiento (auditoria P16 sobre fixtures sinteticos, no medicion real
# ni garantia de tamano final):
# - universo objetivo p10_offline: 3.610 entradas (1.805 pares);
# - entrada representativa estimada con 352 objetivos sinteticos: ~54,5 KB
#   (51,6 minimo / 66,7 max por pareja);
# - snapshot representativo estimado: 196.817.992 B (~187,7 MiB),
#   73,3 % del limite; quedan 71.617.464 B (26,7 % del limite) y el
#   limite es aproximadamente 36,4 % mayor que la estimacion;
#   por tanto cumple el criterio congelado de al menos 20 % libre;
# - techo adversarial de una entrada (esquema 1.0.0 saturado:
#   26 candidatos, catalogo cerrado, identificadores a 64 chars):
#   102.352 B observados en fixtures sinteticos y techo contractual
#   conservador redondeado a 110.000 B;
# - techo adversarial del snapshot (3.610 entradas saturadas):
#   ~379,1 MiB > limite; el limite es deliberadamente inferior como
#   defensa final contra ficheros hostiles. Si un snapshot
#   legitimamente generado superara 256 MiB, la construccion P14
#   falla con razon cerrada (p13_snapshot_build_failed) y se requiere
#   una decision humana explicita de redimensionamiento.
MAX_SNAPSHOT_BYTES: Final = 256 * 1024 * 1024
MAX_ENTRIES: Final = 100_000
CAPACITY_DESIGN_UNIVERSE_ENTRIES: Final = 3_610
CAPACITY_DESIGN_SYNTHETIC_TARGETS_MEASURED: Final = 352
CAPACITY_DESIGN_TYPICAL_ENTRY_BYTES: Final = 54_520
CAPACITY_DESIGN_MAX_ENTRY_BYTES: Final = 110_000
CAPACITY_DESIGN_REPRESENTATIVE_SNAPSHOT_BYTES: Final = 196_817_992
CAPACITY_DESIGN_ADVERSARIAL_SNAPSHOT_BYTES: Final = 397_100_000
# Estimacion conservadora de pico operativo; no medicion de una ejecucion real.
CAPACITY_DESIGN_MEMORY_PEAK_BYTES: Final = 2_500_000_000
SNAPSHOT_PURPOSE: Final = (
    "exclusivo para snapshots privados offline de generacion unica manual"
)
MAX_JSON_DEPTH: Final = 64
MAX_JSON_STRING_LENGTH: Final = 4096
MAX_JSON_ARRAY_ITEMS: Final = 100_000
MAX_JSON_OBJECT_KEYS: Final = 128
MAX_JSON_INTEGER_ABS: Final = 9_007_199_254_740_991

_CIVIL_ISO_DATE: Final = re.compile(r"\d{4}-\d{2}-\d{2}")
_SHA256_UPPER: Final = re.compile(r"[0-9A-F]{64}")
_KEY_CONTROL_CHARACTERS: Final = frozenset(
    tuple(range(0x00, 0x20)) + (0x7F,) + tuple(range(0x80, 0xA0))
)
_KEY_FORBIDDEN_FRAGMENTS: Final = ("/", "\\", "://", "..", "~")

_SNAPSHOT_PAYLOAD_KEYS: Final = frozenset(
    {
        "contract",
        "entry_count",
        "entries",
        "fingerprint",
        "format_version",
        "policies",
        "schema_version",
        "upstream_result_contract",
        "upstream_result_contract_version",
    }
)
_ENTRY_PAYLOAD_KEYS: Final = frozenset(
    {"as_of_date", "entry_fingerprint", "opponent", "player", "result", "result_fingerprint"}
)


class PersistedSnapshotError(RuntimeError):
    """Error interno cerrado de P13.

    No conserva argumentos, query, IDs, rutas, contenido de ficheros ni
    excepciones originales; mapea a un error de servicio P12 en cada
    frontera publica.
    """

    def __init__(self) -> None:
        super().__init__()


class SnapshotNotFoundError(PersistedSnapshotError):
    """La ruta del snapshot no existe."""


class SnapshotUnavailableError(PersistedSnapshotError):
    """El snapshot no es accesible: directory, symlink, I/O, permisos,
    tamano cero o limite de tamano excedido."""


class SnapshotMalformedError(PersistedSnapshotError):
    """Contenido no decodificable o JSON corrupto: no UTF-8, JSON
    invalido, claves duplicadas, NaN/Infinity, profundidad abusiva o
    tipos fuera del arbol contractual."""


class SnapshotIncompatibleError(PersistedSnapshotError):
    """Schema, contract name, version de schema o formato incompatible,
    claves extra/faltantes, tipos JSON incorrectos o catalogo de
    politicas fuera del dominio."""


class SnapshotIntegrityError(PersistedSnapshotError):
    """Violacion de integridad contractual: contador incoherente, orden
    no canonico, claves duplicadas, fingerprint (entrada o snapshot)
    incorrecto, identidad clave-resultado divergente o clave fuera del
    dominio cerrado."""


class SnapshotUpstreamInvalidError(PersistedSnapshotError):
    """El resultado persistido no reconstruye un
    ``TacticalPrioritizationResult`` validado del contrato P10/P11."""


_SNAPSHOT_ERROR_TO_SERVICE_ERROR: Final = MappingProxyType(
    {
        SnapshotNotFoundError: ProviderUnavailableError,
        SnapshotUnavailableError: ProviderUnavailableError,
        SnapshotIncompatibleError: ProviderUnavailableError,
        SnapshotMalformedError: UpstreamContractViolationError,
        SnapshotIntegrityError: UpstreamContractViolationError,
        SnapshotUpstreamInvalidError: UpstreamContractViolationError,
    }
)


def _strict_sha(value: object, field_name: str) -> str:
    if (
        type(value) is not str
        or _SHA256_UPPER.fullmatch(value) is None
    ):
        raise SnapshotIntegrityError()
    return value


def _strict_key_identifier(value: object) -> None:
    if type(value) is not str:
        raise SnapshotIntegrityError()
    if not value or value != value.strip() or len(value) > MAX_IDENTIFIER_LENGTH:
        raise SnapshotIntegrityError()
    if any(ord(character) in _KEY_CONTROL_CHARACTERS for character in value):
        raise SnapshotIntegrityError()
    if any(fragment in value for fragment in _KEY_FORBIDDEN_FRAGMENTS):
        raise SnapshotIntegrityError()


@dataclass(frozen=True)
class TacticalRecommendationSnapshotKey:
    """Clave privada de consulta: exactamente la trada P12."""

    __slots__ = ("player", "opponent", "as_of_date")

    player: str
    opponent: str
    as_of_date: date

    def __post_init__(self) -> None:
        validate_tactical_recommendation_snapshot_key(self)


def validate_tactical_recommendation_snapshot_key(key: object) -> None:
    """Valida la clave privada contra el dominio cerrado de P12."""
    if type(key) is not TacticalRecommendationSnapshotKey:
        raise SnapshotIntegrityError()
    _strict_key_identifier(key.player)
    _strict_key_identifier(key.opponent)
    if key.player == key.opponent:
        raise SnapshotIntegrityError()
    if type(key.as_of_date) is not date:
        raise SnapshotIntegrityError()


def _snapshot_fingerprint_of(payload: dict[str, object]) -> str:
    return sha256(
        SNAPSHOT_SNAPSHOT_DOMAIN + _canonical_json_bytes(payload)
    ).hexdigest().upper()


def _entry_fingerprint_of(
    key: TacticalRecommendationSnapshotKey, result_fingerprint: str
) -> str:
    payload = {
        "as_of_date": key.as_of_date.isoformat(),
        "opponent": key.opponent,
        "player": key.player,
        "result_fingerprint": result_fingerprint,
    }
    return sha256(SNAPSHOT_ENTRY_DOMAIN + _canonical_json_bytes(payload)).hexdigest().upper()


def _pairs_payload(pairs: tuple[tuple[str, ...], ...]) -> list[list[object]]:
    return [[key, value] for key, value in pairs]


def _policy_payload(policy: TacticalScoringPolicy) -> dict[str, object]:
    return {
        "abstention_rules": list(policy.abstention_rules),
        "allowed_scope": policy.allowed_scope,
        "comparability_rule": policy.comparability_rule,
        "contract_version": policy.contract_version,
        "executor_weight": policy.executor_weight,
        "formula": policy.formula,
        "interpretation": policy.interpretation,
        "name": policy.name,
        "opponent_allowed_weight": policy.opponent_allowed_weight,
        "reconciliations": _pairs_payload(policy.reconciliations),
        "uncertainty_method": policy.uncertainty_method,
        "version": policy.version,
    }


def _summary_payload(summary: TacticalEvidenceSummary) -> dict[str, object]:
    return {
        "distinct_matches": summary.distinct_matches,
        "evidence_state": summary.evidence_state,
        "failures": summary.failures,
        "labeled_attempts": summary.labeled_attempts,
        "perspective": summary.perspective,
        "scope": summary.scope,
        "success_rate": summary.success_rate,
        "successes": summary.successes,
        "wilson_lower": summary.wilson_lower,
        "wilson_upper": summary.wilson_upper,
    }


def _candidate_payload(candidate: TacticalScoredCandidate) -> dict[str, object]:
    return {
        "actor": candidate.actor,
        "category": candidate.category,
        "combined_lower": candidate.combined_lower,
        "combined_rate": candidate.combined_rate,
        "combined_upper": candidate.combined_upper,
        "combined_width": candidate.combined_width,
        "comparable": candidate.comparable,
        "contract_version": candidate.contract_version,
        "disagreement": candidate.disagreement,
        "evidence_floor": candidate.evidence_floor,
        "encoder_policy": candidate.encoder_policy.value,
        "executor_evidence": _summary_payload(candidate.executor_evidence),
        "feature_name": candidate.feature_name,
        "match_floor": candidate.match_floor,
        "opponent_allowed_evidence": _summary_payload(candidate.opponent_allowed_evidence),
        "pattern_id": candidate.pattern_id,
        "provenance": _pairs_payload(candidate.provenance),
        "rank_eligible": candidate.rank_eligible,
        "rank_position": candidate.rank_position,
        "query_fingerprint": candidate.query_fingerprint,
        "reason_codes": list(candidate.reason_codes),
        "reconciliations": _pairs_payload(candidate.reconciliations),
        "score_formula": candidate.score_formula,
        "schema_fingerprint": candidate.schema_fingerprint,
        "source_category_fingerprint": candidate.source_category_fingerprint,
        "state": candidate.state.value,
        "tactical_opportunity": candidate.tactical_opportunity,
        "tie_group": candidate.tie_group,
        "uncertainty_label": candidate.uncertainty_label,
    }


def _ranking_payload(ranking: TacticalPatternRanking) -> dict[str, object]:
    return {
        "abstained_candidate_count": ranking.abstained_candidate_count,
        "boundary_tie_expanded": ranking.boundary_tie_expanded,
        "candidates": [_candidate_payload(item) for item in ranking.candidates],
        "contract_version": ranking.contract_version,
        "contractual_purpose": ranking.contractual_purpose,
        "effective_top_k": ranking.effective_top_k,
        "encoder_policy": ranking.encoder_policy.value,
        "order_rule": ranking.order_rule,
        "pattern_id": ranking.pattern_id,
        "provenance": _pairs_payload(ranking.provenance),
        "reason_codes": list(ranking.reason_codes),
        "reconciliations": _pairs_payload(ranking.reconciliations),
        "redundant_representation": ranking.redundant_representation,
        "requested_top_k": ranking.requested_top_k,
        "scored_candidate_count": ranking.scored_candidate_count,
        "scoring_policy_fingerprint": ranking.scoring_policy_fingerprint,
        "state": ranking.state.value,
        "tactical_opportunity": ranking.tactical_opportunity,
        "tie_group_count": ranking.tie_group_count,
        "top_candidates": [_candidate_payload(item) for item in ranking.top_candidates],
    }


def _query_summary_payload(
    summary: TacticalMatchupQuerySummary,
) -> dict[str, object]:
    return {
        "as_of_date": summary.as_of_date,
        "encoder_policy": summary.encoder_policy.value,
        "minimum_labeled_attempts": summary.minimum_labeled_attempts,
        "minimum_matches": summary.minimum_matches,
        "opponent": summary.opponent,
        "player": summary.player,
        "query_fingerprint": summary.query_fingerprint,
        "requested_patterns": list(summary.requested_patterns),
        "scope_strategy": summary.scope_strategy,
        "serve_numbers": list(summary.serve_numbers),
        "window_days": summary.window_days,
    }


def _result_payload(result: TacticalPrioritizationResult) -> dict[str, object]:
    return {
        "abstained_candidate_count": result.abstained_candidate_count,
        "available_pattern_count": result.available_pattern_count,
        "contract_version": result.contract_version,
        "limitations": list(result.limitations),
        "matchup_query": _query_summary_payload(result.matchup_query),
        "partial_or_unavailable_pattern_count": (
            result.partial_or_unavailable_pattern_count
        ),
        "policy": _policy_payload(result.policy),
        "provenance": _pairs_payload(result.provenance),
        "rankings": [_ranking_payload(item) for item in result.rankings],
        "reason_codes": list(result.reason_codes),
        "reconciliations": _pairs_payload(result.reconciliations),
        "redundant_representation": result.redundant_representation,
        "requested_pattern_count": result.requested_pattern_count,
        "schema_fingerprint": result.schema_fingerprint,
        "scored_candidate_count": result.scored_candidate_count,
        "source_evidence_fingerprint": result.source_evidence_fingerprint,
        "state": result.state.value,
        "total_candidate_count": result.total_candidate_count,
    }


def _entry_payload(entry: "PersistedTacticalRecommendationEntry") -> dict[str, object]:
    return {
        "as_of_date": entry.key.as_of_date.isoformat(),
        "entry_fingerprint": entry.entry_fingerprint,
        "opponent": entry.key.opponent,
        "player": entry.key.player,
        "result": _result_payload(entry.result),
        "result_fingerprint": entry.result_fingerprint,
    }


def _snapshot_payload(
    snapshot: "PersistedTacticalRecommendationSnapshot", *, include_fingerprint: bool
) -> dict[str, object]:
    payload: dict[str, object] = {
        "contract": snapshot.contract_name,
        "entry_count": snapshot.entry_count,
        "entries": [_entry_payload(item) for item in snapshot.entries],
        "format_version": snapshot.format_version,
        "policies": [_policy_payload(item) for item in snapshot.policies],
        "schema_version": snapshot.schema_version,
        "upstream_result_contract": snapshot.upstream_result_contract,
        "upstream_result_contract_version": snapshot.upstream_result_contract_version,
    }
    if include_fingerprint:
        payload["fingerprint"] = snapshot.fingerprint
    return payload


@dataclass(frozen=True)
class PersistedTacticalRecommendationEntry:
    """Entrada persistida: clave privada + resultado upstream completo."""

    __slots__ = ("key", "result", "result_fingerprint", "entry_fingerprint")

    key: TacticalRecommendationSnapshotKey
    result: TacticalPrioritizationResult
    result_fingerprint: str
    entry_fingerprint: str

    def __post_init__(self) -> None:
        validate_persisted_tactical_recommendation_entry(self)


def validate_persisted_tactical_recommendation_entry(entry: object) -> None:
    """Valida una entrada reconstruida o construida contra el contrato."""
    if type(entry) is not PersistedTacticalRecommendationEntry:
        raise SnapshotIntegrityError()
    validate_tactical_recommendation_snapshot_key(entry.key)
    if type(entry.result) is not TacticalPrioritizationResult:
        raise SnapshotUpstreamInvalidError()
    upstream_invalid = False
    try:
        validate_tactical_prioritization_result(entry.result)
    except Exception:
        upstream_invalid = True
    if upstream_invalid:
        raise SnapshotUpstreamInvalidError()
    if entry.result.policy != default_tactical_scoring_policy():
        raise SnapshotIntegrityError()
    summary = entry.result.matchup_query
    if (
        summary.player != entry.key.player
        or summary.opponent != entry.key.opponent
        or summary.as_of_date != entry.key.as_of_date.isoformat()
    ):
        raise SnapshotIntegrityError()
    _strict_sha(entry.result_fingerprint, "result_fingerprint")
    upstream_invalid = False
    try:
        expected_result_fingerprint = tactical_prioritization_result_fingerprint(
            entry.result
        )
    except Exception:
        upstream_invalid = True
        expected_result_fingerprint = None
    if upstream_invalid:
        raise SnapshotUpstreamInvalidError()
    if entry.result_fingerprint != expected_result_fingerprint:
        raise SnapshotIntegrityError()
    _strict_sha(entry.entry_fingerprint, "entry_fingerprint")
    if (
        entry.entry_fingerprint
        != _entry_fingerprint_of(entry.key, entry.result_fingerprint)
    ):
        raise SnapshotIntegrityError()


@dataclass(frozen=True)
class PersistedTacticalRecommendationSnapshot:
    """Catalogo privado inmutable, ordenado canonicamente y firmado."""

    __slots__ = (
        "contract_name",
        "schema_version",
        "format_version",
        "entry_count",
        "policies",
        "upstream_result_contract",
        "upstream_result_contract_version",
        "entries",
        "fingerprint",
        "index",
    )

    contract_name: str
    schema_version: str
    format_version: int
    entry_count: int
    policies: tuple[TacticalScoringPolicy, ...]
    upstream_result_contract: str
    upstream_result_contract_version: str
    entries: tuple[PersistedTacticalRecommendationEntry, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        validate_persisted_tactical_recommendation_snapshot(self)
        object.__setattr__(
            self,
            "index",
            MappingProxyType({entry.key: entry for entry in self.entries}),
        )


def _key_sort_value(entry: PersistedTacticalRecommendationEntry) -> tuple[str, str, str]:
    return (
        entry.key.player,
        entry.key.opponent,
        entry.key.as_of_date.isoformat(),
    )


def validate_persisted_tactical_recommendation_snapshot(snapshot: object) -> None:
    """Valida exhaustivamente un snapshot contra el contrato cerrado."""
    if type(snapshot) is not PersistedTacticalRecommendationSnapshot:
        raise SnapshotIntegrityError()
    if snapshot.contract_name != SNAPSHOT_CONTRACT_NAME:
        raise SnapshotIncompatibleError()
    if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotIncompatibleError()
    if type(snapshot.format_version) is not int or (
        snapshot.format_version != SNAPSHOT_FORMAT_VERSION
    ):
        raise SnapshotIncompatibleError()
    if snapshot.upstream_result_contract != SNAPSHOT_UPSTREAM_RESULT_CONTRACT:
        raise SnapshotIncompatibleError()
    if snapshot.upstream_result_contract_version != PRIORITIZATION_CONTRACT_VERSION:
        raise SnapshotIncompatibleError()
    if type(snapshot.policies) is not tuple or len(snapshot.policies) != 1:
        raise SnapshotIncompatibleError()
    if snapshot.policies[0] != default_tactical_scoring_policy():
        raise SnapshotIncompatibleError()
    if (
        type(snapshot.entry_count) is not int
        or not 0 <= snapshot.entry_count <= MAX_ENTRIES
        or snapshot.entry_count != len(snapshot.entries)
    ):
        raise SnapshotIntegrityError()
    if type(snapshot.entries) is not tuple:
        raise SnapshotIntegrityError()
    seen: set[TacticalRecommendationSnapshotKey] = set()
    previous_key: tuple[str, str, str] | None = None
    for entry in snapshot.entries:
        validate_persisted_tactical_recommendation_entry(entry)
        key_value = _key_sort_value(entry)
        if entry.key in seen:
            raise SnapshotIntegrityError()
        seen.add(entry.key)
        if previous_key is not None and key_value <= previous_key:
            raise SnapshotIntegrityError()
        previous_key = key_value
    _strict_sha(snapshot.fingerprint, "fingerprint")
    expected = _snapshot_fingerprint_of(
        _snapshot_payload(snapshot, include_fingerprint=False)
    )
    if snapshot.fingerprint != expected:
        raise SnapshotIntegrityError()


def _reject_json_constant(name: str) -> None:
    raise ValueError(f"constante JSON fuera de contrato: {name}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    seen: set[str] = set()
    for key, _value in pairs:
        if key in seen:
            raise ValueError("claves duplicadas en el JSON del snapshot")
        seen.add(key)
    return dict(pairs)


def _check_json_tree(value: object, *, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise SnapshotMalformedError()
    if value is None or type(value) is bool:
        return
    if type(value) is int:
        if abs(value) > MAX_JSON_INTEGER_ABS:
            raise SnapshotMalformedError()
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise SnapshotMalformedError()
        return
    if type(value) is str:
        if len(value) > MAX_JSON_STRING_LENGTH:
            raise SnapshotMalformedError()
        if any(ord(character) < 0x20 or character == "\x7f" for character in value):
            raise SnapshotMalformedError()
        return
    if type(value) is list:
        if len(value) > MAX_JSON_ARRAY_ITEMS:
            raise SnapshotMalformedError()
        for item in value:
            _check_json_tree(item, depth=depth + 1)
        return
    if type(value) is dict:
        if len(value) > MAX_JSON_OBJECT_KEYS:
            raise SnapshotMalformedError()
        for child_key, child_value in value.items():
            if type(child_key) is not str:
                raise SnapshotMalformedError()
            _check_json_tree(child_value, depth=depth + 1)
        return
    raise SnapshotMalformedError()


def _canonical_json_bytes(structure: dict[str, object]) -> bytes:
    _check_json_tree(structure)
    return json.dumps(
        structure,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _entry_for_result(
    result: TacticalPrioritizationResult,
) -> PersistedTacticalRecommendationEntry:
    summary = result.matchup_query
    iso = summary.as_of_date
    if type(iso) is not str or _CIVIL_ISO_DATE.fullmatch(iso) is None:
        raise SnapshotUpstreamInvalidError()
    invalid_date = False
    try:
        as_of = date.fromisoformat(iso)
    except ValueError:
        invalid_date = True
        as_of = date.min
    if invalid_date:
        raise SnapshotUpstreamInvalidError()
    key = TacticalRecommendationSnapshotKey(summary.player, summary.opponent, as_of)
    upstream_invalid = False
    try:
        result_fingerprint = tactical_prioritization_result_fingerprint(result)
    except Exception:
        upstream_invalid = True
        result_fingerprint = ""
    if upstream_invalid:
        raise SnapshotUpstreamInvalidError()
    entry_fingerprint = _entry_fingerprint_of(key, result_fingerprint)
    return PersistedTacticalRecommendationEntry(
        key, result, result_fingerprint, entry_fingerprint
    )


def build_persisted_tactical_recommendation_snapshot(
    results: object,
) -> PersistedTacticalRecommendationSnapshot:
    """Construye un snapshot firmado a partir de resultados validados.

    El orden de entrada no afecta a bytes ni fingerprint: las entradas
    se ordenan canonicamente. Duplicados exactos o en conflicto se
    rechazan.
    """
    if isinstance(results, (str, bytes, bytearray, dict)):
        raise TypeError("results debe ser una secuencia de resultados.")
    try:
        iterator = iter(results)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError("results debe ser una secuencia de resultados.") from None
    consumption_failed = False
    try:
        materialized = tuple(islice(iterator, MAX_ENTRIES + 1))
    except Exception:
        consumption_failed = True
        materialized = ()
    if consumption_failed:
        raise SnapshotUpstreamInvalidError()
    if len(materialized) > MAX_ENTRIES:
        raise SnapshotIncompatibleError()
    by_key: dict[TacticalRecommendationSnapshotKey, PersistedTacticalRecommendationEntry] = {}
    for result in materialized:
        if type(result) is not TacticalPrioritizationResult:
            raise SnapshotUpstreamInvalidError()
        upstream_invalid = False
        try:
            validate_tactical_prioritization_result(result)
        except Exception:
            upstream_invalid = True
        if upstream_invalid:
            raise SnapshotUpstreamInvalidError()
        entry = _entry_for_result(result)
        if entry.key in by_key:
            raise SnapshotIntegrityError()
        by_key[entry.key] = entry
    entries = tuple(sorted(by_key.values(), key=_key_sort_value))
    payload = {
        "contract": SNAPSHOT_CONTRACT_NAME,
        "entry_count": len(entries),
        "entries": [_entry_payload(item) for item in entries],
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "policies": [_policy_payload(default_tactical_scoring_policy())],
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "upstream_result_contract": SNAPSHOT_UPSTREAM_RESULT_CONTRACT,
        "upstream_result_contract_version": PRIORITIZATION_CONTRACT_VERSION,
    }
    return PersistedTacticalRecommendationSnapshot(
        contract_name=SNAPSHOT_CONTRACT_NAME,
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        format_version=SNAPSHOT_FORMAT_VERSION,
        entry_count=len(entries),
        policies=(default_tactical_scoring_policy(),),
        upstream_result_contract=SNAPSHOT_UPSTREAM_RESULT_CONTRACT,
        upstream_result_contract_version=PRIORITIZATION_CONTRACT_VERSION,
        entries=entries,
        fingerprint=_snapshot_fingerprint_of(payload),
    )


# --------------------------------------------------------------------------
# Reconstruction payload -> objects
# --------------------------------------------------------------------------


def _text(value: object) -> str:
    if type(value) is not str:
        raise SnapshotIncompatibleError()
    return value


def _int(value: object) -> int:
    if type(value) is not int:
        raise SnapshotIncompatibleError()
    return value


def _float(value: object) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise SnapshotIncompatibleError()
    return value


def _bool(value: object) -> bool:
    if type(value) is not bool:
        raise SnapshotIncompatibleError()
    return value


def _opt_int(value: object) -> int | None:
    if value is None:
        return None
    return _int(value)


def _opt_float(value: object) -> float | None:
    if value is None:
        return None
    return _float(value)


def _sha_text(value: object) -> str:
    if type(value) is not str or _SHA256_UPPER.fullmatch(value) is None:
        raise SnapshotIncompatibleError()
    return value


def _str_tuple(value: object) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise SnapshotIncompatibleError()
    return tuple(item for item in value)  # type: ignore[return-value]


def _int_tuple(value: object) -> tuple[int, ...]:
    if type(value) is not list or any(type(item) is not int for item in value):
        raise SnapshotIncompatibleError()
    return tuple(item for item in value)  # type: ignore[return-value]


def _pairs_value(value: object, value_type: type) -> tuple[tuple[str, object], ...]:
    if type(value) is not list:
        raise SnapshotIncompatibleError()
    pairs: list[tuple[str, object]] = []
    for item in value:
        if (
            type(item) is not list
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not value_type
        ):
            raise SnapshotIncompatibleError()
        pairs.append((item[0], item[1]))
    return tuple(pairs)


def _enum_value(value: object, enum_cls: type) -> object:
    if type(value) is not str:
        raise SnapshotIncompatibleError()
    parsed: object | None = None
    try:
        parsed = enum_cls(value)
    except ValueError:
        pass
    if parsed is None:
        raise SnapshotUpstreamInvalidError()
    return parsed


def _check_keys(value: object, contract_cls: type) -> dict[str, object]:
    if type(value) is not dict:
        raise SnapshotIncompatibleError()
    expected = {item.name for item in fields(contract_cls)}
    if set(value) != expected:
        raise SnapshotIncompatibleError()
    return value


def _policy_from_payload(value: object) -> TacticalScoringPolicy:
    data = _check_keys(value, TacticalScoringPolicy)
    return TacticalScoringPolicy(
        contract_version=_text(data["contract_version"]),
        name=_text(data["name"]),
        version=_text(data["version"]),
        executor_weight=_float(data["executor_weight"]),
        opponent_allowed_weight=_float(data["opponent_allowed_weight"]),
        formula=_text(data["formula"]),
        allowed_scope=_text(data["allowed_scope"]),
        uncertainty_method=_text(data["uncertainty_method"]),
        abstention_rules=_str_tuple(data["abstention_rules"]),
        comparability_rule=_text(data["comparability_rule"]),
        interpretation=_text(data["interpretation"]),
        reconciliations=_pairs_value(data["reconciliations"], bool),
    )


def _summary_from_payload(value: object) -> TacticalEvidenceSummary:
    data = _check_keys(value, TacticalEvidenceSummary)
    return TacticalEvidenceSummary(
        perspective=_text(data["perspective"]),
        evidence_state=_text(data["evidence_state"]),
        scope=_text(data["scope"]),
        labeled_attempts=_int(data["labeled_attempts"]),
        successes=_int(data["successes"]),
        failures=_int(data["failures"]),
        success_rate=_opt_float(data["success_rate"]),
        wilson_lower=_opt_float(data["wilson_lower"]),
        wilson_upper=_opt_float(data["wilson_upper"]),
        distinct_matches=_int(data["distinct_matches"]),
    )


def _candidate_from_payload(value: object) -> TacticalScoredCandidate:
    data = _check_keys(value, TacticalScoredCandidate)
    return TacticalScoredCandidate(
        contract_version=_text(data["contract_version"]),
        pattern_id=_text(data["pattern_id"]),
        feature_name=_text(data["feature_name"]),
        category=_text(data["category"]),
        encoder_policy=_enum_value(data["encoder_policy"], TacticalEncodingPolicy),  # type: ignore[arg-type]
        schema_fingerprint=_sha_text(data["schema_fingerprint"]),
        query_fingerprint=_sha_text(data["query_fingerprint"]),
        tactical_opportunity=_text(data["tactical_opportunity"]),
        actor=_text(data["actor"]),
        state=_enum_value(data["state"], TacticalCandidateState),  # type: ignore[arg-type]
        reason_codes=_str_tuple(data["reason_codes"]),
        executor_evidence=_summary_from_payload(data["executor_evidence"]),
        opponent_allowed_evidence=_summary_from_payload(data["opponent_allowed_evidence"]),
        combined_rate=_opt_float(data["combined_rate"]),
        combined_lower=_opt_float(data["combined_lower"]),
        combined_upper=_opt_float(data["combined_upper"]),
        combined_width=_opt_float(data["combined_width"]),
        evidence_floor=_opt_int(data["evidence_floor"]),
        match_floor=_opt_int(data["match_floor"]),
        disagreement=_opt_float(data["disagreement"]),
        uncertainty_label=_text(data["uncertainty_label"]),
        score_formula=_text(data["score_formula"]),
        comparable=_bool(data["comparable"]),
        rank_eligible=_bool(data["rank_eligible"]),
        rank_position=_opt_int(data["rank_position"]),
        tie_group=_opt_int(data["tie_group"]),
        source_category_fingerprint=_sha_text(data["source_category_fingerprint"]),
        provenance=_pairs_value(data["provenance"], str),
        reconciliations=_pairs_value(data["reconciliations"], bool),
    )


def _query_summary_from_payload(value: object) -> TacticalMatchupQuerySummary:
    data = _check_keys(value, TacticalMatchupQuerySummary)
    return TacticalMatchupQuerySummary(
        player=_text(data["player"]),
        opponent=_text(data["opponent"]),
        as_of_date=_text(data["as_of_date"]),
        serve_numbers=_int_tuple(data["serve_numbers"]),
        requested_patterns=_str_tuple(data["requested_patterns"]),
        encoder_policy=_enum_value(data["encoder_policy"], TacticalEncodingPolicy),  # type: ignore[arg-type]
        scope_strategy=_text(data["scope_strategy"]),
        minimum_labeled_attempts=_int(data["minimum_labeled_attempts"]),
        minimum_matches=_int(data["minimum_matches"]),
        window_days=_opt_int(data["window_days"]),
        query_fingerprint=_sha_text(data["query_fingerprint"]),
    )


def _candidate_list(value: object) -> tuple[TacticalScoredCandidate, ...]:
    if type(value) is not list:
        raise SnapshotIncompatibleError()
    return tuple(_candidate_from_payload(item) for item in value)


def _ranking_from_payload(value: object) -> TacticalPatternRanking:
    data = _check_keys(value, TacticalPatternRanking)
    return TacticalPatternRanking(
        contract_version=_text(data["contract_version"]),
        pattern_id=_text(data["pattern_id"]),
        tactical_opportunity=_text(data["tactical_opportunity"]),
        scoring_policy_fingerprint=_sha_text(data["scoring_policy_fingerprint"]),
        encoder_policy=_enum_value(data["encoder_policy"], TacticalEncodingPolicy),  # type: ignore[arg-type]
        candidates=_candidate_list(data["candidates"]),
        top_candidates=_candidate_list(data["top_candidates"]),
        scored_candidate_count=_int(data["scored_candidate_count"]),
        abstained_candidate_count=_int(data["abstained_candidate_count"]),
        requested_top_k=_int(data["requested_top_k"]),
        effective_top_k=_int(data["effective_top_k"]),
        tie_group_count=_int(data["tie_group_count"]),
        boundary_tie_expanded=_bool(data["boundary_tie_expanded"]),
        state=_enum_value(data["state"], TacticalRankingState),  # type: ignore[arg-type]
        reason_codes=_str_tuple(data["reason_codes"]),
        order_rule=_text(data["order_rule"]),
        redundant_representation=_bool(data["redundant_representation"]),
        contractual_purpose=_text(data["contractual_purpose"]),
        provenance=_pairs_value(data["provenance"], str),
        reconciliations=_pairs_value(data["reconciliations"], bool),
    )


def _result_from_payload(value: object) -> TacticalPrioritizationResult:
    data = _check_keys(value, TacticalPrioritizationResult)
    rankings_raw = data["rankings"]
    if type(rankings_raw) is not list:
        raise SnapshotIncompatibleError()
    if type(data["policy"]) is not dict or type(data["matchup_query"]) is not dict:
        raise SnapshotIncompatibleError()
    return TacticalPrioritizationResult(
        contract_version=_text(data["contract_version"]),
        policy=_policy_from_payload(data["policy"]),
        matchup_query=_query_summary_from_payload(data["matchup_query"]),
        schema_fingerprint=_sha_text(data["schema_fingerprint"]),
        source_evidence_fingerprint=_sha_text(data["source_evidence_fingerprint"]),
        state=_enum_value(data["state"], TacticalPrioritizationState),  # type: ignore[arg-type]
        reason_codes=_str_tuple(data["reason_codes"]),
        rankings=tuple(_ranking_from_payload(item) for item in rankings_raw),
        requested_pattern_count=_int(data["requested_pattern_count"]),
        available_pattern_count=_int(data["available_pattern_count"]),
        partial_or_unavailable_pattern_count=_int(
            data["partial_or_unavailable_pattern_count"]
        ),
        total_candidate_count=_int(data["total_candidate_count"]),
        scored_candidate_count=_int(data["scored_candidate_count"]),
        abstained_candidate_count=_int(data["abstained_candidate_count"]),
        redundant_representation=_bool(data["redundant_representation"]),
        limitations=_str_tuple(data["limitations"]),
        reconciliations=_pairs_value(data["reconciliations"], bool),
        provenance=_pairs_value(data["provenance"], str),
    )


def _civil_date_from_iso(value: str) -> date:
    if _CIVIL_ISO_DATE.fullmatch(value) is None:
        raise SnapshotIntegrityError()
    parsed: date | None = None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        pass
    if parsed is None:
        raise SnapshotIntegrityError()
    return parsed


def _entry_from_payload(value: object) -> PersistedTacticalRecommendationEntry:
    if type(value) is not dict or set(value) != _ENTRY_PAYLOAD_KEYS:
        raise SnapshotIncompatibleError()
    player = _text(value["player"])
    opponent = _text(value["opponent"])
    as_of_iso = _text(value["as_of_date"])
    result_fingerprint = _sha_text(value["result_fingerprint"])
    entry_fingerprint = _sha_text(value["entry_fingerprint"])
    key = TacticalRecommendationSnapshotKey(
        player, opponent, _civil_date_from_iso(as_of_iso)
    )
    result = _result_from_payload(value["result"])
    summary = result.matchup_query
    if (
        summary.player != player
        or summary.opponent != opponent
        or summary.as_of_date != as_of_iso
    ):
        raise SnapshotIntegrityError()
    return PersistedTacticalRecommendationEntry(
        key, result, result_fingerprint, entry_fingerprint
    )


def _snapshot_from_structure(structure: object) -> PersistedTacticalRecommendationSnapshot:
    if type(structure) is not dict or set(structure) != _SNAPSHOT_PAYLOAD_KEYS:
        raise SnapshotIncompatibleError()
    _check_json_tree(structure)
    if (
        type(structure["contract"]) is not str
        or structure["contract"] != SNAPSHOT_CONTRACT_NAME
    ):
        raise SnapshotIncompatibleError()
    if (
        type(structure["schema_version"]) is not str
        or structure["schema_version"] != SNAPSHOT_SCHEMA_VERSION
    ):
        raise SnapshotIncompatibleError()
    if (
        type(structure["format_version"]) is not int
        or structure["format_version"] != SNAPSHOT_FORMAT_VERSION
    ):
        raise SnapshotIncompatibleError()
    if (
        type(structure["upstream_result_contract"]) is not str
        or structure["upstream_result_contract"] != SNAPSHOT_UPSTREAM_RESULT_CONTRACT
    ):
        raise SnapshotIncompatibleError()
    if (
        type(structure["upstream_result_contract_version"]) is not str
        or structure["upstream_result_contract_version"] != PRIORITIZATION_CONTRACT_VERSION
    ):
        raise SnapshotIncompatibleError()
    entry_count = structure["entry_count"]
    if (
        type(entry_count) is not int
        or not 0 <= entry_count <= MAX_ENTRIES
    ):
        raise SnapshotIncompatibleError()
    entries_raw = structure["entries"]
    if type(entries_raw) is not list or len(entries_raw) != entry_count:
        raise SnapshotIntegrityError()
    policies_raw = structure["policies"]
    if type(policies_raw) is not list or len(policies_raw) != 1:
        raise SnapshotIncompatibleError()
    default_policy = default_tactical_scoring_policy()
    if _policy_from_payload(policies_raw[0]) != default_policy:
        raise SnapshotIncompatibleError()
    entries = tuple(_entry_from_payload(item) for item in entries_raw)
    snapshot = PersistedTacticalRecommendationSnapshot(
        contract_name=SNAPSHOT_CONTRACT_NAME,
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        format_version=SNAPSHOT_FORMAT_VERSION,
        entry_count=entry_count,
        policies=(default_policy,),
        upstream_result_contract=SNAPSHOT_UPSTREAM_RESULT_CONTRACT,
        upstream_result_contract_version=PRIORITIZATION_CONTRACT_VERSION,
        entries=entries,
        fingerprint=_snapshot_fingerprint_of({k: v for k, v in structure.items() if k != "fingerprint"}),
    )
    if (
        type(structure["fingerprint"]) is not str
        or snapshot.fingerprint != structure["fingerprint"]
    ):
        raise SnapshotIntegrityError()
    return snapshot


def _snapshot_path(value: object) -> Path:
    """Normaliza una ruta local sin ejecutar PathLike arbitrarios."""
    if type(value) is str:
        raw = value
    elif isinstance(value, Path):
        raw = str(value)
    else:
        raise SnapshotIncompatibleError()
    if not raw or raw != raw.strip() or "\x00" in raw:
        raise SnapshotIncompatibleError()
    normalized = raw.replace("\\", "/")
    lowered = normalized.casefold()
    if (
        lowered.startswith(("file:", "vscode:", "http:", "https:"))
        or normalized.startswith("//")
        or normalized == "~"
        or normalized.startswith("~/")
        or any(part == ".." for part in normalized.split("/"))
    ):
        raise SnapshotIncompatibleError()
    return Path(raw)


def _path_has_linked_parent(path: Path) -> bool:
    """Detecta symlinks/junctions existentes en la cadena de padres."""
    for parent in (path.parent, *path.parent.parents):
        try:
            if parent.is_symlink():
                return True
            is_junction = getattr(parent, "is_junction", None)
            if is_junction is not None and is_junction():
                return True
        except OSError:
            return True
    return False


def _read_snapshot_bytes(path: Path) -> bytes:
    """Lee como maximo el limite contractual y reduce carreras con symlinks."""
    failure: type[PersistedSnapshotError] | None = None
    raw: bytes | None = None
    descriptor: int | None = None
    try:
        if _path_has_linked_parent(path):
            raise SnapshotUnavailableError()
        try:
            before = path.lstat()
        except FileNotFoundError:
            raise SnapshotNotFoundError()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise SnapshotUnavailableError()
        if before.st_size == 0 or before.st_size > MAX_SNAPSHOT_BYTES:
            raise SnapshotUnavailableError()
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise SnapshotUnavailableError()
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise SnapshotUnavailableError()
        chunks: list[bytes] = []
        remaining = MAX_SNAPSHOT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if not raw or len(raw) > MAX_SNAPSHOT_BYTES:
            raise SnapshotUnavailableError()
    except PersistedSnapshotError as internal:
        failure = type(internal)
    except OSError:
        failure = SnapshotUnavailableError
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                if failure is None:
                    failure = SnapshotUnavailableError
    if failure is not None:
        raise failure()
    if raw is None:
        raise SnapshotUnavailableError()
    return raw


def load_persisted_tactical_recommendation_snapshot(
    snapshot_path: object,
) -> PersistedTacticalRecommendationSnapshot:
    """Carga y valida un snapshot de disco.

    Levanta los errores internos P13 finos (ausente, ilegible,
    incompatible, corrupto, integridad, upstream invalido). La
    traduccion a errores de servicio P12 la realiza
    ``create_persisted_tactical_recommendation_provider`` (frontera
    publica). Una unica lectura de disco por ciclo de vida del provider;
    no se escribe, regenera ni repara ningun fichero.
    """
    path = _snapshot_path(snapshot_path)
    raw = _read_snapshot_bytes(path)
    malformed = False
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        malformed = True
        text = ""
    if malformed:
        raise SnapshotMalformedError()
    malformed = False
    try:
        structure = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (ValueError, RecursionError):
        malformed = True
        structure = None
    if malformed:
        raise SnapshotMalformedError()
    reconstruction_failure: type[PersistedSnapshotError] | None = None
    reconstructed: PersistedTacticalRecommendationSnapshot | None = None
    try:
        reconstructed = _snapshot_from_structure(structure)
    except PersistedSnapshotError as internal:
        reconstruction_failure = type(internal)
    except (ValueError, TypeError, RecursionError):
        # Bytes parseables que no reconstruyen un resultado upstream
        # validado: violacion contractual, no fallo interno.
        reconstruction_failure = SnapshotUpstreamInvalidError
    if reconstruction_failure is not None:
        raise reconstruction_failure()
    if reconstructed is None:
        raise SnapshotUpstreamInvalidError()
    return reconstructed


def serialize_persisted_tactical_recommendation_snapshot(
    snapshot: object,
) -> bytes:
    """Serializa un snapshot en bytes canonicos deterministas."""
    if type(snapshot) is not PersistedTacticalRecommendationSnapshot:
        raise SnapshotIntegrityError()
    validate_persisted_tactical_recommendation_snapshot(snapshot)
    raw = _canonical_json_bytes(_snapshot_payload(snapshot, include_fingerprint=True))
    if len(raw) > MAX_SNAPSHOT_BYTES:
        raise SnapshotUnavailableError()
    return raw


def persist_persisted_tactical_recommendation_snapshot(
    snapshot: object, destination: object
) -> None:
    """Escribe atomicamente hasta ``os.replace``.

    Un fallo previo preserva el destino anterior. Tras un replace exitoso
    el nuevo destino puede quedar publicado aunque falle el fsync posterior
    del directorio; no se promete rollback despues del punto de commit.
    """
    failure: type[PersistedSnapshotError] | None = None
    destination_path: Path | None = None
    temp_path: Path | None = None
    try:
        if type(snapshot) is not PersistedTacticalRecommendationSnapshot:
            raise SnapshotIncompatibleError()
        validate_persisted_tactical_recommendation_snapshot(snapshot)
        destination_path = _snapshot_path(destination)
        if (
            _path_has_linked_parent(destination_path)
            or destination_path.is_symlink()
            or destination_path.is_dir()
            or not destination_path.parent.is_dir()
        ):
            raise SnapshotUnavailableError()
        raw = serialize_persisted_tactical_recommendation_snapshot(snapshot)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=".persisted-tactical-snapshot-",
            suffix=".tmp",
            dir=os.fspath(destination_path.parent),
        )
        temp_path = Path(temp_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination_path)
        temp_path = None
        try:
            directory_fd = os.open(
                os.fspath(destination_path.parent), os.O_RDONLY
            )
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except PersistedSnapshotError as internal:
        failure = type(internal)
    except OSError:
        failure = SnapshotUnavailableError
    except Exception:
        failure = SnapshotUnavailableError
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass
    if failure is not None:
        raise failure()


def verify_persisted_tactical_recommendation_snapshot(
    snapshot_path: object,
) -> str:
    """Verifica un snapshot persistido y devuelve su fingerprint."""
    return load_persisted_tactical_recommendation_snapshot(snapshot_path).fingerprint


@dataclass(frozen=True)
class PersistedTacticalRecommendationProvider:
    """Provider de solo lectura compatible con el protocolo P12.

    Tras la carga explicita no toca disco: el lookup es una busqueda
    exacta en memoria sobre estructuras profundamente inmutables, segura
    para lecturas concurrentes. Devuelve el mismo objeto inmutable para
    consultas repetidas.
    """

    __slots__ = ("snapshot",)

    snapshot: PersistedTacticalRecommendationSnapshot

    def __post_init__(self) -> None:
        if type(self.snapshot) is not PersistedTacticalRecommendationSnapshot:
            raise SnapshotIntegrityError()
        validate_persisted_tactical_recommendation_snapshot(self.snapshot)

    def fetch_tactical_prioritization(
        self, query: TacticalRecommendationQuery
    ) -> TacticalPrioritizationResult:
        service_error: type | None = None
        result: TacticalPrioritizationResult | None = None
        try:
            if type(query) is not TacticalRecommendationQuery:
                service_error = InvalidRequestError
            else:
                validate_tactical_recommendation_query(query)
                key = TacticalRecommendationSnapshotKey(
                    query.player_id, query.opponent_id, query.as_of_date
                )
                entry = self.snapshot.index.get(key)
                if entry is None:
                    service_error = RecommendationNotFoundError
                else:
                    result = entry.result
        except InvalidRequestError:
            service_error = InvalidRequestError
        except PersistedSnapshotError:
            service_error = InvalidRequestError
        except Exception:
            service_error = InternalServiceError
        if service_error is not None:
            raise service_error()
        return result


def create_persisted_tactical_recommendation_provider(
    snapshot_path: object,
) -> PersistedTacticalRecommendationProvider:
    """Carga explicita del snapshot y construye el provider inmutable."""
    service_error: type[Exception] | None = None
    provider: PersistedTacticalRecommendationProvider | None = None
    try:
        snapshot = load_persisted_tactical_recommendation_snapshot(snapshot_path)
        provider = PersistedTacticalRecommendationProvider(snapshot=snapshot)
    except PersistedSnapshotError as internal:
        service_error = _SNAPSHOT_ERROR_TO_SERVICE_ERROR.get(
            type(internal), InternalServiceError
        )
    except Exception:
        service_error = InternalServiceError
    if service_error is not None:
        raise service_error()
    if provider is None:
        raise InternalServiceError()
    return provider


__all__ = (
    "CAPACITY_DESIGN_ADVERSARIAL_SNAPSHOT_BYTES",
    "CAPACITY_DESIGN_MAX_ENTRY_BYTES",
    "CAPACITY_DESIGN_MEMORY_PEAK_BYTES",
    "CAPACITY_DESIGN_REPRESENTATIVE_SNAPSHOT_BYTES",
    "CAPACITY_DESIGN_SYNTHETIC_TARGETS_MEASURED",
    "CAPACITY_DESIGN_TYPICAL_ENTRY_BYTES",
    "CAPACITY_DESIGN_UNIVERSE_ENTRIES",
    "MAX_ENTRIES",
    "MAX_JSON_ARRAY_ITEMS",
    "MAX_JSON_DEPTH",
    "MAX_JSON_INTEGER_ABS",
    "MAX_JSON_OBJECT_KEYS",
    "MAX_JSON_STRING_LENGTH",
    "MAX_SNAPSHOT_BYTES",
    "PersistedSnapshotError",
    "PersistedTacticalRecommendationEntry",
    "PersistedTacticalRecommendationProvider",
    "PersistedTacticalRecommendationSnapshot",
    "SNAPSHOT_CONTRACT_NAME",
    "SNAPSHOT_ENTRY_DOMAIN",
    "SNAPSHOT_FORMAT_VERSION",
    "SNAPSHOT_PURPOSE",
    "SNAPSHOT_SCHEMA_VERSION",
    "SNAPSHOT_SNAPSHOT_DOMAIN",
    "SNAPSHOT_UPSTREAM_RESULT_CONTRACT",
    "SnapshotIntegrityError",
    "SnapshotIncompatibleError",
    "SnapshotMalformedError",
    "SnapshotNotFoundError",
    "SnapshotUnavailableError",
    "SnapshotUpstreamInvalidError",
    "TacticalRecommendationSnapshotKey",
    "build_persisted_tactical_recommendation_snapshot",
    "create_persisted_tactical_recommendation_provider",
    "load_persisted_tactical_recommendation_snapshot",
    "persist_persisted_tactical_recommendation_snapshot",
    "serialize_persisted_tactical_recommendation_snapshot",
    "validate_persisted_tactical_recommendation_entry",
    "validate_persisted_tactical_recommendation_snapshot",
    "validate_tactical_recommendation_snapshot_key",
    "verify_persisted_tactical_recommendation_snapshot",
)
