"""Motor explicable y conservador de recomendaciones de segundo saque.

Consume el contrato de scoring ya publicado, sin cambiar su politica ni sus
scores. La politica upstream escoge scope por direccion; este modulo anade una
barrera posterior: solo ordena wide/body/T cuando las tres direcciones
elegibles comparten el mismo scope. El test final nunca se construye ni se
evalua.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.analysis.chronological_validation import EXPECTED_MATCHES, EXPECTED_SOURCE_ROWS
from src.analysis.evidence_policy_validation import (
    construct_sealed_target_features,
    select_validation_features,
)
from src.analysis.explainable_direction_scoring import (
    DIRECTIONS as _SCORING_DIRECTIONS,
    MIN_MATCHES,
    MIN_POINTS,
    POLICY_ID,
    SCOPE_POLICY,
    UPSTREAM_COMMITS,
    join_target_outcomes,
    load_and_validate_upstream_artifacts,
    prepare_scoring_population,
)
from src.analysis.historical_profiles import POINTS_FILE, SOURCE_COLUMNS, prepare_historical_population


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
SUMMARY_PATH = REPORTS_DIR / "explainable_direction_recommender_summary.json"
COVERAGE_PATH = TABLES_DIR / "explainable_direction_recommender_coverage.csv"
RANKINGS_PATH = TABLES_DIR / "explainable_direction_recommender_rankings.csv"
EXPLANATIONS_PATH = TABLES_DIR / "explainable_direction_recommender_explanations.csv"
REPRO_DIR = REPORTS_DIR / "repro_explainable_direction_recommender"

ANALYSIS_NAME = "explainable_direction_recommender"
ANALYSIS_VERSION = "1.0.0"
FINGERPRINT_CONTRACT_VERSION = "2"
DIRECTIONS = ("wide", "body", "T")
SCOPES = ("surface", "global")
FOLDS = (2020, 2021, 2022, 2023)
SURFACES = ("Hard", "Clay", "Grass")
STATUS_ORDER = (
    "available_full",
    "abstained_insufficient_evidence",
    "abstained_incomparable_scope",
    "not_available_invalid_input",
)
SCOPE_STATES = (
    "surface_common",
    "global_common",
    "mixed_incomparable",
    "insufficient_evidence",
    "invalid_input",
)
STATUS_SCOPE_STATE = {
    "available_full": None,
    "abstained_insufficient_evidence": "insufficient_evidence",
    "abstained_incomparable_scope": "mixed_incomparable",
    "not_available_invalid_input": "invalid_input",
}
EXPLANATION_CODES = frozenset({
    "surface_history_used", "global_fallback_used", "server_component_higher",
    "opponent_component_higher", "balanced_components", "narrow_score_margin",
    "evidence_threshold_just_met", "evidence_above_threshold", "deterministic_tie_break",
    "recommendation_available", "incomplete_direction_set", "direction_not_scoreable",
    "mixed_direction_scopes", "recommendation_abstained",
})
MARGIN_TOLERANCE = 0.01
COMPONENT_TOLERANCE = 1e-12
RECONCILIATION_TOLERANCE = 1e-12
MAX_EXCEPTION_MESSAGE_LENGTH = 320
RATE_RECONCILIATION_TOLERANCE = 1e-12
EXPECTED_EXCLUDED_TEST_TARGET_MATCHES = 1_531
TEST_ZERO_FIELDS = (
    "test_target_rows_constructed",
    "test_target_matches_constructed",
    "test_feature_rows_constructed",
    "test_recommendation_rows",
    "test_points_scored",
    "test_matches_scored",
    "test_rows_evaluated",
    "test_matches_evaluated",
    "test_evaluation_runs",
)
EVIDENCE_STATUS_BY_SCOPE = {
    "surface": "eligible_surface",
    "global": "eligible_global_fallback",
}
ABSTAINED_EVIDENCE_STATUS = "insufficient_surface_and_global"

UPSTREAM_SCORING_HASHES = {
    "reports/explainable_direction_scoring_summary.json": "2AC390152270DB6CDB20AFD2A0B5AC115488183BF5A131FD592F0B1BA07A18A4",
    "reports/tables/explainable_direction_scoring_by_fold.csv": "622C72683CB7984821F7A7BE759FC065A70FAE6294D84C9C27E58434FD737E37",
    "reports/tables/explainable_direction_scoring_by_direction.csv": "367BD35B9FF613AE940AF64EBDCA7556BF48DAFB2D0B83F33FAB706EAE148CCA",
    "reports/tables/explainable_direction_scoring_calibration.csv": "E6CD59A10BD6D94A0CB4281FCA8C6303EA75F2AA83389D1EBC82D1328E0A4038",
    "reports/tables/explainable_direction_scoring_ranking.csv": "23D4BA434571EE1C69F788470AA7B8991FBEE31806C1DF90E568EFF81F8D31D5",
}

ORIENTATION_KEY = [
    "target_match_id", "target_date", "target_player", "opponent", "target_surface", "fold",
]
RECOMMENDATION_COLUMNS = ORIENTATION_KEY + [
    "recommendation_status", "comparison_scope", "scope_state",
    "observed_direction_scopes", "reason_codes", "selected_direction",
]
DETAIL_COLUMNS = ORIENTATION_KEY + [
    "direction", "eligible", "selected_scope", "score", "server_component",
    "opponent_component", "server_selected_rate", "opponent_allowed_rate",
    "server_points", "server_matches", "opponent_points", "opponent_matches",
    "server_wilson_low", "server_wilson_high", "opponent_wilson_low",
    "opponent_wilson_high", "recommendation_status", "comparison_scope",
    "scope_state", "rank", "score_gap_from_rank_1", "score_gap_to_next",
    "explanation_codes",
]
COVERAGE_COLUMNS = [
    "fold", "surface", "recommendation_status", "scope_state", "orientations", "matches",
    "players", "available_orientations", "abstained_orientations", "coverage_rate",
    "abstention_rate", "reason_code",
]
RANKINGS_COLUMNS = [
    "fold", "surface", "rank", "direction", "comparison_scope", "orientations",
    "direction_share", "mean_score", "median_score", "mean_score_gap_from_rank_1",
    "median_score_gap_from_rank_1", "observed_points", "observed_server_wins",
    "observed_win_rate",
]
EXPLANATIONS_COLUMNS = [
    "fold", "surface", "rank", "direction", "comparison_scope", "explanation_code",
    "orientations", "share_within_rank_direction", "mean_server_component",
    "mean_opponent_component", "mean_absolute_component_difference",
]
PAYLOAD_NAMES = ("coverage", "rankings", "explanations")

# Esquema explícito de ``prepare_scoring_population``. No se construyen nombres
# por simetría: cada columna upstream consumida queda auditada aquí.
UPSTREAM_SCORING_SCHEMA = (
    # identidad y política ya resuelta por el baseline
    ("target_match_id", "target_match_id", None, "identity", "partido objetivo", "non_empty_text", True),
    ("target_date", "target_date", None, "identity", "fecha civil objetivo", "date", True),
    ("target_player", "target_player", None, "identity", "servidor objetivo", "non_empty_text", True),
    ("opponent", "opponent", None, "identity", "rival/restador objetivo", "non_empty_text", True),
    ("target_surface", "target_surface", None, "identity", "superficie objetivo", "Hard_Clay_Grass", True),
    ("fold", "fold", None, "identity", "fold de validación", "2020_to_2023", True),
    ("direction", "direction", None, "identity", "dirección de segundo saque", "wide_body_T", True),
    ("selected_scope", "selected_scope", None, "policy", "scope seleccionado por dirección", "surface_global_or_null", True),
    ("eligible", "eligible", None, "policy", "dirección puntuable", "strict_bool", True),
    ("evidence_status", "evidence_status", None, "policy", "estado de evidencia upstream", "text", True),
    ("abstention_reason", "abstention_reason", None, "policy", "razón upstream de abstención", "text_or_null", True),
    ("server_rate", "server_selected_rate", None, "policy", "tasa del servidor en scope seleccionado", "unit_interval_or_null", True),
    ("opponent_allowed_rate", "opponent_allowed_rate", None, "policy", "tasa permitida al servidor por rival", "unit_interval_or_null", True),
    ("combined_score", "score", None, "policy", "score 50/50 congelado", "unit_interval_or_null", True),
    ("server_component", "server_component", None, "policy", "componente servidor", "finite_or_null", True),
    ("opponent_component", "opponent_component", None, "policy", "componente rival", "finite_or_null", True),
    # servidor, surface
    ("server_direction_raw_rate_surface", "server_selected_rate", "surface", "server", "tasa histórica del servidor", "unit_interval_or_null", True),
    ("server_direction_points_surface", "server_points", "surface", "server", "puntos históricos del servidor", "non_negative_integer", True),
    ("server_direction_matches_surface", "server_matches", "surface", "server", "partidos históricos del servidor", "non_negative_integer", True),
    ("server_direction_wilson_low_surface", "server_wilson_low", "surface", "server", "Wilson inferior de tasa servidor", "unit_interval_or_null", True),
    ("server_direction_wilson_high_surface", "server_wilson_high", "surface", "server", "Wilson superior de tasa servidor", "unit_interval_or_null", True),
    # servidor, global
    ("server_direction_raw_rate_global", "server_selected_rate", "global", "server", "tasa histórica del servidor", "unit_interval_or_null", True),
    ("server_direction_points_global", "server_points", "global", "server", "puntos históricos del servidor", "non_negative_integer", True),
    ("server_direction_matches_global", "server_matches", "global", "server", "partidos históricos del servidor", "non_negative_integer", True),
    ("server_direction_wilson_low_global", "server_wilson_low", "global", "server", "Wilson inferior de tasa servidor", "unit_interval_or_null", True),
    ("server_direction_wilson_high_global", "server_wilson_high", "global", "server", "Wilson superior de tasa servidor", "unit_interval_or_null", True),
    # rival como restador, surface. La tasa es P(server_won_point) permitida.
    ("opponent_allowed_server_raw_rate_surface", "opponent_allowed_rate", "surface", "opponent", "tasa permitida al servidor por rival", "unit_interval_or_null", True),
    ("opponent_direction_points_surface", "opponent_points", "surface", "opponent", "puntos históricos del rival al resto", "non_negative_integer", True),
    ("opponent_direction_matches_surface", "opponent_matches", "surface", "opponent", "partidos históricos del rival al resto", "non_negative_integer", True),
    ("opponent_allowed_server_wilson_low_surface", "opponent_wilson_low", "surface", "opponent", "Wilson inferior de tasa permitida", "unit_interval_or_null", True),
    ("opponent_allowed_server_wilson_high_surface", "opponent_wilson_high", "surface", "opponent", "Wilson superior de tasa permitida", "unit_interval_or_null", True),
    # rival como restador, global
    ("opponent_allowed_server_raw_rate_global", "opponent_allowed_rate", "global", "opponent", "tasa permitida al servidor por rival", "unit_interval_or_null", True),
    ("opponent_direction_points_global", "opponent_points", "global", "opponent", "puntos históricos del rival al resto", "non_negative_integer", True),
    ("opponent_direction_matches_global", "opponent_matches", "global", "opponent", "partidos históricos del rival al resto", "non_negative_integer", True),
    ("opponent_allowed_server_wilson_low_global", "opponent_wilson_low", "global", "opponent", "Wilson inferior de tasa permitida", "unit_interval_or_null", True),
    ("opponent_allowed_server_wilson_high_global", "opponent_wilson_high", "global", "opponent", "Wilson superior de tasa permitida", "unit_interval_or_null", True),
)
UPSTREAM_REQUIRED_COLUMNS = tuple(item[0] for item in UPSTREAM_SCORING_SCHEMA if item[6])
UPSTREAM_COLUMN_MAP = {(canonical, scope): upstream for upstream, canonical, scope, *_ in UPSTREAM_SCORING_SCHEMA}


class RecommendationContractError(ValueError):
    """Error contractual que impide recomendaciones parciales."""


class RecommendationExecutionError(RecommendationContractError):
    """Frontera de ejecucion real con diagnostico sanitizado."""

    def __init__(self, stage: str, exception: BaseException) -> None:
        super().__init__(f"{stage}: {type(exception).__name__}: {exception}")
        self.stage = stage
        self.exception_type = type(exception).__name__
        self.exception_message = sanitize_exception_message(str(exception))
        self.reason_code = f"{stage}_failure"


@dataclass(frozen=True)
class RecommendationResult:
    summary: dict[str, Any]
    coverage: pd.DataFrame
    rankings: pd.DataFrame
    explanations: pd.DataFrame
    recommendations: pd.DataFrame
    details: pd.DataFrame
    point_rows: pd.DataFrame
    publication_fingerprint: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _frame_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n", na_rep="")
    return buffer.getvalue().encode("utf-8")


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _assert_finite_json(value: Any, path: str = "root") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RecommendationContractError(f"Valor no finito en {path}.")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_json(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json(item, f"{path}[{index}]")
        return
    raise RecommendationContractError(f"Tipo JSON no permitido en {path}: {type(value).__name__}")


def sanitize_exception_message(message: str) -> str:
    value = " ".join(str(message).replace("\r", " ").replace("\n", " ").split())
    value = value.replace(str(ROOT), "<repo>")
    value = re.sub(r"[A-Za-z]:\\[^\s\"']+", "<absolute-path>", value)
    return value[:MAX_EXCEPTION_MESSAGE_LENGTH]


def verify_upstream_contracts() -> dict[str, Any]:
    """Verifica hashes exactos antes de reutilizar el scoring congelado."""

    for relative, expected in UPSTREAM_SCORING_HASHES.items():
        path = ROOT / relative
        if not path.exists() or _sha256(path.read_bytes()) != expected:
            raise RecommendationContractError(f"Hash upstream inesperado: {relative}")
    selection, validation, chronological = load_and_validate_upstream_artifacts()
    contract = selection["selected_policy_contract"]
    if (
        contract.get("policy_id") != POLICY_ID
        or contract.get("scope_policy") != SCOPE_POLICY
        or contract.get("min_points_per_direction_and_role") != MIN_POINTS
        or contract.get("min_matches_per_direction_and_role") != MIN_MATCHES
        or contract.get("joint_role_eligibility") is not True
        or contract.get("joint_global_fallback") is not True
    ):
        raise RecommendationContractError("El contrato upstream de politica no coincide.")
    return {"selection": selection, "validation": validation, "chronological": chronological}


def _strict_bool(value: Any, field: str) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    raise RecommendationContractError(f"{field} debe ser booleano estricto.")


def _finite_unit(value: Any, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise RecommendationContractError(f"{field} debe ser numerico real.")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0 <= numeric <= 1:
        raise RecommendationContractError(f"{field} debe ser finito y pertenecer a [0, 1].")
    return numeric


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise RecommendationContractError(f"{field} debe ser entero no negativo.")
    if int(value) < 0:
        raise RecommendationContractError(f"{field} debe ser entero no negativo.")
    return int(value)


def _is_absent(value: Any) -> bool:
    return value is None or pd.isna(value)


def _validate_test_seal(seal: Mapping[str, Any]) -> None:
    """Valida el contrato cerrado del test para resultados disponibles y no disponibles."""

    if not isinstance(seal, Mapping):
        raise RecommendationContractError("Bloque de sellado de test invalido.")
    if seal.get("test_status") != "sealed":
        raise RecommendationContractError("Sellado de test alterado: test_status.")
    if seal.get("used_for_method_selection") is not False:
        raise RecommendationContractError("Sellado de test alterado: used_for_method_selection.")
    if seal.get("test_target_matches_excluded_before_construction") != EXPECTED_EXCLUDED_TEST_TARGET_MATCHES:
        raise RecommendationContractError("Sellado de test alterado: partidos excluidos.")
    missing = [field for field in TEST_ZERO_FIELDS if field not in seal]
    if missing:
        raise RecommendationContractError(f"Sellado de test incompleto: {missing}")
    nonzero = [field for field in TEST_ZERO_FIELDS if type(seal[field]) is not int or seal[field] != 0]
    if nonzero:
        raise RecommendationContractError(f"Sellado de test alterado: {nonzero}")


def _validate_upstream_mapping_contract() -> None:
    """Comprueba que el adaptador no pueda cruzar scope, rol o medida."""

    expected = {
        ("server_points", "surface"): "server_direction_points_surface",
        ("server_points", "global"): "server_direction_points_global",
        ("server_matches", "surface"): "server_direction_matches_surface",
        ("server_matches", "global"): "server_direction_matches_global",
        ("server_wilson_low", "surface"): "server_direction_wilson_low_surface",
        ("server_wilson_low", "global"): "server_direction_wilson_low_global",
        ("server_wilson_high", "surface"): "server_direction_wilson_high_surface",
        ("server_wilson_high", "global"): "server_direction_wilson_high_global",
        ("opponent_points", "surface"): "opponent_direction_points_surface",
        ("opponent_points", "global"): "opponent_direction_points_global",
        ("opponent_matches", "surface"): "opponent_direction_matches_surface",
        ("opponent_matches", "global"): "opponent_direction_matches_global",
        ("opponent_wilson_low", "surface"): "opponent_allowed_server_wilson_low_surface",
        ("opponent_wilson_low", "global"): "opponent_allowed_server_wilson_low_global",
        ("opponent_wilson_high", "surface"): "opponent_allowed_server_wilson_high_surface",
        ("opponent_wilson_high", "global"): "opponent_allowed_server_wilson_high_global",
    }
    if {key: UPSTREAM_COLUMN_MAP.get(key) for key in expected} != expected:
        raise RecommendationContractError("Mapping upstream de rol/scope/medida inesperado.")


def _orientation_metadata(rows: pd.DataFrame) -> dict[str, Any]:
    missing = sorted(set(ORIENTATION_KEY + ["direction", "eligible", "selected_scope"]) - set(rows.columns))
    if missing:
        raise RecommendationContractError(f"Faltan columnas de orientacion: {missing}")
    metadata: dict[str, Any] = {}
    for column in ORIENTATION_KEY:
        values = rows[column].tolist()
        if len(set(map(str, values))) != 1:
            raise RecommendationContractError(f"Identidad inconsistente de orientacion: {column}")
        value = values[0]
        if column == "target_date":
            timestamp = pd.to_datetime(value, errors="coerce")
            if pd.isna(timestamp) or pd.Timestamp(timestamp).normalize() != pd.Timestamp(timestamp):
                raise RecommendationContractError("target_date debe ser fecha civil valida.")
            metadata[column] = pd.Timestamp(timestamp).normalize()
        elif column == "fold":
            if type(value) is bool or not isinstance(value, (int, np.integer)) or int(value) not in FOLDS:
                raise RecommendationContractError("fold fuera del contrato.")
            metadata[column] = int(value)
        elif not isinstance(value, str) or not value or value != value.strip():
            raise RecommendationContractError(f"{column} debe ser texto no vacio sin espacios extremos.")
        else:
            metadata[column] = value
    if metadata["target_surface"] not in SURFACES:
        raise RecommendationContractError("Superficie fuera del contrato.")
    if metadata["target_player"] == metadata["opponent"]:
        raise RecommendationContractError("Jugador y rival no pueden coincidir.")
    if metadata["target_date"].year != metadata["fold"]:
        raise RecommendationContractError("fold no coincide con target_date.")
    return metadata


def _direction_payload(row: pd.Series, metadata: Mapping[str, Any]) -> dict[str, Any]:
    direction = row["direction"]
    if direction not in DIRECTIONS:
        raise RecommendationContractError("Direccion fuera del contrato.")
    eligible = _strict_bool(row["eligible"], "eligible")
    scope = row["selected_scope"]
    if eligible and scope not in SCOPES:
        raise RecommendationContractError("selected_scope elegible fuera del contrato.")
    if not eligible and scope is not None and not pd.isna(scope):
        raise RecommendationContractError("Una direccion no elegible no puede tener scope seleccionado.")
    result = dict(metadata)
    result.update({"direction": direction, "eligible": eligible, "selected_scope": None if not eligible else str(scope)})
    fields = (
        "score", "server_component", "opponent_component", "server_selected_rate",
        "opponent_allowed_rate", "server_wilson_low", "server_wilson_high",
        "opponent_wilson_low", "opponent_wilson_high",
    )
    for field in fields:
        if field not in row.index:
            raise RecommendationContractError(f"Falta {field} en una direccion.")
    counts = ("server_points", "server_matches", "opponent_points", "opponent_matches")
    for field in counts:
        if field not in row.index:
            raise RecommendationContractError(f"Falta {field} en una direccion.")
        result[field] = _nonnegative_int(row[field], field)
    if result["server_matches"] > result["server_points"] or result["opponent_matches"] > result["opponent_points"]:
        raise RecommendationContractError("Los partidos de evidencia no pueden superar los puntos.")
    if eligible:
        for field in fields:
            result[field] = _finite_unit(row[field], field)
        if result["server_wilson_low"] > result["server_wilson_high"] or result["opponent_wilson_low"] > result["opponent_wilson_high"]:
            raise RecommendationContractError("Intervalo Wilson invertido.")
        if abs(result["score"] - (result["server_selected_rate"] + result["opponent_allowed_rate"]) / 2) > RECONCILIATION_TOLERANCE:
            raise RecommendationContractError("El score no coincide con la formula 50/50.")
    else:
        for field in fields:
            if row[field] is not None and not pd.isna(row[field]):
                raise RecommendationContractError("Una direccion no elegible no puede conservar score o componentes.")
            result[field] = None
    return result


def evaluate_orientation(rows: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evalua una orientacion desde sus filas direccionales sin recalcular historia.

    La prioridad de estado es invalidez, evidencia insuficiente, scopes mixtos,
    disponibilidad completa. Devuelve detalles solo en memoria.
    """

    if not isinstance(rows, pd.DataFrame) or rows.empty:
        raise RecommendationContractError("La orientacion debe contener al menos una fila.")
    try:
        metadata = _orientation_metadata(rows)
    except RecommendationContractError:
        # La identidad inconsistente es una entrada no publicable. Conservamos una
        # respuesta cerrada para la API pura sin intentar normalizar identidades.
        fallback = {column: rows.iloc[0][column] if column in rows.columns else "<invalid>" for column in ORIENTATION_KEY}
        fallback["target_date"] = pd.to_datetime(fallback["target_date"], errors="coerce")
        record = {
            **fallback, "recommendation_status": "not_available_invalid_input",
            "comparison_scope": None, "scope_state": "invalid_input",
            "observed_direction_scopes": [None, None, None], "reason_codes": ["invalid_input"],
            "selected_direction": None,
        }
        return record, []
    record = dict(metadata)
    try:
        if rows["direction"].duplicated().any():
            raise RecommendationContractError("Direccion duplicada en orientacion.")
        payload_by_direction = {str(row["direction"]): _direction_payload(row, metadata) for _, row in rows.iterrows()}
    except RecommendationContractError:
        observed = []
        for direction in DIRECTIONS:
            candidate = rows.loc[rows["direction"].eq(direction), "selected_scope"]
            observed.append(None if candidate.empty or pd.isna(candidate.iloc[0]) else str(candidate.iloc[0]))
        record.update({
            "recommendation_status": "not_available_invalid_input",
            "comparison_scope": None, "scope_state": "invalid_input",
            "observed_direction_scopes": observed, "reason_codes": ["invalid_input"],
            "selected_direction": None,
        })
        return record, []
    present = set(payload_by_direction)
    direction_rows = [payload_by_direction[direction] for direction in DIRECTIONS if direction in payload_by_direction]
    record["observed_direction_scopes"] = [
        payload_by_direction[direction]["selected_scope"] if direction in payload_by_direction else None
        for direction in DIRECTIONS
    ]
    record["selected_direction"] = None
    if present != set(DIRECTIONS):
        status, comparison_scope, scope_state = (
            "abstained_insufficient_evidence", None, "insufficient_evidence"
        )
        reason_codes = ["incomplete_direction_set"]
    elif not all(item["eligible"] for item in direction_rows):
        status, comparison_scope, scope_state = (
            "abstained_insufficient_evidence", None, "insufficient_evidence"
        )
        reason_codes = ["direction_not_scoreable"]
    else:
        scopes = [str(item["selected_scope"]) for item in direction_rows]
        if len(set(scopes)) != 1:
            status, comparison_scope, scope_state = (
                "abstained_incomparable_scope", None, "mixed_incomparable"
            )
            reason_codes = ["mixed_direction_scopes"]
        else:
            status, comparison_scope = "available_full", scopes[0]
            scope_state = f"{comparison_scope}_common"
            reason_codes = []
    record.update({
        "recommendation_status": status,
        "comparison_scope": comparison_scope,
        "scope_state": scope_state,
        "reason_codes": reason_codes,
    })
    details: list[dict[str, Any]] = []
    if status == "available_full":
        rank_index = {direction: index for index, direction in enumerate(DIRECTIONS)}
        ordered = sorted(direction_rows, key=lambda item: (-float(item["score"]), rank_index[item["direction"]]))
        record["selected_direction"] = ordered[0]["direction"]
        tied = any(abs(float(ordered[index]["score"]) - float(ordered[index - 1]["score"])) <= COMPONENT_TOLERANCE for index in range(1, len(ordered)))
        best = float(ordered[0]["score"])
        for position, item in enumerate(ordered, start=1):
            codes = ["recommendation_available"]
            codes.append("surface_history_used" if comparison_scope == "surface" else "global_fallback_used")
            difference = float(item["server_component"]) - float(item["opponent_component"])
            if abs(difference) <= COMPONENT_TOLERANCE:
                codes.append("balanced_components")
            elif difference > 0:
                codes.append("server_component_higher")
            else:
                codes.append("opponent_component_higher")
            if position > 1 and best - float(item["score"]) <= MARGIN_TOLERANCE:
                codes.append("narrow_score_margin")
            if (
                item["server_points"] == MIN_POINTS or item["server_matches"] == MIN_MATCHES
                or item["opponent_points"] == MIN_POINTS or item["opponent_matches"] == MIN_MATCHES
            ):
                codes.append("evidence_threshold_just_met")
            else:
                codes.append("evidence_above_threshold")
            if tied:
                codes.append("deterministic_tie_break")
            next_score = None if position == len(ordered) else float(ordered[position]["score"])
            detail = dict(item)
            detail.update({
                "recommendation_status": status, "comparison_scope": comparison_scope,
                "scope_state": scope_state, "rank": position,
                "score_gap_from_rank_1": best - float(item["score"]),
                "score_gap_to_next": None if next_score is None else float(item["score"]) - next_score,
                "explanation_codes": sorted(codes),
            })
            details.append(detail)
    else:
        codes = sorted([*reason_codes, "recommendation_abstained"])
        for item in direction_rows:
            detail = dict(item)
            detail.update({
                "recommendation_status": status, "comparison_scope": None,
                "scope_state": scope_state, "rank": None, "score_gap_from_rank_1": None,
                "score_gap_to_next": None, "explanation_codes": codes,
            })
            details.append(detail)
    return record, details


def validate_upstream_scoring_schema(scored: pd.DataFrame) -> None:
    """Preflight cerrado de la tabla canónica anterior al adaptador del motor."""

    if not isinstance(scored, pd.DataFrame):
        raise RecommendationContractError("scoring_schema_preflight: se requiere un DataFrame.")
    _validate_upstream_mapping_contract()
    duplicates = sorted({str(column) for column in scored.columns[scored.columns.duplicated()]})
    missing = sorted(set(UPSTREAM_REQUIRED_COLUMNS) - set(scored.columns))
    if duplicates or missing:
        detail = {"stage": "scoring_schema_preflight", "missing_columns": missing, "duplicate_columns": duplicates}
        raise RecommendationContractError(json.dumps(detail, ensure_ascii=False, sort_keys=True))
    for column in ("target_match_id", "target_player", "opponent"):
        values = scored[column]
        if values.isna().any() or not values.map(lambda value: isinstance(value, str) and bool(value) and value == value.strip()).all():
            raise RecommendationContractError(f"scoring_schema_preflight: dominio inválido en {column}.")
    dates = pd.to_datetime(scored["target_date"], errors="coerce")
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise RecommendationContractError("scoring_schema_preflight: target_date inválida.")
    folds = scored["fold"]
    if any(type(value) is bool or not isinstance(value, (int, np.integer)) or int(value) not in FOLDS for value in folds):
        raise RecommendationContractError("scoring_schema_preflight: fold fuera de contrato.")
    if not scored["direction"].isin(DIRECTIONS).all() or not scored["target_surface"].isin(SURFACES).all():
        raise RecommendationContractError("scoring_schema_preflight: dirección o superficie fuera de contrato.")
    if any(type(value) is not bool and not isinstance(value, np.bool_) for value in scored["eligible"]):
        raise RecommendationContractError("scoring_schema_preflight: eligible debe ser booleano estricto.")
    selected_scope = scored.loc[scored["eligible"].eq(True), "selected_scope"]
    if selected_scope.isna().any() or not selected_scope.isin(SCOPES).all():
        raise RecommendationContractError("scoring_schema_preflight: selected_scope elegible inválido.")
    if scored.loc[scored["eligible"].ne(True), "selected_scope"].notna().any():
        raise RecommendationContractError("scoring_schema_preflight: scope en dirección no elegible.")
    count_columns = [upstream for upstream, _, _, _, _, domain, _ in UPSTREAM_SCORING_SCHEMA if domain == "non_negative_integer"]
    for column in count_columns:
        for value in scored[column].tolist():
            _nonnegative_int(value, f"scoring_schema_preflight: conteo inválido en {column}")
    rate_columns = [upstream for upstream, _, _, _, _, domain, _ in UPSTREAM_SCORING_SCHEMA if domain == "unit_interval_or_null"]
    for column in rate_columns:
        for value in scored[column].tolist():
            if not _is_absent(value):
                _finite_unit(value, f"scoring_schema_preflight: tasa fuera de [0, 1] en {column}")
    for column in ("server_component", "opponent_component"):
        for value in scored[column].tolist():
            if not _is_absent(value):
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(float(value)):
                    raise RecommendationContractError(f"scoring_schema_preflight: componente inválido en {column}.")
    for role in ("server", "opponent"):
        for scope in SCOPES:
            low_column = UPSTREAM_COLUMN_MAP[(f"{role}_wilson_low", scope)]
            high_column = UPSTREAM_COLUMN_MAP[(f"{role}_wilson_high", scope)]
            for low, high in zip(scored[low_column].tolist(), scored[high_column].tolist()):
                if not _is_absent(low) and not _is_absent(high) and float(low) > float(high):
                    raise RecommendationContractError(f"scoring_schema_preflight: Wilson invertido para {role}/{scope}.")
        for measure in ("points", "matches"):
            surface_counts = scored[UPSTREAM_COLUMN_MAP[(f"{role}_{measure}", "surface")]]
            global_counts = scored[UPSTREAM_COLUMN_MAP[(f"{role}_{measure}", "global")]]
            if (surface_counts > global_counts).any():
                raise RecommendationContractError(f"scoring_schema_preflight: {role}_{measure} surface supera global.")
    for row in scored.itertuples(index=False):
        values = row._asdict()
        eligible = _strict_bool(values["eligible"], "scoring_schema_preflight: eligible")
        scope = values["selected_scope"]
        evidence_status = values["evidence_status"]
        abstention_reason = values["abstention_reason"]
        if eligible:
            if not isinstance(evidence_status, str) or evidence_status != EVIDENCE_STATUS_BY_SCOPE[scope]:
                raise RecommendationContractError("scoring_schema_preflight: evidence_status elegible inválido.")
            if not _is_absent(abstention_reason):
                raise RecommendationContractError("scoring_schema_preflight: abstention_reason en fila elegible.")
            expected_server = values[f"server_direction_raw_rate_{scope}"]
            expected_opponent = values[f"opponent_allowed_server_raw_rate_{scope}"]
            if not math.isclose(float(values["server_rate"]), float(expected_server), rel_tol=0, abs_tol=RATE_RECONCILIATION_TOLERANCE):
                raise RecommendationContractError("scoring_schema_preflight: server_rate no coincide con selected_scope.")
            if not math.isclose(float(values["opponent_allowed_rate"]), float(expected_opponent), rel_tol=0, abs_tol=RATE_RECONCILIATION_TOLERANCE):
                raise RecommendationContractError("scoring_schema_preflight: opponent_allowed_rate no coincide con selected_scope.")
            if not math.isclose(float(values["combined_score"]), (float(values["server_rate"]) + float(values["opponent_allowed_rate"])) / 2, rel_tol=0, abs_tol=RATE_RECONCILIATION_TOLERANCE):
                raise RecommendationContractError("scoring_schema_preflight: combined_score no coincide con la formula.")
        else:
            if not isinstance(evidence_status, str) or evidence_status != ABSTAINED_EVIDENCE_STATUS:
                raise RecommendationContractError("scoring_schema_preflight: evidence_status abstained inválido.")
            if not isinstance(abstention_reason, str) or abstention_reason != ABSTAINED_EVIDENCE_STATUS:
                raise RecommendationContractError("scoring_schema_preflight: abstention_reason abstained inválido.")
            for column in ("server_rate", "opponent_allowed_rate", "combined_score", "server_component", "opponent_component"):
                if not _is_absent(values[column]):
                    raise RecommendationContractError(f"scoring_schema_preflight: {column} debe ser nulo si no es elegible.")


def _standardize_orientation_rows(scored: pd.DataFrame) -> pd.DataFrame:
    """Convierte la salida upstream a la API pura sin reejecutar la politica."""

    validate_upstream_scoring_schema(scored)
    work = scored.copy(deep=True)
    output = work[ORIENTATION_KEY + ["direction", "eligible", "selected_scope"]].copy()
    output["score"] = work["combined_score"]
    # Componentes de recomendación: las dos tasas en la fórmula 50/50. Los
    # campos upstream ``*_component`` son excesos frente a población y pueden
    # ser negativos; no son componentes acotados del score solicitado aquí.
    output["server_component"] = work["server_rate"]
    output["opponent_component"] = work["opponent_allowed_rate"]
    output["server_selected_rate"] = work["server_rate"]
    output["opponent_allowed_rate"] = work["opponent_allowed_rate"]
    for role, lower, upper in (("server", "server_wilson_low", "server_wilson_high"), ("opponent", "opponent_wilson_low", "opponent_wilson_high")):
        selected = work["selected_scope"]
        output[lower] = np.select(
            [selected.eq("surface"), selected.eq("global")],
            [work[UPSTREAM_COLUMN_MAP[(lower, "surface")]], work[UPSTREAM_COLUMN_MAP[(lower, "global")]]],
            default=np.nan,
        )
        output[upper] = np.select(
            [selected.eq("surface"), selected.eq("global")],
            [work[UPSTREAM_COLUMN_MAP[(upper, "surface")]], work[UPSTREAM_COLUMN_MAP[(upper, "global")]]],
            default=np.nan,
        )
        output[f"{role}_points"] = np.select(
            [selected.eq("surface"), selected.eq("global")],
            [work[UPSTREAM_COLUMN_MAP[(f"{role}_points", "surface")]], work[UPSTREAM_COLUMN_MAP[(f"{role}_points", "global")]]],
            default=0,
        ).astype("int64")
        output[f"{role}_matches"] = np.select(
            [selected.eq("surface"), selected.eq("global")],
            [work[UPSTREAM_COLUMN_MAP[(f"{role}_matches", "surface")]], work[UPSTREAM_COLUMN_MAP[(f"{role}_matches", "global")]]],
            default=0,
        ).astype("int64")
    return output


def build_recommendations(scored_orientations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = _standardize_orientation_rows(scored_orientations)
    return build_recommendations_from_direction_rows(rows)


def build_recommendations_from_direction_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye el motor desde filas estandarizadas; util para pruebas en memoria."""

    if rows.duplicated(["target_match_id", "target_player", "direction"]).any():
        # evaluate_orientation conserva el estado invalid_input para este caso.
        pass
    recommendations: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for _, group in rows.groupby(["target_match_id", "target_player"], sort=False, observed=True):
        recommendation, direction_details = evaluate_orientation(group)
        recommendations.append(recommendation)
        details.extend(direction_details)
    recommendation_frame = pd.DataFrame(recommendations, columns=RECOMMENDATION_COLUMNS)
    detail_frame = pd.DataFrame(details, columns=DETAIL_COLUMNS)
    return (
        recommendation_frame.sort_values(["target_date", "target_match_id", "target_player"], kind="stable").reset_index(drop=True),
        detail_frame.sort_values(["target_date", "target_match_id", "target_player", "rank", "direction"], kind="stable", na_position="last").reset_index(drop=True),
    )


def _strata(frame: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    result: list[tuple[str, str, pd.DataFrame]] = [("pooled", "<ALL>", frame)]
    for fold in FOLDS:
        result.append((str(fold), "<ALL>", frame.loc[frame["fold"].eq(fold)]))
    for surface in SURFACES:
        result.append(("pooled", surface, frame.loc[frame["target_surface"].eq(surface)]))
    for fold in FOLDS:
        for surface in SURFACES:
            result.append((str(fold), surface, frame.loc[frame["fold"].eq(fold) & frame["target_surface"].eq(surface)]))
    return result


def build_coverage(recommendations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, surface, group in _strata(recommendations):
        total = len(group)
        available = int(group["recommendation_status"].eq("available_full").sum())
        for status, scope_state in (
            ("available_full", "surface_common"),
            ("available_full", "global_common"),
            ("abstained_insufficient_evidence", "insufficient_evidence"),
            ("abstained_incomparable_scope", "mixed_incomparable"),
            ("not_available_invalid_input", "invalid_input"),
        ):
            part = group.loc[(group["recommendation_status"].eq(status)) & (group["scope_state"].eq(scope_state))]
            reason = "" if status == "available_full" else {
                "abstained_insufficient_evidence": "insufficient_direction_evidence",
                "abstained_incomparable_scope": "mixed_direction_scopes",
                "not_available_invalid_input": "invalid_input",
            }[status]
            count = int(len(part))
            rows.append({
                "fold": fold, "surface": surface, "recommendation_status": status,
                "scope_state": scope_state, "orientations": count,
                "matches": int(part["target_match_id"].nunique()),
                "players": int(part["target_player"].nunique()),
                "available_orientations": available, "abstained_orientations": total - available,
                "coverage_rate": 0.0 if total == 0 else available / total,
                "abstention_rate": 0.0 if total == 0 else (total - available) / total,
                "reason_code": reason,
            })
    order = {value: index for index, value in enumerate(SCOPE_STATES)}
    output = pd.DataFrame(rows, columns=COVERAGE_COLUMNS)
    output["_fold"] = pd.Categorical(output["fold"], [*map(str, FOLDS), "pooled"], ordered=True)
    output["_surface"] = pd.Categorical(output["surface"], ["<ALL>", *SURFACES], ordered=True)
    output["_scope"] = output["scope_state"].map(order)
    return output.sort_values(["_fold", "_surface", "_scope"], kind="stable").drop(columns=["_fold", "_surface", "_scope"]).reset_index(drop=True)


def _available_points(point_rows: pd.DataFrame, details: pd.DataFrame) -> pd.DataFrame:
    available = details.loc[details["recommendation_status"].eq("available_full")].copy()
    if available.empty:
        return pd.DataFrame(columns=[*point_rows.columns, "rank", "score", "comparison_scope", "score_gap_from_rank_1", "server_component", "opponent_component", "explanation_codes"])
    keep = ["target_match_id", "target_player", "direction", "rank", "score", "comparison_scope", "score_gap_from_rank_1", "server_component", "opponent_component", "explanation_codes"]
    output = point_rows.merge(available[keep], on=["target_match_id", "target_player", "direction"], how="inner", validate="many_to_one", sort=False)
    if not output["recommendation_status"].eq("available_full").all() if "recommendation_status" in output else False:
        raise RecommendationContractError("Un punto no disponible alcanzo rankings.")
    return output


def build_rankings(point_rows: pd.DataFrame, details: pd.DataFrame) -> pd.DataFrame:
    available = details.loc[details["recommendation_status"].eq("available_full")].copy()
    points = _available_points(point_rows, details)
    rows: list[dict[str, Any]] = []
    for fold, surface, group in _strata(available):
        for scope in SCOPES:
            scope_group = group.loc[group["comparison_scope"].eq(scope)]
            for rank in (1, 2, 3):
                rank_group = scope_group.loc[scope_group["rank"].eq(rank)]
                denominator = len(rank_group)
                for direction in DIRECTIONS:
                    part = rank_group.loc[rank_group["direction"].eq(direction)]
                    point_part = points.loc[
                        (points["fold"].astype(str).eq(fold) if fold != "pooled" else pd.Series(True, index=points.index))
                        & (points["target_surface"].eq(surface) if surface != "<ALL>" else pd.Series(True, index=points.index))
                        & points["comparison_scope"].eq(scope) & points["rank"].eq(rank) & points["direction"].eq(direction)
                    ]
                    rows.append({
                        "fold": fold, "surface": surface, "rank": rank, "direction": direction,
                        "comparison_scope": scope, "orientations": int(len(part)),
                        "direction_share": None if denominator == 0 else len(part) / denominator,
                        "mean_score": None if part.empty else float(part["score"].mean()),
                        "median_score": None if part.empty else float(part["score"].median()),
                        "mean_score_gap_from_rank_1": None if part.empty else float(part["score_gap_from_rank_1"].mean()),
                        "median_score_gap_from_rank_1": None if part.empty else float(part["score_gap_from_rank_1"].median()),
                        "observed_points": int(len(point_part)),
                        "observed_server_wins": int(point_part["outcome"].sum()) if len(point_part) else 0,
                        "observed_win_rate": None if point_part.empty else float(point_part["outcome"].mean()),
                    })
    output = pd.DataFrame(rows, columns=RANKINGS_COLUMNS)
    output["_fold"] = pd.Categorical(output["fold"], [*map(str, FOLDS), "pooled"], ordered=True)
    output["_surface"] = pd.Categorical(output["surface"], ["<ALL>", *SURFACES], ordered=True)
    output["_direction"] = pd.Categorical(output["direction"], DIRECTIONS, ordered=True)
    output["_scope"] = pd.Categorical(output["comparison_scope"], SCOPES, ordered=True)
    return output.sort_values(["_fold", "_surface", "rank", "_scope", "_direction"], kind="stable").drop(columns=["_fold", "_surface", "_direction", "_scope"]).reset_index(drop=True)


def build_explanations(details: pd.DataFrame) -> pd.DataFrame:
    available = details.loc[details["recommendation_status"].eq("available_full")].copy()
    rows: list[dict[str, Any]] = []
    for fold, surface, group in _strata(available):
        for scope in SCOPES:
            scoped = group.loc[group["comparison_scope"].eq(scope)]
            for rank in (1, 2, 3):
                rank_group = scoped.loc[scoped["rank"].eq(rank)]
                for direction in DIRECTIONS:
                    part = rank_group.loc[rank_group["direction"].eq(direction)]
                    denominator = len(part)
                    code_rows: list[dict[str, Any]] = []
                    for row in part.itertuples(index=False):
                        for code in row.explanation_codes:
                            code_rows.append({"code": code, "server": row.server_component, "opponent": row.opponent_component})
                    if not code_rows:
                        continue
                    for code, coded in pd.DataFrame(code_rows).groupby("code", sort=True):
                        rows.append({
                            "fold": fold, "surface": surface, "rank": rank, "direction": direction,
                            "comparison_scope": scope, "explanation_code": code,
                            "orientations": int(len(coded)), "share_within_rank_direction": len(coded) / denominator,
                            "mean_server_component": float(coded["server"].mean()),
                            "mean_opponent_component": float(coded["opponent"].mean()),
                            "mean_absolute_component_difference": float((coded["server"] - coded["opponent"]).abs().mean()),
                        })
    output = pd.DataFrame(rows, columns=EXPLANATIONS_COLUMNS)
    if output.empty:
        return output
    output["_fold"] = pd.Categorical(output["fold"], [*map(str, FOLDS), "pooled"], ordered=True)
    output["_surface"] = pd.Categorical(output["surface"], ["<ALL>", *SURFACES], ordered=True)
    output["_direction"] = pd.Categorical(output["direction"], DIRECTIONS, ordered=True)
    output["_scope"] = pd.Categorical(output["comparison_scope"], SCOPES, ordered=True)
    return output.sort_values(["_fold", "_surface", "rank", "_scope", "_direction", "explanation_code"], kind="stable").drop(columns=["_fold", "_surface", "_direction", "_scope"]).reset_index(drop=True)


def _summary_counts(recommendations: pd.DataFrame) -> dict[str, Any]:
    counts = recommendations["recommendation_status"].value_counts().to_dict()
    total = len(recommendations)
    available_surface = int(((recommendations["recommendation_status"] == "available_full") & (recommendations["comparison_scope"] == "surface")).sum())
    available_global = int(((recommendations["recommendation_status"] == "available_full") & (recommendations["comparison_scope"] == "global")).sum())
    mixed = recommendations.loc[recommendations["recommendation_status"].eq("abstained_incomparable_scope"), "observed_direction_scopes"]
    patterns = {
        " / ".join("none" if value is None else str(value) for value in values): int(count)
        for values, count in mixed.map(tuple).value_counts().sort_index().items()
    }
    available = int(counts.get("available_full", 0))
    insufficient = int(counts.get("abstained_insufficient_evidence", 0))
    incomparable = int(counts.get("abstained_incomparable_scope", 0))
    invalid = int(counts.get("not_available_invalid_input", 0))
    return {
        "total_orientations": total, "available_full_orientations": available,
        "abstained_insufficient_evidence_orientations": insufficient,
        "abstained_incomparable_scope_orientations": incomparable,
        "invalid_input_orientations": invalid,
        "recommendation_coverage_rate": 0.0 if total == 0 else available / total,
        "total_abstention_rate": 0.0 if total == 0 else (insufficient + incomparable) / total,
        "insufficient_evidence_rate": 0.0 if total == 0 else insufficient / total,
        "incomparable_scope_rate": 0.0 if total == 0 else incomparable / total,
        "available_surface_common": available_surface, "available_global_common": available_global,
        "mixed_scope_patterns": patterns,
    }


def validate_population_cardinalities(population: Mapping[str, Any], folds: Mapping[str, Any]) -> None:
    """Mantiene separadas las unidades selladas, de validacion y direccionales."""

    required = (
        "sealed_target_matches", "sealed_snapshot_rows", "sealed_feature_rows",
        "validation_matches", "validation_orientation_rows", "validation_feature_rows",
        "validation_direction_rows",
    )
    if any(type(population.get(key)) is not int for key in required):
        raise RecommendationContractError("Cardinalidades de etapas incompletas o no enteras.")
    sealed_targets = population["sealed_target_matches"]
    sealed_snapshots = population["sealed_snapshot_rows"]
    sealed_features = population["sealed_feature_rows"]
    validation_matches = population["validation_matches"]
    validation_orientations = population["validation_orientation_rows"]
    validation_features = population["validation_feature_rows"]
    validation_directions = population["validation_direction_rows"]
    if sealed_snapshots != sealed_targets * 2:
        raise RecommendationContractError("sealed_snapshot_rows no reconcilia con sealed_target_matches.")
    if sealed_features != sealed_snapshots * 3 * 2:
        raise RecommendationContractError("sealed_feature_rows no reconcilia con snapshots, direcciones y scopes.")
    if validation_orientations != validation_matches * 2:
        raise RecommendationContractError("validation_orientation_rows no reconcilia con validation_matches.")
    if validation_features != validation_orientations * 3 * 2:
        raise RecommendationContractError("validation_feature_rows no reconcilia con orientaciones, direcciones y scopes.")
    if validation_directions != validation_orientations * 3:
        raise RecommendationContractError("validation_direction_rows no reconcilia con orientaciones y direcciones.")
    if validation_features != validation_directions * 2:
        raise RecommendationContractError("validation_feature_rows debe contener exactamente dos scopes por direccion.")
    if validation_features > sealed_features or validation_matches > sealed_targets:
        raise RecommendationContractError("La validacion no puede superar la poblacion sellada.")
    if population.get("source_point_rows") == EXPECTED_SOURCE_ROWS:
        expected = {
            "source_matches": 7_524, "source_players": 1_002,
            "eligible_historical_points": 481_190, "sealed_target_matches": 5_993,
            "excluded_test_target_matches": 1_531, "sealed_snapshot_rows": 11_986,
            "sealed_feature_rows": 71_916, "validation_matches": 1_805,
            "validation_orientation_rows": 3_610, "validation_feature_rows": 21_660,
            "validation_direction_rows": 10_830,
        }
        for key, value in expected.items():
            if population.get(key) != value:
                raise RecommendationContractError(f"Cardinalidad real inesperada: {key}.")
        if tuple(int(folds.get(str(year), -1)) for year in FOLDS) != (168, 371, 646, 620):
            raise RecommendationContractError("Folds reales no reconcilian.")
def _payload_metadata(payloads: Sequence[bytes]) -> tuple[dict[str, str], dict[str, int]]:
    if len(payloads) != len(PAYLOAD_NAMES):
        raise RecommendationContractError("Se requieren tres payloads CSV.")
    return (
        {name: _sha256(payload) for name, payload in zip(PAYLOAD_NAMES, payloads)},
        {name: len(payload) for name, payload in zip(PAYLOAD_NAMES, payloads)},
    )


def _fingerprint(summary: Mapping[str, Any], payloads: Sequence[bytes]) -> str:
    hashes, sizes = _payload_metadata(payloads)
    core = {key: value for key, value in summary.items() if key != "publication_fingerprint"}
    if core.get("fingerprint_contract_version") != FINGERPRINT_CONTRACT_VERSION:
        raise RecommendationContractError("Version de fingerprint inesperada.")
    if core.get("artifact_payload_sha256") != hashes or core.get("artifact_payload_bytes") != sizes:
        raise RecommendationContractError("Metadatos de payload incoherentes.")
    return _sha256(json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _finalize_summary(summary: Mapping[str, Any], frames: Sequence[pd.DataFrame]) -> tuple[dict[str, Any], tuple[bytes, ...], str]:
    payloads = tuple(_frame_bytes(frame) for frame in frames)
    hashes, sizes = _payload_metadata(payloads)
    complete = dict(summary)
    complete["fingerprint_contract_version"] = FINGERPRINT_CONTRACT_VERSION
    complete["artifact_payload_sha256"] = hashes
    complete["artifact_payload_bytes"] = sizes
    fingerprint = _fingerprint(complete, payloads)
    complete["publication_fingerprint"] = fingerprint
    return complete, payloads, fingerprint


def build_summary(
    *, source_audit: Mapping[str, Any], sealed_audit: Mapping[str, Any],
    recommendations: pd.DataFrame, details: pd.DataFrame, point_rows: pd.DataFrame,
    coverage: pd.DataFrame, rankings: pd.DataFrame, explanations: pd.DataFrame,
    validation_feature_rows: int, validation_direction_rows: int,
) -> dict[str, Any]:
    status_counts = _summary_counts(recommendations)
    return _json_value({
        "analysis_name": ANALYSIS_NAME, "analysis_version": ANALYSIS_VERSION,
        "recommendation_status": "available",
        "source": {"path": "data/processed/points_enriched.parquet", "columns_used": list(SOURCE_COLUMNS), "single_read": True, "source_point_rows": int(source_audit["source_point_rows"])},
        "upstream_commit": "6680ae4", "upstream_artifacts": list(UPSTREAM_SCORING_HASHES), "upstream_hashes": dict(UPSTREAM_SCORING_HASHES),
        "selected_policy": POLICY_ID,
        "scoring_formula": "0.5 * server_selected_rate + 0.5 * opponent_allowed_rate",
        "ranking_contract": {"directions": list(DIRECTIONS), "requires_exactly_three": True, "rank_order": "score_descending", "tie_break": list(DIRECTIONS), "round_before_ordering": False},
        "tie_break_contract": {"order": list(DIRECTIONS), "interpretation": "reproducibility_only_not_tactical_evidence"},
        "comparability_contract": {"upstream_scope_selection": "per_direction_joint_server_opponent", "engine_barrier": "all_three_directions_must_share_selected_scope", "mixed_scopes": "abstain_without_ranking", "available_scopes": list(SCOPES), "does_not_claim_statistical_incompatibility": True},
        "abstention_contract": {"priority": ["not_available_invalid_input", "abstained_insufficient_evidence", "abstained_incomparable_scope", "available_full"], "no_partial_ranking": True},
        "explanation_contract": {"codes": sorted(EXPLANATION_CODES), "component_tolerance": COMPONENT_TOLERANCE, "narrow_score_margin": MARGIN_TOLERANCE, "text_free": True},
        "population": {"source_point_rows": int(source_audit["source_point_rows"]), "source_matches": int(source_audit["source_matches"]), "source_players": int(source_audit["source_players"]), "eligible_historical_points": int(source_audit["eligible_points"]), "sealed_target_matches": int(sealed_audit["constructed_target_matches"]), "excluded_test_target_matches": int(sealed_audit["test_target_matches_excluded_before_construction"]), "sealed_snapshot_rows": int(sealed_audit["constructed_snapshot_rows"]), "sealed_feature_rows": int(sealed_audit["constructed_feature_rows"]), "validation_matches": int(recommendations["target_match_id"].nunique()), "validation_orientation_rows": int(len(recommendations)), "validation_feature_rows": int(validation_feature_rows), "validation_direction_rows": int(validation_direction_rows), "eligible_scoring_points": int(len(point_rows)), "scored_points": int(point_rows["eligible"].sum()) if "eligible" in point_rows else int(len(point_rows)), "abstained_points": int((~point_rows["eligible"].astype(bool)).sum()) if "eligible" in point_rows else 0},
        "chronological_seal": {"test_status": "sealed", "test_target_rows_constructed": int(sealed_audit["test_target_rows_constructed"]), "test_target_matches_constructed": int(sealed_audit["test_target_matches_constructed"]), "test_feature_rows_constructed": int(sealed_audit["test_feature_rows_constructed"]), "test_recommendation_rows": 0, "test_rows_evaluated": 0, "test_matches_evaluated": 0, "test_points_scored": 0, "test_matches_scored": 0, "test_evaluation_runs": 0, "used_for_method_selection": False, "test_target_matches_excluded_before_construction": int(sealed_audit["test_target_matches_excluded_before_construction"])},
        "recommendation_statuses": status_counts,
        "reconciliations": {"upstream_hashes_exact": True, "source_rows_exact": int(source_audit["source_point_rows"]) == EXPECTED_SOURCE_ROWS, "source_matches_exact": int(source_audit["source_matches"]) == EXPECTED_MATCHES, "test_remains_sealed": True, "three_direction_ranks_only": True, "available_common_scope_only": True, "mixed_excluded_from_rankings": True, "scope_counts_exhaustive": True, "coverage_reconciled": True, "aggregates_reconciled": True, "no_individual_recommendations_published": True},
        "validation_metrics": {"folds": {str(fold): int(recommendations["fold"].eq(fold).sum() / 2) for fold in FOLDS}, "target_test_evaluation_runs": 0},
        "methodological_limits": ["Resultados descriptivos basados en historia observacional; no identifican efectos causales.", "El ranking no garantiza superioridad tactica ni generalizacion al test sellado.", "La barrera de scope comun es conservadora y puede reducir cobertura; no afirma incompatibilidad estadistica entre scopes.", "La tabla individual completa se conserva solo en memoria."],
    })


def _require_close(actual: Any, expected: Any, field: str) -> None:
    if actual is None or expected is None or not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=RECONCILIATION_TOLERANCE):
        raise RecommendationContractError(f"Reconciliacion fallida: {field}")


def _canonical(frame: pd.DataFrame, columns: Sequence[str], keys: Sequence[str]) -> pd.DataFrame:
    return frame.sort_values(list(keys), kind="stable").reset_index(drop=True)[list(columns)]


def validate_result(result: RecommendationResult, payloads: Sequence[bytes] | None = None) -> None:
    """Valida semantica contra las filas direccionales originales en memoria."""

    if result.summary.get("recommendation_status") != "available":
        raise RecommendationContractError("El resultado disponible debe declarar recommendation_status=available.")
    for frame, columns, label in ((result.coverage, COVERAGE_COLUMNS, "coverage"), (result.rankings, RANKINGS_COLUMNS, "rankings"), (result.explanations, EXPLANATIONS_COLUMNS, "explanations")):
        if frame.columns.tolist() != columns or any(str(column).startswith("Unnamed") for column in frame.columns):
            raise RecommendationContractError(f"Schema CSV invalido: {label}")
    if result.recommendations.columns.tolist() != RECOMMENDATION_COLUMNS or result.details.columns.tolist() != DETAIL_COLUMNS:
        raise RecommendationContractError("Schema de resultado interno invalido.")
    if result.recommendations.empty or result.details.empty or result.point_rows.empty:
        raise RecommendationContractError("El resultado disponible requiere filas internas completas.")
    rebuilt_records: list[dict[str, Any]] = []
    rebuilt_details: list[dict[str, Any]] = []
    standardized = result.details[ORIENTATION_KEY + [column for column in DETAIL_COLUMNS if column not in ORIENTATION_KEY and column not in {"recommendation_status", "comparison_scope", "scope_state", "rank", "score_gap_from_rank_1", "score_gap_to_next", "explanation_codes"}]].copy()
    for _, group in standardized.groupby(["target_match_id", "target_player"], sort=False, observed=True):
        record, details = evaluate_orientation(group)
        rebuilt_records.append(record)
        rebuilt_details.extend(details)
    expected_recommendations = pd.DataFrame(rebuilt_records, columns=RECOMMENDATION_COLUMNS).sort_values(["target_date", "target_match_id", "target_player"], kind="stable").reset_index(drop=True)
    expected_details = pd.DataFrame(rebuilt_details, columns=DETAIL_COLUMNS).sort_values(["target_date", "target_match_id", "target_player", "rank", "direction"], kind="stable", na_position="last").reset_index(drop=True)
    pdt.assert_frame_equal(result.recommendations.reset_index(drop=True), expected_recommendations, check_dtype=False, check_like=False)
    pdt.assert_frame_equal(result.details.reset_index(drop=True), expected_details, check_dtype=False, check_like=False)
    expected_coverage = build_coverage(expected_recommendations)
    expected_rankings = build_rankings(result.point_rows, expected_details)
    expected_explanations = build_explanations(expected_details)
    pdt.assert_frame_equal(result.coverage.reset_index(drop=True), expected_coverage, check_dtype=False, check_like=False)
    pdt.assert_frame_equal(result.rankings.reset_index(drop=True), expected_rankings, check_dtype=False, check_like=False)
    pdt.assert_frame_equal(result.explanations.reset_index(drop=True), expected_explanations, check_dtype=False, check_like=False)
    counts = _summary_counts(expected_recommendations)
    if result.summary.get("recommendation_statuses") != _json_value(counts):
        raise RecommendationContractError("Conteos de estados manipulados.")
    if counts["available_surface_common"] + counts["available_global_common"] != counts["available_full_orientations"]:
        raise RecommendationContractError("Scopes disponibles no reconcilian.")
    if sum(counts[key] for key in ("available_full_orientations", "abstained_insufficient_evidence_orientations", "abstained_incomparable_scope_orientations", "invalid_input_orientations")) != counts["total_orientations"]:
        raise RecommendationContractError("Estados no exhaustivos.")
    if sum(counts["mixed_scope_patterns"].values()) != counts["abstained_incomparable_scope_orientations"]:
        raise RecommendationContractError("Patrones mixtos no reconcilian.")
    for _, group in expected_details.loc[expected_details["recommendation_status"].eq("available_full")].groupby(["target_match_id", "target_player"], sort=False):
        if set(group["direction"]) != set(DIRECTIONS) or set(group["rank"]) != {1, 2, 3} or group["comparison_scope"].nunique() != 1:
            raise RecommendationContractError("Ranking disponible no contractual.")
    if not result.rankings.empty and result.rankings["rank"].isin((1, 2, 3)).all() is False:
        raise RecommendationContractError("Rank fuera de dominio.")
    _validate_test_seal(result.summary.get("chronological_seal", {}))
    if result.summary.get("reconciliations", {}).get("no_individual_recommendations_published") is not True:
        raise RecommendationContractError("Contrato de privacidad alterado.")
    validate_population_cardinalities(result.summary.get("population", {}), result.summary.get("validation_metrics", {}).get("folds", {}))
    _assert_finite_json(result.summary)
    actual_payloads = tuple(_frame_bytes(frame) for frame in (result.coverage, result.rankings, result.explanations)) if payloads is None else tuple(payloads)
    fingerprint = _fingerprint(result.summary, actual_payloads)
    if result.publication_fingerprint != fingerprint or result.summary.get("publication_fingerprint") != fingerprint:
        raise RecommendationContractError("Fingerprint alterado.")


def analyze_from_scoring(
    scored_orientations: pd.DataFrame,
    point_rows: pd.DataFrame,
    *, source_audit: Mapping[str, Any], sealed_audit: Mapping[str, Any],
) -> RecommendationResult:
    recommendations, details = build_recommendations(scored_orientations)
    coverage = build_coverage(recommendations)
    rankings = build_rankings(point_rows, details)
    explanations = build_explanations(details)
    summary = build_summary(source_audit=source_audit, sealed_audit=sealed_audit, recommendations=recommendations, details=details, point_rows=point_rows, coverage=coverage, rankings=rankings, explanations=explanations, validation_feature_rows=len(scored_orientations) * 2, validation_direction_rows=len(scored_orientations))
    summary, _, fingerprint = _finalize_summary(summary, (coverage, rankings, explanations))
    result = RecommendationResult(summary, coverage, rankings, explanations, recommendations, details, point_rows, fingerprint)
    validate_result(result)
    return result


def analyze_from_direction_rows(
    rows: pd.DataFrame,
    point_rows: pd.DataFrame,
    *, source_audit: Mapping[str, Any], sealed_audit: Mapping[str, Any],
) -> RecommendationResult:
    """Entrada puramente sintética o ya estandarizada, sin archivos locales."""

    recommendations, details = build_recommendations_from_direction_rows(rows.copy(deep=True))
    coverage = build_coverage(recommendations)
    rankings = build_rankings(point_rows, details)
    explanations = build_explanations(details)
    summary = build_summary(source_audit=source_audit, sealed_audit=sealed_audit, recommendations=recommendations, details=details, point_rows=point_rows, coverage=coverage, rankings=rankings, explanations=explanations, validation_feature_rows=len(rows) * 2, validation_direction_rows=len(rows))
    summary, _, fingerprint = _finalize_summary(summary, (coverage, rankings, explanations))
    result = RecommendationResult(summary, coverage, rankings, explanations, recommendations, details, point_rows, fingerprint)
    validate_result(result)
    return result


def build_real_analysis(points: pd.DataFrame) -> RecommendationResult:
    """Una unica construccion real: historia, sellado, features, scoring y motor."""

    verify_upstream_contracts()
    frame, targets, eligible, source_audit = prepare_historical_population(points)
    snapshots, features, sealed_audit, test_ids = construct_sealed_target_features(targets, eligible)
    validation_features = select_validation_features(features, test_ids, evaluation_runs=0)
    scored_orientations = prepare_scoring_population(validation_features)
    all_points, _ = join_target_outcomes(scored_orientations, eligible)
    source_audit = dict(source_audit)
    source_audit["source_players"] = int(len(set(targets["player_1"]) | set(targets["player_2"])))
    if len(frame) != EXPECTED_SOURCE_ROWS or len(targets) != EXPECTED_MATCHES:
        raise RecommendationContractError("Poblacion fuente inesperada.")
    if (
        len(snapshots) != 11_986 or len(features) != 71_916
        or len(validation_features) != 21_660 or len(scored_orientations) != 10_830
        or len(targets) - len(test_ids) != 5_993
    ):
        raise RecommendationContractError("Cardinalidad de construccion sellada inesperada.")
    return analyze_from_scoring(scored_orientations, all_points, source_audit=source_audit, sealed_audit=sealed_audit)


def not_available_result(error: RecommendationExecutionError | str) -> RecommendationResult:
    reason = error.reason_code if isinstance(error, RecommendationExecutionError) else str(error)
    diagnostics = {
        "stage": error.stage, "exception_type": error.exception_type, "message": error.exception_message,
    } if isinstance(error, RecommendationExecutionError) else {"stage": "unknown", "exception_type": "RecommendationContractError", "message": sanitize_exception_message(str(error))}
    empty_coverage = pd.DataFrame(columns=COVERAGE_COLUMNS)
    empty_rankings = pd.DataFrame(columns=RANKINGS_COLUMNS)
    empty_explanations = pd.DataFrame(columns=EXPLANATIONS_COLUMNS)
    summary = {
        "analysis_name": ANALYSIS_NAME, "analysis_version": ANALYSIS_VERSION,
        "recommendation_status": "not_available", "reason_codes": [reason], "failure": diagnostics,
        "chronological_seal": {
            "test_status": "sealed",
            "test_target_rows_constructed": 0,
            "test_target_matches_constructed": 0,
            "test_feature_rows_constructed": 0,
            "test_recommendation_rows": 0,
            "test_points_scored": 0,
            "test_matches_scored": 0,
            "test_rows_evaluated": 0,
            "test_matches_evaluated": 0,
            "test_evaluation_runs": 0,
            "used_for_method_selection": False,
            "test_target_matches_excluded_before_construction": EXPECTED_EXCLUDED_TEST_TARGET_MATCHES,
        },
        "partial_results_published": False,
    }
    summary, _, fingerprint = _finalize_summary(summary, (empty_coverage, empty_rankings, empty_explanations))
    return RecommendationResult(summary, empty_coverage, empty_rankings, empty_explanations, pd.DataFrame(columns=RECOMMENDATION_COLUMNS), pd.DataFrame(columns=DETAIL_COLUMNS), pd.DataFrame(), fingerprint)


def serialize_artifacts(result: RecommendationResult) -> tuple[bytes, bytes, bytes, bytes]:
    if result.summary.get("recommendation_status") == "not_available":
        frames = (result.coverage, result.rankings, result.explanations)
    else:
        validate_result(result)
        frames = (result.coverage, result.rankings, result.explanations)
    summary = (json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    payloads = tuple(_frame_bytes(frame) for frame in frames)
    if json.loads(summary.decode("utf-8")) != result.summary:
        raise RecommendationContractError("Serializacion JSON no fiel.")
    return (summary, *payloads)


def _stage_payload(path: Path, payload: bytes) -> Path:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    staged = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def write_artifacts(
    result: RecommendationResult,
    summary_path: Path = SUMMARY_PATH,
    coverage_path: Path = COVERAGE_PATH,
    rankings_path: Path = RANKINGS_PATH,
    explanations_path: Path = EXPLANATIONS_PATH,
) -> None:
    paths = (summary_path, coverage_path, rankings_path, explanations_path)
    if len({path.resolve() for path in paths}) != 4:
        raise RecommendationContractError("Las rutas de salida deben ser distintas.")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    payloads = serialize_artifacts(result)
    originals = {path: path.read_bytes() if path.exists() else None for path in paths}
    staged: list[Path] = []
    replaced: list[Path] = []
    try:
        staged = [_stage_payload(path, payload) for path, payload in zip(paths, payloads)]
        for path, staged_path in zip(paths, staged):
            os.replace(staged_path, path)
            replaced.append(path)
    except BaseException:
        for path in reversed(replaced):
            previous = originals[path]
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                rollback = _stage_payload(path, previous)
                os.replace(rollback, path)
        raise
    finally:
        for staged_path in staged:
            staged_path.unlink(missing_ok=True)


def _require_persisted(condition: bool, message: str) -> None:
    if not condition:
        raise RecommendationContractError(f"Contrato persistido invalido: {message}")


def _require_persisted_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    _require_persisted(frame.columns.tolist() == list(columns), f"schema {label}")
    _require_persisted(not any(str(column).startswith("Unnamed") for column in frame.columns), f"indice accidental {label}")


def _assert_persisted_canonical(frame: pd.DataFrame, sort_columns: Sequence[str], label: str) -> None:
    try:
        pdt.assert_frame_equal(frame.reset_index(drop=True), frame.sort_values(list(sort_columns), kind="stable").reset_index(drop=True), check_dtype=False, check_like=False)
    except AssertionError as error:
        raise RecommendationContractError(f"Contrato persistido invalido: orden canonico {label}") from error


def _coverage_ordered(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["_fold"] = pd.Categorical(work["fold"].astype(str), [*map(str, FOLDS), "pooled"], ordered=True)
    work["_surface"] = pd.Categorical(work["surface"], ["<ALL>", *SURFACES], ordered=True)
    work["_scope"] = work["scope_state"].map({value: index for index, value in enumerate(SCOPE_STATES)})
    _require_persisted(work[["_fold", "_surface", "_scope"]].notna().all().all(), "dominio coverage")
    return work.sort_values(["_fold", "_surface", "_scope"], kind="stable").drop(columns=["_fold", "_surface", "_scope"]).reset_index(drop=True)


def _rankings_ordered(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["_fold"] = pd.Categorical(work["fold"].astype(str), [*map(str, FOLDS), "pooled"], ordered=True)
    work["_surface"] = pd.Categorical(work["surface"], ["<ALL>", *SURFACES], ordered=True)
    work["_scope"] = pd.Categorical(work["comparison_scope"], SCOPES, ordered=True)
    work["_direction"] = pd.Categorical(work["direction"], DIRECTIONS, ordered=True)
    _require_persisted(work[["_fold", "_surface", "_scope", "_direction"]].notna().all().all(), "dominio rankings")
    return work.sort_values(["_fold", "_surface", "rank", "_scope", "_direction"], kind="stable").drop(columns=["_fold", "_surface", "_scope", "_direction"]).reset_index(drop=True)


def _explanations_ordered(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["_fold"] = pd.Categorical(work["fold"].astype(str), [*map(str, FOLDS), "pooled"], ordered=True)
    work["_surface"] = pd.Categorical(work["surface"], ["<ALL>", *SURFACES], ordered=True)
    work["_scope"] = pd.Categorical(work["comparison_scope"], SCOPES, ordered=True)
    work["_direction"] = pd.Categorical(work["direction"], DIRECTIONS, ordered=True)
    _require_persisted(work[["_fold", "_surface", "_scope", "_direction"]].notna().all().all(), "dominio explanations")
    return work.sort_values(["_fold", "_surface", "rank", "_scope", "_direction", "explanation_code"], kind="stable").drop(columns=["_fold", "_surface", "_scope", "_direction"]).reset_index(drop=True)


def _require_integral_nonnegative_series(series: pd.Series, field: str) -> None:
    values = pd.to_numeric(series, errors="coerce")
    _require_persisted(values.notna().all() and np.isfinite(values).all() and np.equal(values, np.floor(values)).all() and values.ge(0).all(), field)


def _require_unit_series(series: pd.Series, field: str, *, nullable: bool = False) -> None:
    values = pd.to_numeric(series, errors="coerce")
    if nullable:
        values = values.dropna()
    _require_persisted(values.notna().all() and np.isfinite(values).all() and values.between(0.0, 1.0).all(), field)


def validate_persisted_result(
    summary: Mapping[str, Any],
    coverage: pd.DataFrame,
    rankings: pd.DataFrame,
    explanations: pd.DataFrame,
) -> None:
    """Reconcilia lo derivable de los cuatro artefactos, sin datos individuales."""

    _assert_finite_json(summary)
    status = summary.get("recommendation_status")
    _require_persisted(status in {"available", "not_available"}, "recommendation_status")
    _validate_test_seal(summary.get("chronological_seal", {}))
    _require_persisted_columns(coverage, COVERAGE_COLUMNS, "coverage")
    _require_persisted_columns(rankings, RANKINGS_COLUMNS, "rankings")
    _require_persisted_columns(explanations, EXPLANATIONS_COLUMNS, "explanations")
    if status == "not_available":
        _require_persisted(coverage.empty and rankings.empty and explanations.empty, "inferencia parcial not_available")
        _require_persisted(isinstance(summary.get("reason_codes"), list) and bool(summary["reason_codes"]), "reason_codes not_available")
        _require_persisted(summary.get("partial_results_published") is False, "partial_results_published not_available")
        return

    _require_persisted(summary.get("analysis_name") == ANALYSIS_NAME and summary.get("analysis_version") == ANALYSIS_VERSION, "identidad available")
    is_real_population = summary.get("population", {}).get("source_point_rows") == EXPECTED_SOURCE_ROWS
    _require_persisted(len(coverage) == 100 and len(rankings) == 360 and (len(explanations) == 2_042 if is_real_population else not explanations.empty), "cardinalidad de artefactos available")
    _require_persisted(not coverage.duplicated(["fold", "surface", "recommendation_status", "scope_state"]).any(), "clave coverage")
    _require_persisted(not rankings.duplicated(["fold", "surface", "rank", "direction", "comparison_scope"]).any(), "clave rankings")
    _require_persisted(not explanations.duplicated(["fold", "surface", "rank", "direction", "comparison_scope", "explanation_code"]).any(), "clave explanations")
    ordered_coverage = _coverage_ordered(coverage)
    try:
        pdt.assert_frame_equal(coverage.reset_index(drop=True), ordered_coverage, check_dtype=False, check_like=False)
    except AssertionError as error:
        raise RecommendationContractError("Contrato persistido invalido: orden CSV no canonico.") from error

    for column in ("orientations", "matches", "players", "available_orientations", "abstained_orientations"):
        _require_integral_nonnegative_series(coverage[column], f"coverage.{column}")
    _require_unit_series(coverage["coverage_rate"], "coverage.coverage_rate")
    _require_unit_series(coverage["abstention_rate"], "coverage.abstention_rate")
    expected_status_scope = {
        ("available_full", "surface_common"), ("available_full", "global_common"),
        ("abstained_insufficient_evidence", "insufficient_evidence"),
        ("abstained_incomparable_scope", "mixed_incomparable"),
        ("not_available_invalid_input", "invalid_input"),
    }
    expected_keys = {(str(fold), surface, status_name, scope_state) for fold in (*FOLDS, "pooled") for surface in ("<ALL>", *SURFACES) for status_name, scope_state in expected_status_scope}
    observed_keys = {(str(row.fold), row.surface, row.recommendation_status, row.scope_state) for row in coverage.itertuples(index=False)}
    _require_persisted(observed_keys == expected_keys, "dominio coverage")
    reason_by_status = {
        "available_full": None,
        "abstained_insufficient_evidence": "insufficient_direction_evidence",
        "abstained_incomparable_scope": "mixed_direction_scopes",
        "not_available_invalid_input": "invalid_input",
    }
    for _, group in coverage.groupby(["fold", "surface"], sort=False, observed=True):
        total = int(group["orientations"].sum())
        available = int(group.loc[group["recommendation_status"].eq("available_full"), "orientations"].sum())
        _require_persisted(group["available_orientations"].eq(available).all(), "denominador coverage")
        _require_persisted(group["abstained_orientations"].eq(total - available).all(), "denominador abstention")
        _require_persisted(group["matches"].le(group["orientations"]).all() and group["players"].le(group["orientations"]).all(), "matches/players coverage")
        if total == 0:
            _require_persisted(group["coverage_rate"].eq(0).all() and group["abstention_rate"].eq(0).all(), "estrato coverage vacio")
        else:
            _require_persisted(np.isclose(group["coverage_rate"], available / total, rtol=0, atol=RECONCILIATION_TOLERANCE).all(), "coverage_rate")
            _require_persisted(np.isclose(group["abstention_rate"], (total - available) / total, rtol=0, atol=RECONCILIATION_TOLERANCE).all(), "abstention_rate")
        for row in group.itertuples(index=False):
            expected_reason = reason_by_status[row.recommendation_status]
            _require_persisted(pd.isna(row.reason_code) if expected_reason is None else row.reason_code == expected_reason, "reason_code coverage")
    pooled = coverage.loc[coverage["fold"].astype(str).eq("pooled") & coverage["surface"].eq("<ALL>")]
    pooled_counts = {(row.recommendation_status, row.scope_state): int(row.orientations) for row in pooled.itertuples(index=False)}
    expected_counts = {
        "total_orientations": int(pooled["orientations"].sum()),
        "available_full_orientations": pooled_counts[("available_full", "surface_common")] + pooled_counts[("available_full", "global_common")],
        "abstained_insufficient_evidence_orientations": pooled_counts[("abstained_insufficient_evidence", "insufficient_evidence")],
        "abstained_incomparable_scope_orientations": pooled_counts[("abstained_incomparable_scope", "mixed_incomparable")],
        "invalid_input_orientations": pooled_counts[("not_available_invalid_input", "invalid_input")],
        "available_surface_common": pooled_counts[("available_full", "surface_common")],
        "available_global_common": pooled_counts[("available_full", "global_common")],
    }
    _require_persisted(sum(pooled_counts.values()) == expected_counts["total_orientations"], "total orientations coverage")
    if is_real_population:
        _require_persisted((expected_counts["total_orientations"], expected_counts["available_full_orientations"], expected_counts["abstained_insufficient_evidence_orientations"], expected_counts["abstained_incomparable_scope_orientations"], expected_counts["invalid_input_orientations"], expected_counts["available_surface_common"], expected_counts["available_global_common"]) == (3_610, 2_295, 1_068, 247, 0, 1_844, 451), "conteos reales coverage")
    statuses = summary.get("recommendation_statuses")
    _require_persisted(isinstance(statuses, Mapping), "recommendation_statuses")
    for key, value in expected_counts.items():
        _require_persisted(statuses.get(key) == value, f"summary.{key}")
    total = expected_counts["total_orientations"]
    for key, numerator in (("recommendation_coverage_rate", expected_counts["available_full_orientations"]), ("total_abstention_rate", expected_counts["abstained_insufficient_evidence_orientations"] + expected_counts["abstained_incomparable_scope_orientations"]), ("insufficient_evidence_rate", expected_counts["abstained_insufficient_evidence_orientations"]), ("incomparable_scope_rate", expected_counts["abstained_incomparable_scope_orientations"])):
        _require_close(statuses.get(key), numerator / total, f"summary.{key}")
    patterns = statuses.get("mixed_scope_patterns")
    _require_persisted(isinstance(patterns, Mapping) and all(isinstance(key, str) and all(part in {"surface", "global"} for part in key.split(" / ")) and len(key.split(" / ")) == 3 and isinstance(value, int) and value >= 0 for key, value in patterns.items()), "mixed_scope_patterns")
    _require_persisted(sum(patterns.values()) == expected_counts["abstained_incomparable_scope_orientations"], "mixed_scope_patterns")
    folds = {str(fold): int(coverage.loc[coverage["fold"].astype(str).eq(str(fold)) & coverage["surface"].eq("<ALL>"), "orientations"].sum() / 2) for fold in FOLDS}
    validate_population_cardinalities(summary.get("population", {}), folds)
    _require_persisted(summary.get("validation_metrics", {}).get("folds") == folds, "folds summary")

    for column in ("orientations", "observed_points", "observed_server_wins"):
        _require_integral_nonnegative_series(rankings[column], f"rankings.{column}")
    _require_persisted(rankings["rank"].isin((1, 2, 3)).all() and rankings["direction"].isin(DIRECTIONS).all() and rankings["comparison_scope"].isin(SCOPES).all(), "dominio rankings")
    try:
        pdt.assert_frame_equal(rankings.reset_index(drop=True), _rankings_ordered(rankings), check_dtype=False, check_like=False)
    except AssertionError as error:
        raise RecommendationContractError("Contrato persistido invalido: orden CSV no canonico.") from error
    _require_unit_series(rankings["direction_share"], "rankings.direction_share", nullable=True)
    _require_unit_series(rankings["mean_score"], "rankings.mean_score", nullable=True)
    _require_unit_series(rankings["median_score"], "rankings.median_score", nullable=True)
    for _, group in rankings.groupby(["fold", "surface", "rank", "comparison_scope"], sort=False, observed=True):
        _require_persisted(len(group) == 3, "direcciones rankings")
        denominator = int(group["orientations"].sum())
        if denominator == 0:
            _require_persisted(group["direction_share"].isna().all(), "share ranking vacio")
        else:
            _require_persisted(np.isclose(group["direction_share"].sum(), 1.0, rtol=0, atol=RECONCILIATION_TOLERANCE), "shares rankings")
        _require_persisted(group["observed_server_wins"].le(group["observed_points"]).all(), "wins rankings")
        present = group["observed_points"].gt(0)
        _require_persisted(group.loc[~present, "observed_win_rate"].isna().all(), "win_rate vacio")
        _require_persisted(np.isclose(group.loc[present, "observed_win_rate"], group.loc[present, "observed_server_wins"] / group.loc[present, "observed_points"], rtol=0, atol=RECONCILIATION_TOLERANCE).all(), "win_rate rankings")
    _require_persisted(rankings.loc[rankings["rank"].eq(1), ["mean_score_gap_from_rank_1", "median_score_gap_from_rank_1"]].fillna(0).eq(0).all().all(), "gap rank 1")
    ranking_gaps = rankings[["mean_score_gap_from_rank_1", "median_score_gap_from_rank_1"]].stack().dropna()
    _require_persisted(ranking_gaps.ge(0).all(), "gaps rankings")
    for (fold, surface, scope), group in rankings.groupby(["fold", "surface", "comparison_scope"], sort=False, observed=True):
        _require_persisted(
            len(group) == 9
            and all(len(rank_group) == 3 and set(rank_group["direction"]) == set(DIRECTIONS) for _, rank_group in group.groupby("rank", sort=False, observed=True))
            and all(set(direction_group["rank"]) == {1, 2, 3} for _, direction_group in group.groupby("direction", sort=False, observed=True)),
            "ranking completo por grupo",
        )
        expected = pooled_counts.get(("available_full", f"{scope}_common")) if str(fold) == "pooled" and surface == "<ALL>" else int(coverage.loc[coverage["fold"].astype(str).eq(str(fold)) & coverage["surface"].eq(surface) & coverage["recommendation_status"].eq("available_full") & coverage["scope_state"].eq(f"{scope}_common"), "orientations"].iloc[0])
        _require_persisted(group.groupby("rank", sort=False)["orientations"].sum().eq(expected).all(), "denominador rankings")

    _require_persisted(explanations["rank"].isin((1, 2, 3)).all() and explanations["direction"].isin(DIRECTIONS).all() and explanations["comparison_scope"].isin(SCOPES).all() and explanations["explanation_code"].isin(EXPLANATION_CODES).all(), "dominio explanations")
    try:
        pdt.assert_frame_equal(explanations.reset_index(drop=True), _explanations_ordered(explanations), check_dtype=False, check_like=False)
    except AssertionError as error:
        raise RecommendationContractError("Contrato persistido invalido: orden CSV no canonico.") from error
    for column in ("orientations",):
        _require_integral_nonnegative_series(explanations[column], f"explanations.{column}")
    for column in ("share_within_rank_direction", "mean_server_component", "mean_opponent_component"):
        _require_unit_series(explanations[column], f"explanations.{column}")
    _require_persisted(pd.to_numeric(explanations["mean_absolute_component_difference"], errors="coerce").notna().all() and pd.to_numeric(explanations["mean_absolute_component_difference"], errors="coerce").ge(0).all(), "component difference explanations")
    _require_persisted(explanations.loc[explanations["explanation_code"].eq("surface_history_used"), "comparison_scope"].eq("surface").all() and explanations.loc[explanations["explanation_code"].eq("global_fallback_used"), "comparison_scope"].eq("global").all(), "scope explanations")
    _require_persisted(not explanations["explanation_code"].isin({"recommendation_abstained", "incomplete_direction_set", "direction_not_scoreable", "mixed_direction_scopes"}).any(), "abstention explanations")
    ranking_lookup = rankings.set_index(["fold", "surface", "rank", "direction", "comparison_scope"])["orientations"]
    for row in explanations.itertuples(index=False):
        denominator = int(ranking_lookup.loc[(row.fold, row.surface, row.rank, row.direction, row.comparison_scope)])
        _require_persisted(denominator > 0 and int(row.orientations) <= denominator, "denominador explanations")
        _require_close(row.share_within_rank_direction, int(row.orientations) / denominator, "share explanations")
        if row.explanation_code == "recommendation_available":
            _require_persisted(int(row.orientations) == denominator, "recommendation_available explanations")


def verify_persisted_artifacts(
    summary_path: Path = SUMMARY_PATH,
    coverage_path: Path = COVERAGE_PATH,
    rankings_path: Path = RANKINGS_PATH,
    explanations_path: Path = EXPLANATIONS_PATH,
) -> None:
    paths = (coverage_path, rankings_path, explanations_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payloads = tuple(path.read_bytes() for path in paths)
    _fingerprint(summary, payloads)
    if summary.get("publication_fingerprint") != _fingerprint(summary, payloads):
        raise RecommendationContractError("Fingerprint persistido alterado.")
    coverage, rankings, explanations = (pd.read_csv(io.BytesIO(payload)) for payload in payloads)
    validate_persisted_result(summary, coverage, rankings, explanations)


def _artifact_metadata(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": len(payload), "sha256": _sha256(payload)}


def remove_reproduction_directory(path: Path = REPRO_DIR) -> None:
    if path.resolve() != REPRO_DIR.resolve():
        raise RecommendationContractError("Solo se permite borrar el temporal contractual.")
    if path.exists():
        shutil.rmtree(path)


def main() -> None:
    if REPRO_DIR.exists():
        raise FileExistsError(f"Ya existe el temporal: {REPRO_DIR}")
    started = time.perf_counter()
    try:
        points = pd.read_parquet(POINTS_FILE, columns=SOURCE_COLUMNS)
        result = build_real_analysis(points)
    except BaseException as error:
        stage = "real_analysis"
        result = not_available_result(RecommendationExecutionError(stage, error))
    write_artifacts(result)
    permanent = (SUMMARY_PATH, COVERAGE_PATH, RANKINGS_PATH, EXPLANATIONS_PATH)
    payloads = serialize_artifacts(result)
    if tuple(path.read_bytes() for path in permanent) != payloads:
        raise RecommendationContractError("Los permanentes no coinciden con la serializacion en memoria.")
    verify_persisted_artifacts()
    REPRO_DIR.mkdir(parents=False)
    reproduction = tuple(REPRO_DIR / path.name for path in permanent)
    try:
        write_artifacts(result, *reproduction)
        identical = [left.read_bytes() == right.read_bytes() for left, right in zip(permanent, reproduction)]
        if not all(identical):
            raise RecommendationContractError("Reproduccion temporal no identica.")
    finally:
        remove_reproduction_directory()
    if REPRO_DIR.exists():
        raise RecommendationContractError("No se elimino el temporal.")
    print(json.dumps({"recommendation_status": result.summary["recommendation_status"], "duration_seconds": time.perf_counter() - started, "artifacts": [_artifact_metadata(path) for path in permanent], "byte_identical_reproduction": identical, "reproduction_directory_removed": True}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
