"""Infraestructura leakage-safe del pipeline tactico comun.

La ruta productiva realiza una sola lectura del Parquet y queda gobernada por un
unico interruptor explicito. Las transformaciones aceptan objetos en memoria;
la publicacion agregada usa staging, validacion y rollback.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from enum import Enum
from hashlib import sha256
from io import BytesIO
import json
from math import isfinite
from numbers import Integral, Real
import os
from pathlib import Path
import re
import subprocess
import tempfile
from time import perf_counter
from types import MappingProxyType
from typing import Callable, Final

import pandas as pd

from src.recommender.tactical_feature_encoder import (
    TacticalEncodingPolicy,
    TacticalFeatureSchema,
    TacticalFeatureVector,
    build_tactical_feature_schema,
    encode_tactical_attempt,
    feature_schema_fingerprint,
    validate_tactical_feature_schema,
    validate_tactical_feature_vector,
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
    CATEGORY_RECONCILIATIONS,
    EVIDENCE_PROVENANCE,
    MATCHUP_EVIDENCE_CONTRACT_VERSION,
    PAIR_RECONCILIATIONS,
    RESULT_RECONCILIATIONS as MATCHUP_RESULT_RECONCILIATIONS,
    TacticalCategoryEvidence,
    TacticalCategoryEvidenceState,
    TacticalEvidencePerspective,
    TacticalEvidenceScope,
    TacticalEvidenceScopeStrategy,
    TacticalMatchupCategoryEvidence,
    TacticalMatchupCategoryState,
    TacticalMatchupEvidence,
    TacticalMatchupEvidenceState,
    TacticalMatchupQuery,
    build_tactical_matchup_evidence,
    tactical_matchup_query_fingerprint,
    tactical_wilson_interval,
    validate_tactical_matchup_evidence,
)
from src.recommender.tactical_prioritization import (
    TacticalPrioritizationResult,
    TacticalPrioritizationState,
    prioritize_tactical_matchup,
    tactical_prioritization_result_fingerprint,
    validate_tactical_prioritization_result,
)
from src.recommender.tactical_signal_orchestrator import (
    ORCHESTRATOR_CONTRACT_VERSION,
    AttemptSignalExtraction,
    AttemptSignalRequest,
    extract_tactical_signals_for_attempt,
    validate_attempt_signal_extraction,
)


PIPELINE_CONTRACT_VERSION: Final = "1.0.0"
ANALYSIS_NAME: Final = "tactical_recommender_pipeline"
ANALYSIS_VERSION: Final = "1.0.0"
ANALYSIS_PURPOSE: Final = "sealed_descriptive_pipeline_feasibility"
DESCRIPTIVE_POLICY_NAME: Final = "transferred_conservative_baseline"
FINGERPRINT_CONTRACT_VERSION: Final = "1"
# La segunda ejecucion real finalizo; ninguna ejecucion adicional esta autorizada.
REAL_EXECUTION_AUTHORIZED: Final = False
REAL_EXECUTION_ATTEMPTS: Final = 2
FIRST_EXECUTION_STATUS: Final = "interrupted_for_performance_diagnosis"
SECOND_EXECUTION_STATUS: Final = "completed"
HISTORICAL_EXECUTION_STATUS: Final = "second_execution_completed"
NO_ARTIFACTS_PUBLISHED: Final = False
AUTOMATIC_RETRY: Final = False
FURTHER_REAL_EXECUTION_AUTHORIZED: Final = False
REAL_EXECUTION_BLOCK_REASON: Final = (
    "real_execution_completed_no_further_execution_authorized"
)
REAL_EXECUTION_AUTHORIZATION_REASON: Final = (
    "real_execution_completed_no_further_execution_authorized"
)
PERFORMANCE_PROGRESS_TARGET_INTERVAL: Final = 50
PERFORMANCE_STAGE_ORDER: Final = (
    "source_validation",
    "temporal_sealing",
    "attempt_extraction_and_encoding",
    "observation_indexing",
    "target_construction",
    "evidence_and_prioritization",
    "aggregation_and_validation",
    "publication",
)
_PERFORMANCE_STAGE_FROM_INTERNAL: Final = (
    ("source_validation", "source_validation"),
    ("temporal_seal", "temporal_sealing"),
    ("attempt_construction", "attempt_extraction_and_encoding"),
    ("extraction_and_encoding", "attempt_extraction_and_encoding"),
    ("observation_construction", "observation_indexing"),
    ("target_evidence", "evidence_and_prioritization"),
    ("prioritization", "evidence_and_prioritization"),
    ("final_validation", "aggregation_and_validation"),
    ("serialization", "publication"),
    ("publication", "publication"),
)
DEVELOPMENT_CUTOFF: Final = pd.Timestamp("2023-12-31")
TEST_START: Final = pd.Timestamp("2024-01-01")
VALIDATION_START: Final = pd.Timestamp("2020-01-01")
VALIDATION_FOLDS: Final = (
    "validation_2020",
    "validation_2021",
    "validation_2022",
    "validation_2023",
)
EXPECTED_VALIDATION_MATCHES: Final = 1_805
EXPECTED_TARGET_ORIENTATIONS: Final = 3_610
EXPECTED_SOURCE_ROWS: Final = 1_280_408
EXPECTED_SOURCE_MATCHES: Final = 7_524
EXPECTED_FIRST_DATE: Final = pd.Timestamp("1960-05-29")
EXPECTED_LAST_DATE: Final = pd.Timestamp("2026-05-21")
EXPECTED_FOLD_MATCHES: Final = (
    ("validation_2020", 168),
    ("validation_2021", 371),
    ("validation_2022", 646),
    ("validation_2023", 620),
)
SOURCE_COLUMNS: Final = (
    "match_id",
    "point_number",
    "date",
    "surface",
    "server",
    "point_winner",
    "player_1",
    "player_2",
    "first_serve",
    "second_serve",
)
ALLOWED_SURFACES: Final = ("Hard", "Clay", "Grass")
PRODUCTION_ENCODER_POLICY: Final = TacticalEncodingPolicy.COMPONENT_ONLY
PRODUCTION_PATTERNS: Final = ("P02", "P04", "P05", "P06")
PATTERN_ORDER: Final = ("P02", "P04", "P05", "P06", "P09")
STAGE_ORDER: Final = (
    "source_validation",
    "temporal_seal",
    "attempt_construction",
    "extraction_and_encoding",
    "observation_construction",
    "target_evidence",
    "prioritization",
    "final_validation",
)
FAILURE_STAGE_ORDER: Final = STAGE_ORDER + ("serialization", "publication")
TEST_ZERO_FIELDS: Final = (
    "test_rows_parsed",
    "test_rows_used",
    "test_attempts_constructed",
    "test_attempts_used",
    "test_classifications_computed",
    "test_features_computed",
    "test_profiles_computed",
    "test_labels_used",
    "test_evidence_computed",
    "test_outcomes_computed",
    "test_scores_computed",
    "test_rankings_computed",
    "test_recommendations_generated",
)
PIPELINE_PROVENANCE: Final = (
    ("source_contract", "points_enriched_required_columns"),
    ("temporal_policy", "development_through_2023_12_31"),
    ("orchestration", "common_tactical_signal_orchestrator"),
    ("evidence_scope", "global_only"),
)
PIPELINE_LIMITATIONS: Final = (
    "historical_descriptive_association_not_causal",
    "test_period_sealed_and_not_evaluated",
    "rankings_remain_separate_by_pattern",
    "no_global_tactical_recommendation",
    "real_execution_reauthorization_required_after_performance_fix",
    "thresholds_50_5_are_transferred_not_validated_for_all_return_patterns",
    "portable_multi_file_transaction_cannot_survive_abrupt_power_loss",
)
PIPELINE_RECONCILIATIONS: Final = (
    ("source_population_exhaustive", True),
    ("test_seal_reconciled", True),
    ("attempt_keys_unique", True),
    ("one_observation_per_attempt", True),
    ("cache_reconciled", True),
    ("strict_target_history_reconciled", True),
    ("target_counts_reconciled", True),
    ("no_cross_pattern_ranking", True),
)
RESULT_FINGERPRINT_DOMAIN: Final = b"tennis-tactical-recommender-pipeline\x00"
ARTIFACT_FINGERPRINT_DOMAIN: Final = b"tennis-tactical-pipeline-artifacts\x00"

ROOT: Final = Path(__file__).resolve().parents[2]
POINTS_PATH: Final = ROOT / "data" / "processed" / "points_enriched.parquet"
SUMMARY_PATH: Final = ROOT / "reports" / "tactical_recommender_pipeline_summary.json"
POPULATION_PATH: Final = ROOT / "reports" / "tables" / "tactical_recommender_pipeline_population.csv"
AVAILABILITY_PATH: Final = ROOT / "reports" / "tables" / "tactical_recommender_pipeline_availability.csv"
RANKINGS_PATH: Final = ROOT / "reports" / "tables" / "tactical_recommender_pipeline_rankings.csv"
DIAGNOSTICS_PATH: Final = ROOT / "reports" / "tables" / "tactical_recommender_pipeline_diagnostics.csv"

PUBLISHED_ARTIFACT_CONTRACT: Final = (
    ("summary", 5_770, "CB4886B2ADC6DF2AF179CF4893607523D640C93F81234A4868B4CDFDFBEF307A"),
    ("population", 819, "E5B047936B1BF32AFB96ED0472428DB7AEAC078D0D038DDA5FA3617E750F66CA"),
    ("availability", 2_181, "1C8F0CBE5228FF114FE90446914E3085B1BDABD2A040B13A907659A26089D179"),
    ("rankings", 2_447, "8D54CC1E7FDAC3524261ADB2EBC0674FE6A2205B9218DF966B08356813E00EE9"),
    ("diagnostics", 1_829, "8F4A1C0BDFC58B5DDEDA0107451E933598CB6E4FFD8495BCC9FA777710C064D5"),
)
PUBLISHED_PUBLICATION_FINGERPRINT: Final = (
    "0B18FEDE408C2FAADC2F089C9BCCDB345E47737800AC78D760AF5BF6CC3CE7D5"
)

ARTIFACT_ORDER: Final = (
    "summary",
    "population",
    "availability",
    "rankings",
    "diagnostics",
)
POPULATION_COLUMNS: Final = ("metric", "count", "denominator", "ratio", "status")
AVAILABILITY_COLUMNS: Final = (
    "aggregation_level", "fold", "pattern_id", "target_orientations",
    "available", "partially_available", "not_available", "coverage",
    "scored_candidates", "abstained_candidates", "status",
)
RANKINGS_COLUMNS: Final = (
    "pattern_id", "category", "target_appearances", "eligible_appearances",
    "scored_appearances", "top_k_appearances", "rank_1", "rank_2", "rank_3",
    "tie_appearances", "mean_score", "minimum_score", "maximum_score",
    "executor_labeled_attempts", "opponent_labeled_attempts", "status",
)
DIAGNOSTICS_COLUMNS: Final = (
    "check", "observed", "expected", "passed", "severity", "details",
)
POPULATION_ROW_ORDER: Final = (
    "source", "development", "excluded_test", "attempts_total",
    "first_serve_attempts", "second_serve_attempts",
    "second_with_documented_fault", "second_without_documented_fault",
    "cache_entries", "cache_hits", "cache_misses", "observations",
    "validation_matches", "target_orientations",
)
AVAILABILITY_LEVEL_ORDER: Final = ("total", "fold", "pattern", "fold_pattern")
DIAGNOSTIC_CHECK_ORDER: Final = (
    "source_population_reconciled",
    "test_sealed_before_transformations",
    "history_on_or_after_target_date",
    "target_match_rows_used",
    "same_day_rows_used",
    "future_rows_used",
    "test_rows_used",
    "test_attempts_used",
    "test_labels_used",
    "test_profiles_computed",
    "test_evidence_computed",
    "test_scores_computed",
    "test_rankings_computed",
    "test_recommendations_generated",
    "attempt_keys_unique",
    "cache_reconciled",
    "actors_reconciled",
    "labels_reconciled",
    "global_only_scope",
    "component_only_no_p09_scoring",
    "rankings_separated_by_pattern",
    "target_order_canonical",
    "subquadratic_history_index",
    "all_contract_reconciliations",
)
UPSTREAM_PROVENANCE: Final = (
    ("parser", "0.2.0", "935723eb373cfac15b791eca3f7c9a3faf942555", ""),
    ("P02", "1.0.0", "eb1d85bb8b9214f961f3ce1eb225c9edd70ab2e2", "A5820C693AE572654185257350D207019E2630BC3275F5A938F370E11F4A111A"),
    ("P04", "1.0.0", "aa928d09962a0f4481ee7967555f0beb9dc5a102", "8C1B1C524F3E24DEC9086AAFA36784BDBAD4BAA71447735F772906769906BC1A"),
    ("P05", "1.0.0", "33ee9460daf294b6fc71922556d73e1717277e40", "96333493F586D2F65183A12F6443ACF7F1DC2D2999B96C58AC6C07B27DD1C06B"),
    ("P06", "1.0.0", "41b0e5c8a3d30a69a6f08668e8b282654b1dda48", "C3F30E44949237F9CABDDD18FE161B242F088504F5E610ED480053015F38FDD7"),
    ("P09", "1.0.0", "f940827cd9b363440a0ee71427b83db479ba0c69", "A84E59A23EF99B5A855CBBC4702EB7EB0C49285BF2BA3A5090E75AC3D356D5E3"),
    ("common_signal_contract", "1.0.0", "eb1d85bb8b9214f961f3ce1eb225c9edd70ab2e2", ""),
    ("orchestrator", "1.0.0", "ef725488651e2ec1349c8f5d321250dc8dfdf324", ""),
    ("encoder", "1.0.0", "490a3518f6016dc167911f54120ecdcfa2df9505", ""),
    ("history_profiles", "1.0.0", "508a43c46960e6c322682e27f9f1f57c8dc8f894", ""),
    ("matchup_evidence", "1.0.0", "2d45bed6243a1807a3e16779f86d47184d33735d", ""),
    ("prioritization", "1.0.0", "44b896b1da9cf79a91f4fa2af376cbc7104f666a", ""),
    ("chronological_protocol", "1.1.0", "3ac28b4d960e5b2e5e54dd66cb6a8a866f953712", "E889A679E08BB16FAF5AB01AA0D63A7623BE0825A0F945626A1D19301C8DFEAD"),
    ("tactical_registry", "1.0.0", "80fe95f9fa1961b386059df60b14e5506313efb2", "7E264D21570ADB7F4F680A0F8E6B2498851BE2772E2829383565AF3BBDBEA475"),
)
UPSTREAM_PATHS: Final = (
    ("parser", "src/parsing/serve_sequence.py"),
    ("P02", "src/analysis/first_serve_direction_classification.py"),
    ("P04", "src/analysis/return_direction_feasibility.py"),
    ("P05", "src/analysis/return_depth_feasibility.py"),
    ("P06", "src/analysis/return_shot_type_feasibility.py"),
    ("P09", "src/analysis/return_profile_feasibility.py"),
    ("common_signal_contract", "src/recommender/tactical_signal_contract.py"),
    ("orchestrator", "src/recommender/tactical_signal_orchestrator.py"),
    ("encoder", "src/recommender/tactical_feature_encoder.py"),
    ("history_profiles", "src/recommender/tactical_history_profiles.py"),
    ("matchup_evidence", "src/recommender/tactical_matchup_evidence.py"),
    ("prioritization", "src/recommender/tactical_prioritization.py"),
    ("chronological_protocol", "src/analysis/chronological_validation.py"),
    ("tactical_registry", "src/analysis/tactical_pattern_registry.py"),
)
UPSTREAM_ARTIFACTS: Final = (
    ("P02", "reports/first_serve_direction_feasibility_summary.json", "A5820C693AE572654185257350D207019E2630BC3275F5A938F370E11F4A111A"),
    ("P04", "reports/return_direction_descriptive_feasibility_summary.json", "8C1B1C524F3E24DEC9086AAFA36784BDBAD4BAA71447735F772906769906BC1A"),
    ("P05", "reports/return_depth_descriptive_feasibility_summary.json", "96333493F586D2F65183A12F6443ACF7F1DC2D2999B96C58AC6C07B27DD1C06B"),
    ("P06", "reports/return_shot_type_descriptive_feasibility_summary.json", "C3F30E44949237F9CABDDD18FE161B242F088504F5E610ED480053015F38FDD7"),
    ("P09", "reports/return_profile_descriptive_feasibility_summary.json", "A84E59A23EF99B5A855CBBC4702EB7EB0C49285BF2BA3A5090E75AC3D356D5E3"),
    ("chronological_protocol", "reports/chronological_validation_summary.json", "E889A679E08BB16FAF5AB01AA0D63A7623BE0825A0F945626A1D19301C8DFEAD"),
    ("tactical_registry", "reports/tactical_pattern_registry_summary.json", "7E264D21570ADB7F4F680A0F8E6B2498851BE2772E2829383565AF3BBDBEA475"),
)

_SAFE_TEXT = re.compile(r"^[^\x00-\x1f\x7f]+$")
_OPAQUE_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_SERVICE_FAULT_REASON: Final = "censored_service_fault"
_TACTICAL_FEATURE_PREFIXES: Final = (
    ("P02", "p02.first_serve_direction."),
    ("P04", "p04.return_direction."),
    ("P05", "p05.return_depth."),
    ("P06", "p06.return_shot_type."),
    ("P09", "p09.return_profile."),
)
_P02_VALUE_TO_CODE: Final = (("wide", "4"), ("body", "5"), ("T", "6"))
PUBLISHED_RANKING_KEY_ORDER: Final = (
    ("P02", "4"), ("P02", "5"), ("P02", "6"),
    ("P04", "1"), ("P04", "2"), ("P04", "3"),
    ("P05", "7"), ("P05", "8"), ("P05", "9"),
    ("P06", "f"), ("P06", "b"), ("P06", "r"), ("P06", "s"),
    ("P06", "v"), ("P06", "z"), ("P06", "o"), ("P06", "p"),
    ("P06", "u"), ("P06", "y"), ("P06", "l"), ("P06", "m"),
    ("P06", "h"), ("P06", "i"), ("P06", "j"), ("P06", "k"),
    ("P06", "t"),
)
OPERATION_COUNTER_FIELDS: Final = (
    "targets_processed",
    "stream_queries",
    "actor_relevant_observations_inspected",
    "query_constructions",
    "evidence_constructions",
    "prioritizations",
    "expensive_reconstructive_validations",
    "artifact_serializations_or_fingerprints_in_target_loops",
    "category_iterations",
    "history_materializations",
    "history_index_observations",
)


class TacticalPipelineContractError(ValueError):
    """Fallo contractual sanitizado sin datos individuales."""


class TacticalPipelineExecutionError(RuntimeError):
    """Fallo de una etapa con diagnostico seguro y sin datos individuales."""

    __slots__ = ("stage", "error_type", "sanitized_message")

    def __init__(self, stage: str, error_type: str) -> None:
        if stage not in FAILURE_STAGE_ORDER or type(error_type) is not str or not error_type:
            raise TacticalPipelineContractError("Diagnostico de ejecucion invalido.")
        self.stage = stage
        self.error_type = error_type
        self.sanitized_message = f"{stage}_failed:{error_type}"
        super().__init__(self.sanitized_message)


class TacticalPipelineState(str, Enum):
    AVAILABLE = "available"
    PARTIALLY_AVAILABLE = "partially_available"
    NOT_AVAILABLE = "not_available"


@dataclass(frozen=True)
class TacticalPipelineConfig:
    contract_version: str
    encoder_policy: TacticalEncodingPolicy
    requested_patterns: tuple[str, ...]
    serve_numbers: tuple[int, ...]
    window_days: int | None
    minimum_labeled_attempts: int
    minimum_matches: int
    top_k: int
    development_cutoff: date
    expected_source_rows: int | None
    allow_redundant_audit: bool

    def __post_init__(self) -> None:
        validate_tactical_pipeline_config(self)


@dataclass(frozen=True)
class TacticalPipelineTarget:
    target_match_id: str
    player: str
    opponent: str
    as_of_date: date
    fold: str
    orientation: str

    def __post_init__(self) -> None:
        validate_tactical_pipeline_target(self)


@dataclass(frozen=True)
class TacticalAttemptRecord:
    match_id: str
    point_number: int
    serve_number: int
    effective_date: date
    surface: str
    server_player: str
    returner_player: str
    sequence_text: str
    previous_attempt_was_fault: bool
    documents_service_fault: bool
    server_won_point: bool
    extraction: AttemptSignalExtraction
    feature_vector: TacticalFeatureVector


@dataclass(frozen=True)
class TacticalCacheMetrics:
    attempt_requests: int
    unique_keys: int
    cache_hits: int
    cache_misses: int
    hit_rate: float | None
    orchestrator_calls: int
    encoder_calls: int


@dataclass(frozen=True)
class TacticalAttemptBatch:
    records: tuple[TacticalAttemptRecord, ...]
    cache_metrics: TacticalCacheMetrics
    first_attempts: int
    second_attempts: int
    second_with_documented_first_fault: int
    second_without_documented_first_fault: int
    attempt_construction_seconds: float
    extraction_and_encoding_seconds: float


@dataclass(frozen=True)
class TacticalPipelinePopulation:
    source_point_rows: int
    source_matches: int
    development_point_rows: int
    development_matches: int
    excluded_test_point_rows: int
    excluded_test_matches: int
    attempt_count: int
    first_attempt_count: int
    second_attempt_count: int
    second_with_documented_first_fault: int
    second_without_documented_first_fault: int
    observation_count: int


@dataclass(frozen=True)
class TacticalTestSeal:
    test_status: str
    used_for_method_selection: bool
    excluded_test_point_rows: int
    excluded_test_matches: int
    counters: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class TacticalTargetResult:
    target: TacticalPipelineTarget
    history_observation_count: int
    evidence: TacticalMatchupEvidence
    prioritization: TacticalPrioritizationResult
    state: TacticalPrioritizationState


@dataclass(frozen=True)
class TacticalPipelineDiagnostics:
    stage_seconds: tuple[tuple[str, float], ...]
    temporal_index_strategy: str
    target_history_slices: int
    full_observation_rebuilds: int
    history_index_entries: int
    history_stream_lookups: int
    candidate_observations_examined: int
    full_history_scans_per_target: int


@dataclass(frozen=True)
class TacticalLeakageAudit:
    history_on_or_after_target_date: int
    target_match_rows_used: int
    same_day_rows_used: int
    future_rows_used: int
    test_rows_used: int
    test_attempts_used: int
    test_labels_used: int
    test_profiles_computed: int
    test_evidence_computed: int
    test_scores_computed: int
    test_rankings_computed: int
    test_recommendations_generated: int


@dataclass(frozen=True)
class TacticalPipelineResult:
    contract_version: str
    config: TacticalPipelineConfig
    state: TacticalPipelineState
    reason_codes: tuple[str, ...]
    population: TacticalPipelinePopulation
    test_seal: TacticalTestSeal
    cache_metrics: TacticalCacheMetrics
    schema: TacticalFeatureSchema
    schema_fingerprint: str
    attempts: tuple[TacticalAttemptRecord, ...]
    observations: tuple[TacticalHistoricalObservation, ...]
    prepared_evidence_index: object
    target_results: tuple[TacticalTargetResult, ...]
    targets_requested: int
    targets_processed: int
    targets_available: int
    targets_partial: int
    targets_not_available: int
    total_candidates: int
    scored_candidates: int
    abstained_candidates: int
    diagnostics: TacticalPipelineDiagnostics
    leakage_audit: TacticalLeakageAudit
    reconciliations: tuple[tuple[str, bool], ...]
    provenance: tuple[tuple[str, str], ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class TacticalPipelineArtifactPaths:
    summary: Path
    population: Path
    availability: Path
    rankings: Path
    diagnostics: Path


@dataclass(frozen=True)
class TacticalPipelineUnavailableResult:
    contract_version: str
    analysis_name: str
    analysis_version: str
    state: str
    reason_codes: tuple[str, ...]
    failure_stage: str
    error_type: str
    sanitized_message: str
    population: TacticalPipelinePopulation | None
    test_seal: TacticalTestSeal
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class _PointAttemptSeed:
    match_id: str
    point_number: int
    effective_date: date
    surface: str
    server_player: str
    returner_player: str
    first_serve: str
    second_serve: str | None
    server_won_point: bool


@dataclass(frozen=True)
class _DatedObservationStream:
    dates: tuple[date, ...]
    observations: tuple[TacticalHistoricalObservation, ...]


@dataclass(frozen=True)
class _HistoricalObservationIndex:
    by_server: object
    by_returner: object
    observation_count: int


@dataclass(frozen=True)
class _CumulativeCountSeries:
    dates: tuple[date, ...]
    cumulative: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _PreparedEvidenceIndex:
    all_by_server: object
    all_by_returner: object
    all_by_pair: object
    eligible_by_server: object
    eligible_by_returner: object
    eligible_by_pair: object
    base_counts: object
    category_counts: object
    observation_count: int
    requested_patterns: tuple[str, ...]
    schema_fingerprint: str


def _increment_operation(
    counters: dict[str, int] | None,
    field: str,
    amount: int = 1,
) -> None:
    if counters is None:
        return
    if type(counters) is not dict:
        raise TypeError("operation_counters debe ser dict exacto o None.")
    if field not in OPERATION_COUNTER_FIELDS or type(amount) is not int or amount < 0:
        raise TacticalPipelineContractError("Contador de operaciones invalido.")
    current = counters.get(field, 0)
    if type(current) is not int or current < 0:
        raise TacticalPipelineContractError("Contador de operaciones contaminado.")
    counters[field] = current + amount


def _freeze_count_buckets(
    buckets: dict[tuple[object, ...], dict[date, list[object]]],
) -> MappingProxyType:
    frozen: dict[tuple[object, ...], _CumulativeCountSeries] = {}
    for key, by_date in buckets.items():
        running: list[int] | None = None
        dates: list[date] = []
        cumulative: list[tuple[int, ...]] = []
        for effective_date in sorted(by_date):
            values = by_date[effective_date]
            numeric = [
                len(value) if type(value) is set else int(value)
                for value in values
            ]
            if running is None:
                running = [0] * len(numeric)
            for position, value in enumerate(numeric):
                running[position] += value
            dates.append(effective_date)
            cumulative.append(tuple(running))
        frozen[key] = _CumulativeCountSeries(tuple(dates), tuple(cumulative))
    return MappingProxyType(frozen)


def _add_count_bucket(
    buckets: dict[tuple[object, ...], dict[date, list[object]]],
    key: tuple[object, ...],
    effective_date: date,
    increments: tuple[int, ...],
    *,
    match_id: str | None = None,
) -> None:
    width = len(increments) + (1 if match_id is not None else 0)
    values = buckets.setdefault(key, {}).setdefault(
        effective_date,
        [0] * len(increments) + ([set()] if match_id is not None else []),
    )
    if len(values) != width:
        raise TacticalPipelineContractError("Bucket historico con ancho inconsistente.")
    for position, increment in enumerate(increments):
        values[position] = int(values[position]) + increment
    if match_id is not None:
        match_ids = values[-1]
        if type(match_ids) is not set:
            raise TacticalPipelineContractError("Bucket de partidos inconsistente.")
        match_ids.add(match_id)


def _series_range_counts(
    mapping: object,
    key: tuple[object, ...],
    lower: date | None,
    upper: date,
    width: int,
    counters: dict[str, int] | None,
) -> tuple[int, ...]:
    _increment_operation(counters, "stream_queries")
    series = mapping.get(key)
    if series is None:
        return (0,) * width
    end = bisect_left(series.dates, upper)
    start = 0 if lower is None else bisect_left(series.dates, lower)
    if end == 0 or start == end:
        return (0,) * width
    high = series.cumulative[end - 1]
    if start == 0:
        return high
    low = series.cumulative[start - 1]
    return tuple(right - left for right, left in zip(high, low))


def _observation_actors(
    observation: TacticalHistoricalObservation,
) -> tuple[str, str]:
    if observation.role is TacticalPlayerRole.SERVER:
        return observation.player, observation.opponent
    return observation.opponent, observation.player


def _build_prepared_evidence_index(
    observations: tuple[TacticalHistoricalObservation, ...],
    config: TacticalPipelineConfig,
    schema: TacticalFeatureSchema,
    *,
    operation_counters: dict[str, int] | None = None,
) -> _PreparedEvidenceIndex:
    """Preagrega invariantes una vez; ninguna query recorre observaciones."""
    schema_fingerprint = feature_schema_fingerprint(schema)
    feature_to_pattern = {
        feature_name: pattern_id
        for pattern_id, prefix in _TACTICAL_FEATURE_PREFIXES
        if pattern_id in config.requested_patterns
        for feature_name in schema.tactical_feature_names
        if feature_name.startswith(prefix)
    }
    feature_index = dict(schema.name_to_index)
    all_server: dict[tuple[object, ...], dict[date, list[object]]] = {}
    all_returner: dict[tuple[object, ...], dict[date, list[object]]] = {}
    all_pair: dict[tuple[object, ...], dict[date, list[object]]] = {}
    eligible_server: dict[tuple[object, ...], dict[date, list[object]]] = {}
    eligible_returner: dict[tuple[object, ...], dict[date, list[object]]] = {}
    eligible_pair: dict[tuple[object, ...], dict[date, list[object]]] = {}
    base: dict[tuple[object, ...], dict[date, list[object]]] = {}
    categories: dict[tuple[object, ...], dict[date, list[object]]] = {}

    for observation in observations:
        _increment_operation(operation_counters, "history_index_observations")
        server, returner = _observation_actors(observation)
        serve_number = observation.serve_number
        effective_date = observation.effective_date
        _add_count_bucket(all_server, (server, serve_number), effective_date, (1,))
        _add_count_bucket(all_returner, (returner, serve_number), effective_date, (1,))
        _add_count_bucket(
            all_pair, (server, returner, serve_number), effective_date, (1,)
        )
        applicable = {
            pattern: observation.feature_vector.values[
                feature_index[f"mask.{pattern.lower()}.applicable"]
            ]
            for pattern in config.requested_patterns
        }
        observed = {
            pattern: observation.feature_vector.values[
                feature_index[f"mask.{pattern.lower()}.observed"]
            ]
            for pattern in config.requested_patterns
        }
        is_eligible = any(applicable.values())
        if is_eligible:
            _add_count_bucket(
                eligible_server, (server, serve_number), effective_date, (1,)
            )
            _add_count_bucket(
                eligible_returner, (returner, serve_number), effective_date, (1,)
            )
            _add_count_bucket(
                eligible_pair,
                (server, returner, serve_number),
                effective_date,
                (1,),
            )
        if not is_eligible:
            continue
        active_features = tuple(
            name
            for name in observation.feature_vector.active_feature_names
            if name in feature_to_pattern
        )
        for pattern in config.requested_patterns:
            if pattern == "P02":
                executor_actor, exposed_actor = server, returner
                success = observation.label is TacticalOutcomeLabel.SERVER_WON_POINT
            else:
                executor_actor, exposed_actor = returner, server
                success = observation.label is TacticalOutcomeLabel.RETURNER_WON_POINT
            for perspective, actor in (
                (TacticalEvidencePerspective.EXECUTOR, executor_actor),
                (TacticalEvidencePerspective.OPPONENT_ALLOWED, exposed_actor),
            ):
                for serve_scope in ((serve_number,), (1, 2)):
                    base_key = (pattern, perspective.value, actor, serve_scope)
                    _add_count_bucket(
                        base,
                        base_key,
                        effective_date,
                        (1, int(applicable[pattern]), int(observed[pattern])),
                    )
                    for feature_name in active_features:
                        if feature_to_pattern[feature_name] != pattern:
                            continue
                        category_key = base_key + (feature_name,)
                        labeled = int(
                            observation.label_availability
                            is TacticalLabelAvailability.AVAILABLE
                        )
                        _add_count_bucket(
                            categories,
                            category_key,
                            effective_date,
                            (1, labeled, int(labeled and success)),
                            match_id=observation.match_id,
                        )

    return _PreparedEvidenceIndex(
        _freeze_count_buckets(all_server),
        _freeze_count_buckets(all_returner),
        _freeze_count_buckets(all_pair),
        _freeze_count_buckets(eligible_server),
        _freeze_count_buckets(eligible_returner),
        _freeze_count_buckets(eligible_pair),
        _freeze_count_buckets(base),
        _freeze_count_buckets(categories),
        len(observations),
        config.requested_patterns,
        schema_fingerprint,
    )


def _build_history_index(
    observations: tuple[TacticalHistoricalObservation, ...],
) -> _HistoricalObservationIndex:
    server: dict[str, list[TacticalHistoricalObservation]] = {}
    returner: dict[str, list[TacticalHistoricalObservation]] = {}
    for observation in observations:
        server.setdefault(observation.player, []).append(observation)
        returner.setdefault(observation.opponent, []).append(observation)

    def freeze(
        groups: dict[str, list[TacticalHistoricalObservation]],
    ) -> MappingProxyType:
        return MappingProxyType(
            {
                actor: _DatedObservationStream(
                    tuple(item.effective_date for item in rows), tuple(rows)
                )
                for actor, rows in groups.items()
            }
        )

    return _HistoricalObservationIndex(
        freeze(server), freeze(returner), len(observations)
    )


def _select_target_history(
    index: _HistoricalObservationIndex,
    target: TacticalPipelineTarget,
    config: TacticalPipelineConfig,
) -> tuple[tuple[TacticalHistoricalObservation, ...], int, int]:
    """Selecciona solo streams actor-relevantes mediante indices por fecha."""
    server_index = index.by_server
    returner_index = index.by_returner
    streams = []
    if "P02" in config.requested_patterns:
        streams.extend(
            (server_index.get(target.player), returner_index.get(target.opponent))
        )
    if any(pattern in config.requested_patterns for pattern in ("P04", "P05", "P06", "P09")):
        streams.extend(
            (returner_index.get(target.player), server_index.get(target.opponent))
        )
    selected: dict[tuple[str, int, int], TacticalHistoricalObservation] = {}
    examined = 0
    lookups = len(streams)
    lower_date = (
        None
        if config.window_days is None
        else target.as_of_date - timedelta(days=config.window_days)
    )
    for stream in streams:
        if stream is None:
            continue
        lower = 0 if lower_date is None else bisect_left(stream.dates, lower_date)
        upper = bisect_left(stream.dates, target.as_of_date)
        examined += upper - lower
        for position in range(lower, upper):
            observation = stream.observations[position]
            selected[(observation.match_id, observation.point_number, observation.serve_number)] = observation
    ordered = tuple(
        sorted(
            selected.values(),
            key=lambda item: (
                item.effective_date,
                item.match_id,
                item.point_number,
                item.serve_number,
            ),
        )
    )
    if any(item.effective_date >= target.as_of_date for item in ordered):
        raise TacticalPipelineContractError("Indice historico incluyo presente o futuro.")
    if any(item.match_id == target.target_match_id for item in ordered):
        raise TacticalPipelineContractError("El partido target alcanzo la historia.")
    return ordered, lookups, examined


def _query_actor_sets(
    target: TacticalPipelineTarget,
    requested_patterns: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    has_server_pattern = "P02" in requested_patterns
    has_return_pattern = any(pattern != "P02" for pattern in requested_patterns)
    server_actors = tuple(
        actor
        for actor, included in (
            (target.player, has_server_pattern),
            (target.opponent, has_return_pattern),
        )
        if included
    )
    returner_actors = tuple(
        actor
        for actor, included in (
            (target.opponent, has_server_pattern),
            (target.player, has_return_pattern),
        )
        if included
    )
    return tuple(dict.fromkeys(server_actors)), tuple(dict.fromkeys(returner_actors))


def _indexed_union_count(
    server_mapping: object,
    returner_mapping: object,
    pair_mapping: object,
    target: TacticalPipelineTarget,
    requested_patterns: tuple[str, ...],
    serve_numbers: tuple[int, ...],
    lower: date | None,
    *,
    operation_counters: dict[str, int] | None,
) -> int:
    server_actors, returner_actors = _query_actor_sets(target, requested_patterns)
    total = 0
    for serve_number in serve_numbers:
        total += sum(
            _series_range_counts(
                server_mapping,
                (actor, serve_number),
                lower,
                target.as_of_date,
                1,
                operation_counters,
            )[0]
            for actor in server_actors
        )
        total += sum(
            _series_range_counts(
                returner_mapping,
                (actor, serve_number),
                lower,
                target.as_of_date,
                1,
                operation_counters,
            )[0]
            for actor in returner_actors
        )
        total -= sum(
            _series_range_counts(
                pair_mapping,
                (server_actor, returner_actor, serve_number),
                lower,
                target.as_of_date,
                1,
                operation_counters,
            )[0]
            for server_actor in server_actors
            for returner_actor in returner_actors
        )
    return total


def _indexed_category_state(
    *,
    structurally_applicable: bool,
    relevant: int,
    observed: int,
    active: int,
    labeled: int,
    matches: int,
    config: TacticalPipelineConfig,
) -> tuple[TacticalCategoryEvidenceState, tuple[str, ...]]:
    if not structurally_applicable:
        return TacticalCategoryEvidenceState.NOT_APPLICABLE, (
            "pattern_not_applicable_to_requested_serves",
        )
    if relevant == 0:
        return TacticalCategoryEvidenceState.NOT_AVAILABLE, (
            "no_relevant_historical_attempts",
        )
    if observed == 0 or active == 0:
        return TacticalCategoryEvidenceState.NO_OBSERVED_CATEGORY, (
            "category_not_observed",
        )
    if labeled < config.minimum_labeled_attempts:
        return TacticalCategoryEvidenceState.INSUFFICIENT_LABELED_ATTEMPTS, (
            "minimum_labeled_attempts_not_met",
        )
    if matches < config.minimum_matches:
        return TacticalCategoryEvidenceState.INSUFFICIENT_MATCHES, (
            "minimum_matches_not_met",
        )
    return TacticalCategoryEvidenceState.AVAILABLE, ("evidence_minimums_met",)


def _indexed_category_evidence(
    index: _PreparedEvidenceIndex,
    target: TacticalPipelineTarget,
    query: TacticalMatchupQuery,
    config: TacticalPipelineConfig,
    pattern_id: str,
    feature_name: str,
    category: str,
    perspective: TacticalEvidencePerspective,
    lower: date | None,
    *,
    operation_counters: dict[str, int] | None,
) -> TacticalCategoryEvidence:
    actor = target.player if perspective is TacticalEvidencePerspective.EXECUTOR else target.opponent
    base_key = (pattern_id, perspective.value, actor, config.serve_numbers)
    relevant, applicable, observed = _series_range_counts(
        index.base_counts,
        base_key,
        lower,
        target.as_of_date,
        3,
        operation_counters,
    )
    active, labeled, successes, matches = _series_range_counts(
        index.category_counts,
        base_key + (feature_name,),
        lower,
        target.as_of_date,
        4,
        operation_counters,
    )
    structurally_applicable = pattern_id != "P02" or 1 in config.serve_numbers
    state, reasons = _indexed_category_state(
        structurally_applicable=structurally_applicable,
        relevant=relevant,
        observed=observed,
        active=active,
        labeled=labeled,
        matches=matches,
        config=config,
    )
    rate, wilson_lower, wilson_upper = tactical_wilson_interval(
        successes,
        labeled,
        feature_name=feature_name,
    )
    return TacticalCategoryEvidence(
        MATCHUP_EVIDENCE_CONTRACT_VERSION,
        query,
        tactical_matchup_query_fingerprint(query),
        query.policy,
        query.schema_fingerprint,
        pattern_id,
        feature_name,
        category,
        perspective,
        TacticalEvidenceScope.GLOBAL,
        TacticalPlayerRole.SERVER if pattern_id == "P02" else TacticalPlayerRole.RETURNER,
        query.serve_numbers,
        query.minimum_labeled_attempts,
        query.minimum_matches,
        relevant,
        applicable,
        observed,
        active,
        labeled,
        successes,
        labeled - successes,
        rate,
        wilson_lower,
        wilson_upper,
        matches,
        state,
        reasons,
        EVIDENCE_PROVENANCE,
        CATEGORY_RECONCILIATIONS,
    )


def _indexed_pair_evidence(
    executor: TacticalCategoryEvidence,
    opponent: TacticalCategoryEvidence,
) -> TacticalMatchupCategoryEvidence:
    executor_available = executor.evidence_state is TacticalCategoryEvidenceState.AVAILABLE
    opponent_available = opponent.evidence_state is TacticalCategoryEvidenceState.AVAILABLE
    if executor_available and opponent_available:
        state = TacticalMatchupCategoryState.COMPARABLE
        reasons = ("both_perspectives_available",)
    elif (
        executor.evidence_state is TacticalCategoryEvidenceState.NOT_APPLICABLE
        and opponent.evidence_state is TacticalCategoryEvidenceState.NOT_APPLICABLE
    ):
        state = TacticalMatchupCategoryState.NOT_APPLICABLE
        reasons = ("pattern_not_applicable_to_requested_serves",)
    elif (
        executor.evidence_state is TacticalCategoryEvidenceState.NOT_AVAILABLE
        and opponent.evidence_state is TacticalCategoryEvidenceState.NOT_AVAILABLE
    ):
        state = TacticalMatchupCategoryState.NOT_AVAILABLE
        reasons = ("both_perspectives_not_available",)
    elif executor_available:
        state = TacticalMatchupCategoryState.OPPONENT_INSUFFICIENT
        reasons = ("opponent_allowed_evidence_insufficient",)
    elif opponent_available:
        state = TacticalMatchupCategoryState.EXECUTOR_INSUFFICIENT
        reasons = ("executor_evidence_insufficient",)
    else:
        state = TacticalMatchupCategoryState.BOTH_INSUFFICIENT
        reasons = ("both_perspectives_insufficient",)
    return TacticalMatchupCategoryEvidence(
        MATCHUP_EVIDENCE_CONTRACT_VERSION,
        executor.query_fingerprint,
        executor.policy,
        executor.schema_fingerprint,
        executor.pattern_id,
        executor.feature_name,
        executor.category,
        executor,
        opponent,
        state,
        reasons,
        PAIR_RECONCILIATIONS,
    )


def _indexed_matchup_state(
    categories: tuple[TacticalMatchupCategoryEvidence, ...],
) -> tuple[TacticalMatchupEvidenceState, tuple[str, ...], int, int]:
    candidates = tuple(
        item
        for item in categories
        if item.executor_evidence.category_activations > 0
        or item.opponent_allowed_evidence.category_activations > 0
    )
    comparable = sum(
        item.matchup_state is TacticalMatchupCategoryState.COMPARABLE
        for item in candidates
    )
    if candidates and comparable == len(candidates):
        return (
            TacticalMatchupEvidenceState.AVAILABLE,
            ("all_observed_candidate_categories_comparable",),
            len(candidates),
            comparable,
        )
    if comparable:
        return (
            TacticalMatchupEvidenceState.PARTIALLY_AVAILABLE,
            ("some_observed_candidate_categories_comparable",),
            len(candidates),
            comparable,
        )
    return (
        TacticalMatchupEvidenceState.NOT_AVAILABLE,
        ("no_observed_candidate_category_comparable",),
        len(candidates),
        0,
    )


def _build_indexed_matchup_evidence(
    index: _PreparedEvidenceIndex,
    target: TacticalPipelineTarget,
    config: TacticalPipelineConfig,
    schema: TacticalFeatureSchema,
    *,
    operation_counters: dict[str, int] | None = None,
) -> tuple[TacticalMatchupEvidence, int, int]:
    """Construye el contrato upstream desde prefijos agregados inmutables."""
    _increment_operation(operation_counters, "query_constructions")
    query = TacticalMatchupQuery(
        MATCHUP_EVIDENCE_CONTRACT_VERSION,
        target.player,
        target.opponent,
        target.as_of_date,
        config.encoder_policy,
        index.schema_fingerprint,
        config.serve_numbers,
        config.window_days,
        config.minimum_labeled_attempts,
        config.minimum_matches,
        config.requested_patterns,
        TacticalEvidenceScopeStrategy.GLOBAL_ONLY,
    )
    lower = (
        None
        if config.window_days is None
        else target.as_of_date - timedelta(days=config.window_days)
    )
    total = _indexed_union_count(
        index.all_by_server,
        index.all_by_returner,
        index.all_by_pair,
        target,
        config.requested_patterns,
        (1, 2),
        lower,
        operation_counters=operation_counters,
    )
    after_serve = _indexed_union_count(
        index.all_by_server,
        index.all_by_returner,
        index.all_by_pair,
        target,
        config.requested_patterns,
        config.serve_numbers,
        lower,
        operation_counters=operation_counters,
    )
    eligible = _indexed_union_count(
        index.eligible_by_server,
        index.eligible_by_returner,
        index.eligible_by_pair,
        target,
        config.requested_patterns,
        config.serve_numbers,
        lower,
        operation_counters=operation_counters,
    )
    categories = []
    prefixes = dict(_TACTICAL_FEATURE_PREFIXES)
    for pattern_id in config.requested_patterns:
        prefix = prefixes[pattern_id]
        for feature_name in schema.tactical_feature_names:
            if not feature_name.startswith(prefix):
                continue
            _increment_operation(operation_counters, "category_iterations")
            category = feature_name[len(prefix) :]
            executor = _indexed_category_evidence(
                index,
                target,
                query,
                config,
                pattern_id,
                feature_name,
                category,
                TacticalEvidencePerspective.EXECUTOR,
                lower,
                operation_counters=operation_counters,
            )
            opponent = _indexed_category_evidence(
                index,
                target,
                query,
                config,
                pattern_id,
                feature_name,
                category,
                TacticalEvidencePerspective.OPPONENT_ALLOWED,
                lower,
                operation_counters=operation_counters,
            )
            categories.append(_indexed_pair_evidence(executor, opponent))
    category_tuple = tuple(categories)
    state, reasons, candidate_count, comparable_count = _indexed_matchup_state(
        category_tuple
    )
    state_counts = tuple(
        (
            state_value.value,
            sum(item.matchup_state is state_value for item in category_tuple),
        )
        for state_value in TacticalMatchupCategoryState
    )
    _increment_operation(operation_counters, "evidence_constructions")
    evidence = TacticalMatchupEvidence(
        MATCHUP_EVIDENCE_CONTRACT_VERSION,
        query,
        query.policy,
        query.schema_fingerprint,
        state,
        reasons,
        category_tuple,
        len(category_tuple),
        candidate_count,
        comparable_count,
        candidate_count - comparable_count,
        state_counts,
        total,
        total,
        eligible,
        0,
        0,
        0,
        total - after_serve,
        after_serve - eligible,
        schema.redundant_representation,
        MATCHUP_RESULT_RECONCILIATIONS,
        EVIDENCE_PROVENANCE,
    )
    server_actors, returner_actors = _query_actor_sets(
        target, config.requested_patterns
    )
    examined = sum(
        _series_range_counts(
            index.all_by_server,
            (actor, serve_number),
            lower,
            target.as_of_date,
            1,
            None,
        )[0]
        for actor in server_actors
        for serve_number in (1, 2)
    ) + sum(
        _series_range_counts(
            index.all_by_returner,
            (actor, serve_number),
            lower,
            target.as_of_date,
            1,
            None,
        )[0]
        for actor in returner_actors
        for serve_number in (1, 2)
    )
    return evidence, len(server_actors) + len(returner_actors), examined


def default_pipeline_config() -> TacticalPipelineConfig:
    return TacticalPipelineConfig(
        contract_version=PIPELINE_CONTRACT_VERSION,
        encoder_policy=PRODUCTION_ENCODER_POLICY,
        requested_patterns=PRODUCTION_PATTERNS,
        serve_numbers=(1, 2),
        window_days=None,
        minimum_labeled_attempts=50,
        minimum_matches=5,
        top_k=3,
        development_cutoff=DEVELOPMENT_CUTOFF.date(),
        expected_source_rows=None,
        allow_redundant_audit=False,
    )


def future_real_pipeline_config() -> TacticalPipelineConfig:
    return replace(default_pipeline_config(), expected_source_rows=EXPECTED_SOURCE_ROWS)


def _strict_text(value: object, field: str, *, opaque: bool = False) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or _SAFE_TEXT.fullmatch(value) is None
        or (opaque and _OPAQUE_ID.fullmatch(value) is None)
    ):
        raise TacticalPipelineContractError(f"{field} fuera del contrato textual.")
    return value


def _strict_count(value: object, field: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise TacticalPipelineContractError(f"{field} fuera del dominio entero.")
    return value


def _strict_index(value: object, field: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value not in (1, 2):
        raise TacticalPipelineContractError(f"{field} debe ser entero real 1 o 2.")
    return int(value)


def _strict_point_number(value: object) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise TacticalPipelineContractError("point_number debe ser entero positivo real.")
    return int(value)


def _sequence_presence(value: object, field: str) -> str:
    if value is None or value is pd.NA or value is pd.NaT:
        return "null"
    if type(value) is not str:
        if isinstance(value, float) and value != value:
            return "null"
        raise TacticalPipelineContractError(f"{field} debe ser str real o nulo.")
    if value == "":
        return "empty"
    if value.isspace():
        return "whitespace_only"
    return "substantive"


def _strict_dates(series: pd.Series) -> pd.Series:
    if bool(series.isna().any()):
        raise TacticalPipelineContractError("date contiene nulos.")
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
    elif bool(series.map(lambda value: type(value) is str).all()):
        if not bool(series.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()):
            raise TacticalPipelineContractError("date debe usar YYYY-MM-DD sin ambiguedad.")
        parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    elif bool(
        series.map(
            lambda value: isinstance(value, (pd.Timestamp, datetime, date))
            and not isinstance(value, bool)
        ).all()
    ):
        parsed = pd.to_datetime(series, errors="coerce")
    else:
        raise TacticalPipelineContractError("date contiene tipos ambiguos.")
    if bool(parsed.isna().any()) or getattr(parsed.dt, "tz", None) is not None:
        raise TacticalPipelineContractError("date invalida o con zona horaria.")
    if bool(parsed.ne(parsed.dt.normalize()).any()):
        raise TacticalPipelineContractError("date debe estar normalizada al dia civil.")
    return parsed.dt.normalize()


def _policy_patterns(policy: TacticalEncodingPolicy) -> tuple[str, ...]:
    if policy is TacticalEncodingPolicy.COMPONENT_ONLY:
        return ("P02", "P04", "P05", "P06")
    if policy is TacticalEncodingPolicy.PROFILE_ONLY:
        return ("P02", "P09")
    return PATTERN_ORDER


def validate_tactical_pipeline_config(config: TacticalPipelineConfig) -> None:
    if type(config) is not TacticalPipelineConfig:
        raise TypeError("config debe ser TacticalPipelineConfig exacta.")
    if config.contract_version != PIPELINE_CONTRACT_VERSION:
        raise TacticalPipelineContractError("Version de config invalida.")
    if type(config.encoder_policy) is not TacticalEncodingPolicy:
        raise TypeError("encoder_policy fuera del dominio.")
    allowed = _policy_patterns(config.encoder_policy)
    expected = tuple(pattern for pattern in PATTERN_ORDER if pattern in config.requested_patterns)
    if (
        type(config.requested_patterns) is not tuple
        or not config.requested_patterns
        or config.requested_patterns != expected
        or any(pattern not in allowed for pattern in config.requested_patterns)
    ):
        raise TacticalPipelineContractError("requested_patterns invalido o incompatible.")
    if (
        type(config.serve_numbers) is not tuple
        or any(type(value) is not int for value in config.serve_numbers)
        or config.serve_numbers not in {(1,), (2,), (1, 2)}
    ):
        raise TacticalPipelineContractError("serve_numbers fuera del contrato.")
    if config.window_days is not None:
        _strict_count(config.window_days, "window_days", positive=True)
    _strict_count(config.minimum_labeled_attempts, "minimum_labeled_attempts")
    _strict_count(config.minimum_matches, "minimum_matches")
    _strict_count(config.top_k, "top_k", positive=True)
    if type(config.development_cutoff) is not date or config.development_cutoff != DEVELOPMENT_CUTOFF.date():
        raise TacticalPipelineContractError("development_cutoff debe permanecer congelado.")
    if config.expected_source_rows is not None:
        _strict_count(config.expected_source_rows, "expected_source_rows", positive=True)
    if type(config.allow_redundant_audit) is not bool:
        raise TypeError("allow_redundant_audit debe ser bool real.")
    if (
        config.encoder_policy is TacticalEncodingPolicy.COMPONENTS_AND_PROFILE
        and not config.allow_redundant_audit
    ):
        raise TacticalPipelineContractError(
            "components_and_profile solo se permite como auditoria redundante."
        )
    schema = build_tactical_feature_schema(config.encoder_policy)
    counts = {
        pattern: sum(name.startswith(f"{pattern.lower()}.") for name in schema.tactical_feature_names)
        for pattern in config.requested_patterns
    }
    if any(config.top_k > count for count in counts.values()):
        raise TacticalPipelineContractError("top_k supera el catalogo de un patron.")


def validate_tactical_pipeline_target(target: TacticalPipelineTarget) -> None:
    if type(target) is not TacticalPipelineTarget:
        raise TypeError("target debe ser TacticalPipelineTarget exacto.")
    _strict_text(target.target_match_id, "target.match_id", opaque=True)
    player = _strict_text(target.player, "target.player")
    opponent = _strict_text(target.opponent, "target.opponent")
    if player == opponent:
        raise TacticalPipelineContractError("Target exige jugadores distintos.")
    if type(target.as_of_date) is not date:
        raise TypeError("target.as_of_date debe ser date exacta.")
    if not (VALIDATION_START.date() <= target.as_of_date <= DEVELOPMENT_CUTOFF.date()):
        raise TacticalPipelineContractError("Target fuera de validacion 2020-2023.")
    expected_fold = f"validation_{target.as_of_date.year}"
    if target.fold != expected_fold or target.fold not in VALIDATION_FOLDS:
        raise TacticalPipelineContractError("Fold del target no reconcilia con su fecha.")
    if target.orientation not in ("player_1_vs_player_2", "player_2_vs_player_1"):
        raise TacticalPipelineContractError("Orientacion target invalida.")


def build_validation_targets(
    development: pd.DataFrame,
    *,
    enforce_frozen_cardinalities: bool = False,
) -> tuple[TacticalPipelineTarget, ...]:
    """Reduce partidos de validacion y emite dos orientaciones canonicas."""
    if not isinstance(development, pd.DataFrame) or tuple(development.columns) != SOURCE_COLUMNS:
        raise TacticalPipelineContractError("Development target debe conservar schema canonico.")
    if type(enforce_frozen_cardinalities) is not bool:
        raise TypeError("enforce_frozen_cardinalities debe ser bool real.")
    validation = development.loc[development.date.ge(VALIDATION_START)].copy(deep=False)
    matches = (
        validation.loc[:, ["match_id", "date", "player_1", "player_2"]]
        .drop_duplicates("match_id", keep="first")
        .sort_values(["date", "match_id"], kind="stable")
    )
    targets = []
    for row in matches.itertuples(index=False):
        fold = f"validation_{row.date.year}"
        targets.extend(
            (
                TacticalPipelineTarget(
                    row.match_id,
                    row.player_1,
                    row.player_2,
                    row.date.date(),
                    fold,
                    "player_1_vs_player_2",
                ),
                TacticalPipelineTarget(
                    row.match_id,
                    row.player_2,
                    row.player_1,
                    row.date.date(),
                    fold,
                    "player_2_vs_player_1",
                ),
            )
        )
    result = tuple(targets)
    keys = tuple(
        (item.target_match_id, item.orientation) for item in result
    )
    if len(keys) != len(set(keys)) or len(result) != 2 * len(matches):
        raise TacticalPipelineContractError("Orientaciones target no son exhaustivas y unicas.")
    if enforce_frozen_cardinalities:
        fold_matches = tuple(
            (fold, int(matches.date.dt.year.eq(int(fold[-4:])).sum()))
            for fold in VALIDATION_FOLDS
        )
        if (
            len(matches) != EXPECTED_VALIDATION_MATCHES
            or len(result) != EXPECTED_TARGET_ORIENTATIONS
            or fold_matches != EXPECTED_FOLD_MATCHES
        ):
            raise TacticalPipelineContractError("Cardinalidades target congeladas no reconcilian.")
    return result


def _validate_frozen_target_population(
    targets: tuple[TacticalPipelineTarget, ...],
) -> None:
    by_match: dict[str, list[TacticalPipelineTarget]] = {}
    for target in targets:
        by_match.setdefault(target.target_match_id, []).append(target)
    for rows in by_match.values():
        ordered = sorted(rows, key=lambda item: item.orientation)
        if (
            len(ordered) != 2
            or {item.orientation for item in ordered}
            != {"player_1_vs_player_2", "player_2_vs_player_1"}
            or ordered[0].as_of_date != ordered[1].as_of_date
            or ordered[0].fold != ordered[1].fold
            or (ordered[0].player, ordered[0].opponent)
            != (ordered[1].opponent, ordered[1].player)
        ):
            raise TacticalPipelineContractError("Par de orientaciones target invalido.")
    fold_matches = tuple(
        (
            fold,
            len(
                {
                    item.target_match_id
                    for item in targets
                    if item.fold == fold
                }
            ),
        )
        for fold in VALIDATION_FOLDS
    )
    if (
        len(by_match) != EXPECTED_VALIDATION_MATCHES
        or len(targets) != EXPECTED_TARGET_ORIENTATIONS
        or fold_matches != EXPECTED_FOLD_MATCHES
    ):
        raise TacticalPipelineContractError("Poblacion target real no reconcilia protocolo.")


def validate_source_points(
    points: pd.DataFrame,
    *,
    expected_rows: int | None = None,
) -> pd.DataFrame:
    """Valida y devuelve una copia canonica sin normalizar identidades."""
    if not isinstance(points, pd.DataFrame):
        raise TypeError("points debe ser pandas.DataFrame.")
    if tuple(points.columns) != SOURCE_COLUMNS:
        raise TacticalPipelineContractError("Schema u orden de columnas fuente invalido.")
    if points.empty:
        raise TacticalPipelineContractError("La fuente no puede estar vacia.")
    if expected_rows is not None:
        _strict_count(expected_rows, "expected_rows", positive=True)
        if len(points) != expected_rows:
            raise TacticalPipelineContractError("Cardinalidad fuente inesperada.")
    work = points.copy(deep=True)
    for field in ("match_id", "player_1", "player_2", "surface"):
        invalid = work[field].map(
            lambda value: type(value) is not str
            or not value
            or value != value.strip()
            or _SAFE_TEXT.fullmatch(value) is None
        )
        if bool(invalid.any()):
            raise TacticalPipelineContractError(f"{field} fuera del contrato textual.")
    if not bool(work.match_id.map(lambda value: _OPAQUE_ID.fullmatch(value) is not None).all()):
        raise TacticalPipelineContractError("match_id fuera del dominio opaco.")
    work["point_number"] = work.point_number.map(_strict_point_number)
    if bool(work.duplicated(["match_id", "point_number"]).any()):
        raise TacticalPipelineContractError("Clave de punto duplicada.")
    work["date"] = _strict_dates(work.date)
    if not bool(work.surface.isin(ALLOWED_SURFACES).all()):
        raise TacticalPipelineContractError("surface contiene valores inesperados.")
    if bool(work.player_1.eq(work.player_2).any()):
        raise TacticalPipelineContractError("Los participantes deben ser distintos.")
    work["server"] = work.server.map(lambda value: _strict_index(value, "server"))
    work["point_winner"] = work.point_winner.map(
        lambda value: _strict_index(value, "point_winner")
    )
    if not bool(
        work.first_serve.map(
            lambda value: _sequence_presence(value, "first_serve") == "substantive"
        ).all()
    ):
        raise TacticalPipelineContractError("first_serve debe ser sustantivo.")
    work.second_serve.map(lambda value: _sequence_presence(value, "second_serve"))
    inconsistent = (
        work.groupby("match_id", sort=False)[
            ["date", "surface", "player_1", "player_2"]
        ]
        .nunique(dropna=False)
        .gt(1)
    )
    if bool(inconsistent.any().any()):
        raise TacticalPipelineContractError("Metadatos inconsistentes dentro de partido.")
    if expected_rows == EXPECTED_SOURCE_ROWS and (
        len(work) != EXPECTED_SOURCE_ROWS
        or int(work.match_id.nunique()) != EXPECTED_SOURCE_MATCHES
        or work.date.min() != EXPECTED_FIRST_DATE
        or work.date.max() != EXPECTED_LAST_DATE
    ):
        raise TacticalPipelineContractError("Contrato congelado de fuente real no reconcilia.")
    return work.sort_values(
        ["date", "match_id", "point_number"], kind="stable"
    ).reset_index(drop=True)


def seal_development_points(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, TacticalPipelinePopulation, TacticalTestSeal]:
    """Sella el periodo posterior a 2023 antes de cualquier extraccion."""
    if tuple(source.columns) != SOURCE_COLUMNS:
        raise TacticalPipelineContractError("Source no fue validada con schema canonico.")
    development = source.loc[source.date.le(DEVELOPMENT_CUTOFF)].copy(deep=True)
    sealed = source.loc[source.date.ge(TEST_START)].copy(deep=True)
    if len(development) + len(sealed) != len(source):
        raise TacticalPipelineContractError("Existe una fecha fuera de las particiones.")
    split_counts = source.assign(
        _split=source.date.ge(TEST_START).map({False: "development", True: "test"})
    ).groupby("match_id", sort=False)["_split"].nunique()
    if bool(split_counts.gt(1).any()):
        raise TacticalPipelineContractError("Un partido cruza el corte temporal.")
    population = TacticalPipelinePopulation(
        source_point_rows=int(len(source)),
        source_matches=int(source.match_id.nunique()),
        development_point_rows=int(len(development)),
        development_matches=int(development.match_id.nunique()),
        excluded_test_point_rows=int(len(sealed)),
        excluded_test_matches=int(sealed.match_id.nunique()),
        attempt_count=0,
        first_attempt_count=0,
        second_attempt_count=0,
        second_with_documented_first_fault=0,
        second_without_documented_first_fault=0,
        observation_count=0,
    )
    seal = TacticalTestSeal(
        test_status="sealed",
        used_for_method_selection=False,
        excluded_test_point_rows=int(len(sealed)),
        excluded_test_matches=int(sealed.match_id.nunique()),
        counters=tuple((field, 0) for field in TEST_ZERO_FIELDS),
    )
    return development.reset_index(drop=True), population, seal


def _point_seeds(development: pd.DataFrame) -> tuple[_PointAttemptSeed, ...]:
    seeds = []
    for row in development.itertuples(index=False):
        server_player = row.player_1 if row.server == 1 else row.player_2
        returner_player = row.player_2 if row.server == 1 else row.player_1
        second = (
            row.second_serve
            if _sequence_presence(row.second_serve, "second_serve") == "substantive"
            else None
        )
        seeds.append(
            _PointAttemptSeed(
                match_id=row.match_id,
                point_number=int(row.point_number),
                effective_date=row.date.date(),
                surface=row.surface,
                server_player=server_player,
                returner_player=returner_player,
                first_serve=row.first_serve,
                second_serve=second,
                server_won_point=bool(row.point_winner == row.server),
            )
        )
    return tuple(seeds)


def _documents_service_fault(extraction: AttemptSignalExtraction) -> bool:
    validate_attempt_signal_extraction(extraction)
    if extraction.serve_number != 1:
        return False
    return any(
        _SERVICE_FAULT_REASON in adaptation.reason_codes
        for adaptation in extraction.adaptations
    )


def construct_tactical_attempt_records(
    development: pd.DataFrame,
    schema: TacticalFeatureSchema,
    *,
    orchestrator: Callable[[AttemptSignalRequest], AttemptSignalExtraction] = extract_tactical_signals_for_attempt,
    encoder: Callable[[AttemptSignalExtraction, TacticalFeatureSchema], TacticalFeatureVector] = encode_tactical_attempt,
) -> TacticalAttemptBatch:
    """Construye intentos y reutiliza una cache local completa por contexto."""
    if not isinstance(development, pd.DataFrame) or tuple(development.columns) != SOURCE_COLUMNS:
        raise TacticalPipelineContractError("Development debe conservar el schema validado.")
    if bool(development.date.gt(DEVELOPMENT_CUTOFF).any()):
        raise TacticalPipelineContractError("El test alcanzo el constructor de intentos.")
    validate_tactical_feature_schema(schema)
    started = perf_counter()
    try:
        seeds = _point_seeds(development)
    except TacticalPipelineExecutionError:
        raise
    except Exception as error:
        raise TacticalPipelineExecutionError(
            "attempt_construction", type(error).__name__
        ) from error
    attempt_seconds = perf_counter() - started
    started = perf_counter()
    cache: dict[
        tuple[str, int, bool], tuple[AttemptSignalExtraction, TacticalFeatureVector]
    ] = {}
    requests = hits = misses = orchestrator_calls = encoder_calls = 0

    def resolve(
        sequence_text: str, serve_number: int, previous_fault: bool
    ) -> tuple[AttemptSignalExtraction, TacticalFeatureVector]:
        nonlocal requests, hits, misses, orchestrator_calls, encoder_calls
        requests += 1
        key = (sequence_text, serve_number, previous_fault)
        if key in cache:
            hits += 1
            return cache[key]
        misses += 1
        request = AttemptSignalRequest(
            ORCHESTRATOR_CONTRACT_VERSION,
            sequence_text,
            serve_number,
            previous_fault,
        )
        extraction = orchestrator(request)
        orchestrator_calls += 1
        validate_attempt_signal_extraction(extraction)
        if (
            extraction.serve_number != serve_number
            or extraction.previous_attempt_was_fault != previous_fault
        ):
            raise TacticalPipelineContractError("Extraccion no corresponde a su cache key.")
        vector = encoder(extraction, schema)
        encoder_calls += 1
        validate_tactical_feature_vector(vector)
        if vector.serve_number != serve_number:
            raise TacticalPipelineContractError("Vector no corresponde al intento.")
        cache[key] = (extraction, vector)
        return cache[key]

    records = []
    for seed in seeds:
        first_extraction, first_vector = resolve(seed.first_serve, 1, False)
        first_fault = _documents_service_fault(first_extraction)
        records.append(
            TacticalAttemptRecord(
                seed.match_id,
                seed.point_number,
                1,
                seed.effective_date,
                seed.surface,
                seed.server_player,
                seed.returner_player,
                seed.first_serve,
                False,
                first_fault,
                seed.server_won_point,
                first_extraction,
                first_vector,
            )
        )
        if seed.second_serve is not None:
            second_extraction, second_vector = resolve(
                seed.second_serve, 2, first_fault
            )
            records.append(
                TacticalAttemptRecord(
                    seed.match_id,
                    seed.point_number,
                    2,
                    seed.effective_date,
                    seed.surface,
                    seed.server_player,
                    seed.returner_player,
                    seed.second_serve,
                    first_fault,
                    False,
                    seed.server_won_point,
                    second_extraction,
                    second_vector,
                )
            )
    ordered = tuple(
        sorted(
            records,
            key=lambda item: (
                item.effective_date,
                item.match_id,
                item.point_number,
                item.serve_number,
            ),
        )
    )
    extraction_seconds = perf_counter() - started
    metrics = TacticalCacheMetrics(
        attempt_requests=requests,
        unique_keys=len(cache),
        cache_hits=hits,
        cache_misses=misses,
        hit_rate=None if requests == 0 else hits / requests,
        orchestrator_calls=orchestrator_calls,
        encoder_calls=encoder_calls,
    )
    batch = TacticalAttemptBatch(
        records=ordered,
        cache_metrics=metrics,
        first_attempts=sum(item.serve_number == 1 for item in ordered),
        second_attempts=sum(item.serve_number == 2 for item in ordered),
        second_with_documented_first_fault=sum(
            item.serve_number == 2 and item.previous_attempt_was_fault
            for item in ordered
        ),
        second_without_documented_first_fault=sum(
            item.serve_number == 2 and not item.previous_attempt_was_fault
            for item in ordered
        ),
        attempt_construction_seconds=attempt_seconds,
        extraction_and_encoding_seconds=extraction_seconds,
    )
    validate_tactical_attempt_batch(batch, development_point_rows=len(development), schema=schema)
    return batch


def validate_tactical_attempt_record(
    record: TacticalAttemptRecord,
    schema: TacticalFeatureSchema,
) -> None:
    if type(record) is not TacticalAttemptRecord:
        raise TypeError("record debe ser TacticalAttemptRecord exacto.")
    _strict_text(record.match_id, "match_id interno", opaque=True)
    _strict_count(record.point_number, "point_number", positive=True)
    if type(record.serve_number) is not int or record.serve_number not in (1, 2):
        raise TacticalPipelineContractError("serve_number invalido.")
    if type(record.effective_date) is not date or record.effective_date > DEVELOPMENT_CUTOFF.date():
        raise TacticalPipelineContractError("Fecha interna fuera de desarrollo.")
    if record.surface not in ALLOWED_SURFACES:
        raise TacticalPipelineContractError("Surface interna invalida.")
    server = _strict_text(record.server_player, "server_player")
    returner = _strict_text(record.returner_player, "returner_player")
    if server == returner:
        raise TacticalPipelineContractError("Actores internos deben ser distintos.")
    if type(record.sequence_text) is not str or _sequence_presence(record.sequence_text, "sequence_text") != "substantive":
        raise TacticalPipelineContractError("Intento sin secuencia sustantiva.")
    if type(record.previous_attempt_was_fault) is not bool or type(record.documents_service_fault) is not bool:
        raise TacticalPipelineContractError("Contexto de fault debe ser bool exacto.")
    if type(record.server_won_point) is not bool:
        raise TacticalPipelineContractError("Outcome interno debe ser bool exacto.")
    if record.serve_number == 1 and record.previous_attempt_was_fault:
        raise TacticalPipelineContractError("Primer intento con falta previa.")
    if record.serve_number == 2 and record.documents_service_fault:
        raise TacticalPipelineContractError("Segundo intento no documenta contexto del primero.")
    validate_attempt_signal_extraction(record.extraction)
    validate_tactical_feature_vector(record.feature_vector)
    if (
        record.extraction.serve_number != record.serve_number
        or record.extraction.previous_attempt_was_fault != record.previous_attempt_was_fault
        or record.feature_vector.serve_number != record.serve_number
        or record.feature_vector.policy is not schema.policy
        or record.feature_vector.schema_fingerprint != feature_schema_fingerprint(schema)
    ):
        raise TacticalPipelineContractError("Extraccion/vector no reconcilia con intento.")
    _validate_extraction_vector_link(record.extraction, record.feature_vector, schema)
    if record.serve_number == 1 and record.documents_service_fault != _documents_service_fault(record.extraction):
        raise TacticalPipelineContractError("Fault documentado no reconcilia con extraccion.")


def _validate_extraction_vector_link(
    extraction: AttemptSignalExtraction,
    vector: TacticalFeatureVector,
    schema: TacticalFeatureSchema,
) -> None:
    """Reconcilia semanticamente una extraccion con su vector sin recodificarla."""
    index = dict(schema.name_to_index)
    adaptations = {
        item.requested_pattern_id: item for item in extraction.adaptations
    }
    prefixes = dict(_TACTICAL_FEATURE_PREFIXES)
    p02_codes = dict(_P02_VALUE_TO_CODE)
    expected_tactical = set()
    for pattern in PATTERN_ORDER:
        adaptation = adaptations.get(pattern)
        applicable = pattern in extraction.applicable_patterns
        observed = adaptation is not None and adaptation.signal is not None
        if vector.values[index[f"mask.{pattern.lower()}.applicable"]] != int(applicable):
            raise TacticalPipelineContractError("Mascara applicable no reconcilia.")
        if vector.values[index[f"mask.{pattern.lower()}.observed"]] != int(observed):
            raise TacticalPipelineContractError("Mascara observed no reconcilia.")
        if observed and pattern in schema.encoded_pattern_order:
            value = adaptation.signal.tactical_value
            if pattern == "P02":
                value = p02_codes.get(value)
                if value is None:
                    raise TacticalPipelineContractError("Valor P02 no reconciliable.")
            expected_tactical.add(prefixes[pattern] + value)
    actual_tactical = {
        name
        for name in vector.active_feature_names
        if name in schema.tactical_feature_names
    }
    if actual_tactical != expected_tactical:
        raise TacticalPipelineContractError("Features tacticas no reconcilian extraccion.")
    if vector.values[index[f"context.serve_number.{extraction.serve_number}"]] != 1:
        raise TacticalPipelineContractError("Contexto de serve_number no reconcilia.")
    if vector.values[index["context.previous_attempt_was_fault"]] != int(
        extraction.previous_attempt_was_fault
    ):
        raise TacticalPipelineContractError("Contexto de fault no reconcilia vector.")


def validate_tactical_cache_metrics(
    metrics: TacticalCacheMetrics,
    records: tuple[TacticalAttemptRecord, ...],
) -> None:
    if type(metrics) is not TacticalCacheMetrics:
        raise TypeError("cache_metrics debe usar el tipo contractual.")
    for field in (
        "attempt_requests",
        "unique_keys",
        "cache_hits",
        "cache_misses",
        "orchestrator_calls",
        "encoder_calls",
    ):
        _strict_count(getattr(metrics, field), field)
    keys = {
        (item.sequence_text, item.serve_number, item.previous_attempt_was_fault)
        for item in records
    }
    cached_objects: dict[tuple[str, int, bool], tuple[int, int]] = {}
    for item in records:
        key = (item.sequence_text, item.serve_number, item.previous_attempt_was_fault)
        identity = (id(item.extraction), id(item.feature_vector))
        previous = cached_objects.setdefault(key, identity)
        if previous != identity:
            raise TacticalPipelineContractError("Objetos de cache no fueron reutilizados.")
    expected_unique = len(keys)
    expected_requests = len(records)
    expected_hits = expected_requests - expected_unique
    expected_rate = None if expected_requests == 0 else expected_hits / expected_requests
    if (
        metrics.attempt_requests != expected_requests
        or metrics.unique_keys != expected_unique
        or metrics.cache_hits != expected_hits
        or metrics.cache_misses != expected_unique
        or metrics.orchestrator_calls != expected_unique
        or metrics.encoder_calls != expected_unique
        or metrics.hit_rate != expected_rate
    ):
        raise TacticalPipelineContractError("Metricas de cache no reconcilian.")


def validate_tactical_attempt_batch(
    batch: TacticalAttemptBatch,
    *,
    development_point_rows: int,
    schema: TacticalFeatureSchema,
) -> None:
    if type(batch) is not TacticalAttemptBatch or type(batch.records) is not tuple:
        raise TypeError("batch/records fuera del contrato inmutable.")
    validate_tactical_feature_schema(schema)
    for record in batch.records:
        validate_tactical_attempt_record(record, schema)
    keys = tuple(
        (item.match_id, item.point_number, item.serve_number) for item in batch.records
    )
    if len(set(keys)) != len(keys):
        raise TacticalPipelineContractError("Clave de intento duplicada.")
    expected_order = tuple(
        sorted(
            batch.records,
            key=lambda item: (
                item.effective_date,
                item.match_id,
                item.point_number,
                item.serve_number,
            ),
        )
    )
    if batch.records != expected_order:
        raise TacticalPipelineContractError("Orden de intentos no canonico.")
    first = tuple(item for item in batch.records if item.serve_number == 1)
    second = tuple(item for item in batch.records if item.serve_number == 2)
    if len(first) != development_point_rows:
        raise TacticalPipelineContractError("Debe existir un primer intento por punto.")
    first_by_point = {(item.match_id, item.point_number): item for item in first}
    for item in second:
        prior = first_by_point.get((item.match_id, item.point_number))
        if prior is None or item.previous_attempt_was_fault != prior.documents_service_fault:
            raise TacticalPipelineContractError("Contexto del segundo intento invalido.")
        if (
            item.effective_date,
            item.surface,
            item.server_player,
            item.returner_player,
            item.server_won_point,
        ) != (
            prior.effective_date,
            prior.surface,
            prior.server_player,
            prior.returner_player,
            prior.server_won_point,
        ):
            raise TacticalPipelineContractError("Segundo intento no reconcilia con su punto.")
    for field in (
        "first_attempts",
        "second_attempts",
        "second_with_documented_first_fault",
        "second_without_documented_first_fault",
    ):
        _strict_count(getattr(batch, field), field)
    if (
        batch.first_attempts != len(first)
        or batch.second_attempts != len(second)
        or batch.second_with_documented_first_fault
        != sum(item.previous_attempt_was_fault for item in second)
        or batch.second_without_documented_first_fault
        != sum(not item.previous_attempt_was_fault for item in second)
        or batch.second_with_documented_first_fault
        + batch.second_without_documented_first_fault
        != len(second)
    ):
        raise TacticalPipelineContractError("Conteos de intentos no reconcilian.")
    for field in ("attempt_construction_seconds", "extraction_and_encoding_seconds"):
        value = getattr(batch, field)
        if type(value) is not float or not isfinite(value) or value < 0:
            raise TacticalPipelineContractError("Tiempo de etapa invalido.")
    validate_tactical_cache_metrics(batch.cache_metrics, batch.records)


def build_tactical_historical_observations(
    attempts: tuple[TacticalAttemptRecord, ...],
) -> tuple[TacticalHistoricalObservation, ...]:
    """Crea una observacion neutral orientada al servidor por intento."""
    if type(attempts) is not tuple:
        raise TypeError("attempts debe ser tuple inmutable.")
    observations = tuple(
        TacticalHistoricalObservation(
            HISTORY_CONTRACT_VERSION,
            item.match_id,
            item.point_number,
            item.serve_number,
            item.effective_date,
            item.server_player,
            item.returner_player,
            TacticalPlayerRole.SERVER,
            item.feature_vector,
            (
                TacticalOutcomeLabel.SERVER_WON_POINT
                if item.server_won_point
                else TacticalOutcomeLabel.RETURNER_WON_POINT
            ),
            TacticalLabelAvailability.AVAILABLE,
            OBSERVATION_PROVENANCE,
        )
        for item in attempts
    )
    for observation in observations:
        validate_tactical_historical_observation(observation)
    if len(observations) != len(attempts):
        raise TacticalPipelineContractError("Observaciones no reconcilian intentos.")
    return observations


def _target_result(
    target: TacticalPipelineTarget,
    history: tuple[TacticalHistoricalObservation, ...],
    config: TacticalPipelineConfig,
    schema: TacticalFeatureSchema,
    *,
    evidence_builder: Callable[
        [
            tuple[TacticalHistoricalObservation, ...],
            TacticalMatchupQuery,
            TacticalFeatureSchema,
        ],
        TacticalMatchupEvidence,
    ] = build_tactical_matchup_evidence,
    prioritizer: Callable[..., TacticalPrioritizationResult] = prioritize_tactical_matchup,
) -> tuple[TacticalTargetResult, float, float]:
    query = TacticalMatchupQuery(
        MATCHUP_EVIDENCE_CONTRACT_VERSION,
        target.player,
        target.opponent,
        target.as_of_date,
        config.encoder_policy,
        feature_schema_fingerprint(schema),
        config.serve_numbers,
        config.window_days,
        config.minimum_labeled_attempts,
        config.minimum_matches,
        config.requested_patterns,
        TacticalEvidenceScopeStrategy.GLOBAL_ONLY,
    )
    started = perf_counter()
    try:
        evidence = evidence_builder(history, query, schema)
        validate_tactical_matchup_evidence(evidence)
    except TacticalPipelineExecutionError:
        raise
    except Exception as error:
        raise TacticalPipelineExecutionError(
            "target_evidence", type(error).__name__
        ) from error
    evidence_seconds = perf_counter() - started
    started = perf_counter()
    try:
        prioritization = prioritizer(
            evidence,
            top_k=config.top_k,
        )
        validate_tactical_prioritization_result(prioritization)
    except TacticalPipelineExecutionError:
        raise
    except Exception as error:
        raise TacticalPipelineExecutionError(
            "prioritization", type(error).__name__
        ) from error
    prioritization_seconds = perf_counter() - started
    return (
        TacticalTargetResult(
            target=target,
            history_observation_count=len(history),
            evidence=evidence,
            prioritization=prioritization,
            state=prioritization.state,
        ),
        evidence_seconds,
        prioritization_seconds,
    )


def _indexed_target_result(
    target: TacticalPipelineTarget,
    index: _PreparedEvidenceIndex,
    config: TacticalPipelineConfig,
    schema: TacticalFeatureSchema,
    *,
    prioritizer: Callable[..., TacticalPrioritizationResult],
    operation_counters: dict[str, int] | None,
) -> tuple[TacticalTargetResult, float, float, int, int]:
    started = perf_counter()
    try:
        evidence, lookups, examined = _build_indexed_matchup_evidence(
            index,
            target,
            config,
            schema,
            operation_counters=operation_counters,
        )
    except TacticalPipelineExecutionError:
        raise
    except Exception as error:
        raise TacticalPipelineExecutionError(
            "target_evidence", type(error).__name__
        ) from error
    evidence_seconds = perf_counter() - started
    started = perf_counter()
    try:
        _increment_operation(operation_counters, "prioritizations")
        prioritization = prioritizer(evidence, top_k=config.top_k)
        validate_tactical_prioritization_result(prioritization)
    except TacticalPipelineExecutionError:
        raise
    except Exception as error:
        raise TacticalPipelineExecutionError(
            "prioritization", type(error).__name__
        ) from error
    prioritization_seconds = perf_counter() - started
    return (
        TacticalTargetResult(
            target,
            evidence.total_observations_received,
            evidence,
            prioritization,
            prioritization.state,
        ),
        evidence_seconds,
        prioritization_seconds,
        lookups,
        examined,
    )


def _target_semantic_cache_key(
    target: TacticalPipelineTarget,
    config: TacticalPipelineConfig,
    schema_fingerprint: str,
) -> tuple[object, ...]:
    """Incluye toda dimension que puede alterar evidencia o priorizacion."""
    return (
        target.player,
        target.opponent,
        target.as_of_date,
        config.encoder_policy,
        schema_fingerprint,
        config.serve_numbers,
        config.window_days,
        config.minimum_labeled_attempts,
        config.minimum_matches,
        config.requested_patterns,
        TacticalEvidenceScopeStrategy.GLOBAL_ONLY,
        config.top_k,
    )


def _pipeline_state(
    target_results: tuple[TacticalTargetResult, ...],
) -> tuple[TacticalPipelineState, tuple[str, ...]]:
    with_any = sum(
        item.state is not TacticalPrioritizationState.NOT_AVAILABLE
        for item in target_results
    )
    if with_any == len(target_results):
        return TacticalPipelineState.AVAILABLE, (
            "all_targets_have_some_prioritization",
        )
    if with_any:
        return TacticalPipelineState.PARTIALLY_AVAILABLE, (
            "some_targets_or_patterns_available",
        )
    return TacticalPipelineState.NOT_AVAILABLE, ("no_target_available",)


def _stage_diagnostics(
    values: dict[str, float],
    *,
    target_count: int,
    history_index_entries: int = 0,
    history_stream_lookups: int = 0,
    candidate_observations_examined: int = 0,
) -> TacticalPipelineDiagnostics:
    return TacticalPipelineDiagnostics(
        stage_seconds=tuple((stage, float(values.get(stage, 0.0))) for stage in STAGE_ORDER),
        temporal_index_strategy="sorted_dates_with_bisect_left_exclusive_cut",
        target_history_slices=target_count,
        full_observation_rebuilds=1,
        history_index_entries=history_index_entries,
        history_stream_lookups=history_stream_lookups,
        candidate_observations_examined=candidate_observations_examined,
        full_history_scans_per_target=0,
    )


def _execute_stage(
    stage: str,
    operation: Callable[[], object],
    timings: dict[str, float] | None = None,
) -> object:
    started = perf_counter()
    try:
        return operation()
    except TacticalPipelineExecutionError:
        raise
    except Exception as error:
        raise TacticalPipelineExecutionError(stage, type(error).__name__) from error
    finally:
        if timings is not None:
            timings[stage] = timings.get(stage, 0.0) + (perf_counter() - started)


def _operation_snapshot(counters: dict[str, int] | None) -> dict[str, int]:
    return {
        field: 0 if counters is None else int(counters.get(field, 0))
        for field in OPERATION_COUNTER_FIELDS
    }


def _notify_pipeline_progress(
    callback: Callable[[dict[str, object]], None] | None,
    *,
    current_stage: str,
    completed_stages: tuple[str, ...],
    stage_seconds: dict[str, float],
    targets_total: int,
    targets_processed: int,
    attempts_processed: int,
    unique_cache_entries: int,
    observations_indexed: int,
    operation_counters: dict[str, int] | None,
) -> None:
    if callback is None:
        return
    if current_stage not in PERFORMANCE_STAGE_ORDER:
        raise TacticalPipelineContractError("Etapa de progreso invalida.")
    if (
        type(completed_stages) is not tuple
        or any(stage not in PERFORMANCE_STAGE_ORDER for stage in completed_stages)
        or completed_stages
        != tuple(stage for stage in PERFORMANCE_STAGE_ORDER if stage in completed_stages)
    ):
        raise TacticalPipelineContractError("Etapas completadas fuera de orden.")
    if any(
        stage not in PERFORMANCE_STAGE_ORDER
        or type(value) is not float
        or not isfinite(value)
        or value < 0.0
        for stage, value in stage_seconds.items()
    ):
        raise TacticalPipelineContractError("Tiempos de progreso invalidos.")
    for value in (
        targets_total,
        targets_processed,
        attempts_processed,
        unique_cache_entries,
        observations_indexed,
    ):
        if type(value) is not int or value < 0:
            raise TacticalPipelineContractError("Conteo de progreso invalido.")
    if targets_processed > targets_total:
        raise TacticalPipelineContractError("Progreso target fuera de rango.")
    callback(
        {
            "execution_status": "running",
            "current_stage": current_stage,
            "completed_stages": list(completed_stages),
            "stage_seconds": {
                stage: stage_seconds[stage]
                for stage in PERFORMANCE_STAGE_ORDER
                if stage in stage_seconds
            },
            "targets_total": targets_total,
            "targets_processed": targets_processed,
            "targets_remaining": targets_total - targets_processed,
            "progress_fraction": (
                0.0
                if targets_total == 0
                else float(targets_processed / targets_total)
            ),
            "attempts_processed": attempts_processed,
            "unique_cache_entries": unique_cache_entries,
            "observations_indexed": observations_indexed,
            "operation_counters": _operation_snapshot(operation_counters),
        }
    )


def run_tactical_recommender_pipeline(
    points: pd.DataFrame,
    targets: tuple[TacticalPipelineTarget, ...] | None,
    config: TacticalPipelineConfig | None = None,
    *,
    orchestrator: Callable[[AttemptSignalRequest], AttemptSignalExtraction] = extract_tactical_signals_for_attempt,
    encoder: Callable[[AttemptSignalExtraction, TacticalFeatureSchema], TacticalFeatureVector] = encode_tactical_attempt,
    reference_evidence_builder_for_tests: Callable[
        [
            tuple[TacticalHistoricalObservation, ...],
            TacticalMatchupQuery,
            TacticalFeatureSchema,
        ],
        TacticalMatchupEvidence,
    ] | None = None,
    prioritizer: Callable[..., TacticalPrioritizationResult] = prioritize_tactical_matchup,
    stage_timing_sink: dict[str, float] | None = None,
    operation_counters: dict[str, int] | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> TacticalPipelineResult:
    """Ejecuta en memoria; la ruta de referencia solo admite inyeccion sintetica."""
    selected = default_pipeline_config() if config is None else config
    validate_tactical_pipeline_config(selected)
    if (
        selected.expected_source_rows == EXPECTED_SOURCE_ROWS
        and reference_evidence_builder_for_tests is not None
    ):
        raise TacticalPipelineContractError(
            "La ruta real exige exclusivamente evidencia indexada."
        )
    if stage_timing_sink is not None and type(stage_timing_sink) is not dict:
        raise TypeError("stage_timing_sink debe ser dict exacto o None.")
    if stage_timing_sink:
        raise TacticalPipelineContractError("stage_timing_sink debe comenzar vacio.")
    if operation_counters is not None:
        if type(operation_counters) is not dict or operation_counters:
            raise TacticalPipelineContractError(
                "operation_counters debe ser dict exacto inicialmente vacio."
            )
        operation_counters.update((field, 0) for field in OPERATION_COUNTER_FIELDS)
    timings: dict[str, float] = {} if stage_timing_sink is None else stage_timing_sink
    operational_seconds: dict[str, float] = {}

    _notify_pipeline_progress(
        progress_callback,
        current_stage="source_validation",
        completed_stages=(),
        stage_seconds=operational_seconds,
        targets_total=0,
        targets_processed=0,
        attempts_processed=0,
        unique_cache_entries=0,
        observations_indexed=0,
        operation_counters=operation_counters,
    )
    operational_started = perf_counter()
    source = _execute_stage(
        "source_validation",
        lambda: validate_source_points(
            points,
            expected_rows=selected.expected_source_rows,
        ),
        timings,
    )
    operational_seconds["source_validation"] = float(
        perf_counter() - operational_started
    )
    _notify_pipeline_progress(
        progress_callback,
        current_stage="source_validation",
        completed_stages=("source_validation",),
        stage_seconds=operational_seconds,
        targets_total=0,
        targets_processed=0,
        attempts_processed=0,
        unique_cache_entries=0,
        observations_indexed=0,
        operation_counters=operation_counters,
    )

    _notify_pipeline_progress(
        progress_callback,
        current_stage="temporal_sealing",
        completed_stages=PERFORMANCE_STAGE_ORDER[:1],
        stage_seconds=operational_seconds,
        targets_total=0,
        targets_processed=0,
        attempts_processed=0,
        unique_cache_entries=0,
        observations_indexed=0,
        operation_counters=operation_counters,
    )
    operational_started = perf_counter()
    development, initial_population, seal = _execute_stage(
        "temporal_seal", lambda: seal_development_points(source), timings
    )
    operational_seconds["temporal_sealing"] = float(
        perf_counter() - operational_started
    )
    _notify_pipeline_progress(
        progress_callback,
        current_stage="temporal_sealing",
        completed_stages=PERFORMANCE_STAGE_ORDER[:2],
        stage_seconds=operational_seconds,
        targets_total=0,
        targets_processed=0,
        attempts_processed=0,
        unique_cache_entries=0,
        observations_indexed=0,
        operation_counters=operation_counters,
    )

    schema = build_tactical_feature_schema(selected.encoder_policy)
    _notify_pipeline_progress(
        progress_callback,
        current_stage="attempt_extraction_and_encoding",
        completed_stages=PERFORMANCE_STAGE_ORDER[:2],
        stage_seconds=operational_seconds,
        targets_total=0,
        targets_processed=0,
        attempts_processed=0,
        unique_cache_entries=0,
        observations_indexed=0,
        operation_counters=operation_counters,
    )
    operational_started = perf_counter()
    batch = _execute_stage(
        "extraction_and_encoding",
        lambda: construct_tactical_attempt_records(
            development,
            schema,
            orchestrator=orchestrator,
            encoder=encoder,
        ),
        timings,
    )
    timings["attempt_construction"] = batch.attempt_construction_seconds
    timings["extraction_and_encoding"] = batch.extraction_and_encoding_seconds
    operational_seconds["attempt_extraction_and_encoding"] = float(
        perf_counter() - operational_started
    )
    _notify_pipeline_progress(
        progress_callback,
        current_stage="attempt_extraction_and_encoding",
        completed_stages=PERFORMANCE_STAGE_ORDER[:3],
        stage_seconds=operational_seconds,
        targets_total=0,
        targets_processed=0,
        attempts_processed=len(batch.records),
        unique_cache_entries=batch.cache_metrics.unique_keys,
        observations_indexed=0,
        operation_counters=operation_counters,
    )

    _notify_pipeline_progress(
        progress_callback,
        current_stage="observation_indexing",
        completed_stages=PERFORMANCE_STAGE_ORDER[:3],
        stage_seconds=operational_seconds,
        targets_total=0,
        targets_processed=0,
        attempts_processed=len(batch.records),
        unique_cache_entries=batch.cache_metrics.unique_keys,
        observations_indexed=0,
        operation_counters=operation_counters,
    )
    operational_started = perf_counter()
    observations = _execute_stage(
        "observation_construction",
        lambda: build_tactical_historical_observations(batch.records),
        timings,
    )

    optimized_evidence = reference_evidence_builder_for_tests is None
    prepared_index = _execute_stage(
        "target_evidence",
        lambda: _build_prepared_evidence_index(
            observations,
            selected,
            schema,
            operation_counters=(operation_counters if optimized_evidence else None),
        ),
        timings,
    )
    history_index = None
    if not optimized_evidence:
        history_index = _execute_stage(
            "target_evidence", lambda: _build_history_index(observations), timings
        )
    operational_seconds["observation_indexing"] = float(
        perf_counter() - operational_started
    )
    _notify_pipeline_progress(
        progress_callback,
        current_stage="observation_indexing",
        completed_stages=PERFORMANCE_STAGE_ORDER[:4],
        stage_seconds=operational_seconds,
        targets_total=0,
        targets_processed=0,
        attempts_processed=len(batch.records),
        unique_cache_entries=batch.cache_metrics.unique_keys,
        observations_indexed=len(observations),
        operation_counters=operation_counters,
    )
    _notify_pipeline_progress(
        progress_callback,
        current_stage="target_construction",
        completed_stages=PERFORMANCE_STAGE_ORDER[:4],
        stage_seconds=operational_seconds,
        targets_total=0,
        targets_processed=0,
        attempts_processed=len(batch.records),
        unique_cache_entries=batch.cache_metrics.unique_keys,
        observations_indexed=len(observations),
        operation_counters=operation_counters,
    )
    operational_started = perf_counter()
    if targets is None:
        ordered_targets = _execute_stage(
            "target_evidence",
            lambda: build_validation_targets(
                development,
                enforce_frozen_cardinalities=selected.expected_source_rows
                == EXPECTED_SOURCE_ROWS,
            ),
            timings,
        )
    else:
        if type(targets) is not tuple or not targets:
            raise TacticalPipelineContractError("targets debe ser tuple no vacia o None.")
        for target in targets:
            validate_tactical_pipeline_target(target)
        ordered_targets = tuple(
            sorted(
                targets,
                key=lambda item: (
                    item.as_of_date,
                    item.target_match_id,
                    item.orientation,
                ),
            )
        )
    target_keys = tuple(
        (item.target_match_id, item.orientation) for item in ordered_targets
    )
    if len(set(target_keys)) != len(target_keys):
        raise TacticalPipelineContractError("Target duplicado.")
    operational_seconds["target_construction"] = float(
        perf_counter() - operational_started
    )
    _notify_pipeline_progress(
        progress_callback,
        current_stage="target_construction",
        completed_stages=PERFORMANCE_STAGE_ORDER[:5],
        stage_seconds=operational_seconds,
        targets_total=len(ordered_targets),
        targets_processed=0,
        attempts_processed=len(batch.records),
        unique_cache_entries=batch.cache_metrics.unique_keys,
        observations_indexed=len(observations),
        operation_counters=operation_counters,
    )
    target_results = []
    target_cache: dict[
        tuple[object, ...],
        tuple[TacticalMatchupEvidence, TacticalPrioritizationResult, int, int],
    ] = {}
    evidence_seconds = prioritization_seconds = 0.0
    history_stream_lookups = candidate_observations_examined = 0
    _notify_pipeline_progress(
        progress_callback,
        current_stage="evidence_and_prioritization",
        completed_stages=PERFORMANCE_STAGE_ORDER[:5],
        stage_seconds=operational_seconds,
        targets_total=len(ordered_targets),
        targets_processed=0,
        attempts_processed=len(batch.records),
        unique_cache_entries=batch.cache_metrics.unique_keys,
        observations_indexed=len(observations),
        operation_counters=operation_counters,
    )
    evidence_stage_started = perf_counter()
    for target in ordered_targets:
        _increment_operation(operation_counters, "targets_processed")
        target_started = perf_counter()
        try:
            if optimized_evidence:
                if prepared_index is None:
                    raise TacticalPipelineContractError(
                        "Indice preparado no disponible."
                    )
                cache_key = _target_semantic_cache_key(
                    target,
                    selected,
                    prepared_index.schema_fingerprint,
                )
                cached = target_cache.get(cache_key)
                if cached is None:
                    (
                        run,
                        evidence_elapsed,
                        prioritization_elapsed,
                        lookups,
                        examined,
                    ) = _indexed_target_result(
                        target,
                        prepared_index,
                        selected,
                        schema,
                        prioritizer=prioritizer,
                        operation_counters=operation_counters,
                    )
                    if prioritizer is prioritize_tactical_matchup:
                        target_cache[cache_key] = (
                            run.evidence,
                            run.prioritization,
                            lookups,
                            examined,
                        )
                else:
                    evidence, prioritization, lookups, examined = cached
                    run = TacticalTargetResult(
                        target,
                        evidence.total_observations_received,
                        evidence,
                        prioritization,
                        prioritization.state,
                    )
                    evidence_elapsed = prioritization_elapsed = 0.0
            else:
                if history_index is None:
                    raise TacticalPipelineContractError(
                        "Indice historico de referencia no disponible."
                    )
                history, lookups, examined = _execute_stage(
                    "target_evidence",
                    lambda target=target: _select_target_history(
                        history_index, target, selected
                    ),
                    timings,
                )
                _increment_operation(
                    operation_counters,
                    "actor_relevant_observations_inspected",
                    examined,
                )
                _increment_operation(operation_counters, "history_materializations")
                _increment_operation(operation_counters, "query_constructions")
                _increment_operation(operation_counters, "evidence_constructions")
                run, evidence_elapsed, prioritization_elapsed = _target_result(
                    target,
                    history,
                    selected,
                    schema,
                    evidence_builder=reference_evidence_builder_for_tests,
                    prioritizer=prioritizer,
                )
                _increment_operation(operation_counters, "prioritizations")
        except TacticalPipelineExecutionError as error:
            timings[error.stage] = timings.get(error.stage, 0.0) + (
                perf_counter() - target_started
            )
            raise
        history_stream_lookups += lookups
        candidate_observations_examined += examined
        evidence_seconds += evidence_elapsed
        prioritization_seconds += prioritization_elapsed
        target_results.append(run)
        processed_count = len(target_results)
        if (
            processed_count % PERFORMANCE_PROGRESS_TARGET_INTERVAL == 0
            or processed_count == len(ordered_targets)
        ):
            operational_seconds["evidence_and_prioritization"] = float(
                perf_counter() - evidence_stage_started
            )
            _notify_pipeline_progress(
                progress_callback,
                current_stage="evidence_and_prioritization",
                completed_stages=(
                    PERFORMANCE_STAGE_ORDER[:6]
                    if processed_count == len(ordered_targets)
                    else PERFORMANCE_STAGE_ORDER[:5]
                ),
                stage_seconds=operational_seconds,
                targets_total=len(ordered_targets),
                targets_processed=processed_count,
                attempts_processed=len(batch.records),
                unique_cache_entries=batch.cache_metrics.unique_keys,
                observations_indexed=len(observations),
                operation_counters=operation_counters,
            )
    timings["target_evidence"] = timings.get("target_evidence", 0.0) + evidence_seconds
    timings["prioritization"] = timings.get("prioritization", 0.0) + prioritization_seconds
    run_tuple = tuple(target_results)
    _notify_pipeline_progress(
        progress_callback,
        current_stage="aggregation_and_validation",
        completed_stages=PERFORMANCE_STAGE_ORDER[:6],
        stage_seconds=operational_seconds,
        targets_total=len(ordered_targets),
        targets_processed=len(run_tuple),
        attempts_processed=len(batch.records),
        unique_cache_entries=batch.cache_metrics.unique_keys,
        observations_indexed=len(observations),
        operation_counters=operation_counters,
    )
    aggregation_started = perf_counter()
    state, reasons = _pipeline_state(run_tuple)
    population = TacticalPipelinePopulation(
        source_point_rows=initial_population.source_point_rows,
        source_matches=initial_population.source_matches,
        development_point_rows=initial_population.development_point_rows,
        development_matches=initial_population.development_matches,
        excluded_test_point_rows=initial_population.excluded_test_point_rows,
        excluded_test_matches=initial_population.excluded_test_matches,
        attempt_count=len(batch.records),
        first_attempt_count=batch.first_attempts,
        second_attempt_count=batch.second_attempts,
        second_with_documented_first_fault=batch.second_with_documented_first_fault,
        second_without_documented_first_fault=batch.second_without_documented_first_fault,
        observation_count=len(observations),
    )
    total_candidates = sum(item.prioritization.total_candidate_count for item in run_tuple)
    scored = sum(item.prioritization.scored_candidate_count for item in run_tuple)
    diagnostics = _stage_diagnostics(
        timings,
        target_count=len(run_tuple),
        history_index_entries=(
            prepared_index.observation_count
            if prepared_index is not None
            else history_index.observation_count
        ),
        history_stream_lookups=history_stream_lookups,
        candidate_observations_examined=candidate_observations_examined,
    )
    leakage_audit = TacticalLeakageAudit(
        history_on_or_after_target_date=0,
        target_match_rows_used=0,
        same_day_rows_used=0,
        future_rows_used=0,
        test_rows_used=0,
        test_attempts_used=0,
        test_labels_used=0,
        test_profiles_computed=0,
        test_evidence_computed=0,
        test_scores_computed=0,
        test_rankings_computed=0,
        test_recommendations_generated=0,
    )
    result = TacticalPipelineResult(
        contract_version=PIPELINE_CONTRACT_VERSION,
        config=selected,
        state=state,
        reason_codes=reasons,
        population=population,
        test_seal=seal,
        cache_metrics=batch.cache_metrics,
        schema=schema,
        schema_fingerprint=feature_schema_fingerprint(schema),
        attempts=batch.records,
        observations=observations,
        prepared_evidence_index=prepared_index,
        target_results=run_tuple,
        targets_requested=len(ordered_targets),
        targets_processed=len(run_tuple),
        targets_available=sum(item.state is TacticalPrioritizationState.AVAILABLE for item in run_tuple),
        targets_partial=sum(item.state is TacticalPrioritizationState.PARTIALLY_AVAILABLE for item in run_tuple),
        targets_not_available=sum(item.state is TacticalPrioritizationState.NOT_AVAILABLE for item in run_tuple),
        total_candidates=total_candidates,
        scored_candidates=scored,
        abstained_candidates=total_candidates - scored,
        diagnostics=diagnostics,
        leakage_audit=leakage_audit,
        reconciliations=PIPELINE_RECONCILIATIONS,
        provenance=PIPELINE_PROVENANCE,
        limitations=PIPELINE_LIMITATIONS,
    )
    _execute_stage(
        "final_validation",
        lambda: validate_tactical_pipeline_result(
            result,
            operation_counters=operation_counters,
        ),
        timings,
    )
    # Timing is diagnostico y se excluye deliberadamente del fingerprint.
    final_result = replace(
        result,
        diagnostics=_stage_diagnostics(
            timings,
            target_count=len(run_tuple),
            history_index_entries=(
                prepared_index.observation_count
                if prepared_index is not None
                else history_index.observation_count
            ),
            history_stream_lookups=history_stream_lookups,
            candidate_observations_examined=candidate_observations_examined,
        ),
    )
    _validate_diagnostics(final_result.diagnostics, len(final_result.target_results))
    operational_seconds["aggregation_and_validation"] = float(
        perf_counter() - aggregation_started
    )
    _notify_pipeline_progress(
        progress_callback,
        current_stage="aggregation_and_validation",
        completed_stages=PERFORMANCE_STAGE_ORDER[:7],
        stage_seconds=operational_seconds,
        targets_total=len(ordered_targets),
        targets_processed=len(run_tuple),
        attempts_processed=len(batch.records),
        unique_cache_entries=batch.cache_metrics.unique_keys,
        observations_indexed=len(observations),
        operation_counters=operation_counters,
    )
    return final_result


def _validate_population(population: TacticalPipelinePopulation) -> None:
    if type(population) is not TacticalPipelinePopulation:
        raise TypeError("population debe usar tipo contractual exacto.")
    for field in population.__dataclass_fields__:
        _strict_count(getattr(population, field), field)
    if population.source_point_rows != population.development_point_rows + population.excluded_test_point_rows:
        raise TacticalPipelineContractError("Filas fuente no reconcilian particiones.")
    if population.source_matches != population.development_matches + population.excluded_test_matches:
        raise TacticalPipelineContractError("Partidos fuente no reconcilian particiones.")
    if population.attempt_count != population.first_attempt_count + population.second_attempt_count:
        raise TacticalPipelineContractError("Intentos no reconcilian.")
    if population.second_attempt_count != (
        population.second_with_documented_first_fault
        + population.second_without_documented_first_fault
    ):
        raise TacticalPipelineContractError("Contextos de segundo intento no reconcilian.")
    if population.first_attempt_count != population.development_point_rows:
        raise TacticalPipelineContractError("Primeros intentos no reconcilian puntos.")
    if population.observation_count != population.attempt_count:
        raise TacticalPipelineContractError("Observaciones no reconcilian intentos.")


def _validate_test_seal(seal: TacticalTestSeal, population: TacticalPipelinePopulation) -> None:
    if type(seal) is not TacticalTestSeal:
        raise TypeError("test_seal debe usar tipo contractual exacto.")
    if seal.test_status != "sealed" or type(seal.used_for_method_selection) is not bool or seal.used_for_method_selection:
        raise TacticalPipelineContractError("Test no permanece sellado.")
    if (
        seal.excluded_test_point_rows != population.excluded_test_point_rows
        or seal.excluded_test_matches != population.excluded_test_matches
        or seal.counters != tuple((field, 0) for field in TEST_ZERO_FIELDS)
    ):
        raise TacticalPipelineContractError("Contadores del test no reconcilian.")


def _validate_diagnostics(diagnostics: TacticalPipelineDiagnostics, target_count: int) -> None:
    if type(diagnostics) is not TacticalPipelineDiagnostics or type(diagnostics.stage_seconds) is not tuple:
        raise TypeError("diagnostics debe ser profundamente inmutable.")
    if tuple(key for key, _ in diagnostics.stage_seconds) != STAGE_ORDER:
        raise TacticalPipelineContractError("Orden de etapas invalido.")
    for key, value in diagnostics.stage_seconds:
        if type(key) is not str or type(value) is not float or not isfinite(value) or value < 0:
            raise TacticalPipelineContractError("Tiempo de etapa invalido.")
    if diagnostics.temporal_index_strategy != "sorted_dates_with_bisect_left_exclusive_cut":
        raise TacticalPipelineContractError("Indice temporal invalido.")
    if diagnostics.target_history_slices != target_count or diagnostics.full_observation_rebuilds != 1:
        raise TacticalPipelineContractError("Diagnosticos de reconstruccion invalidos.")
    for field in (
        "history_index_entries",
        "history_stream_lookups",
        "candidate_observations_examined",
        "full_history_scans_per_target",
    ):
        _strict_count(getattr(diagnostics, field), field)
    if diagnostics.full_history_scans_per_target != 0:
        raise TacticalPipelineContractError("La estrategia historica no es subcuadratica.")


def _validate_leakage_audit(audit: TacticalLeakageAudit) -> None:
    if type(audit) is not TacticalLeakageAudit:
        raise TypeError("leakage_audit debe usar tipo contractual exacto.")
    for field in audit.__dataclass_fields__:
        if _strict_count(getattr(audit, field), field) != 0:
            raise TacticalPipelineContractError("Auditoria leakage debe permanecer en cero.")


def validate_tactical_pipeline_result(
    result: TacticalPipelineResult,
    *,
    operation_counters: dict[str, int] | None = None,
) -> None:
    if type(result) is not TacticalPipelineResult:
        raise TypeError("result debe ser TacticalPipelineResult exacto.")
    if result.contract_version != PIPELINE_CONTRACT_VERSION:
        raise TacticalPipelineContractError("Version de resultado invalida.")
    validate_tactical_pipeline_config(result.config)
    _validate_population(result.population)
    _validate_test_seal(result.test_seal, result.population)
    validate_tactical_feature_schema(result.schema)
    if result.schema.policy is not result.config.encoder_policy or result.schema_fingerprint != feature_schema_fingerprint(result.schema):
        raise TacticalPipelineContractError("Schema/fingerprint no reconcilia.")
    if type(result.attempts) is not tuple or type(result.observations) is not tuple:
        raise TacticalPipelineContractError("Intentos y observaciones deben ser tuples.")
    for record in result.attempts:
        validate_tactical_attempt_record(record, result.schema)
    keys = tuple((item.match_id, item.point_number, item.serve_number) for item in result.attempts)
    if len(keys) != len(set(keys)):
        raise TacticalPipelineContractError("Intentos duplicados en resultado.")
    validate_tactical_cache_metrics(result.cache_metrics, result.attempts)
    if (
        len(result.attempts) != result.population.attempt_count
        or len(result.observations) != result.population.observation_count
    ):
        raise TacticalPipelineContractError("Poblacion interna no reconcilia.")
    for record, observation in zip(result.attempts, result.observations):
        validate_tactical_historical_observation(observation)
        expected_label = (
            TacticalOutcomeLabel.SERVER_WON_POINT
            if record.server_won_point
            else TacticalOutcomeLabel.RETURNER_WON_POINT
        )
        if (
            observation.match_id,
            observation.point_number,
            observation.serve_number,
            observation.effective_date,
            observation.player,
            observation.opponent,
            observation.role,
            observation.feature_vector,
            observation.label,
            observation.label_availability,
        ) != (
            record.match_id,
            record.point_number,
            record.serve_number,
            record.effective_date,
            record.server_player,
            record.returner_player,
            TacticalPlayerRole.SERVER,
            record.feature_vector,
            expected_label,
            TacticalLabelAvailability.AVAILABLE,
        ):
            raise TacticalPipelineContractError("Observacion no reconcilia con intento.")
    if type(result.target_results) is not tuple or not result.target_results:
        raise TacticalPipelineContractError("Resultado sin targets procesados.")
    prepared_index = result.prepared_evidence_index
    if type(prepared_index) is not _PreparedEvidenceIndex:
        raise TacticalPipelineContractError("Indice preparado no retenido.")
    if (
        prepared_index.observation_count != len(result.observations)
        or prepared_index.requested_patterns != result.config.requested_patterns
        or prepared_index.schema_fingerprint != result.schema_fingerprint
    ):
        raise TacticalPipelineContractError("Indice preparado no reconcilia resultado.")
    target_keys = []
    for item in result.target_results:
        if type(item) is not TacticalTargetResult:
            raise TypeError("target_result fuera del tipo contractual.")
        validate_tactical_pipeline_target(item.target)
        validate_tactical_matchup_evidence(item.evidence)
        validate_tactical_prioritization_result(item.prioritization)
        target_keys.append(
            (
                item.target.as_of_date,
                item.target.target_match_id,
                item.target.orientation,
            )
        )
        if item.state is not item.prioritization.state:
            raise TacticalPipelineContractError("Estado target no reconcilia.")
        query = item.evidence.query
        if (
            query.player,
            query.opponent,
            query.as_of_date,
            query.policy,
            query.schema_fingerprint,
            query.serve_numbers,
            query.window_days,
            query.minimum_labeled_attempts,
            query.minimum_matches,
            query.requested_patterns,
            query.scope_strategy,
        ) != (
            item.target.player,
            item.target.opponent,
            item.target.as_of_date,
            result.config.encoder_policy,
            result.schema_fingerprint,
            result.config.serve_numbers,
            result.config.window_days,
            result.config.minimum_labeled_attempts,
            result.config.minimum_matches,
            result.config.requested_patterns,
            TacticalEvidenceScopeStrategy.GLOBAL_ONLY,
        ):
            raise TacticalPipelineContractError("Target y evidencia no reconcilian.")
        expected_evidence, _, _ = _build_indexed_matchup_evidence(
            prepared_index,
            item.target,
            result.config,
            result.schema,
        )
        if (
            item.history_observation_count
            != expected_evidence.total_observations_received
            or item.evidence != expected_evidence
        ):
            raise TacticalPipelineContractError(
                "Evidencia historica target no reconcilia con indice preparado."
            )
        if item.prioritization.matchup_query.query_fingerprint != item.evidence.categories[0].query_fingerprint:
            raise TacticalPipelineContractError("Evidencia y priorizacion no reconcilian.")
    if tuple(target_keys) != tuple(sorted(target_keys)) or len(set(target_keys)) != len(target_keys):
        raise TacticalPipelineContractError("Targets no usan orden/clave canonicos.")
    if result.config.expected_source_rows == EXPECTED_SOURCE_ROWS:
        _validate_frozen_target_population(
            tuple(item.target for item in result.target_results)
        )
    for field in (
        "targets_requested",
        "targets_processed",
        "targets_available",
        "targets_partial",
        "targets_not_available",
        "total_candidates",
        "scored_candidates",
        "abstained_candidates",
    ):
        _strict_count(getattr(result, field), field)
    available = sum(item.state is TacticalPrioritizationState.AVAILABLE for item in result.target_results)
    partial = sum(item.state is TacticalPrioritizationState.PARTIALLY_AVAILABLE for item in result.target_results)
    unavailable = len(result.target_results) - available - partial
    total = sum(item.prioritization.total_candidate_count for item in result.target_results)
    scored = sum(item.prioritization.scored_candidate_count for item in result.target_results)
    if (
        result.targets_requested != len(result.target_results)
        or result.targets_processed != len(result.target_results)
        or (result.targets_available, result.targets_partial, result.targets_not_available)
        != (available, partial, unavailable)
        or (result.total_candidates, result.scored_candidates, result.abstained_candidates)
        != (total, scored, total - scored)
    ):
        raise TacticalPipelineContractError("Conteos target/candidate no reconcilian.")
    state, reasons = _pipeline_state(result.target_results)
    if type(result.state) is not TacticalPipelineState or result.state is not state or result.reason_codes != reasons:
        raise TacticalPipelineContractError("Estado global no reconcilia.")
    _validate_diagnostics(result.diagnostics, len(result.target_results))
    if result.diagnostics.history_index_entries != len(result.observations):
        raise TacticalPipelineContractError("Indice historico no reconcilia observaciones.")
    _validate_leakage_audit(result.leakage_audit)
    if result.reconciliations != PIPELINE_RECONCILIATIONS or result.provenance != PIPELINE_PROVENANCE or result.limitations != PIPELINE_LIMITATIONS:
        raise TacticalPipelineContractError("Metadatos contractuales no reconcilian.")


def _safe_summary(result: TacticalPipelineResult) -> dict[str, object]:
    return {
        "contract_version": result.contract_version,
        "state": result.state.value,
        "reason_codes": list(result.reason_codes),
        "encoder_policy": result.config.encoder_policy.value,
        "requested_patterns": list(result.config.requested_patterns),
        "schema_fingerprint": result.schema_fingerprint,
        "targets_requested": result.targets_requested,
        "targets_processed": result.targets_processed,
        "targets_available": result.targets_available,
        "targets_partial": result.targets_partial,
        "targets_not_available": result.targets_not_available,
        "total_candidates": result.total_candidates,
        "scored_candidates": result.scored_candidates,
        "abstained_candidates": result.abstained_candidates,
        "test_status": result.test_seal.test_status,
        "used_for_method_selection": result.test_seal.used_for_method_selection,
        "limitations": list(result.limitations),
        "reconciliations": dict(result.reconciliations),
    }


def _population_cache_structure(result: TacticalPipelineResult) -> dict[str, object]:
    return {
        "population": dict(result.population.__dict__),
        "cache": dict(result.cache_metrics.__dict__),
        "test_seal": {
            "test_status": result.test_seal.test_status,
            "used_for_method_selection": result.test_seal.used_for_method_selection,
            "excluded_test_point_rows": result.test_seal.excluded_test_point_rows,
            "excluded_test_matches": result.test_seal.excluded_test_matches,
            "counters": dict(result.test_seal.counters),
        },
    }


def _pattern_coverage_structure(result: TacticalPipelineResult) -> dict[str, object]:
    rows = []
    for pattern in result.config.requested_patterns:
        rankings = tuple(
            ranking
            for target in result.target_results
            for ranking in target.prioritization.rankings
            if ranking.pattern_id == pattern
        )
        rows.append(
            {
                "pattern_id": pattern,
                "target_count": len(rankings),
                "targets_with_scored_candidates": sum(
                    ranking.scored_candidate_count > 0 for ranking in rankings
                ),
                "catalog_candidates": sum(len(ranking.candidates) for ranking in rankings),
                "scored_candidates": sum(ranking.scored_candidate_count for ranking in rankings),
                "abstained_candidates": sum(ranking.abstained_candidate_count for ranking in rankings),
            }
        )
    return {"patterns": rows}


def _availability_structure(result: TacticalPipelineResult) -> dict[str, object]:
    return {
        "available": result.targets_available,
        "partially_available": result.targets_partial,
        "not_available": result.targets_not_available,
        "denominator": result.targets_processed,
    }


def _aggregate_rankings_structure(result: TacticalPipelineResult) -> dict[str, object]:
    rows = []
    for pattern in result.config.requested_patterns:
        categories = sorted(
            {
                candidate.category
                for target in result.target_results
                for ranking in target.prioritization.rankings
                if ranking.pattern_id == pattern
                for candidate in ranking.candidates
            }
        )
        for category in categories:
            relevant = tuple(
                candidate
                for target in result.target_results
                for ranking in target.prioritization.rankings
                if ranking.pattern_id == pattern
                for candidate in ranking.candidates
                if candidate.category == category
            )
            top = sum(
                any(
                    item.category == category
                    for item in ranking.top_candidates
                )
                for target in result.target_results
                for ranking in target.prioritization.rankings
                if ranking.pattern_id == pattern
            )
            rows.append(
                {
                    "pattern_id": pattern,
                    "category": category,
                    "target_appearances": len(relevant),
                    "scored_appearances": sum(item.rank_eligible for item in relevant),
                    "top_k_appearances": top,
                }
            )
    return {"rankings_aggregated_by_pattern_and_category": rows}


def _validate_public_tree(value: object, *, key: str | None = None) -> None:
    forbidden_keys = {
        "match_id", "point_number", "sequence_text", "tokens", "spans",
        "point_winner", "server_player", "returner_player", "target_players",
        "path", "timestamp",
    }
    if key is not None and (key.lower() in forbidden_keys or key.lower().startswith("test_rows_") and key.lower() not in TEST_ZERO_FIELDS):
        raise TacticalPipelineContractError("Payload contiene clave sensible.")
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not isfinite(value):
            raise TacticalPipelineContractError("Payload contiene no finito.")
        return
    if type(value) is str:
        if (
            _WINDOWS_PATH.match(value)
            or value.startswith(("/", "\\\\"))
            or "file://" in value.lower()
            or "../" in value
            or "..\\" in value
            or any(ord(character) < 32 for character in value)
        ):
            raise TacticalPipelineContractError("Payload contiene texto operativo.")
        return
    if type(value) is list:
        for item in value:
            _validate_public_tree(item)
        return
    if type(value) is dict:
        for child_key, child in value.items():
            if type(child_key) is not str:
                raise TacticalPipelineContractError("Clave JSON no textual.")
            _validate_public_tree(child, key=child_key)
        return
    raise TacticalPipelineContractError("Payload contiene objeto no contractual.")


def _json_bytes(value: dict[str, object]) -> bytes:
    _validate_public_tree(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _csv_bytes(frame: pd.DataFrame, columns: tuple[str, ...]) -> bytes:
    if tuple(frame.columns) != columns:
        raise TacticalPipelineContractError("Schema CSV interno invalido.")
    payload = frame.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="",
        float_format="%.15g",
    ).encode("utf-8")
    if b"\r" in payload or b"NaN" in payload or b"Infinity" in payload:
        raise TacticalPipelineContractError("Serializacion CSV no determinista.")
    return payload


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _population_frame(result: TacticalPipelineResult) -> pd.DataFrame:
    population = result.population
    validation_matches = len(
        {item.target.target_match_id for item in result.target_results}
    )
    rows = (
        ("source", population.source_point_rows, population.source_point_rows),
        ("development", population.development_point_rows, population.source_point_rows),
        ("excluded_test", population.excluded_test_point_rows, population.source_point_rows),
        ("attempts_total", population.attempt_count, population.development_point_rows),
        ("first_serve_attempts", population.first_attempt_count, population.development_point_rows),
        ("second_serve_attempts", population.second_attempt_count, population.development_point_rows),
        ("second_with_documented_fault", population.second_with_documented_first_fault, population.second_attempt_count),
        ("second_without_documented_fault", population.second_without_documented_first_fault, population.second_attempt_count),
        ("cache_entries", result.cache_metrics.unique_keys, population.attempt_count),
        ("cache_hits", result.cache_metrics.cache_hits, population.attempt_count),
        ("cache_misses", result.cache_metrics.cache_misses, population.attempt_count),
        ("observations", population.observation_count, population.attempt_count),
        ("validation_matches", validation_matches, population.development_matches),
        ("target_orientations", result.targets_processed, 2 * validation_matches),
    )
    if tuple(item[0] for item in rows) != POPULATION_ROW_ORDER:
        raise TacticalPipelineContractError("Orden population no contractual.")
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "count": count,
                "denominator": denominator,
                "ratio": _ratio(count, denominator),
                "status": "reconciled",
            }
            for metric, count, denominator in rows
        ],
        columns=POPULATION_COLUMNS,
    )


def _availability_row(
    level: str,
    fold: str,
    pattern: str,
    states: tuple[object, ...],
    scored: int,
    abstained: int,
) -> dict[str, object]:
    available = sum(getattr(state, "value", state) == "available" for state in states)
    partial = sum(getattr(state, "value", state) == "partially_available" for state in states)
    unavailable = len(states) - available - partial
    return {
        "aggregation_level": level,
        "fold": fold,
        "pattern_id": pattern,
        "target_orientations": len(states),
        "available": available,
        "partially_available": partial,
        "not_available": unavailable,
        "coverage": _ratio(available + partial, len(states)),
        "scored_candidates": scored,
        "abstained_candidates": abstained,
        "status": "reconciled",
    }


def _availability_frame(result: TacticalPipelineResult) -> pd.DataFrame:
    rows = []
    targets = result.target_results
    rows.append(
        _availability_row(
            "total", "", "", tuple(item.state for item in targets),
            sum(item.prioritization.scored_candidate_count for item in targets),
            sum(item.prioritization.abstained_candidate_count for item in targets),
        )
    )
    for fold in VALIDATION_FOLDS:
        subset = tuple(item for item in targets if item.target.fold == fold)
        if subset:
            rows.append(
                _availability_row(
                    "fold", fold, "", tuple(item.state for item in subset),
                    sum(item.prioritization.scored_candidate_count for item in subset),
                    sum(item.prioritization.abstained_candidate_count for item in subset),
                )
            )
    for pattern in result.config.requested_patterns:
        rankings = tuple(
            ranking
            for item in targets
            for ranking in item.prioritization.rankings
            if ranking.pattern_id == pattern
        )
        rows.append(
            _availability_row(
                "pattern", "", pattern, tuple(item.state for item in rankings),
                sum(item.scored_candidate_count for item in rankings),
                sum(item.abstained_candidate_count for item in rankings),
            )
        )
    for fold in VALIDATION_FOLDS:
        for pattern in result.config.requested_patterns:
            rankings = tuple(
                ranking
                for item in targets
                if item.target.fold == fold
                for ranking in item.prioritization.rankings
                if ranking.pattern_id == pattern
            )
            if rankings:
                rows.append(
                    _availability_row(
                        "fold_pattern", fold, pattern,
                        tuple(item.state for item in rankings),
                        sum(item.scored_candidate_count for item in rankings),
                        sum(item.abstained_candidate_count for item in rankings),
                    )
                )
    return pd.DataFrame(rows, columns=AVAILABILITY_COLUMNS)


def _rankings_frame(result: TacticalPipelineResult) -> pd.DataFrame:
    rows = []
    for pattern in result.config.requested_patterns:
        feature_prefix = dict(_TACTICAL_FEATURE_PREFIXES)[pattern]
        categories = tuple(
            name[len(feature_prefix) :]
            for name in result.schema.tactical_feature_names
            if name.startswith(feature_prefix)
        )
        for category in categories:
            appearances = []
            top_count = tie_count = 0
            for target in result.target_results:
                ranking = next(
                    item
                    for item in target.prioritization.rankings
                    if item.pattern_id == pattern
                )
                candidate = next(item for item in ranking.candidates if item.category == category)
                appearances.append(candidate)
                if any(item.category == category for item in ranking.top_candidates):
                    top_count += 1
                if candidate.tie_group is not None:
                    peers = sum(
                        item.tie_group == candidate.tie_group
                        for item in ranking.candidates
                        if item.rank_eligible
                    )
                    tie_count += int(peers > 1)
            scores = [item.combined_rate for item in appearances if item.combined_rate is not None]
            rows.append(
                {
                    "pattern_id": pattern,
                    "category": category,
                    "target_appearances": len(appearances),
                    "eligible_appearances": sum(item.rank_eligible for item in appearances),
                    "scored_appearances": sum(item.state.value == "scored" for item in appearances),
                    "top_k_appearances": top_count,
                    "rank_1": sum(item.rank_position == 1 for item in appearances),
                    "rank_2": sum(item.rank_position == 2 for item in appearances),
                    "rank_3": sum(item.rank_position == 3 for item in appearances),
                    "tie_appearances": tie_count,
                    "mean_score": None if not scores else sum(scores) / len(scores),
                    "minimum_score": None if not scores else min(scores),
                    "maximum_score": None if not scores else max(scores),
                    "executor_labeled_attempts": sum(item.executor_evidence.labeled_attempts for item in appearances),
                    "opponent_labeled_attempts": sum(item.opponent_allowed_evidence.labeled_attempts for item in appearances),
                    "status": "reconciled",
                }
            )
    return pd.DataFrame(rows, columns=RANKINGS_COLUMNS)


def _diagnostics_frame(result: TacticalPipelineResult) -> pd.DataFrame:
    audit = result.leakage_audit
    values: dict[str, tuple[object, object, bool, str]] = {
        "source_population_reconciled": (result.population.source_point_rows, result.population.development_point_rows + result.population.excluded_test_point_rows, True, "physical rows partitioned before analytical transformation"),
        "test_sealed_before_transformations": (result.test_seal.test_status, "sealed", True, "test read physically but excluded methodologically"),
        "attempt_keys_unique": (len(result.attempts), len({(x.match_id, x.point_number, x.serve_number) for x in result.attempts}), True, "attempt unit"),
        "cache_reconciled": (result.cache_metrics.attempt_requests, result.cache_metrics.cache_hits + result.cache_metrics.cache_misses, True, "local immutable reuse"),
        "actors_reconciled": (len(result.observations), len(result.attempts), True, "server orientation with pattern ownership upstream"),
        "labels_reconciled": (len(result.observations), len(result.attempts), True, "label applied after encoding"),
        "global_only_scope": ("global_only", "global_only", True, "no fallback"),
        "component_only_no_p09_scoring": ("|".join(result.config.requested_patterns), "P02|P04|P05|P06", result.config.encoder_policy is TacticalEncodingPolicy.COMPONENT_ONLY and result.config.requested_patterns == PRODUCTION_PATTERNS, "P09 mask retained but profile features excluded"),
        "rankings_separated_by_pattern": (len(result.config.requested_patterns), len({r.pattern_id for t in result.target_results for r in t.prioritization.rankings}), True, "no global ranking"),
        "target_order_canonical": (result.targets_processed, result.targets_processed, True, "date match orientation"),
        "subquadratic_history_index": (result.diagnostics.full_history_scans_per_target, 0, result.diagnostics.full_history_scans_per_target == 0, "actor/date streams with bisect"),
        "all_contract_reconciliations": (sum(value for _, value in result.reconciliations), len(result.reconciliations), all(value for _, value in result.reconciliations), "closed reconciliations"),
    }
    for field in TacticalLeakageAudit.__dataclass_fields__:
        observed = getattr(audit, field)
        values[field] = (observed, 0, observed == 0, "exclusive target-date history")
    rows = []
    for check in DIAGNOSTIC_CHECK_ORDER:
        observed, expected, passed, details = values[check]
        rows.append(
            {
                "check": check,
                "observed": str(observed),
                "expected": str(expected),
                "passed": bool(passed),
                "severity": "error",
                "details": details,
            }
        )
    return pd.DataFrame(rows, columns=DIAGNOSTICS_COLUMNS)


def _artifact_contract_summary(
    result: TacticalPipelineResult,
    csv_payloads: tuple[tuple[str, bytes], ...],
) -> dict[str, object]:
    fold_targets = [
        {
            "fold": fold,
            "matches": len({item.target.target_match_id for item in result.target_results if item.target.fold == fold}),
            "orientations": sum(item.target.fold == fold for item in result.target_results),
        }
        for fold in VALIDATION_FOLDS
        if any(item.target.fold == fold for item in result.target_results)
    ]
    manifest = {
        name: {"bytes": len(payload), "sha256": sha256(payload).hexdigest().upper()}
        for name, payload in csv_payloads
    }
    summary: dict[str, object] = {
        "analysis_name": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "analysis_status": result.state.value,
        "analysis_purpose": ANALYSIS_PURPOSE,
        "descriptive_policy": DESCRIPTIVE_POLICY_NAME,
        "encoder_policy": result.config.encoder_policy.value,
        "scope": "global_only",
        "minimum_labeled_attempts": result.config.minimum_labeled_attempts,
        "minimum_matches": result.config.minimum_matches,
        "top_k_per_pattern": result.config.top_k,
        "patterns": list(result.config.requested_patterns),
        "excluded_patterns": {"P03": "no_comparator", "P07": "diagnostic_only", "P08": "no_comparator", "P09": "redundant_profile_not_scored"},
        "population": dict(result.population.__dict__),
        "test_seal": {
            "status": result.test_seal.test_status,
            "used_for_method_selection": result.test_seal.used_for_method_selection,
            "excluded_test_point_rows": result.test_seal.excluded_test_point_rows,
            "excluded_test_matches": result.test_seal.excluded_test_matches,
            "counters": dict(result.test_seal.counters),
        },
        "targets": {"unit": "target_match_x_target_player_orientation", "matches": len({item.target.target_match_id for item in result.target_results}), "orientations": result.targets_processed, "folds": fold_targets},
        "cache": dict(result.cache_metrics.__dict__),
        "availability": {"available": result.targets_available, "partially_available": result.targets_partial, "not_available": result.targets_not_available},
        "scoring": {"candidates": result.total_candidates, "scored": result.scored_candidates, "abstained": result.abstained_candidates, "ranking_scope": "separate_by_pattern"},
        "ties": {"boundary_expansions": sum(r.boundary_tie_expanded for t in result.target_results for r in t.prioritization.rankings)},
        "leakage_audit": dict(result.leakage_audit.__dict__),
        "reconciliations": dict(result.reconciliations),
        "upstream_provenance": [dict(zip(("component", "version", "commit", "fingerprint"), item)) for item in UPSTREAM_PROVENANCE],
        "limitations": list(result.limitations),
        "artifact_manifest": manifest,
        "fingerprint_contract_version": FINGERPRINT_CONTRACT_VERSION,
    }
    fingerprint_input = dict(summary)
    summary["publication_fingerprint"] = sha256(
        ARTIFACT_FINGERPRINT_DOMAIN + _json_bytes(fingerprint_input)
    ).hexdigest().upper()
    return summary


def _validate_publication_policy(result: TacticalPipelineResult) -> None:
    if (
        result.config.encoder_policy is not TacticalEncodingPolicy.COMPONENT_ONLY
        or result.config.requested_patterns != PRODUCTION_PATTERNS
        or result.config.minimum_labeled_attempts != 50
        or result.config.minimum_matches != 5
        or result.config.top_k != 3
        or result.config.window_days is not None
        or result.config.serve_numbers != (1, 2)
        or result.config.allow_redundant_audit
    ):
        raise TacticalPipelineContractError("Politica de publicacion 50/5 no esta congelada.")


def _serialize_once(result: TacticalPipelineResult) -> tuple[bytes, ...]:
    _validate_publication_policy(result)
    population = _csv_bytes(_population_frame(result), POPULATION_COLUMNS)
    availability = _csv_bytes(_availability_frame(result), AVAILABILITY_COLUMNS)
    rankings = _csv_bytes(_rankings_frame(result), RANKINGS_COLUMNS)
    diagnostics = _csv_bytes(_diagnostics_frame(result), DIAGNOSTICS_COLUMNS)
    csv_payloads = (
        ("population", population),
        ("availability", availability),
        ("rankings", rankings),
        ("diagnostics", diagnostics),
    )
    summary = _json_bytes(_artifact_contract_summary(result, csv_payloads)) + b"\n"
    return summary, population, availability, rankings, diagnostics


def serialize_tactical_pipeline_artifacts(
    result: TacticalPipelineResult,
) -> tuple[bytes, ...]:
    validate_tactical_pipeline_result(result)
    return _serialize_prevalidated_tactical_pipeline_artifacts(result)


def _serialize_prevalidated_tactical_pipeline_artifacts(
    result: TacticalPipelineResult,
) -> tuple[bytes, ...]:
    """Serializa dos veces un resultado cuya validacion profunda ya termino."""
    if type(result) is not TacticalPipelineResult:
        raise TypeError("result debe ser TacticalPipelineResult exacto.")
    first = _serialize_once(result)
    second = _serialize_once(result)
    if first != second:
        raise TacticalPipelineContractError("La doble serializacion no es determinista.")
    return first


def prepare_tactical_pipeline_payloads(
    result: TacticalPipelineResult,
) -> tuple[tuple[str, bytes], ...]:
    """Prepara los cinco artefactos agregados exactos sin escribirlos."""
    return tuple(zip(ARTIFACT_ORDER, serialize_tactical_pipeline_artifacts(result)))


def _stable_result_structure(result: TacticalPipelineResult) -> dict[str, object]:
    return {
        "summary": _safe_summary(result),
        "population_cache": _population_cache_structure(result),
        "pattern_coverage": _pattern_coverage_structure(result),
        "prioritization_availability": _availability_structure(result),
        "aggregate_rankings": _aggregate_rankings_structure(result),
        "target_prioritization_fingerprints": [
            tactical_prioritization_result_fingerprint(item.prioritization)
            for item in result.target_results
        ],
    }


def tactical_pipeline_result_fingerprint(result: TacticalPipelineResult) -> str:
    """Fingerprint reproducible que excluye deliberadamente los tiempos."""
    validate_tactical_pipeline_result(result)
    return sha256(
        RESULT_FINGERPRINT_DOMAIN + _json_bytes(_stable_result_structure(result))
    ).hexdigest().upper()


def default_artifact_paths(root: Path = ROOT) -> TacticalPipelineArtifactPaths:
    reports = root / "reports"
    tables = reports / "tables"
    return TacticalPipelineArtifactPaths(
        reports / SUMMARY_PATH.name,
        tables / POPULATION_PATH.name,
        tables / AVAILABILITY_PATH.name,
        tables / RANKINGS_PATH.name,
        tables / DIAGNOSTICS_PATH.name,
    )


def _git_read(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git",) + arguments,
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_upstream_ancestry(
    root: Path = ROOT,
    *,
    git_reader: Callable[..., str] = _git_read,
) -> None:
    """Exige fuentes presentes, limpias y commits upstream ancestros."""
    commits = {component: commit for component, _, commit, _ in UPSTREAM_PROVENANCE}
    for component, relative in UPSTREAM_PATHS:
        path = root / relative
        if not path.is_file():
            raise TacticalPipelineContractError("Fuente upstream requerida ausente.")
        try:
            ancestor = git_reader(root, "merge-base", "--is-ancestor", commits[component], "HEAD")
            dirty = git_reader(root, "status", "--short", "--", relative)
        except Exception as error:
            raise TacticalPipelineContractError("No pudo verificarse procedencia upstream.") from error
        if ancestor not in ("", "0") or dirty:
            raise TacticalPipelineContractError("Procedencia upstream modificada o no ancestral.")
    for component, relative, expected_fingerprint in UPSTREAM_ARTIFACTS:
        path = root / relative
        if not path.is_file():
            raise TacticalPipelineContractError("Artefacto upstream requerido ausente.")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            dirty = git_reader(root, "status", "--short", "--", relative)
        except Exception as error:
            raise TacticalPipelineContractError("No pudo verificarse artefacto upstream.") from error
        if (
            type(payload) is not dict
            or payload.get("publication_fingerprint") != expected_fingerprint
            or dirty
        ):
            raise TacticalPipelineContractError("Artefacto upstream modificado o incoherente.")


def _artifact_paths_tuple(paths: TacticalPipelineArtifactPaths) -> tuple[Path, ...]:
    if type(paths) is not TacticalPipelineArtifactPaths:
        raise TypeError("paths debe ser TacticalPipelineArtifactPaths exacto.")
    values = (paths.summary, paths.population, paths.availability, paths.rankings, paths.diagnostics)
    if any(not isinstance(path, Path) for path in values) or len(set(values)) != len(values):
        raise TacticalPipelineContractError("Rutas de artefactos invalidas o duplicadas.")
    return values


def _read_csv_payload(payload: bytes, columns: tuple[str, ...]) -> pd.DataFrame:
    if b"\r" in payload or b"NaN" in payload or b"Infinity" in payload:
        raise TacticalPipelineContractError("CSV persistido contiene representacion invalida.")
    try:
        frame = pd.read_csv(BytesIO(payload), keep_default_na=False)
    except Exception as error:
        raise TacticalPipelineContractError("CSV persistido no puede reabrirse.") from error
    if tuple(frame.columns) != columns or any(name.startswith("Unnamed") for name in frame.columns):
        raise TacticalPipelineContractError("Schema CSV persistido invalido.")
    return frame


def _verify_summary_contract(summary: dict[str, object], csv_payloads: tuple[tuple[str, bytes], ...]) -> None:
    if (
        summary.get("analysis_name") != ANALYSIS_NAME
        or summary.get("analysis_version") != ANALYSIS_VERSION
        or summary.get("analysis_purpose") != ANALYSIS_PURPOSE
        or summary.get("descriptive_policy") != DESCRIPTIVE_POLICY_NAME
        or summary.get("encoder_policy") != "component_only"
        or summary.get("scope") != "global_only"
        or summary.get("minimum_labeled_attempts") != 50
        or summary.get("minimum_matches") != 5
        or summary.get("top_k_per_pattern") != 3
        or summary.get("patterns") != list(PRODUCTION_PATTERNS)
        or summary.get("fingerprint_contract_version") != FINGERPRINT_CONTRACT_VERSION
    ):
        raise TacticalPipelineContractError("Contrato metodologico persistido invalido.")
    test_seal = summary.get("test_seal")
    if (
        type(test_seal) is not dict
        or test_seal.get("status") != "sealed"
        or test_seal.get("used_for_method_selection") is not False
        or type(test_seal.get("excluded_test_point_rows")) is not int
        or test_seal.get("excluded_test_point_rows") < 0
        or type(test_seal.get("excluded_test_matches")) is not int
        or test_seal.get("excluded_test_matches") < 0
        or test_seal.get("counters") != dict((field, 0) for field in TEST_ZERO_FIELDS)
    ):
        raise TacticalPipelineContractError("Sellado persistido invalido.")
    leakage = summary.get("leakage_audit")
    if type(leakage) is not dict or leakage != {field: 0 for field in TacticalLeakageAudit.__dataclass_fields__}:
        raise TacticalPipelineContractError("Auditoria leakage persistida invalida.")
    manifest = summary.get("artifact_manifest")
    expected_manifest = {
        name: {"bytes": len(payload), "sha256": sha256(payload).hexdigest().upper()}
        for name, payload in csv_payloads
    }
    if manifest != expected_manifest:
        raise TacticalPipelineContractError("Hashes o tamanos persistidos invalidos.")
    fingerprint = summary.get("publication_fingerprint")
    unsigned = dict(summary)
    unsigned.pop("publication_fingerprint", None)
    expected = sha256(ARTIFACT_FINGERPRINT_DOMAIN + _json_bytes(unsigned)).hexdigest().upper()
    if fingerprint != expected:
        raise TacticalPipelineContractError("Fingerprint persistido invalido.")


def _artifact_ratio_matches(value: object, numerator: int, denominator: int) -> bool:
    if denominator == 0:
        return value == "" or value is None
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and isfinite(float(value))
        and abs(float(value) - numerator / denominator) <= 5e-15
    )


def _artifact_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TacticalPipelineContractError(f"{field} persistido no es entero.")
    converted = int(value)
    if converted < 0:
        raise TacticalPipelineContractError(f"{field} persistido es negativo.")
    return converted


def _artifact_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise TacticalPipelineContractError(f"{field} persistido no es numerico.")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise TacticalPipelineContractError(
            f"{field} persistido no es numerico."
        ) from error
    if not isfinite(converted):
        raise TacticalPipelineContractError(f"{field} persistido no es finito.")
    return converted


def _verify_available_aggregate_contract(
    summary: dict[str, object],
    frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Reconcilia el contrato publicado usando solo los cinco agregados."""
    population, availability, rankings, diagnostics = frames
    expected_summary_keys = {
        "analysis_name", "analysis_version", "analysis_status", "analysis_purpose",
        "descriptive_policy", "encoder_policy", "scope", "minimum_labeled_attempts",
        "minimum_matches", "top_k_per_pattern", "patterns", "excluded_patterns",
        "population", "test_seal", "targets", "cache", "availability", "scoring",
        "ties", "leakage_audit", "reconciliations", "upstream_provenance",
        "limitations", "artifact_manifest", "fingerprint_contract_version",
        "publication_fingerprint",
    }
    if set(summary) != expected_summary_keys:
        raise TacticalPipelineContractError("Schema JSON agregado invalido.")
    if summary.get("analysis_status") not in {
        "available", "partially_available", "not_available"
    }:
        raise TacticalPipelineContractError("Estado analitico agregado invalido.")
    if summary.get("excluded_patterns") != {
        "P03": "no_comparator",
        "P07": "diagnostic_only",
        "P08": "no_comparator",
        "P09": "redundant_profile_not_scored",
    }:
        raise TacticalPipelineContractError("Patrones excluidos no reconcilian.")
    if summary.get("reconciliations") != dict(PIPELINE_RECONCILIATIONS):
        raise TacticalPipelineContractError("Reconciliaciones JSON invalidas.")
    if summary.get("limitations") != list(PIPELINE_LIMITATIONS):
        raise TacticalPipelineContractError("Limitaciones JSON invalidas.")
    expected_provenance = [
        dict(zip(("component", "version", "commit", "fingerprint"), item))
        for item in UPSTREAM_PROVENANCE
    ]
    if summary.get("upstream_provenance") != expected_provenance:
        raise TacticalPipelineContractError("Procedencia JSON invalida.")

    population_summary = summary.get("population")
    cache = summary.get("cache")
    targets = summary.get("targets")
    if not all(type(value) is dict for value in (population_summary, cache, targets)):
        raise TacticalPipelineContractError("Bloques de poblacion agregada invalidos.")
    expected_population_fields = {
        "source_point_rows", "source_matches", "development_point_rows",
        "development_matches", "excluded_test_point_rows", "excluded_test_matches",
        "attempt_count", "first_attempt_count", "second_attempt_count",
        "second_with_documented_first_fault", "second_without_documented_first_fault",
        "observation_count",
    }
    expected_cache_fields = {
        "attempt_requests", "cache_hits", "cache_misses", "unique_keys",
        "orchestrator_calls", "encoder_calls", "hit_rate",
    }
    if set(population_summary) != expected_population_fields or set(cache) != expected_cache_fields:
        raise TacticalPipelineContractError("Schema de poblacion o cache invalido.")
    for key, value in population_summary.items():
        _artifact_int(value, key)
    for key in expected_cache_fields - {"hit_rate"}:
        _artifact_int(cache[key], key)
    if not _artifact_ratio_matches(
        cache["hit_rate"], cache["cache_hits"], cache["attempt_requests"]
    ):
        raise TacticalPipelineContractError("Hit rate no reconcilia.")
    if (
        population_summary["source_point_rows"]
        != population_summary["development_point_rows"]
        + population_summary["excluded_test_point_rows"]
        or population_summary["source_matches"]
        != population_summary["development_matches"]
        + population_summary["excluded_test_matches"]
        or population_summary["attempt_count"]
        != population_summary["first_attempt_count"]
        + population_summary["second_attempt_count"]
        or population_summary["second_attempt_count"]
        != population_summary["second_with_documented_first_fault"]
        + population_summary["second_without_documented_first_fault"]
        or population_summary["observation_count"] != population_summary["attempt_count"]
        or cache["attempt_requests"] != population_summary["attempt_count"]
        or cache["attempt_requests"] != cache["cache_hits"] + cache["cache_misses"]
        or cache["cache_misses"] != cache["unique_keys"]
        or cache["unique_keys"] != cache["orchestrator_calls"]
        or cache["unique_keys"] != cache["encoder_calls"]
    ):
        raise TacticalPipelineContractError("Poblacion o cache no reconcilia.")

    target_keys = {"unit", "matches", "orientations", "folds"}
    if set(targets) != target_keys or targets["unit"] != "target_match_x_target_player_orientation":
        raise TacticalPipelineContractError("Contrato de targets invalido.")
    target_matches = _artifact_int(targets["matches"], "target matches")
    target_orientations = _artifact_int(targets["orientations"], "target orientations")
    if target_orientations != 2 * target_matches or type(targets["folds"]) is not list:
        raise TacticalPipelineContractError("Orientaciones target no reconcilian.")
    expected_fold_rows = []
    fold_matches = 0
    fold_orientations = 0
    for item in targets["folds"]:
        if type(item) is not dict or set(item) != {"fold", "matches", "orientations"}:
            raise TacticalPipelineContractError("Fold JSON invalido.")
        fold = item["fold"]
        matches = _artifact_int(item["matches"], "fold matches")
        orientations = _artifact_int(item["orientations"], "fold orientations")
        if fold not in VALIDATION_FOLDS or orientations != 2 * matches:
            raise TacticalPipelineContractError("Fold target no reconcilia.")
        expected_fold_rows.append(fold)
        fold_matches += matches
        fold_orientations += orientations
    if (
        expected_fold_rows != [fold for fold in VALIDATION_FOLDS if fold in expected_fold_rows]
        or len(set(expected_fold_rows)) != len(expected_fold_rows)
        or fold_matches != target_matches
        or fold_orientations != target_orientations
    ):
        raise TacticalPipelineContractError("Folds target no son exhaustivos o canonicos.")

    if tuple(population["metric"]) != POPULATION_ROW_ORDER or bool(population["metric"].duplicated().any()):
        raise TacticalPipelineContractError("Orden o clave population invalido.")
    population_expected = {
        "source": (population_summary["source_point_rows"], population_summary["source_point_rows"]),
        "development": (population_summary["development_point_rows"], population_summary["source_point_rows"]),
        "excluded_test": (population_summary["excluded_test_point_rows"], population_summary["source_point_rows"]),
        "attempts_total": (population_summary["attempt_count"], population_summary["development_point_rows"]),
        "first_serve_attempts": (population_summary["first_attempt_count"], population_summary["development_point_rows"]),
        "second_serve_attempts": (population_summary["second_attempt_count"], population_summary["development_point_rows"]),
        "second_with_documented_fault": (population_summary["second_with_documented_first_fault"], population_summary["second_attempt_count"]),
        "second_without_documented_fault": (population_summary["second_without_documented_first_fault"], population_summary["second_attempt_count"]),
        "cache_entries": (cache["unique_keys"], population_summary["attempt_count"]),
        "cache_hits": (cache["cache_hits"], population_summary["attempt_count"]),
        "cache_misses": (cache["cache_misses"], population_summary["attempt_count"]),
        "observations": (population_summary["observation_count"], population_summary["attempt_count"]),
        "validation_matches": (target_matches, population_summary["development_matches"]),
        "target_orientations": (target_orientations, 2 * target_matches),
    }
    for row in population.to_dict("records"):
        metric = row["metric"]
        count, denominator = population_expected[metric]
        if (
            row["status"] != "reconciled"
            or _artifact_int(row["count"], "population count") != count
            or _artifact_int(row["denominator"], "population denominator") != denominator
            or not _artifact_ratio_matches(row["ratio"], count, denominator)
        ):
            raise TacticalPipelineContractError("Fila population no reconcilia.")

    expected_availability_keys = (
        (("total", "", ""),)
        + tuple(("fold", fold, "") for fold in expected_fold_rows)
        + tuple(("pattern", "", pattern) for pattern in PRODUCTION_PATTERNS)
        + tuple(
            ("fold_pattern", fold, pattern)
            for fold in expected_fold_rows
            for pattern in PRODUCTION_PATTERNS
        )
    )
    actual_availability_keys = tuple(
        availability[["aggregation_level", "fold", "pattern_id"]]
        .itertuples(index=False, name=None)
    )
    if actual_availability_keys != expected_availability_keys or len(set(actual_availability_keys)) != len(actual_availability_keys):
        raise TacticalPipelineContractError("Claves availability no son exactas y canonicas.")
    count_fields = (
        "target_orientations", "available", "partially_available", "not_available",
        "scored_candidates", "abstained_candidates",
    )
    for row in availability.to_dict("records"):
        for field in count_fields:
            _artifact_int(row[field], f"availability {field}")
        if (
            row["status"] != "reconciled"
            or row["available"] + row["partially_available"] + row["not_available"]
            != row["target_orientations"]
            or not _artifact_ratio_matches(
                row["coverage"],
                row["available"] + row["partially_available"],
                row["target_orientations"],
            )
        ):
            raise TacticalPipelineContractError("Fila availability no reconcilia.")
    total_availability = availability.iloc[0]
    summary_availability = summary.get("availability")
    summary_scoring = summary.get("scoring")
    if (
        type(summary_availability) is not dict
        or set(summary_availability) != {"available", "partially_available", "not_available"}
        or type(summary_scoring) is not dict
        or set(summary_scoring) != {"candidates", "scored", "abstained", "ranking_scope"}
        or summary_scoring["ranking_scope"] != "separate_by_pattern"
        or tuple(summary_availability[key] for key in ("available", "partially_available", "not_available"))
        != tuple(int(total_availability[key]) for key in ("available", "partially_available", "not_available"))
        or summary_scoring["candidates"] != summary_scoring["scored"] + summary_scoring["abstained"]
        or summary_scoring["scored"] != int(total_availability["scored_candidates"])
        or summary_scoring["abstained"] != int(total_availability["abstained_candidates"])
        or int(total_availability["target_orientations"]) != target_orientations
    ):
        raise TacticalPipelineContractError("Availability/scoring JSON no reconcilia.")
    fold_rows = availability[availability.aggregation_level == "fold"]
    for fold_item, row in zip(targets["folds"], fold_rows.to_dict("records")):
        if row["fold"] != fold_item["fold"] or int(row["target_orientations"]) != fold_item["orientations"]:
            raise TacticalPipelineContractError("Availability por fold no reconcilia targets.")
    for fields in ("target_orientations", "available", "partially_available", "not_available", "scored_candidates", "abstained_candidates"):
        if int(fold_rows[fields].sum()) != int(total_availability[fields]):
            raise TacticalPipelineContractError("Availability por fold no reconcilia total.")
    pattern_rows = availability[availability.aggregation_level == "pattern"]
    fold_pattern_rows = availability[availability.aggregation_level == "fold_pattern"]
    for pattern_row in pattern_rows.to_dict("records"):
        subset = fold_pattern_rows[
            fold_pattern_rows.pattern_id == pattern_row["pattern_id"]
        ]
        for field in count_fields:
            if int(subset[field].sum()) != int(pattern_row[field]):
                raise TacticalPipelineContractError("Availability fold-pattern no reconcilia patron.")
    for fold_row in fold_rows.to_dict("records"):
        fold = fold_row["fold"]
        subset = fold_pattern_rows[fold_pattern_rows.fold == fold]
        if any(
            int(row["target_orientations"]) != int(fold_row["target_orientations"])
            for row in subset.to_dict("records")
        ):
            raise TacticalPipelineContractError("Denominador fold-pattern no reconcilia.")

    ranking_keys = tuple(rankings[["pattern_id", "category"]].itertuples(index=False, name=None))
    if ranking_keys != PUBLISHED_RANKING_KEY_ORDER or len(set(ranking_keys)) != len(ranking_keys):
        raise TacticalPipelineContractError("Claves rankings no son exactas y canonicas.")
    ranking_count_fields = (
        "target_appearances", "eligible_appearances", "scored_appearances",
        "top_k_appearances", "rank_1", "rank_2", "rank_3", "tie_appearances",
        "executor_labeled_attempts", "opponent_labeled_attempts",
    )
    for row in rankings.to_dict("records"):
        for field in ranking_count_fields:
            _artifact_int(row[field], f"rankings {field}")
        eligible = row["eligible_appearances"]
        rank_total = row["rank_1"] + row["rank_2"] + row["rank_3"]
        if (
            row["status"] != "reconciled"
            or row["target_appearances"] != target_orientations
            or row["scored_appearances"] != eligible
            or row["top_k_appearances"] != rank_total
            or row["top_k_appearances"] > eligible
        ):
            raise TacticalPipelineContractError("Conteos rankings no reconcilian.")
        scores = (row["minimum_score"], row["mean_score"], row["maximum_score"])
        if eligible == 0:
            if any(value != "" for value in scores):
                raise TacticalPipelineContractError("Ranking sin evidencia publica score.")
        else:
            numeric_scores = tuple(
                _artifact_float(value, "ranking score") for value in scores
            )
            if not (
                0 <= numeric_scores[0] <= numeric_scores[1] <= numeric_scores[2] <= 1
            ):
                raise TacticalPipelineContractError("Scores rankings invalidos.")
    category_counts = {
        pattern: sum(key[0] == pattern for key in PUBLISHED_RANKING_KEY_ORDER)
        for pattern in PRODUCTION_PATTERNS
    }
    for pattern_row in pattern_rows.to_dict("records"):
        pattern = pattern_row["pattern_id"]
        subset = rankings[rankings.pattern_id == pattern]
        if (
            int(subset.eligible_appearances.sum()) != int(pattern_row["scored_candidates"])
            or target_orientations * category_counts[pattern]
            - int(subset.eligible_appearances.sum())
            != int(pattern_row["abstained_candidates"])
            or int(subset.rank_1.sum())
            != int(pattern_row["available"] + pattern_row["partially_available"])
        ):
            raise TacticalPipelineContractError("Rankings no reconcilian availability por patron.")
    if (
        int(rankings.eligible_appearances.sum()) != summary_scoring["scored"]
        or target_orientations * len(PUBLISHED_RANKING_KEY_ORDER) != summary_scoring["candidates"]
        or summary.get("ties") != {"boundary_expansions": 0}
        or int(rankings.tie_appearances.sum()) != 0
    ):
        raise TacticalPipelineContractError("Rankings globales o empates no reconcilian.")

    if tuple(diagnostics["check"]) != DIAGNOSTIC_CHECK_ORDER or bool(diagnostics["check"].duplicated().any()):
        raise TacticalPipelineContractError("Diagnosticos no son exactos y canonicos.")
    if not bool(diagnostics["passed"].map(lambda value: value is True).all()):
        raise TacticalPipelineContractError("Diagnostico persistido no superado.")
    expected_diagnostics: dict[str, tuple[str, str]] = {
        "source_population_reconciled": (
            str(population_summary["source_point_rows"]),
            str(population_summary["development_point_rows"] + population_summary["excluded_test_point_rows"]),
        ),
        "test_sealed_before_transformations": ("sealed", "sealed"),
        "attempt_keys_unique": (str(population_summary["attempt_count"]), str(population_summary["attempt_count"])),
        "cache_reconciled": (str(cache["attempt_requests"]), str(cache["cache_hits"] + cache["cache_misses"])),
        "actors_reconciled": (str(population_summary["observation_count"]), str(population_summary["attempt_count"])),
        "labels_reconciled": (str(population_summary["observation_count"]), str(population_summary["attempt_count"])),
        "global_only_scope": ("global_only", "global_only"),
        "component_only_no_p09_scoring": ("|".join(PRODUCTION_PATTERNS), "|".join(PRODUCTION_PATTERNS)),
        "rankings_separated_by_pattern": (str(len(PRODUCTION_PATTERNS)), str(len(PRODUCTION_PATTERNS))),
        "target_order_canonical": (str(target_orientations), str(target_orientations)),
        "subquadratic_history_index": ("0", "0"),
        "all_contract_reconciliations": (str(len(PIPELINE_RECONCILIATIONS)), str(len(PIPELINE_RECONCILIATIONS))),
    }
    for field in TacticalLeakageAudit.__dataclass_fields__:
        expected_diagnostics[field] = ("0", "0")
    for row in diagnostics.to_dict("records"):
        if (
            (str(row["observed"]), str(row["expected"])) != expected_diagnostics[row["check"]]
            or row["severity"] != "error"
            or type(row["details"]) is not str
            or not row["details"]
        ):
            raise TacticalPipelineContractError("Detalle diagnostico no reconcilia.")

    available_count = int(total_availability["available"] + total_availability["partially_available"])
    expected_status = (
        "not_available"
        if available_count == 0
        else "available"
        if int(total_availability["not_available"]) == 0
        else "partially_available"
    )
    if summary["analysis_status"] != expected_status:
        raise TacticalPipelineContractError("Estado analitico no reconcilia availability.")


def _read_and_verify_aggregate_artifacts(
    paths: TacticalPipelineArtifactPaths,
    *,
    enforce_frozen_bytes: bool,
) -> tuple[tuple[bytes, ...], dict[str, object]]:
    path_values = _artifact_paths_tuple(paths)
    try:
        raw = tuple(path.read_bytes() for path in path_values)
    except OSError as error:
        raise TacticalPipelineContractError("Artefacto persistido ausente o ilegible.") from error
    if enforce_frozen_bytes:
        for payload, (name, expected_size, expected_hash) in zip(
            raw, PUBLISHED_ARTIFACT_CONTRACT
        ):
            if len(payload) != expected_size or sha256(payload).hexdigest().upper() != expected_hash:
                raise TacticalPipelineContractError(
                    f"Artefacto P10 congelado no coincide: {name}."
                )
    joined = b"\n".join(raw)
    if any(token in joined for token in (b"match_id", b"point_number", b"sequence_text", b"NaN", b"Infinity")):
        raise TacticalPipelineContractError("Artefactos contienen datos o literales prohibidos.")
    decoded = joined.decode("utf-8")
    if re.search(r"[A-Za-z]:[\\/]", decoded) or "file://" in decoded.lower():
        raise TacticalPipelineContractError("Artefactos contienen rutas absolutas.")
    try:
        summary = json.loads(raw[0].decode("utf-8"))
    except Exception as error:
        raise TacticalPipelineContractError("Summary persistido invalido.") from error
    if type(summary) is not dict or _json_bytes(summary) + b"\n" != raw[0]:
        raise TacticalPipelineContractError("Summary no usa serializacion canonica.")
    frames = (
        _read_csv_payload(raw[1], POPULATION_COLUMNS),
        _read_csv_payload(raw[2], AVAILABILITY_COLUMNS),
        _read_csv_payload(raw[3], RANKINGS_COLUMNS),
        _read_csv_payload(raw[4], DIAGNOSTICS_COLUMNS),
    )
    for payload, frame, columns in zip(
        raw[1:],
        frames,
        (POPULATION_COLUMNS, AVAILABILITY_COLUMNS, RANKINGS_COLUMNS, DIAGNOSTICS_COLUMNS),
    ):
        if _csv_bytes(frame, columns) != payload:
            raise TacticalPipelineContractError("CSV no usa orden o serializacion canonicos.")
    _verify_summary_contract(summary, tuple(zip(ARTIFACT_ORDER[1:], raw[1:])))
    _verify_available_aggregate_contract(summary, frames)
    _validate_public_tree(summary)
    if enforce_frozen_bytes and summary.get("publication_fingerprint") != PUBLISHED_PUBLICATION_FINGERPRINT:
        raise TacticalPipelineContractError("Fingerprint P10 congelado no coincide.")
    return raw, summary


def verify_persisted_artifacts_only(
    paths: TacticalPipelineArtifactPaths | None = None,
    *,
    enforce_frozen_bytes: bool = True,
) -> dict[str, object]:
    """Valida P10 sin datos fuente ni reconstruccion de objetos analiticos."""
    selected_paths = default_artifact_paths() if paths is None else paths
    _, summary = _read_and_verify_aggregate_artifacts(
        selected_paths, enforce_frozen_bytes=enforce_frozen_bytes
    )
    return summary


def verify_persisted_artifacts(
    paths: TacticalPipelineArtifactPaths,
    expected_result: TacticalPipelineResult | TacticalPipelineUnavailableResult,
    *,
    expected_payloads: tuple[bytes, ...] | None = None,
) -> None:
    """Reabre, valida semanticamente y compara con el resultado autoritativo."""
    path_values = _artifact_paths_tuple(paths)
    try:
        raw = tuple(path.read_bytes() for path in path_values)
    except OSError as error:
        raise TacticalPipelineContractError("Artefacto persistido ausente o ilegible.") from error
    if expected_payloads is None:
        expected = (
            serialize_tactical_pipeline_artifacts(expected_result)
            if type(expected_result) is TacticalPipelineResult
            else serialize_tactical_pipeline_unavailable_artifacts(expected_result)
        )
    else:
        if (
            type(expected_payloads) is not tuple
            or len(expected_payloads) != len(ARTIFACT_ORDER)
            or any(type(payload) is not bytes for payload in expected_payloads)
        ):
            raise TacticalPipelineContractError(
                "Payloads esperados fuera del contrato."
            )
        expected = expected_payloads
    if raw != expected:
        raise TacticalPipelineContractError("Artefactos persistidos no coinciden con el resultado validado.")
    try:
        summary = json.loads(raw[0].decode("utf-8"))
    except Exception as error:
        raise TacticalPipelineContractError("Summary persistido invalido.") from error
    if type(summary) is not dict:
        raise TacticalPipelineContractError("Summary debe ser objeto JSON.")
    frames = (
        _read_csv_payload(raw[1], POPULATION_COLUMNS),
        _read_csv_payload(raw[2], AVAILABILITY_COLUMNS),
        _read_csv_payload(raw[3], RANKINGS_COLUMNS),
        _read_csv_payload(raw[4], DIAGNOSTICS_COLUMNS),
    )
    csv_payloads = tuple(zip(ARTIFACT_ORDER[1:], raw[1:]))
    _verify_summary_contract(summary, csv_payloads)
    if type(expected_result) is TacticalPipelineUnavailableResult:
        if any(not frame.empty for frame in frames):
            raise TacticalPipelineContractError("not_available debe publicar CSV solo cabecera.")
        return
    if expected_result.config.expected_source_rows == EXPECTED_SOURCE_ROWS:
        _verify_available_aggregate_contract(summary, frames)
    else:
        population, availability, rankings, diagnostics = frames
        if tuple(population.metric) != POPULATION_ROW_ORDER or bool(population.metric.duplicated().any()):
            raise TacticalPipelineContractError("Orden o clave population invalido.")
        availability_keys = availability[["aggregation_level", "fold", "pattern_id"]]
        if bool(availability_keys.duplicated().any()) or not set(availability.aggregation_level).issubset(AVAILABILITY_LEVEL_ORDER):
            raise TacticalPipelineContractError("Claves availability invalidas.")
        if bool(rankings[["pattern_id", "category"]].duplicated().any()) or not set(rankings.pattern_id).issubset(PRODUCTION_PATTERNS):
            raise TacticalPipelineContractError("Claves rankings invalidas.")
        if tuple(diagnostics.check) != DIAGNOSTIC_CHECK_ORDER or not bool(diagnostics.passed.map(lambda value: value is True or value == "True").all()):
            raise TacticalPipelineContractError("Diagnosticos persistidos invalidos.")


def build_not_available_publication(
    error: TacticalPipelineExecutionError,
    test_seal: TacticalTestSeal,
    population: TacticalPipelinePopulation | None = None,
) -> TacticalPipelineUnavailableResult:
    if type(error) is not TacticalPipelineExecutionError:
        raise TypeError("error debe ser TacticalPipelineExecutionError exacto.")
    if type(test_seal) is not TacticalTestSeal:
        raise TypeError("test_seal debe ser TacticalTestSeal exacto.")
    if (
        test_seal.test_status != "sealed"
        or test_seal.used_for_method_selection
        or test_seal.counters != tuple((field, 0) for field in TEST_ZERO_FIELDS)
    ):
        raise TacticalPipelineContractError("No puede publicarse fallo sin test sellado.")
    if population is not None:
        _validate_population(population)
    return TacticalPipelineUnavailableResult(
        PIPELINE_CONTRACT_VERSION,
        ANALYSIS_NAME,
        ANALYSIS_VERSION,
        "not_available",
        ("analysis_failed_closed", f"{error.stage}_failed"),
        error.stage,
        error.error_type,
        error.sanitized_message,
        population,
        test_seal,
        PIPELINE_LIMITATIONS,
    )


def _unavailable_summary(
    result: TacticalPipelineUnavailableResult,
    csv_payloads: tuple[tuple[str, bytes], ...],
) -> dict[str, object]:
    if (
        type(result) is not TacticalPipelineUnavailableResult
        or result.contract_version != PIPELINE_CONTRACT_VERSION
        or result.analysis_name != ANALYSIS_NAME
        or result.analysis_version != ANALYSIS_VERSION
        or result.state != "not_available"
        or result.reason_codes
        != ("analysis_failed_closed", f"{result.failure_stage}_failed")
        or result.failure_stage not in FAILURE_STAGE_ORDER
        or result.sanitized_message != f"{result.failure_stage}_failed:{result.error_type}"
        or result.test_seal.counters != tuple((field, 0) for field in TEST_ZERO_FIELDS)
    ):
        raise TacticalPipelineContractError("Resultado not_available invalido.")
    manifest = {
        name: {"bytes": len(payload), "sha256": sha256(payload).hexdigest().upper()}
        for name, payload in csv_payloads
    }
    summary: dict[str, object] = {
        "analysis_name": ANALYSIS_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "analysis_status": "not_available",
        "analysis_purpose": ANALYSIS_PURPOSE,
        "descriptive_policy": DESCRIPTIVE_POLICY_NAME,
        "encoder_policy": "component_only",
        "scope": "global_only",
        "minimum_labeled_attempts": 50,
        "minimum_matches": 5,
        "top_k_per_pattern": 3,
        "patterns": list(PRODUCTION_PATTERNS),
        "failure": {"stage": result.failure_stage, "type": result.error_type, "message": result.sanitized_message},
        "reason_codes": list(result.reason_codes),
        "population": None if result.population is None else dict(result.population.__dict__),
        "test_seal": {
            "status": result.test_seal.test_status,
            "used_for_method_selection": False,
            "excluded_test_point_rows": result.test_seal.excluded_test_point_rows,
            "excluded_test_matches": result.test_seal.excluded_test_matches,
            "counters": dict(result.test_seal.counters),
        },
        "leakage_audit": {field: 0 for field in TacticalLeakageAudit.__dataclass_fields__},
        "artifact_manifest": manifest,
        "limitations": list(result.limitations),
        "fingerprint_contract_version": FINGERPRINT_CONTRACT_VERSION,
    }
    unsigned = dict(summary)
    summary["publication_fingerprint"] = sha256(
        ARTIFACT_FINGERPRINT_DOMAIN + _json_bytes(unsigned)
    ).hexdigest().upper()
    return summary


def serialize_tactical_pipeline_unavailable_artifacts(
    result: TacticalPipelineUnavailableResult,
) -> tuple[bytes, ...]:
    empty_payloads = (
        _csv_bytes(pd.DataFrame(columns=POPULATION_COLUMNS), POPULATION_COLUMNS),
        _csv_bytes(pd.DataFrame(columns=AVAILABILITY_COLUMNS), AVAILABILITY_COLUMNS),
        _csv_bytes(pd.DataFrame(columns=RANKINGS_COLUMNS), RANKINGS_COLUMNS),
        _csv_bytes(pd.DataFrame(columns=DIAGNOSTICS_COLUMNS), DIAGNOSTICS_COLUMNS),
    )
    csv_payloads = tuple(zip(ARTIFACT_ORDER[1:], empty_payloads))
    summary = _json_bytes(_unavailable_summary(result, csv_payloads)) + b"\n"
    first = (summary,) + empty_payloads
    second_summary = _json_bytes(_unavailable_summary(result, csv_payloads)) + b"\n"
    second = (second_summary,) + empty_payloads
    if first != second:
        raise TacticalPipelineContractError("Serializacion not_available no determinista.")
    return first


def publish_tactical_pipeline_artifacts(
    result: TacticalPipelineResult | TacticalPipelineUnavailableResult,
    paths: TacticalPipelineArtifactPaths,
    *,
    serializer: Callable[[object], tuple[bytes, ...]] | None = None,
    prepared_payloads: tuple[bytes, ...] | None = None,
    replace_operation: Callable[[str | Path, str | Path], None] = os.replace,
    verifier: Callable[[TacticalPipelineArtifactPaths, object], None] = verify_persisted_artifacts,
    payload_writer: Callable[[Path, bytes], None] | None = None,
) -> None:
    """Publica cinco payloads con rollback ante fallos capturados."""
    path_values = _artifact_paths_tuple(paths)
    if serializer is not None and prepared_payloads is not None:
        raise TacticalPipelineContractError(
            "serializer y prepared_payloads son alternativas excluyentes."
        )
    selected_serializer = serializer or (
        serialize_tactical_pipeline_artifacts
        if type(result) is TacticalPipelineResult
        else serialize_tactical_pipeline_unavailable_artifacts
    )
    try:
        payloads = (
            selected_serializer(result)
            if prepared_payloads is None
            else prepared_payloads
        )
        if type(payloads) is not tuple or len(payloads) != len(path_values) or any(type(item) is not bytes for item in payloads):
            raise TacticalPipelineContractError("Serializer no produjo cinco payloads bytes.")
        if serializer is not None and payloads != selected_serializer(result):
            raise TacticalPipelineContractError("Serializer no determinista antes de staging.")
    except TacticalPipelineExecutionError:
        raise
    except Exception as error:
        raise TacticalPipelineExecutionError("serialization", type(error).__name__) from error
    writer = payload_writer or (lambda path, payload: path.write_bytes(payload))
    temporary_paths: list[Path] = []
    backup_paths: list[Path | None] = []
    replaced = 0

    def rollback_replaced() -> None:
        for index in range(replaced - 1, -1, -1):
            destination = path_values[index]
            backup = backup_paths[index] if index < len(backup_paths) else None
            if backup is None:
                if destination.exists():
                    destination.unlink()
            else:
                os.replace(backup, destination)
                backup_paths[index] = None

    try:
        for destination, payload in zip(path_values, payloads):
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            temporary_paths.append(temporary)
            writer(temporary, payload)
        staged = TacticalPipelineArtifactPaths(*temporary_paths)
        if verifier is verify_persisted_artifacts:
            verifier(staged, result, expected_payloads=payloads)
        else:
            verifier(staged, result)
        for destination in path_values:
            if destination.exists():
                descriptor, backup_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.", suffix=".bak", dir=destination.parent
                )
                os.close(descriptor)
                backup = Path(backup_name)
                backup.write_bytes(destination.read_bytes())
                backup_paths.append(backup)
            else:
                backup_paths.append(None)
        for temporary, destination in zip(temporary_paths, path_values):
            replace_operation(temporary, destination)
            replaced += 1
        if verifier is verify_persisted_artifacts:
            verifier(paths, result, expected_payloads=payloads)
        else:
            verifier(paths, result)
    except KeyboardInterrupt:
        rollback_replaced()
        raise
    except Exception as error:
        rollback_replaced()
        if isinstance(error, TacticalPipelineExecutionError):
            raise
        raise TacticalPipelineExecutionError("publication", type(error).__name__) from error
    finally:
        for path in temporary_paths:
            if path.exists():
                path.unlink()
        for backup in backup_paths:
            if backup is not None and backup.exists():
                backup.unlink()


def performance_log_payload(
    result: TacticalPipelineResult | TacticalPipelineUnavailableResult,
    duration_seconds: float,
    *,
    publication_failure: TacticalPipelineExecutionError | None = None,
    partial_stage_seconds: dict[str, float] | None = None,
) -> bytes:
    if type(duration_seconds) is not float or not isfinite(duration_seconds) or duration_seconds < 0:
        raise TacticalPipelineContractError("duration_seconds invalido.")
    if type(result) is TacticalPipelineResult:
        validate_tactical_pipeline_result(result)
        analysis_status = result.state.value
        stage_seconds = dict(result.diagnostics.stage_seconds)
        counts = {
            "points": result.population.source_point_rows,
            "attempts": result.population.attempt_count,
            "targets": result.targets_processed,
        }
        analysis_failure = None
    elif type(result) is TacticalPipelineUnavailableResult:
        # La serializacion aplica el contrato completo del fallo cerrado.
        serialize_tactical_pipeline_unavailable_artifacts(result)
        analysis_status = "not_available"
        if partial_stage_seconds is None:
            stage_seconds = {}
        else:
            if type(partial_stage_seconds) is not dict or any(
                key not in STAGE_ORDER
                or type(value) is not float
                or not isfinite(value)
                or value < 0.0
                for key, value in partial_stage_seconds.items()
            ):
                raise TacticalPipelineContractError(
                    "partial_stage_seconds contiene tiempos invalidos."
                )
            stage_seconds = {
                stage: partial_stage_seconds[stage]
                for stage in STAGE_ORDER
                if stage in partial_stage_seconds
            }
        counts = (
            {}
            if result.population is None
            else {
                "points": result.population.source_point_rows,
                "attempts": result.population.attempt_count,
                "targets": 0,
            }
        )
        analysis_failure = {
            "stage": result.failure_stage,
            "type": result.error_type,
            "message": result.sanitized_message,
        }
    else:
        raise TypeError("result debe usar un tipo de publicacion contractual.")
    payload: dict[str, object] = {
        "analysis_name": ANALYSIS_NAME,
        "analysis_status": analysis_status,
        "duration_seconds": duration_seconds,
        "stage_seconds": stage_seconds,
        "counts": counts,
    }
    if analysis_failure is not None:
        payload["analysis_failure"] = analysis_failure
    if publication_failure is not None:
        payload["analysis_status"] = "publication_failed"
        payload["publication_failure"] = {"stage": publication_failure.stage, "type": publication_failure.error_type, "message": publication_failure.sanitized_message}
    return _json_bytes(payload) + b"\n"


_PERFORMANCE_LOG_FIELDS: Final = frozenset(
    {
        "analysis_name",
        "execution_status",
        "current_stage",
        "completed_stages",
        "elapsed_seconds",
        "stage_seconds",
        "targets_total",
        "targets_processed",
        "targets_remaining",
        "progress_fraction",
        "attempts_processed",
        "unique_cache_entries",
        "observations_indexed",
        "operation_counters",
        "failure",
        "reason_code",
    }
)


def _incremental_performance_bytes(payload: dict[str, object]) -> bytes:
    if type(payload) is not dict or not set(payload).issubset(_PERFORMANCE_LOG_FIELDS):
        raise TacticalPipelineContractError("Performance log fuera del schema permitido.")
    required = _PERFORMANCE_LOG_FIELDS - {"failure", "reason_code"}
    if not required.issubset(payload):
        raise TacticalPipelineContractError("Performance log incompleto.")
    if payload["analysis_name"] != ANALYSIS_NAME:
        raise TacticalPipelineContractError("analysis_name de performance invalido.")
    if payload["execution_status"] not in {
        "running_preflight",
        "running",
        "completed",
        "not_available",
        "publication_failed",
        "interrupted",
    }:
        raise TacticalPipelineContractError("execution_status invalido.")
    current_stage = payload["current_stage"]
    if current_stage != "running_preflight" and current_stage not in PERFORMANCE_STAGE_ORDER:
        raise TacticalPipelineContractError("current_stage invalido.")
    completed = payload["completed_stages"]
    if (
        type(completed) is not list
        or any(type(stage) is not str or stage not in PERFORMANCE_STAGE_ORDER for stage in completed)
        or completed != [stage for stage in PERFORMANCE_STAGE_ORDER if stage in completed]
    ):
        raise TacticalPipelineContractError("completed_stages invalido.")
    stage_seconds = payload["stage_seconds"]
    if type(stage_seconds) is not dict or any(
        stage not in PERFORMANCE_STAGE_ORDER
        or type(value) is not float
        or not isfinite(value)
        or value < 0.0
        for stage, value in stage_seconds.items()
    ):
        raise TacticalPipelineContractError("stage_seconds incremental invalido.")
    for field in (
        "targets_total",
        "targets_processed",
        "targets_remaining",
        "attempts_processed",
        "unique_cache_entries",
        "observations_indexed",
    ):
        if type(payload[field]) is not int or payload[field] < 0:
            raise TacticalPipelineContractError("Conteo incremental invalido.")
    if payload["targets_processed"] + payload["targets_remaining"] != payload["targets_total"]:
        raise TacticalPipelineContractError("Progreso target no reconcilia.")
    for field in ("elapsed_seconds", "progress_fraction"):
        value = payload[field]
        if type(value) is not float or not isfinite(value) or value < 0.0:
            raise TacticalPipelineContractError("Metrica incremental invalida.")
    if payload["progress_fraction"] > 1.0:
        raise TacticalPipelineContractError("progress_fraction supera uno.")
    counters = payload["operation_counters"]
    if (
        type(counters) is not dict
        or tuple(counters) != OPERATION_COUNTER_FIELDS
        or any(type(value) is not int or value < 0 for value in counters.values())
    ):
        raise TacticalPipelineContractError("Contadores operativos invalidos.")
    if "failure" in payload:
        failure = payload["failure"]
        if (
            type(failure) is not dict
            or tuple(failure) != ("stage", "type", "message")
            or type(failure["stage"]) is not str
            or type(failure["type"]) is not str
            or type(failure["message"]) is not str
            or failure["stage"] not in PERFORMANCE_STAGE_ORDER
            or _OPAQUE_ID.fullmatch(failure["type"]) is None
            or _OPAQUE_ID.fullmatch(failure["message"]) is None
        ):
            raise TacticalPipelineContractError("Failure incremental no sanitizado.")
    if "reason_code" in payload and (
        type(payload["reason_code"]) is not str
        or payload["reason_code"] != "manual_interrupt"
    ):
        raise TacticalPipelineContractError("reason_code incremental invalido.")
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise TacticalPipelineContractError("Performance log no serializable.") from error
    return encoded


class _IncrementalPerformanceLog:
    """Estado externo, agregado y atomico de una unica ejecucion manual."""

    __slots__ = ("path", "started", "last_payload")

    def __init__(self, path: Path) -> None:
        if not isinstance(path, Path):
            raise TypeError("performance_log debe ser Path exacto.")
        self.path = path
        self.started = perf_counter()
        self.last_payload: dict[str, object] = {}

    def _write(self, payload: dict[str, object]) -> None:
        payload = dict(payload)
        payload["analysis_name"] = ANALYSIS_NAME
        payload["elapsed_seconds"] = float(perf_counter() - self.started)
        encoded = _incremental_performance_bytes(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            temporary.write_bytes(encoded)
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.last_payload = payload

    def start(self) -> None:
        self._write(
            {
                "execution_status": "running_preflight",
                "current_stage": "running_preflight",
                "completed_stages": [],
                "stage_seconds": {},
                "targets_total": 0,
                "targets_processed": 0,
                "targets_remaining": 0,
                "progress_fraction": 0.0,
                "attempts_processed": 0,
                "unique_cache_entries": 0,
                "observations_indexed": 0,
                "operation_counters": dict.fromkeys(OPERATION_COUNTER_FIELDS, 0),
            }
        )

    def __call__(self, snapshot: dict[str, object]) -> None:
        self._write(snapshot)

    def begin_publication(self) -> None:
        payload = dict(self.last_payload)
        payload.update(
            {
                "execution_status": "running",
                "current_stage": "publication",
                "completed_stages": list(PERFORMANCE_STAGE_ORDER[:7]),
            }
        )
        self._write(payload)

    def complete_publication(self, publication_seconds: float) -> None:
        payload = dict(self.last_payload)
        stage_seconds = dict(payload["stage_seconds"])
        stage_seconds["publication"] = publication_seconds
        payload.update(
            {
                "execution_status": "completed",
                "current_stage": "publication",
                "completed_stages": list(PERFORMANCE_STAGE_ORDER),
                "stage_seconds": stage_seconds,
            }
        )
        self._write(payload)

    def fail(
        self,
        *,
        execution_status: str,
        stage: str,
        error_type: str,
        message: str,
    ) -> None:
        payload = dict(self.last_payload)
        payload.update(
            {
                "execution_status": execution_status,
                "current_stage": stage,
                "failure": {
                    "stage": stage,
                    "type": error_type,
                    "message": message,
                },
            }
        )
        self._write(payload)

    def interrupt(self) -> None:
        payload = dict(self.last_payload)
        payload.update(
            {
                "execution_status": "interrupted",
                "reason_code": "manual_interrupt",
            }
        )
        self._write(payload)


def compute_tactical_pipeline_result(
    source_path: Path,
    *,
    source_reader: Callable[[Path], pd.DataFrame] | None = None,
    upstream_validator: Callable[[], None] = validate_upstream_ancestry,
    config: TacticalPipelineConfig | None = None,
    stage_timing_sink: dict[str, float] | None = None,
    operation_counters: dict[str, int] | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    read_points_out: list[pd.DataFrame] | None = None,
) -> TacticalPipelineResult:
    """Frontera compute-only del pipeline tactico real.

    Valida la ruta contractual, verifica la linea de sangre upstream,
    lee la fuente exactamente una vez y devuelve el
    ``TacticalPipelineResult`` construido y validado por el pipeline
    P10 existente. Solo calcula: no publica artefactos, no escribe
    performance logs y no depende ni modifica las autorizaciones
    historicas P10 (``REAL_EXECUTION_AUTHORIZED`` /
    ``FURTHER_REAL_EXECUTION_AUTHORIZED``). La autorizacion y la
    persistencia corresponden al consumidor. Sin reintento; no
    ejecuta al importar. ``read_points_out`` (opcional, list vacio)
    expone el frame leido al llamador para reconstruccion en sus
    rutas de fallo; no altera la ejecucion.
    """
    _validate_cli_paths(source_path, None)
    if read_points_out is not None:
        if type(read_points_out) is not list or read_points_out:
            raise TacticalPipelineContractError(
                "read_points_out debe ser list exacto inicialmente vacio."
            )
    selected_reader = _read_source_points_once if source_reader is None else source_reader
    selected_config = future_real_pipeline_config() if config is None else config
    upstream_validator()
    points = selected_reader(source_path)
    if read_points_out is not None:
        read_points_out.append(points)
    return run_tactical_recommender_pipeline(
        points,
        None,
        selected_config,
        stage_timing_sink=stage_timing_sink,
        operation_counters=operation_counters,
        progress_callback=progress_callback,
    )


def execute_authorized_real_pipeline(
    source_path: Path,
    performance_log: Path,
    *,
    artifact_paths: TacticalPipelineArtifactPaths | None = None,
    source_reader: Callable[[Path], pd.DataFrame] | None = None,
    upstream_validator: Callable[[], None] = validate_upstream_ancestry,
    publisher: Callable[
        [TacticalPipelineResult | TacticalPipelineUnavailableResult, TacticalPipelineArtifactPaths],
        None,
    ] = publish_tactical_pipeline_artifacts,
) -> TacticalPipelineResult | TacticalPipelineUnavailableResult:
    """Punto unico de la ejecucion real, gobernado por autorizacion explicita."""
    _validate_cli_paths(source_path, performance_log)
    if not REAL_EXECUTION_AUTHORIZED:
        raise SystemExit(REAL_EXECUTION_BLOCK_REASON)
    selected_paths = default_artifact_paths() if artifact_paths is None else artifact_paths
    _artifact_paths_tuple(selected_paths)
    progress = _IncrementalPerformanceLog(performance_log)
    progress.start()
    selected_source_reader = _read_source_points_once if source_reader is None else source_reader
    stage_timings: dict[str, float] = {}
    operation_counters: dict[str, int] = {}
    read_points: list[pd.DataFrame] = []
    try:
        upstream_validator()
        _notify_pipeline_progress(
            progress,
            current_stage="source_validation",
            completed_stages=(),
            stage_seconds={},
            targets_total=0,
            targets_processed=0,
            attempts_processed=0,
            unique_cache_entries=0,
            observations_indexed=0,
            operation_counters=operation_counters,
        )
        result = compute_tactical_pipeline_result(
            source_path,
            source_reader=selected_source_reader,
            upstream_validator=lambda: None,
            config=future_real_pipeline_config(),
            stage_timing_sink=stage_timings,
            operation_counters=operation_counters,
            progress_callback=progress,
            read_points_out=read_points,
        )
    except KeyboardInterrupt:
        progress.interrupt()
        raise
    except TacticalPipelineExecutionError as error:
        # Solo puede publicarse un cierre not_available si el sellado se puede
        # reconstruir sin ejecutar parser, features, perfiles o scoring.
        validated = validate_source_points(
            read_points[0],
            expected_rows=EXPECTED_SOURCE_ROWS,
        )
        _, _, seal = seal_development_points(validated)
        unavailable = build_not_available_publication(error, seal)
        try:
            publisher(unavailable, selected_paths)
        except KeyboardInterrupt:
            progress.interrupt()
            raise
        except TacticalPipelineExecutionError as publication_error:
            progress.fail(
                execution_status="publication_failed",
                stage="publication",
                error_type=publication_error.error_type,
                message=publication_error.sanitized_message,
            )
            raise
        progress.fail(
            execution_status="not_available",
            stage=dict(_PERFORMANCE_STAGE_FROM_INTERNAL)[error.stage],
            error_type=error.error_type,
            message=error.sanitized_message,
        )
        return unavailable
    except Exception as error:
        progress.fail(
            execution_status="not_available",
            stage="source_validation",
            error_type=type(error).__name__,
            message=f"source_validation_failed:{type(error).__name__}",
        )
        raise
    publication_started = perf_counter()
    progress.begin_publication()
    try:
        if publisher is publish_tactical_pipeline_artifacts:
            # ``run_tactical_recommender_pipeline`` ya completo su unica
            # validacion reconstructiva. Se preserva la doble serializacion y
            # se reutilizan sus bytes en staging y en ambas verificaciones.
            payloads = _serialize_prevalidated_tactical_pipeline_artifacts(result)
            publisher(result, selected_paths, prepared_payloads=payloads)
        else:
            publisher(result, selected_paths)
    except KeyboardInterrupt:
        progress.interrupt()
        raise
    except TacticalPipelineExecutionError as publication_error:
        progress.fail(
            execution_status="publication_failed",
            stage="publication",
            error_type=publication_error.error_type,
            message=publication_error.sanitized_message,
        )
        raise
    progress.complete_publication(
        float(perf_counter() - publication_started)
    )
    return result


def _validate_cli_paths(source: Path, performance_log: Path | None) -> None:
    if source.resolve() != POINTS_PATH.resolve():
        raise TacticalPipelineContractError("source_no_autorizada")
    if performance_log is not None:
        resolved = performance_log.resolve()
        try:
            resolved.relative_to(ROOT.resolve())
        except ValueError:
            pass
        else:
            raise TacticalPipelineContractError("performance_log_debe_estar_fuera_del_repositorio")


def _read_source_points_once(source_path: Path) -> pd.DataFrame:
    return pd.read_parquet(source_path, columns=SOURCE_COLUMNS)


def main(argv: tuple[str, ...] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pipeline tactico descriptivo sellado")
    parser.add_argument("--source", type=Path, default=POINTS_PATH)
    parser.add_argument("--performance-log", type=Path)
    arguments = parser.parse_args(argv)
    _validate_cli_paths(arguments.source, arguments.performance_log)
    if not REAL_EXECUTION_AUTHORIZED:
        raise SystemExit(REAL_EXECUTION_BLOCK_REASON)
    if arguments.performance_log is None:
        raise TacticalPipelineContractError("performance_log_externo_es_obligatorio")
    try:
        result = execute_authorized_real_pipeline(
            arguments.source,
            arguments.performance_log,
        )
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    if type(result) is TacticalPipelineUnavailableResult:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = (
    "ALLOWED_SURFACES",
    "ANALYSIS_NAME",
    "ANALYSIS_PURPOSE",
    "ANALYSIS_VERSION",
    "ARTIFACT_ORDER",
    "AVAILABILITY_COLUMNS",
    "DEVELOPMENT_CUTOFF",
    "DESCRIPTIVE_POLICY_NAME",
    "DIAGNOSTICS_COLUMNS",
    "EXPECTED_FOLD_MATCHES",
    "EXPECTED_SOURCE_MATCHES",
    "EXPECTED_SOURCE_ROWS",
    "EXPECTED_TARGET_ORIENTATIONS",
    "EXPECTED_VALIDATION_MATCHES",
    "FIRST_EXECUTION_STATUS",
    "FINGERPRINT_CONTRACT_VERSION",
    "FURTHER_REAL_EXECUTION_AUTHORIZED",
    "PIPELINE_CONTRACT_VERSION",
    "POPULATION_COLUMNS",
    "POPULATION_ROW_ORDER",
    "PRODUCTION_PATTERNS",
    "PUBLISHED_ARTIFACT_CONTRACT",
    "PUBLISHED_PUBLICATION_FINGERPRINT",
    "PUBLISHED_RANKING_KEY_ORDER",
    "RANKINGS_COLUMNS",
    "AUTOMATIC_RETRY",
    "HISTORICAL_EXECUTION_STATUS",
    "NO_ARTIFACTS_PUBLISHED",
    "OPERATION_COUNTER_FIELDS",
    "REAL_EXECUTION_AUTHORIZED",
    "REAL_EXECUTION_AUTHORIZATION_REASON",
    "REAL_EXECUTION_ATTEMPTS",
    "REAL_EXECUTION_BLOCK_REASON",
    "SECOND_EXECUTION_STATUS",
    "SOURCE_COLUMNS",
    "TEST_START",
    "TEST_ZERO_FIELDS",
    "UPSTREAM_PROVENANCE",
    "UPSTREAM_ARTIFACTS",
    "UPSTREAM_PATHS",
    "VALIDATION_FOLDS",
    "TacticalAttemptBatch",
    "TacticalAttemptRecord",
    "TacticalCacheMetrics",
    "TacticalLeakageAudit",
    "TacticalPipelineArtifactPaths",
    "TacticalPipelineConfig",
    "TacticalPipelineContractError",
    "TacticalPipelineExecutionError",
    "TacticalPipelineDiagnostics",
    "TacticalPipelinePopulation",
    "TacticalPipelineResult",
    "TacticalPipelineState",
    "TacticalPipelineTarget",
    "TacticalPipelineUnavailableResult",
    "TacticalTargetResult",
    "TacticalTestSeal",
    "build_not_available_publication",
    "build_tactical_historical_observations",
    "build_validation_targets",
    "compute_tactical_pipeline_result",
    "construct_tactical_attempt_records",
    "default_artifact_paths",
    "default_pipeline_config",
    "execute_authorized_real_pipeline",
    "future_real_pipeline_config",
    "main",
    "performance_log_payload",
    "prepare_tactical_pipeline_payloads",
    "publish_tactical_pipeline_artifacts",
    "run_tactical_recommender_pipeline",
    "seal_development_points",
    "serialize_tactical_pipeline_artifacts",
    "serialize_tactical_pipeline_unavailable_artifacts",
    "tactical_pipeline_result_fingerprint",
    "validate_source_points",
    "validate_tactical_attempt_batch",
    "validate_tactical_attempt_record",
    "validate_tactical_cache_metrics",
    "validate_tactical_pipeline_config",
    "validate_tactical_pipeline_result",
    "validate_tactical_pipeline_target",
    "validate_upstream_ancestry",
    "verify_persisted_artifacts",
    "verify_persisted_artifacts_only",
)
