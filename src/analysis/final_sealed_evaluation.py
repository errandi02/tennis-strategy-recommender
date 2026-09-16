"""P20: especificacion congelada de la evaluacion final unica del test
2024-2026, previa a cualquier autorizacion real (P21).

Este modulo NO lee Parquet/CSV, NO abre el snapshot privado, NO ejecuta
P10 real y NO evalua el test. Congela, antes de leer una sola fila del
test, la configuracion, la temporalidad y las metricas que regiran una
futura evaluacion real separada y explicitamente autorizada por un
humano (P21). Import sin efectos: construir este modulo no produce
I/O ni instancia ningun lector real.

Auditoria contractual (evidencia de codigo, fuente autoritativa; no se
asume nada desde informes previos):

- El test 2024-2026 se define exclusivamente en
  ``src/analysis/chronological_validation.py``: ``TEST_START =
  2024-01-01``, ``TEST_END = EXPECTED_LAST_DATE = 2026-05-21``,
  poblacion congelada ``EXPECTED_SPLIT_MATCHES["test"] = 1531``. Ese
  modulo define TAMBIEN dos protocolos con ``is_final_test=True`` sobre
  la MISMA poblacion (``rolling_and_frozen_test_share_population``,
  ``rows_must_not_be_summed_as_independent_populations``):
  ``final_test_rolling`` (``history_update_policy=
  update_after_complete_date``, el mismo protocolo rolling-origin ya
  usado y validado en 2020-2023) y ``final_test_frozen``
  (``history_update_policy=frozen_at_2023-12-31``,
  ``performance_metrics_executed: False``, nombrado
  ``frozen_sensitivity_contract``). La nomenclatura
  (``*_sensitivity_contract``) establece que el protocolo PRINCIPAL es
  el rolling (identico al ya validado en 2020-2023) y el frozen es una
  sensibilidad secundaria sobre la misma poblacion, nunca una segunda
  poblacion independiente.
- La configuracion realmente congelada en el pipeline productivo P10-
  P12 (fuente autoritativa: ``src/recommender/
  tactical_recommendation_contract.py``, ``src/recommender/
  tactical_prioritization.py`` y ``src/recommender/
  tactical_matchup_evidence.py``) es: ``encoder_policy=component_only``,
  ``evidence_scope=global_only`` (el enum
  ``TacticalEvidenceScopeStrategy`` de
  ``tactical_matchup_evidence.py`` SOLO define ``GLOBAL_ONLY``; no
  existe ``SURFACE_ONLY`` ni ``SURFACE_THEN_GLOBAL`` en el pipeline
  productivo), ``minimum_labeled_activations=50``,
  ``minimum_distinct_matches=5``, combinacion ``50/50`` exacta
  (``executor_weight=opponent_allowed_weight=0.5``), sin baseline
  poblacional interno (``"no_population_baseline_imputed"`` en
  ``tactical_prioritization.py``), ``requested_top_k=3``,
  ``ranking_scope=independent_within_pattern``,
  ``global_cross_pattern_ranking=False``. La politica
  ``surface_then_global`` que aparece en ``evidence_policy_selection.py``
  /``evidence_policy_validation.py`` pertenece a una rama exploratoria
  distinta (el motor explicable de UNA sola direccion, P02) que nunca
  se integro en el recomendador P10 de cuatro patrones: no hay
  discrepancia real una vez verificada la fuente autoritativa, solo dos
  lineas de trabajo separadas.
- El objeto que produce resultados target-level es
  ``TacticalTargetResult`` (``target``, ``total_observations_received``,
  ``evidence``, ``prioritization``, ``state``), agregado en
  ``TacticalPipelineResult.target_results`` por
  ``compute_tactical_pipeline_result`` (``src/analysis/
  tactical_recommender_pipeline.py``). Esa frontera es compute-only
  (sin publicacion, sin performance log, sin depender de
  ``REAL_EXECUTION_AUTHORIZED``); YA es reutilizable sin modificar.
- No existe hoy ``build_test_targets``: ``build_validation_targets``
  (misma fuente) filtra explicitamente
  ``development.date.ge(VALIDATION_START)`` sobre la particion
  ``development`` (que ya excluyo el test como ``excluded_test_*``
  aguas arriba). Construir targets del test exigiria una funcion
  simetrica sobre una particion ``test`` explicita, hoy inexistente.
  Ese es el cambio upstream minimo identificado para una futura P21;
  P20 NO lo implementa.
- Labels reales disponibles: el unico precedente de evaluacion
  contra un label observado (``server_won_point``, punto a punto) es
  ``src/analysis/explainable_direction_scoring.py``, y SOLO para P02
  (direccion del primer saque) contra los folds de validacion
  2020-2023: calcula Brier/log-loss/calibracion comparando el score
  del modelo contra ``server_won_point`` renombrado ``outcome``. No
  existe un pipeline de scoring/calibracion equivalente para P04/P05/
  P06 (solo feasibility descriptiva). Por eso Brier/log-loss y la
  comparacion contra baseline poblacional se congelan aqui como
  metricas primarias EXCLUSIVAS de P02; para P04/P05/P06 las metricas
  primarias se limitan a cobertura/disponibilidad (ya calculables
  honestamente desde el contrato P11 sin definir ningun label nuevo).
- Train/historia termina como maximo en 2023-12-31 por construccion:
  ``VALIDATION_END = TRAIN_END`` + estructura de ``assign_split``
  (ninguna fecha de test puede etiquetarse train/validation) y por el
  ``TacticalLeakageAudit`` de P10, cuyos ocho campos deben ser
  exactamente cero (``test_rows_used=0`` incluido) en cualquier
  ejecucion real.
- Un partido de test no influye en otro por: (a) el ``calendar_day_rule``
  ya validado ("todas las partidas de una fecha se evaluan antes de
  que esa fecha actualice la historia"), compartido por el protocolo
  rolling del test con las 4 validaciones 2020-2023 ya auditadas: no
  hay influencia intradia; (b) en el protocolo frozen, la historia
  jamas se actualiza durante el test, luego ningun partido de test
  puede alimentar la evaluacion de otro.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Final


FINAL_SEALED_EVALUATION_CONTRACT_VERSION: Final = "1.0.0"
FINAL_SEALED_EVALUATION_BASE_COMMIT: Final = (
    "0b7c9c610bee91b57abd108db8fe3c97f7988980"
)

# --------------------------------------------------------------------- #
# Temporalidad exacta (fuente: src/analysis/chronological_validation.py) #
# --------------------------------------------------------------------- #

TEST_START_DATE: Final = "2024-01-01"
TEST_END_DATE: Final = "2026-05-21"
EXPECTED_TEST_MATCHES: Final = 1_531
# Misma convencion de orientaciones que build_validation_targets: dos
# orientaciones (player_1_vs_player_2 / player_2_vs_player_1) por
# partido. Valor derivado, no un nuevo dato: 1_531 * 2.
EXPECTED_TEST_ORIENTATIONS: Final = 3_062

PRIMARY_TEMPORAL_PROTOCOL: Final = "rolling_origin"
PRIMARY_HISTORY_UPDATE_POLICY: Final = "update_after_complete_date"
SENSITIVITY_TEMPORAL_PROTOCOL: Final = "frozen"
SENSITIVITY_HISTORY_UPDATE_POLICY: Final = "frozen_at_2023-12-31"
PROTOCOLS_SHARE_POPULATION: Final = True

# --------------------------------------------------------------------- #
# Configuracion congelada del pipeline productivo P10-P12                #
# (fuente: tactical_recommendation_contract.py / tactical_prioritization #
# .py / tactical_matchup_evidence.py -- valores citados, no inventados)  #
# --------------------------------------------------------------------- #

PATTERNS: Final = ("P02", "P04", "P05", "P06")
ENCODER_POLICY: Final = "component_only"
EVIDENCE_SCOPE: Final = "global_only"
FALLBACK_POLICY: Final = "none_single_scope_global_only"
MINIMUM_LABELED_ACTIVATIONS: Final = 50
MINIMUM_DISTINCT_MATCHES: Final = 5
COMBINATION: Final = "equal_weight_executor_opponent"
SCORE_FORMULA: Final = (
    "combined_rate = 0.5 * executor_success_rate + "
    "0.5 * opponent_allowed_success_rate"
)
EXECUTOR_WEIGHT: Final = 0.5
OPPONENT_ALLOWED_WEIGHT: Final = 0.5
REQUESTED_TOP_K: Final = 3
RANKING_SCOPE: Final = "independent_within_pattern"
GLOBAL_CROSS_PATTERN_RANKING: Final = False

# --------------------------------------------------------------------- #
# Metricas congeladas antes de abrir el test                             #
# --------------------------------------------------------------------- #

# Calculables honestamente para los cuatro patrones desde el contrato
# publico P11 (status/scored/abstained/reconciliations), sin definir
# ningun label nuevo.
PRIMARY_METRICS_ALL_PATTERNS: Final = (
    "orientation_coverage_by_status",
    "pattern_coverage_scored_vs_abstained",
    "reconciliation_by_year_within_test",
)
# Unico precedente de codigo con label observado (server_won_point) y
# scoring/calibracion: explainable_direction_scoring.py, solo P02.
PRIMARY_METRICS_P02_ONLY: Final = (
    "brier_score_vs_server_won_point_label",
    "log_loss_vs_server_won_point_label",
    "population_baseline_comparison",
)
SECONDARY_METRICS_ALL_PATTERNS: Final = (
    "wilson_interval_width_distribution",
    "rank_and_tie_distribution",
    "abstention_reason_code_distribution",
    "coverage_by_surface_and_period",
)
SECONDARY_METRICS_P02_ONLY: Final = ("descriptive_calibration_bins",)

FORBIDDEN_ACTIONS: Final = (
    "select_new_evidence_policy_or_scope",
    "change_minimum_thresholds",
    "recalibrate_or_retrain_after_observing_test",
    "select_metrics_post_hoc_for_favorable_outcome",
    "declare_causal_effect",
    "convert_abstention_to_success_or_failure_without_prior_definition",
    "repeat_or_partially_repeat_the_evaluation",
    "modify_model_or_policy_after_observing_any_test_result",
)

ABSTENTION_RULES: Final = (
    "both_perspectives_must_be_available",
    "rates_and_wilson_bounds_must_be_present",
    "no_imputation_or_partial_score",
    "abstained_candidates_counted_in_denominator_never_dropped",
)

FAILURE_RULES: Final = (
    "upstream_contract_violation_counted_as_error_bucket_"
    "excluded_from_performance_denominators",
    "missing_label_marks_metric_not_computable_not_zero_not_one",
    "partial_2026_reported_separately_never_pooled_as_complete_year",
)

SINGLE_EVALUATION_RULE: Final = (
    "exactly_one_authorized_real_evaluation_run_per_protocol_ever_"
    "no_repeats_no_partial_reruns"
)
NO_POST_HOC_MODIFICATION_RULE: Final = (
    "model_policy_thresholds_and_metric_selection_are_frozen_before_"
    "authorization_and_must_not_change_after_any_test_result_is_"
    "observed_for_any_reason"
)

# Denominadores y tratamiento exactos exigidos por P20. Mapa cerrado
# (clave -> regla textual), sin logica implicita en otro lugar.
METRIC_DENOMINATOR_RULES: Final = MappingProxyType(
    {
        "orientation_coverage_by_status": (
            "denominator_is_expected_test_orientations_fixed_3062; "
            "every orientation belongs to exactly one of "
            "available/partially_available/not_available"
        ),
        "pattern_coverage_scored_vs_abstained": (
            "denominator_is_total_candidate_count_per_pattern_from_"
            "P11_card.total_options; never recomputed independently"
        ),
        "brier_score_vs_server_won_point_label": (
            "denominator_is_count_of_P02_ranked_options_with_a_valid_"
            "server_won_point_label_in_the_evaluated_match_only; "
            "missing_label_excludes_from_denominator_not_zero_fill"
        ),
        "log_loss_vs_server_won_point_label": (
            "same_denominator_as_brier_score_vs_server_won_point_label"
        ),
        "population_baseline_comparison": (
            "denominator_is_same_P02_ranked_population_as_brier_"
            "score_vs_server_won_point_label_for_a_fair_delta"
        ),
        "abstained_options": (
            "always_counted_in_orientation_or_pattern_denominator_of_"
            "origin_never_excluded_never_treated_as_failure"
        ),
        "missing_label": (
            "marks_the_specific_metric_not_computable_for_that_row_"
            "excluded_from_that_metric_denominator_only"
        ),
        "partially_available_orientations": (
            "counted_in_orientation_coverage_denominator_excluded_"
            "from_pattern_scored_vs_abstained_aggregation_when_the_"
            "pattern_itself_is_not_available_for_that_orientation"
        ),
        "ties": (
            "tie_group_members_counted_once_each_via_existing_"
            "tie_group_and_tie_group_count_fields_never_broken_"
            "arbitrarily_never_double_counted"
        ),
        "incompatible_or_upstream_error_results": (
            "counted_in_a_separate_error_bucket_over_the_full_"
            "population_excluded_from_all_performance_metric_"
            "denominators"
        ),
        "year_2026_partial": (
            "reported_as_its_own_bucket_2026_partial_through_2026-05-21_"
            "never_pooled_with_2024_or_2025_as_if_complete"
        ),
    }
)


class FinalSealedEvaluationContractError(RuntimeError):
    """Especificacion congelada P20 fuera de su propio contrato."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class FinalSealedEvaluationSpecification:
    """Especificacion frozen e inmutable de la futura evaluacion (P21)."""

    contract_version: str
    base_commit: str
    test_start: str
    test_end: str
    expected_test_matches: int
    expected_test_orientations: int
    patterns: tuple[str, ...]
    encoder_policy: str
    evidence_scope: str
    fallback_policy: str
    minimum_labeled_activations: int
    minimum_distinct_matches: int
    combination: str
    score_formula: str
    executor_weight: float
    opponent_allowed_weight: float
    requested_top_k: int
    ranking_scope: str
    global_cross_pattern_ranking: bool
    primary_temporal_protocol: str
    primary_history_update_policy: str
    sensitivity_temporal_protocol: str
    sensitivity_history_update_policy: str
    protocols_share_population: bool
    primary_metrics_all_patterns: tuple[str, ...]
    primary_metrics_p02_only: tuple[str, ...]
    secondary_metrics_all_patterns: tuple[str, ...]
    secondary_metrics_p02_only: tuple[str, ...]
    abstention_rules: tuple[str, ...]
    failure_rules: tuple[str, ...]
    single_evaluation_rule: str
    no_post_hoc_modification_rule: str
    forbidden_actions: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_final_sealed_evaluation_specification(self)


def default_final_sealed_evaluation_specification() -> (
    FinalSealedEvaluationSpecification
):
    """Unica fabrica: siempre los mismos literales congelados."""
    return FinalSealedEvaluationSpecification(
        contract_version=FINAL_SEALED_EVALUATION_CONTRACT_VERSION,
        base_commit=FINAL_SEALED_EVALUATION_BASE_COMMIT,
        test_start=TEST_START_DATE,
        test_end=TEST_END_DATE,
        expected_test_matches=EXPECTED_TEST_MATCHES,
        expected_test_orientations=EXPECTED_TEST_ORIENTATIONS,
        patterns=PATTERNS,
        encoder_policy=ENCODER_POLICY,
        evidence_scope=EVIDENCE_SCOPE,
        fallback_policy=FALLBACK_POLICY,
        minimum_labeled_activations=MINIMUM_LABELED_ACTIVATIONS,
        minimum_distinct_matches=MINIMUM_DISTINCT_MATCHES,
        combination=COMBINATION,
        score_formula=SCORE_FORMULA,
        executor_weight=EXECUTOR_WEIGHT,
        opponent_allowed_weight=OPPONENT_ALLOWED_WEIGHT,
        requested_top_k=REQUESTED_TOP_K,
        ranking_scope=RANKING_SCOPE,
        global_cross_pattern_ranking=GLOBAL_CROSS_PATTERN_RANKING,
        primary_temporal_protocol=PRIMARY_TEMPORAL_PROTOCOL,
        primary_history_update_policy=PRIMARY_HISTORY_UPDATE_POLICY,
        sensitivity_temporal_protocol=SENSITIVITY_TEMPORAL_PROTOCOL,
        sensitivity_history_update_policy=SENSITIVITY_HISTORY_UPDATE_POLICY,
        protocols_share_population=PROTOCOLS_SHARE_POPULATION,
        primary_metrics_all_patterns=PRIMARY_METRICS_ALL_PATTERNS,
        primary_metrics_p02_only=PRIMARY_METRICS_P02_ONLY,
        secondary_metrics_all_patterns=SECONDARY_METRICS_ALL_PATTERNS,
        secondary_metrics_p02_only=SECONDARY_METRICS_P02_ONLY,
        abstention_rules=ABSTENTION_RULES,
        failure_rules=FAILURE_RULES,
        single_evaluation_rule=SINGLE_EVALUATION_RULE,
        no_post_hoc_modification_rule=NO_POST_HOC_MODIFICATION_RULE,
        forbidden_actions=FORBIDDEN_ACTIONS,
    )


def validate_final_sealed_evaluation_specification(
    spec: FinalSealedEvaluationSpecification,
) -> None:
    """Defensa contra edicion accidental: cada campo == su literal congelado."""
    if type(spec) is not FinalSealedEvaluationSpecification:
        raise FinalSealedEvaluationContractError(
            "Especificacion P20 debe ser FinalSealedEvaluationSpecification."
        )
    expected = {
        "contract_version": FINAL_SEALED_EVALUATION_CONTRACT_VERSION,
        "base_commit": FINAL_SEALED_EVALUATION_BASE_COMMIT,
        "test_start": TEST_START_DATE,
        "test_end": TEST_END_DATE,
        "expected_test_matches": EXPECTED_TEST_MATCHES,
        "expected_test_orientations": EXPECTED_TEST_ORIENTATIONS,
        "patterns": PATTERNS,
        "encoder_policy": ENCODER_POLICY,
        "evidence_scope": EVIDENCE_SCOPE,
        "fallback_policy": FALLBACK_POLICY,
        "minimum_labeled_activations": MINIMUM_LABELED_ACTIVATIONS,
        "minimum_distinct_matches": MINIMUM_DISTINCT_MATCHES,
        "combination": COMBINATION,
        "score_formula": SCORE_FORMULA,
        "executor_weight": EXECUTOR_WEIGHT,
        "opponent_allowed_weight": OPPONENT_ALLOWED_WEIGHT,
        "requested_top_k": REQUESTED_TOP_K,
        "ranking_scope": RANKING_SCOPE,
        "global_cross_pattern_ranking": GLOBAL_CROSS_PATTERN_RANKING,
        "primary_temporal_protocol": PRIMARY_TEMPORAL_PROTOCOL,
        "primary_history_update_policy": PRIMARY_HISTORY_UPDATE_POLICY,
        "sensitivity_temporal_protocol": SENSITIVITY_TEMPORAL_PROTOCOL,
        "sensitivity_history_update_policy": SENSITIVITY_HISTORY_UPDATE_POLICY,
        "protocols_share_population": PROTOCOLS_SHARE_POPULATION,
        "primary_metrics_all_patterns": PRIMARY_METRICS_ALL_PATTERNS,
        "primary_metrics_p02_only": PRIMARY_METRICS_P02_ONLY,
        "secondary_metrics_all_patterns": SECONDARY_METRICS_ALL_PATTERNS,
        "secondary_metrics_p02_only": SECONDARY_METRICS_P02_ONLY,
        "abstention_rules": ABSTENTION_RULES,
        "failure_rules": FAILURE_RULES,
        "single_evaluation_rule": SINGLE_EVALUATION_RULE,
        "no_post_hoc_modification_rule": NO_POST_HOC_MODIFICATION_RULE,
        "forbidden_actions": FORBIDDEN_ACTIONS,
    }
    for field in fields(spec):
        actual = getattr(spec, field.name)
        want = expected[field.name]
        if actual != want or type(actual) is not type(want):
            raise FinalSealedEvaluationContractError(
                f"Campo congelado divergente: {field.name}."
            )
    if spec.executor_weight != 0.5 or spec.opponent_allowed_weight != 0.5:
        raise FinalSealedEvaluationContractError(
            "La combinacion debe ser 50/50 exacta."
        )
    if spec.expected_test_orientations != spec.expected_test_matches * 2:
        raise FinalSealedEvaluationContractError(
            "Orientaciones esperadas deben ser el doble de los partidos."
        )
    if set(spec.patterns) != {"P02", "P04", "P05", "P06"}:
        raise FinalSealedEvaluationContractError(
            "Los cuatro patrones congelados deben ser P02/P04/P05/P06."
        )


_CONFIGURATION_FINGERPRINT_FIELDS: Final = (
    "encoder_policy",
    "evidence_scope",
    "fallback_policy",
    "minimum_labeled_activations",
    "minimum_distinct_matches",
    "combination",
    "score_formula",
    "executor_weight",
    "opponent_allowed_weight",
    "requested_top_k",
    "ranking_scope",
    "global_cross_pattern_ranking",
)
_CONFIGURATION_FINGERPRINT_DOMAIN: Final = (
    b"tennis-final-sealed-evaluation-configuration\x00"
)
_SPECIFICATION_FINGERPRINT_DOMAIN: Final = (
    b"tennis-final-sealed-evaluation-specification\x00"
)


def _canonical_json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def compute_configuration_fingerprint(
    spec: FinalSealedEvaluationSpecification,
) -> str:
    """SHA-256 mayuscula solo sobre policy/scoring/thresholds/fallback/top-k.

    Independiente de fechas, poblacion o catalogo de metricas: cambia
    unicamente si la configuracion de puntuacion del recomendador
    cambia. Funcion pura, recomputable por un tercero sin este modulo.
    """
    payload = {name: getattr(spec, name) for name in _CONFIGURATION_FINGERPRINT_FIELDS}
    digest = hashlib.sha256(
        _CONFIGURATION_FINGERPRINT_DOMAIN + _canonical_json_bytes(payload)
    ).hexdigest()
    return digest.upper()


def compute_specification_fingerprint(
    spec: FinalSealedEvaluationSpecification,
) -> str:
    """SHA-256 mayuscula sobre la especificacion congelada completa."""
    payload = {field.name: getattr(spec, field.name) for field in fields(spec)}
    digest = hashlib.sha256(
        _SPECIFICATION_FINGERPRINT_DOMAIN + _canonical_json_bytes(payload)
    ).hexdigest()
    return digest.upper()


# --------------------------------------------------------------------- #
# Puerta de autorizacion unica (P20 -> P21): ninguna via alternativa     #
# --------------------------------------------------------------------- #

REAL_TEST_EVALUATION_AUTHORIZED: Final = False

# P23: cierre inmediato tras el UNICO intento autorizado (concedido
# tras una auditoria completa de P20-P22 que encontro y corrigio dos
# defectos de senal/cierre -- ausencia de manejo de SIGTERM y
# ``sys.argv`` no propagado a ``main()`` en
# ``final_sealed_evaluation_runner.py`` -- ANTES de autorizar, nunca
# despues). Ese intento se inicio exactamente una vez en Mac, el
# supervisor y el proceso analitico terminaron, y fallo ANTES de
# publicar el bundle: exit code exacto ``1``, sin
# ``reports/final_evaluation/`` creado, log externo de 0 bytes, sin
# procesos residuales, sin SIGINT/SIGTERM conocido -- se clasifica como
# ejecucion FALLIDA, no interrumpida.
#
# Diagnostico pendiente: esta autorizacion no se reutiliza. Ninguna
# ejecucion adicional queda autorizada por este cierre; una futura
# evaluacion exigiria una decision humana y una autorizacion
# INDEPENDIENTES (repitiendo el preflight completo), nunca un
# reintento del mismo proceso ni una reapertura silenciosa de la
# puerta.
REAL_TEST_EVALUATION_AUTHORIZATION_REASON: Final = (
    "real_test_evaluation_failed_pending_diagnosis_no_further_execution_authorized"
)

# Intento consumido: PREVIOUS=1 (el unico intento autorizado ya ocurrio),
# COMPLETED=0 (fallo antes de publicar el bundle, nunca llego a
# completarse), INTERRUPTED=0 (fue un fallo, no una interrupcion por
# senal conocida), RETRIES=0 (sin reintento automatico bajo ninguna
# circunstancia). No existe en este contrato un contador explicito de
# fallos separado: el fallo queda representado por esta combinacion
# exacta (previous=1, completed=0, interrupted=0) junto con la razon de
# cierre de arriba.
PREVIOUS_REAL_TEST_EVALUATIONS: Final = 1
COMPLETED_REAL_TEST_EVALUATIONS: Final = 0
INTERRUPTED_REAL_TEST_EVALUATIONS: Final = 0
AUTOMATIC_RETRIES_PERFORMED: Final = 0

# Sin reintento automatico bajo ninguna circunstancia: el fallo del
# unico intento autorizado consume esa autorizacion (regla de cierre);
# una segunda ejecucion exige una NUEVA decision humana explicita,
# nunca un reintento silencioso del mismo proceso.
AUTOMATIC_RETRY: Final = False
AUTOMATIC_RETRY_POLICY: Final = "single_manual_execution_without_automatic_retry"

REAL_TEST_EVALUATION_BLOCK_REASON_CODE: Final = (
    "real_test_evaluation_not_yet_authorized"
)
REAL_TEST_EVALUATION_BLOCK_REASON: Final = (
    "P20 bloqueado: la evaluacion real del test 2024-2026 no esta "
    "autorizada (REAL_TEST_EVALUATION_AUTHORIZED=False). Requiere una "
    "decision humana separada (P21)."
)


def run_real_test_evaluation(*_args: object, **_kwargs: object) -> None:
    """Frontera controlada por la puerta unica: aborta ANTES de
    cualquier I/O unicamente cuando ``REAL_TEST_EVALUATION_AUTHORIZED``
    vale ``False``.

    Firma deliberadamente generica (``*_args``/``**_kwargs``): no
    acepta ninguna ruta real ni configuracion externa. Estado VIGENTE
    (P23, cierre tras fallo): ``False`` -- el unico intento autorizado
    ya ocurrio (en Mac) y fallo ANTES de publicar el bundle (exit code
    ``1``, sin ``reports/final_evaluation/`` creado; ver razon exacta
    en ``REAL_TEST_EVALUATION_AUTHORIZATION_REASON`` y el historial en
    ``PREVIOUS_REAL_TEST_EVALUATIONS=1``/``COMPLETED_REAL_TEST_
    EVALUATIONS=0``). Diagnostico pendiente: ninguna nueva ejecucion
    queda autorizada por este cierre; una futura evaluacion exigiria
    una decision humana y una autorizacion independientes. En este
    estado esta funcion aborta antes de tocar ``os.environ``, Parquet,
    CSV, el snapshot privado o cualquier lector real; no hay ninguna
    otra via de ejecucion en ningun caso (ni variable de entorno, ni
    flag, ni segunda constante).

    P22 (``src.analysis.final_sealed_evaluation_runner``) completo la
    frontera productiva real: el ``import`` es local (dentro de esta
    funcion, no a nivel de modulo) para que ``final_sealed_evaluation``
    siga sin ningun efecto ni dependencia de pandas/pyarrow al
    importarse, y para evitar un ciclo de importacion con el runner
    (que si importa este modulo).
    """
    if not REAL_TEST_EVALUATION_AUTHORIZED:
        raise SystemExit(REAL_TEST_EVALUATION_BLOCK_REASON)
    from src.analysis.final_sealed_evaluation_runner import (  # noqa: PLC0415
        execute_real_sealed_test_evaluation,
    )

    execute_real_sealed_test_evaluation()


PREFLIGHT_MANIFEST_PATH: Final = (
    Path(__file__).resolve().parents[2] / "reports" / "final_evaluation_preflight.json"
)


def build_preflight_manifest() -> dict[str, object]:
    """Manifiesto puro (sin I/O): status preflight_only, cero resultados reales."""
    spec = default_final_sealed_evaluation_specification()
    return {
        "contract_name": "final_evaluation_preflight",
        "contract_version": spec.contract_version,
        "status": "preflight_only",
        "authorized": REAL_TEST_EVALUATION_AUTHORIZED,
        "test_evaluation_runs": COMPLETED_REAL_TEST_EVALUATIONS,
        "previous_real_test_evaluations": PREVIOUS_REAL_TEST_EVALUATIONS,
        "interrupted_real_test_evaluations": INTERRUPTED_REAL_TEST_EVALUATIONS,
        "automatic_retries_performed": AUTOMATIC_RETRIES_PERFORMED,
        "base_commit": spec.base_commit,
        "fingerprints": {
            "configuration_fingerprint": compute_configuration_fingerprint(spec),
            "specification_fingerprint": compute_specification_fingerprint(spec),
        },
        "temporality": {
            "test_start": spec.test_start,
            "test_end": spec.test_end,
            "primary_temporal_protocol": spec.primary_temporal_protocol,
            "primary_history_update_policy": spec.primary_history_update_policy,
            "sensitivity_temporal_protocol": spec.sensitivity_temporal_protocol,
            "sensitivity_history_update_policy": (
                spec.sensitivity_history_update_policy
            ),
            "protocols_share_population": spec.protocols_share_population,
        },
        "configuration": {
            "patterns": list(spec.patterns),
            "encoder_policy": spec.encoder_policy,
            "evidence_scope": spec.evidence_scope,
            "fallback_policy": spec.fallback_policy,
            "minimum_labeled_activations": spec.minimum_labeled_activations,
            "minimum_distinct_matches": spec.minimum_distinct_matches,
            "combination": spec.combination,
            "score_formula": spec.score_formula,
            "requested_top_k": spec.requested_top_k,
            "ranking_scope": spec.ranking_scope,
            "global_cross_pattern_ranking": spec.global_cross_pattern_ranking,
        },
        "expected_population": {
            "expected_test_matches": spec.expected_test_matches,
            "expected_test_orientations": spec.expected_test_orientations,
        },
        "metrics": {
            "primary_all_patterns": list(spec.primary_metrics_all_patterns),
            "primary_p02_only": list(spec.primary_metrics_p02_only),
            "secondary_all_patterns": list(spec.secondary_metrics_all_patterns),
            "secondary_p02_only": list(spec.secondary_metrics_p02_only),
            "denominator_rules": dict(METRIC_DENOMINATOR_RULES),
        },
        "rules": {
            "abstention_rules": list(spec.abstention_rules),
            "failure_rules": list(spec.failure_rules),
            "single_evaluation_rule": spec.single_evaluation_rule,
            "no_post_hoc_modification_rule": spec.no_post_hoc_modification_rule,
            "forbidden_actions": list(spec.forbidden_actions),
        },
        "no_real_metrics_observed": True,
        "no_target_level_data": True,
        "no_identities": True,
    }


def publish_preflight_manifest(path: Path | None = None) -> Path:
    """Escribe el manifiesto de forma atomica (mkstemp + os.replace).

    Sin resultados reales en ningun caso: ``authorized`` refleja la
    constante del modulo tal cual (``True`` solo mientras una unica
    ejecucion manual siga autorizada y pendiente; ``False`` en
    cualquier otro momento, incluido tras el cierre posterior a un
    intento completado, fallido o interrumpido). Los cuatro contadores
    de ejecucion reflejan el historial exacto tal cual (no son siempre
    cero: tras el cierre de un intento fallido,
    ``PREVIOUS_REAL_TEST_EVALUATIONS`` pasa a contar ese intento aunque
    ``COMPLETED``/``INTERRUPTED`` sigan en cero). Este manifiesto no
    lee ningun dato de origen ni ejecuta nada; su unica entrada es la
    especificacion congelada de este modulo.
    """
    destination = PREFLIGHT_MANIFEST_PATH if path is None else path
    manifest = build_preflight_manifest()
    payload = json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=".final_evaluation_preflight-", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    except Exception:
        try:
            os.remove(temp_name)
        except OSError:
            pass
        raise
    return destination


if __name__ == "__main__":
    publish_preflight_manifest()


__all__ = (
    "ABSTENTION_RULES",
    "COMBINATION",
    "ENCODER_POLICY",
    "EVIDENCE_SCOPE",
    "EXECUTOR_WEIGHT",
    "EXPECTED_TEST_MATCHES",
    "EXPECTED_TEST_ORIENTATIONS",
    "FAILURE_RULES",
    "FALLBACK_POLICY",
    "FINAL_SEALED_EVALUATION_BASE_COMMIT",
    "FINAL_SEALED_EVALUATION_CONTRACT_VERSION",
    "FORBIDDEN_ACTIONS",
    "GLOBAL_CROSS_PATTERN_RANKING",
    "METRIC_DENOMINATOR_RULES",
    "MINIMUM_DISTINCT_MATCHES",
    "MINIMUM_LABELED_ACTIVATIONS",
    "NO_POST_HOC_MODIFICATION_RULE",
    "OPPONENT_ALLOWED_WEIGHT",
    "PATTERNS",
    "PREFLIGHT_MANIFEST_PATH",
    "PRIMARY_HISTORY_UPDATE_POLICY",
    "PRIMARY_METRICS_ALL_PATTERNS",
    "PRIMARY_METRICS_P02_ONLY",
    "PRIMARY_TEMPORAL_PROTOCOL",
    "PROTOCOLS_SHARE_POPULATION",
    "RANKING_SCOPE",
    "REAL_TEST_EVALUATION_AUTHORIZED",
    "REAL_TEST_EVALUATION_AUTHORIZATION_REASON",
    "REAL_TEST_EVALUATION_BLOCK_REASON",
    "REAL_TEST_EVALUATION_BLOCK_REASON_CODE",
    "REQUESTED_TOP_K",
    "SCORE_FORMULA",
    "SECONDARY_METRICS_ALL_PATTERNS",
    "SECONDARY_METRICS_P02_ONLY",
    "SENSITIVITY_HISTORY_UPDATE_POLICY",
    "SENSITIVITY_TEMPORAL_PROTOCOL",
    "SINGLE_EVALUATION_RULE",
    "TEST_END_DATE",
    "TEST_START_DATE",
    "AUTOMATIC_RETRIES_PERFORMED",
    "AUTOMATIC_RETRY",
    "AUTOMATIC_RETRY_POLICY",
    "COMPLETED_REAL_TEST_EVALUATIONS",
    "INTERRUPTED_REAL_TEST_EVALUATIONS",
    "PREVIOUS_REAL_TEST_EVALUATIONS",
    "FinalSealedEvaluationContractError",
    "FinalSealedEvaluationSpecification",
    "build_preflight_manifest",
    "compute_configuration_fingerprint",
    "compute_specification_fingerprint",
    "default_final_sealed_evaluation_specification",
    "publish_preflight_manifest",
    "run_real_test_evaluation",
    "validate_final_sealed_evaluation_specification",
)
