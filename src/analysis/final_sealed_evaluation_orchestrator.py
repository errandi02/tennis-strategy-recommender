"""P22: orquestador compute-only -- estructuras P21/P22 adaptadas ->
resultado agregado, inmutable y determinista. Sin I/O: nunca lee ni
escribe archivos, nunca abre el snapshot y nunca consulta variables de
entorno.

Ejecuta los dos protocolos congelados por P20 sobre la MISMA
poblacion de targets: ``rolling_origin`` (principal) y
``frozen_at_2023-12-31`` (sensibilidad), reutilizando exclusivamente
el evaluador compute-only P21 (``evaluate_sealed_test_population``,
``assert_protocols_share_population``) y sus agregadores de metricas.
Las metricas de Brier/log-loss/baseline P02 se derivan uniendo los
labels reales observados (``P02ObservedAttempt``, adaptador P22) con
la puntuacion PRE-partido que cada protocolo asigno a esa categoria
concreta -- nunca con el top-ranked si la categoria observada no
coincide, y nunca con una categoria sin score valido (candidato no
``SCORED``): esos casos quedan fuera del denominador, no se imputan.

Decision metodologica humana explicita y congelada (sustituye la
decision unilateral de la version anterior de este modulo; fijada
ANTES de ejecutar sobre datos reales, sin reconsiderarse tras observar
resultados -- P20/P21 no especifican este procedimiento, por lo que se
fija aqui como contrato propio de P22, sin tocar
``final_sealed_evaluation.py``):

- la baseline poblacional P02 es una TASA FIJA UNICA;
- calculada EXCLUSIVAMENTE con intentos P02 etiquetados de desarrollo
  (``adapted.development_p02_observed_attempts``: fecha maxima
  2023-12-31 por construccion, ya que proceden de
  ``construct_tactical_attempt_records``, cuyo gate temporal de P10
  lo garantiza);
- IDENTICA para ``rolling_origin`` y ``frozen`` (se calcula UNA sola
  vez, fuera del bucle por protocolo, y se pasa igual a ambos);
- sin ningun outcome del periodo de test;
- ``rate=None`` (nunca un valor por defecto como 0.5) si el
  denominador etiquetado es cero;
- numerador, denominador y la regla textual se incluyen explicitamente
  en el resultado agregado y en la serializacion (P22, paso 4).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Final

from src.analysis.final_sealed_evaluation import (
    compute_configuration_fingerprint,
    compute_specification_fingerprint,
    default_final_sealed_evaluation_specification,
)
from src.analysis.final_sealed_evaluation_adapter import (
    AdaptedSealedEvaluationInput,
    P02ObservedAttempt,
)
from src.analysis.final_sealed_evaluation_boundary import (
    FROZEN_PROTOCOL,
    ROLLING_PROTOCOL,
    P02LabeledAttempt,
    SealedTestEvaluationResult,
    SealedTestTargetEvaluation,
    aggregate_abstention_reason_codes,
    aggregate_coverage_by_surface_and_period,
    aggregate_error_bucket,
    aggregate_orientation_coverage_by_status,
    aggregate_pattern_coverage_scored_vs_abstained,
    aggregate_rank_and_tie_distribution,
    aggregate_reconciliation_by_year,
    aggregate_wilson_interval_width_distribution,
    assert_protocols_share_population,
    build_test_targets,
    compute_p02_brier_and_log_loss,
    compute_p02_population_baseline_comparison,
    evaluate_sealed_test_population,
)
from src.recommender.tactical_prioritization import TacticalCandidateState


FINAL_SEALED_EVALUATION_ORCHESTRATOR_CONTRACT_VERSION: Final = "1.0.0"

_P02_PATTERN_ID: Final = "P02"

# Contrato humano congelado (ver docstring del modulo): texto exacto de
# la regla, estable e independiente de cualquier valor observado.
P02_POPULATION_BASELINE_RULE: Final = (
    "fixed_rate_from_labeled_p02_development_attempts_only; "
    "max_date_2023-12-31_by_construction; "
    "identical_for_rolling_origin_and_frozen; "
    "no_test_period_outcomes; "
    "null_if_labeled_denominator_is_zero"
)
_P02_BASELINE_FINGERPRINT_DOMAIN: Final = (
    b"tennis-final-sealed-evaluation-p02-population-baseline\x00"
)


class FinalSealedEvaluationOrchestratorError(RuntimeError):
    """Rechazo cerrado del orquestador P22 (sin fugas)."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class P02PopulationBaseline:
    """Baseline poblacional P02: tasa fija, calculada una sola vez,
    identica para ambos protocolos. ``rate`` es ``None`` (nunca un
    valor por defecto) cuando ``denominator`` es cero."""

    rule: str
    numerator: int
    denominator: int
    rate: float | None

    def __post_init__(self) -> None:
        if self.rule != P02_POPULATION_BASELINE_RULE:
            raise FinalSealedEvaluationOrchestratorError(
                "Regla de baseline P02 fuera del contrato congelado."
            )
        if type(self.numerator) is not int or type(self.denominator) is not int:
            raise FinalSealedEvaluationOrchestratorError(
                "Numerador/denominador de baseline P02 deben ser int reales."
            )
        if self.numerator < 0 or self.denominator < 0 or self.numerator > self.denominator:
            raise FinalSealedEvaluationOrchestratorError(
                "Numerador/denominador de baseline P02 no reconcilian."
            )
        if self.denominator == 0:
            if self.rate is not None:
                raise FinalSealedEvaluationOrchestratorError(
                    "Baseline P02 sin denominador debe tener rate=None."
                )
        else:
            if type(self.rate) is not float or self.rate != self.numerator / self.denominator:
                raise FinalSealedEvaluationOrchestratorError(
                    "Baseline P02 rate no reconcilia con numerador/denominador."
                )


def compute_p02_population_baseline_fingerprint(baseline: P02PopulationBaseline) -> str:
    """SHA-256 mayuscula, independiente, sobre la baseline congelada."""
    payload = {field.name: getattr(baseline, field.name) for field in fields(baseline)}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(_P02_BASELINE_FINGERPRINT_DOMAIN + blob).hexdigest().upper()


@dataclass(frozen=True, slots=True)
class ProtocolMetricsSummary:
    protocol: str
    targets_total: int
    coverage_by_status: MappingProxyType
    pattern_coverage: MappingProxyType
    reconciliation_by_year: MappingProxyType
    wilson_width_distribution: MappingProxyType
    rank_and_tie_distribution: MappingProxyType
    abstention_reason_codes: MappingProxyType
    coverage_by_surface_and_period: MappingProxyType
    error_bucket: MappingProxyType
    p02_performance: MappingProxyType
    p02_population_baseline_comparison: MappingProxyType


@dataclass(frozen=True, slots=True)
class FinalSealedEvaluationOutcome:
    contract_version: str
    configuration_fingerprint: str
    specification_fingerprint: str
    p02_population_baseline: P02PopulationBaseline
    p02_population_baseline_fingerprint: str
    population_reconciled: bool
    targets_total: int
    rolling: ProtocolMetricsSummary
    frozen: ProtocolMetricsSummary


def _freeze(value: dict) -> MappingProxyType:
    return MappingProxyType(dict(value))


def compute_p02_population_baseline(
    development_p02_observed: tuple[P02ObservedAttempt, ...],
) -> P02PopulationBaseline:
    """Tasa fija unica: SOLO intentos P02 etiquetados de desarrollo.

    Pura funcion del numerador/denominador; no depende del protocolo
    ni de ningun dato del periodo de test."""
    if type(development_p02_observed) is not tuple:
        raise FinalSealedEvaluationOrchestratorError(
            "development_p02_observed debe ser tuple exacto."
        )
    numerator = sum(1 for item in development_p02_observed if item.server_won_point)
    denominator = len(development_p02_observed)
    rate = None if denominator == 0 else numerator / denominator
    return P02PopulationBaseline(P02_POPULATION_BASELINE_RULE, numerator, denominator, rate)


def _build_p02_labeled_attempts(
    p02_observed: tuple[P02ObservedAttempt, ...],
    result: SealedTestEvaluationResult,
) -> tuple[P02LabeledAttempt, ...]:
    """Une DIRECCION observada (``P02ObservedAttempt.category``, de
    ``parse_and_classify_first_serve_direction``) + OUTCOME observado
    (``server_won_point``, campo independiente del mismo intento) con
    el SCORE PRE-PARTIDO exacto de esa categoria (nunca el top-ranked
    si no coincide, nunca el de otra categoria). El intento evaluado
    NUNCA puede aparecer en su propia historia: ``candidate.
    combined_rate`` proviene de ``evaluation.prioritization``, ya
    construida por P21 exclusivamente con observaciones estrictamente
    anteriores (``visible_observations_for_target``); esta funcion no
    anade ninguna observacion nueva, solo LEE el resultado ya sellado."""
    index: dict[tuple[str, str], SealedTestTargetEvaluation] = {
        (item.target.target_match_id, item.target.player): item
        for item in result.evaluations
        if isinstance(item, SealedTestTargetEvaluation)
    }
    labeled: list[P02LabeledAttempt] = []
    for observed in p02_observed:
        evaluation = index.get((observed.target_match_id, observed.player))
        if evaluation is None:
            continue
        ranking = next(
            (
                item for item in evaluation.prioritization.rankings
                if item.pattern_id == _P02_PATTERN_ID
            ),
            None,
        )
        if ranking is None:
            continue
        candidate = next(
            (item for item in ranking.candidates if item.category == observed.category),
            None,
        )
        if (
            candidate is None
            or candidate.state is not TacticalCandidateState.SCORED
            or candidate.combined_rate is None
        ):
            continue
        labeled.append(
            P02LabeledAttempt(
                observed.target_match_id,
                observed.player,
                observed.category,
                candidate.combined_rate,
                observed.server_won_point,
            )
        )
    return tuple(labeled)


def _p02_baseline_comparison_payload(
    labeled_attempts: tuple[P02LabeledAttempt, ...],
    baseline: P02PopulationBaseline,
) -> dict[str, object]:
    common = {
        "population_baseline_rule": baseline.rule,
        "population_baseline_numerator": baseline.numerator,
        "population_baseline_denominator": baseline.denominator,
        "population_baseline_rate": baseline.rate,
    }
    if baseline.rate is None:
        return {
            **common,
            "denominator": 0,
            "delta_brier_vs_population": None,
            "delta_log_loss_vs_population": None,
        }
    comparison = compute_p02_population_baseline_comparison(labeled_attempts, baseline.rate)
    return {**common, **comparison}


def _summarize_protocol(
    result: SealedTestEvaluationResult,
    p02_observed: tuple[P02ObservedAttempt, ...],
    baseline: P02PopulationBaseline,
) -> ProtocolMetricsSummary:
    labeled_attempts = _build_p02_labeled_attempts(p02_observed, result)
    p02_performance = compute_p02_brier_and_log_loss(labeled_attempts)
    p02_baseline_comparison = _p02_baseline_comparison_payload(labeled_attempts, baseline)
    return ProtocolMetricsSummary(
        result.protocol,
        result.targets_total,
        _freeze(aggregate_orientation_coverage_by_status(result)),
        _freeze(aggregate_pattern_coverage_scored_vs_abstained(result)),
        _freeze(aggregate_reconciliation_by_year(result)),
        _freeze(aggregate_wilson_interval_width_distribution(result)),
        _freeze(aggregate_rank_and_tie_distribution(result)),
        _freeze(aggregate_abstention_reason_codes(result)),
        _freeze(aggregate_coverage_by_surface_and_period(result)),
        _freeze(aggregate_error_bucket(result)),
        _freeze(p02_performance),
        _freeze(p02_baseline_comparison),
    )


def evaluate_final_sealed_test(
    adapted: AdaptedSealedEvaluationInput,
) -> FinalSealedEvaluationOutcome:
    """Funcion pura: estructuras P22/P21 adaptadas -> resultado agregado.

    No lee ni escribe archivos. Ejecuta ``rolling_origin`` (principal)
    y ``frozen`` (sensibilidad) sobre EXACTAMENTE la misma poblacion de
    targets (comprobado con ``assert_protocols_share_population``, no
    solo asumido). La baseline poblacional P02 se calcula UNA sola vez
    y se reutiliza IDENTICA en ambos protocolos."""
    if type(adapted) is not AdaptedSealedEvaluationInput:
        raise FinalSealedEvaluationOrchestratorError(
            "adapted debe ser AdaptedSealedEvaluationInput exacto."
        )
    spec = default_final_sealed_evaluation_specification()
    targets = build_test_targets(
        adapted.test_match_rows, enforce_frozen_cardinalities=True
    )
    rolling = evaluate_sealed_test_population(
        targets,
        adapted.development_observations,
        adapted.test_observations,
        adapted.schema,
        protocol=ROLLING_PROTOCOL,
    )
    frozen = evaluate_sealed_test_population(
        targets,
        adapted.development_observations,
        adapted.test_observations,
        adapted.schema,
        protocol=FROZEN_PROTOCOL,
    )
    assert_protocols_share_population(rolling, frozen)
    baseline = compute_p02_population_baseline(adapted.development_p02_observed_attempts)
    rolling_summary = _summarize_protocol(rolling, adapted.test_p02_observed_attempts, baseline)
    frozen_summary = _summarize_protocol(frozen, adapted.test_p02_observed_attempts, baseline)
    return FinalSealedEvaluationOutcome(
        FINAL_SEALED_EVALUATION_ORCHESTRATOR_CONTRACT_VERSION,
        compute_configuration_fingerprint(spec),
        compute_specification_fingerprint(spec),
        baseline,
        compute_p02_population_baseline_fingerprint(baseline),
        True,
        len(targets),
        rolling_summary,
        frozen_summary,
    )


__all__ = (
    "FINAL_SEALED_EVALUATION_ORCHESTRATOR_CONTRACT_VERSION",
    "P02_POPULATION_BASELINE_RULE",
    "FinalSealedEvaluationOrchestratorError",
    "FinalSealedEvaluationOutcome",
    "P02PopulationBaseline",
    "ProtocolMetricsSummary",
    "compute_p02_population_baseline",
    "compute_p02_population_baseline_fingerprint",
    "evaluate_final_sealed_test",
)
