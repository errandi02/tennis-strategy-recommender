"""Tests sinteticos P22: orquestador compute-only (estructuras P21/P22
adaptadas -> resultado agregado). Ningun test lee Parquet/CSV reales
ni ejecuta P10; toda entrada es sintetica (reutiliza los helpers de
``test_final_sealed_evaluation_adapter``).

Cubre: rolling y frozen sobre poblacion identica, reconciliacion de
poblacion, agregacion exacta de las metricas P20, denominadores P02
(labels ausentes -> None, candidato no-scored excluido), determinismo,
y AST sin I/O de import.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import src.analysis.final_sealed_evaluation_adapter as adapter
import src.analysis.final_sealed_evaluation_orchestrator as orchestrator
from test_final_sealed_evaluation_adapter import (
    _small_points_dataframe,
    small_cardinalities,  # noqa: F401 (fixture reexportada)
)


_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- #
# A. Import sin efectos y AST                                            #
# --------------------------------------------------------------------- #


def test_import_subprocess_sin_efectos(tmp_path) -> None:
    sentinel = tmp_path / "sentinela.txt"
    sentinel.write_text("x", encoding="utf-8")
    before = {item.name for item in tmp_path.iterdir()}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable, "-c",
            "import src.analysis.final_sealed_evaluation_orchestrator as o; "
            "print(o.FINAL_SEALED_EVALUATION_ORCHESTRATOR_CONTRACT_VERSION)",
        ],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "1.0.0"
    assert completed.stderr == ""
    after = {item.name for item in tmp_path.iterdir()}
    assert after == before


def test_ast_no_readers_no_environ() -> None:
    tree = ast.parse(Path(orchestrator.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {
                "environ", "getenv", "argv", "read_parquet", "read_csv", "open",
            }, f"Acceso prohibido: {node.attr}"


def test_ast_no_pandas_import() -> None:
    tree = ast.parse(Path(orchestrator.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(name == "pandas" or name.startswith("pandas.") for name in imported)


# --------------------------------------------------------------------- #
# B. Rolling/frozen sobre poblacion identica, reconciliacion             #
# --------------------------------------------------------------------- #


@pytest.fixture()
def small_outcome(small_cardinalities):
    df = _small_points_dataframe(train=2, validation=2, test=2, first_serve="4")
    adapted = adapter.adapt_real_points_dataframe(df)
    return orchestrator.evaluate_final_sealed_test(adapted)


def test_outcome_reconciles_population(small_outcome) -> None:
    assert small_outcome.population_reconciled is True
    assert small_outcome.targets_total == small_outcome.rolling.targets_total
    assert small_outcome.targets_total == small_outcome.frozen.targets_total


def test_outcome_distinguishes_rolling_and_frozen_protocol(small_outcome) -> None:
    assert small_outcome.rolling.protocol == "rolling_origin"
    assert small_outcome.frozen.protocol == "frozen"


def test_outcome_has_fingerprints(small_outcome) -> None:
    assert len(small_outcome.configuration_fingerprint) == 64
    assert len(small_outcome.specification_fingerprint) == 64


def test_outcome_is_frozen_dataclass(small_outcome) -> None:
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        small_outcome.targets_total = 999


# --------------------------------------------------------------------- #
# C. Metricas P20 agregadas correctamente en ambos protocolos            #
# --------------------------------------------------------------------- #


def test_both_protocols_expose_all_p20_metric_buckets(small_outcome) -> None:
    for summary in (small_outcome.rolling, small_outcome.frozen):
        assert set(summary.coverage_by_status["by_status"]) == {
            "available", "partially_available", "not_available",
        }
        assert set(summary.pattern_coverage) == {"P02", "P04", "P05", "P06"}
        assert "brier_score" in summary.p02_performance
        assert "log_loss" in summary.p02_performance
        assert "delta_brier_vs_population" in summary.p02_population_baseline_comparison


def test_frozen_error_bucket_denominator_matches_targets(small_outcome) -> None:
    assert (
        small_outcome.frozen.error_bucket["denominator"] == small_outcome.frozen.targets_total
    )


# --------------------------------------------------------------------- #
# D. Denominadores P02: labels ausentes -> None, no fabricados            #
# --------------------------------------------------------------------- #


def test_p02_performance_none_when_no_recognizable_direction(small_cardinalities) -> None:
    df = _small_points_dataframe(train=2, validation=2, test=2, first_serve="n4")
    adapted = adapter.adapt_real_points_dataframe(df)
    outcome = orchestrator.evaluate_final_sealed_test(adapted)
    assert outcome.rolling.p02_performance["denominator"] == 0
    assert outcome.rolling.p02_performance["brier_score"] is None
    assert outcome.rolling.p02_performance["log_loss"] is None


def test_p04_p05_p06_never_expose_predictive_performance_fields(small_outcome) -> None:
    for summary in (small_outcome.rolling, small_outcome.frozen):
        for pattern, values in summary.pattern_coverage.items():
            if pattern == "P02":
                continue
            assert "brier_score" not in values
            assert "log_loss" not in values


# --------------------------------------------------------------------- #
# E. Determinismo                                                        #
# --------------------------------------------------------------------- #


def test_orchestrator_deterministic_for_same_adapted_input(small_cardinalities) -> None:
    df = _small_points_dataframe(train=2, validation=2, test=2, first_serve="4")
    adapted = adapter.adapt_real_points_dataframe(df)
    outcome_a = orchestrator.evaluate_final_sealed_test(adapted)
    outcome_b = orchestrator.evaluate_final_sealed_test(adapted)
    assert outcome_a == outcome_b


def test_orchestrator_rejects_wrong_input_type() -> None:
    with pytest.raises(orchestrator.FinalSealedEvaluationOrchestratorError):
        orchestrator.evaluate_final_sealed_test(object())


# --------------------------------------------------------------------- #
# F. Bloqueo 3: contrato de baseline poblacional P02 (decision humana)   #
# --------------------------------------------------------------------- #


def test_baseline_rule_matches_frozen_contract_text() -> None:
    assert orchestrator.P02_POPULATION_BASELINE_RULE == (
        "fixed_rate_from_labeled_p02_development_attempts_only; "
        "max_date_2023-12-31_by_construction; "
        "identical_for_rolling_origin_and_frozen; "
        "no_test_period_outcomes; "
        "null_if_labeled_denominator_is_zero"
    )


def test_compute_baseline_null_when_denominator_zero() -> None:
    baseline = orchestrator.compute_p02_population_baseline(())
    assert baseline.numerator == 0
    assert baseline.denominator == 0
    assert baseline.rate is None


def test_compute_baseline_exact_numerator_denominator() -> None:
    attempts = (
        adapter.P02ObservedAttempt("m1", "Alice", "wide", True),
        adapter.P02ObservedAttempt("m2", "Alice", "body", False),
        adapter.P02ObservedAttempt("m3", "Bob", "wide", True),
    )
    baseline = orchestrator.compute_p02_population_baseline(attempts)
    assert baseline.numerator == 2
    assert baseline.denominator == 3
    assert baseline.rate == pytest.approx(2 / 3)


def test_baseline_dataclass_rejects_tampered_rate() -> None:
    import dataclasses

    baseline = orchestrator.compute_p02_population_baseline(
        (adapter.P02ObservedAttempt("m1", "Alice", "wide", True),)
    )
    with pytest.raises(orchestrator.FinalSealedEvaluationOrchestratorError):
        dataclasses.replace(baseline, rate=0.999999)


def test_baseline_dataclass_rejects_foreign_rule_text() -> None:
    with pytest.raises(orchestrator.FinalSealedEvaluationOrchestratorError):
        orchestrator.P02PopulationBaseline("a different rule", 1, 1, 1.0)


def test_baseline_fingerprint_is_deterministic_and_independent_of_protocol() -> None:
    baseline_a = orchestrator.compute_p02_population_baseline(
        (adapter.P02ObservedAttempt("m1", "Alice", "wide", True),)
    )
    baseline_b = orchestrator.compute_p02_population_baseline(
        (adapter.P02ObservedAttempt("m1", "Alice", "wide", True),)
    )
    fp_a = orchestrator.compute_p02_population_baseline_fingerprint(baseline_a)
    fp_b = orchestrator.compute_p02_population_baseline_fingerprint(baseline_b)
    assert fp_a == fp_b
    assert len(fp_a) == 64


def test_rolling_and_frozen_receive_identical_baseline(small_outcome) -> None:
    """Adversarial: fallaria si rolling y frozen recibiesen baselines
    distintas."""
    rolling_baseline = small_outcome.rolling.p02_population_baseline_comparison
    frozen_baseline = small_outcome.frozen.p02_population_baseline_comparison
    for key in (
        "population_baseline_rule",
        "population_baseline_numerator",
        "population_baseline_denominator",
        "population_baseline_rate",
    ):
        assert rolling_baseline[key] == frozen_baseline[key]


def test_baseline_denominator_matches_exactly_development_p02_observed(
    small_cardinalities,
) -> None:
    """Adversarial: fallaria si la baseline incorporase cualquier fila
    del periodo de test (posterior a 2023-12-31)."""
    df = _small_points_dataframe(train=2, validation=2, test=2, first_serve="4")
    adapted = adapter.adapt_real_points_dataframe(df)
    outcome = orchestrator.evaluate_final_sealed_test(adapted)
    assert outcome.p02_population_baseline.denominator == len(
        adapted.development_p02_observed_attempts
    )
    assert outcome.p02_population_baseline.numerator == sum(
        1 for item in adapted.development_p02_observed_attempts if item.server_won_point
    )


# --------------------------------------------------------------------- #
# G. Bloqueo 2: direccion/outcome/score nunca se intercambian            #
# --------------------------------------------------------------------- #


def test_labeled_attempt_score_matches_its_own_observed_category(small_outcome) -> None:
    """Adversarial: fallaria si se usase el score de otra categoria (por
    ejemplo, el top-ranked) en vez del de la categoria REALMENTE
    observada."""
    df = _small_points_dataframe(train=2, validation=2, test=2, first_serve="4")
    adapted = adapter.adapt_real_points_dataframe(df)
    outcome = orchestrator.evaluate_final_sealed_test(adapted)
    from src.analysis.final_sealed_evaluation_boundary import evaluate_sealed_test_population
    from src.analysis.final_sealed_evaluation_boundary import (
        ROLLING_PROTOCOL,
        SealedTestTargetEvaluation,
        build_test_targets,
    )

    targets = build_test_targets(adapted.test_match_rows, enforce_frozen_cardinalities=True)
    rolling = evaluate_sealed_test_population(
        targets, adapted.development_observations, adapted.test_observations,
        adapted.schema, protocol=ROLLING_PROTOCOL,
    )
    labeled = orchestrator._build_p02_labeled_attempts(
        adapted.test_p02_observed_attempts, rolling
    )
    index = {
        (item.target.target_match_id, item.target.player): item
        for item in rolling.evaluations
        if isinstance(item, SealedTestTargetEvaluation)
    }
    for item in labeled:
        matching_observed = [
            observed for observed in adapted.test_p02_observed_attempts
            if observed.target_match_id == item.target_match_id
            and observed.player == item.player
        ]
        assert matching_observed, "Etiqueta sin observacion de origen."
        # El outcome debe proceder EXACTAMENTE del intento observado
        # original, nunca de otro.
        assert item.observed_server_won_point == matching_observed[0].server_won_point
        # El score debe ser el de LA MISMA categoria observada.
        evaluation = index[(item.target_match_id, item.player)]
        ranking = next(
            r for r in evaluation.prioritization.rankings if r.pattern_id == "P02"
        )
        candidate = next(c for c in ranking.candidates if c.category == item.category)
        assert item.predicted_score == candidate.combined_rate
        # Y NO debe coincidir por accidente con el de otra categoria
        # distinta (si hay mas de una categoria con score valido).
        other_scored = [
            c for c in ranking.candidates
            if c.category != item.category and c.combined_rate is not None
        ]
        for other in other_scored:
            if other.combined_rate != candidate.combined_rate:
                assert item.predicted_score != other.combined_rate


def test_single_test_match_never_sees_its_own_points_as_history(monkeypatch) -> None:
    """Adversarial: fallaria si el intento evaluado entrase en su
    propia historia."""
    monkeypatch.setattr(adapter, "EXPECTED_TRAIN_MATCHES", 2)
    monkeypatch.setattr(adapter, "EXPECTED_VALIDATION_MATCHES", 2)
    monkeypatch.setattr(adapter, "EXPECTED_TEST_MATCHES", 1)
    monkeypatch.setattr(adapter, "EXPECTED_TEST_ORIENTATIONS", 2)
    import src.analysis.final_sealed_evaluation as p20

    monkeypatch.setattr(p20, "EXPECTED_TEST_MATCHES", 1)
    monkeypatch.setattr(p20, "EXPECTED_TEST_ORIENTATIONS", 2)
    df = _small_points_dataframe(train=2, validation=2, test=1, first_serve="4")
    adapted = adapter.adapt_real_points_dataframe(df)
    from src.analysis.final_sealed_evaluation_boundary import (
        ROLLING_PROTOCOL,
        SealedTestTargetEvaluation,
        build_test_targets,
        evaluate_sealed_test_population,
    )

    targets = build_test_targets(adapted.test_match_rows, enforce_frozen_cardinalities=True)
    rolling = evaluate_sealed_test_population(
        targets, adapted.development_observations, adapted.test_observations,
        adapted.schema, protocol=ROLLING_PROTOCOL,
    )
    for item in rolling.evaluations:
        assert isinstance(item, SealedTestTargetEvaluation)
        assert item.visible_test_observation_count == 0


def test_two_same_day_test_matches_never_see_each_other(small_cardinalities, monkeypatch) -> None:
    """Adversarial: fallaria si se incorporase un outcome del mismo dia
    (dos partidos de test en la misma fecha)."""
    df = _small_points_dataframe(train=2, validation=2, test=1, first_serve="4")
    test_rows = df[df["match_id"].str.startswith("test")]
    same_day = test_rows.copy()
    same_day["match_id"] = same_day["match_id"] + "-same-day"
    same_day["player_1"] = "Carol"
    same_day["player_2"] = "Dave"
    combined = pd.concat([df, same_day], ignore_index=True)

    import src.analysis.final_sealed_evaluation as p20

    monkeypatch.setattr(adapter, "EXPECTED_TEST_MATCHES", 2)
    monkeypatch.setattr(adapter, "EXPECTED_TEST_ORIENTATIONS", 4)
    monkeypatch.setattr(p20, "EXPECTED_TEST_MATCHES", 2)
    monkeypatch.setattr(p20, "EXPECTED_TEST_ORIENTATIONS", 4)

    adapted = adapter.adapt_real_points_dataframe(combined)
    from src.analysis.final_sealed_evaluation_boundary import (
        ROLLING_PROTOCOL,
        SealedTestTargetEvaluation,
        build_test_targets,
        evaluate_sealed_test_population,
    )

    targets = build_test_targets(adapted.test_match_rows, enforce_frozen_cardinalities=True)
    rolling = evaluate_sealed_test_population(
        targets, adapted.development_observations, adapted.test_observations,
        adapted.schema, protocol=ROLLING_PROTOCOL,
    )
    for item in rolling.evaluations:
        assert isinstance(item, SealedTestTargetEvaluation)
        assert item.visible_test_observation_count == 0
