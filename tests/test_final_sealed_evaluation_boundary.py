"""Tests sinteticos P21: frontera segura y evaluador compute-only del
test sellado 2024-01-01..2026-05-21. Ningun test lee Parquet/CSV
reales, abre el snapshot privado, consulta variables de entorno ni
ejecuta P10 (``run_tactical_recommender_pipeline`` /
``compute_tactical_pipeline_result``): toda observacion y partido es
sintetico, construido con el mismo mecanismo semantico ya validado en
``test_tactical_recommendation_service.py`` (attempts -> orquestador
-> encoder -> evidencia -> priorizacion).

Cubre: limites exactos de fechas y +/-1 dia, mezcla de splits, orden
de entrada arbitrario, duplicados, columnas ausentes/tipo incorrecto,
exactamente dos orientaciones, varios partidos en la misma fecha,
cambio de historia solo tras cerrar el dia (calendar_day_rule), frozen
sin actualizacion, igualdad de poblacion rolling/frozen, label P02
presente/ausente, probabilidad 0/1 y proteccion de log-loss,
abstained/partially_available/error upstream, empates/rankings, 2024,
2025 y 2026 parcial separados, determinismo ante orden de entrada
equivalente, autorizacion cerrada sin bypass, y AST sin I/O de import.
"""

from __future__ import annotations

import ast
import math
import os
import subprocess
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

import pytest

import src.analysis.final_sealed_evaluation_boundary as boundary
from src.recommender.tactical_feature_encoder import (
    TacticalEncodingPolicy,
    build_tactical_feature_schema,
)
from src.recommender.tactical_prioritization import TacticalPrioritizationState
from test_tactical_recommendation_service import (
    _attempt,
    _full_history,
    _observations,
)


_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA = build_tactical_feature_schema(TacticalEncodingPolicy.COMPONENT_ONLY)


@lru_cache(maxsize=4)
def _dev_observations(player: str = "Alice", opponent: str = "Bob"):
    """Cacheado: _full_history genera ~2000 intentos sinteticos y
    codificarlos es el unico coste no trivial de este archivo; sin
    cache, cada test que lo invoca repetiria esa codificacion completa."""
    return _observations(_full_history(0, player, opponent), _SCHEMA)


def _test_attempt(
    *, match_id: str, effective_date: date, server: str, returner: str, success: bool = True
):
    return _attempt(
        match_id=match_id,
        point_number=1,
        effective_date=effective_date,
        server=server,
        returner=returner,
        sequence="h4#",
        success=success,
    )


def _row(match_id: str, when: date, player_1="Alice", player_2="Bob", surface="Hard"):
    return boundary.SealedTestMatchRow(match_id, when, player_1, player_2, surface)


# --------------------------------------------------------------------- #
# A. Import sin efectos e invariantes AST                                #
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
            "import src.analysis.final_sealed_evaluation_boundary as b; "
            "print(b.REAL_TEST_EVALUATION_AUTHORIZED)",
        ],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "True"
    assert completed.stderr == ""
    after = {item.name for item in tmp_path.iterdir()}
    assert after == before


def test_ast_no_pandas_no_environ_no_readers() -> None:
    source = Path(boundary.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)
    banned_modules = {"pandas", "pyarrow", "argparse", "subprocess", "requests", "httpx"}
    assert not (
        imported
        & {name for bad in banned_modules for name in imported if name == bad or name.startswith(bad + ".")}
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {
                "environ", "getenv", "argv", "read_parquet", "read_csv",
            }, f"Acceso prohibido: {node.attr}"


def test_ast_no_module_level_calls() -> None:
    tree = ast.parse(Path(boundary.__file__).read_text(encoding="utf-8"))
    for statement in tree.body:
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and type(statement.value.value) is str
        ):
            continue
        assert isinstance(
            statement,
            (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign),
        ), f"Estado de modulo no permitido: {type(statement).__name__}"


def test_no_upstream_pipeline_symbols_referenced() -> None:
    """P21 no reutiliza run_tactical_recommender_pipeline /
    compute_tactical_pipeline_result / seal_development_points: no
    tocan la frontera de archivo de P10. Se comprueba en codigo
    ejecutable (imports/nombres), no en el docstring que los explica."""
    tree = ast.parse(Path(boundary.__file__).read_text(encoding="utf-8"))
    forbidden = {
        "run_tactical_recommender_pipeline",
        "compute_tactical_pipeline_result",
        "seal_development_points",
        "TacticalPipelineTarget",
    }
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            used.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
    assert used.isdisjoint(forbidden)


# --------------------------------------------------------------------- #
# B. Validacion de filas/targets: limites exactos y +/-1 dia             #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "when", [date(2024, 1, 1), date(2026, 5, 21)], ids=["start", "end"]
)
def test_match_row_accepts_exact_boundaries(when) -> None:
    row = _row("m1", when)
    targets = boundary.build_test_targets((row,))
    assert len(targets) == 2


@pytest.mark.parametrize(
    "when", [date(2023, 12, 31), date(2026, 5, 22)], ids=["one_before_start", "one_after_end"]
)
def test_match_row_rejects_one_day_outside_range(when) -> None:
    row = _row("m1", when)
    with pytest.raises(boundary.FinalSealedEvaluationBoundaryError):
        boundary.build_test_targets((row,))


def test_match_row_rejects_same_player() -> None:
    with pytest.raises(boundary.FinalSealedEvaluationBoundaryError):
        _row("m1", date(2024, 6, 1), player_1="Alice", player_2="Alice")


def test_match_row_rejects_invalid_surface() -> None:
    with pytest.raises(boundary.FinalSealedEvaluationBoundaryError):
        _row("m1", date(2024, 6, 1), surface="Indoor")


def test_target_rejects_non_date_as_of_date() -> None:
    with pytest.raises(boundary.FinalSealedEvaluationBoundaryError):
        boundary.SealedTestTarget(
            "m1", "Alice", "Bob", "2024-06-01", "player_1_vs_player_2", "Hard"
        )


def test_target_rejects_bad_orientation() -> None:
    with pytest.raises(boundary.FinalSealedEvaluationBoundaryError):
        boundary.SealedTestTarget(
            "m1", "Alice", "Bob", date(2024, 6, 1), "sideways", "Hard"
        )


# --------------------------------------------------------------------- #
# C. build_test_targets: orientaciones, splits mezclados, duplicados     #
# --------------------------------------------------------------------- #


def test_build_test_targets_exactly_two_orientations_ordered() -> None:
    targets = boundary.build_test_targets((_row("m1", date(2024, 6, 1)),))
    assert len(targets) == 2
    assert {t.orientation for t in targets} == set(boundary.ORIENTATIONS)
    first, second = targets
    assert (first.player, first.opponent) == ("Alice", "Bob")
    assert (second.player, second.opponent) == ("Bob", "Alice")


def test_build_test_targets_rejects_validation_or_train_rows_mixed_in() -> None:
    rows = (_row("m1", date(2024, 6, 1)), _row("m2", date(2023, 6, 1)))
    with pytest.raises(boundary.FinalSealedEvaluationBoundaryError):
        boundary.build_test_targets(rows)


def test_build_test_targets_rejects_duplicate_match_id() -> None:
    rows = (_row("m1", date(2024, 6, 1)), _row("m1", date(2024, 6, 2)))
    with pytest.raises(boundary.FinalSealedEvaluationBoundaryError):
        boundary.build_test_targets(rows)


def test_build_test_targets_rejects_wrong_row_type() -> None:
    with pytest.raises(boundary.FinalSealedEvaluationBoundaryError):
        boundary.build_test_targets(({"match_id": "m1"},))


def test_build_test_targets_rejects_empty() -> None:
    with pytest.raises(boundary.FinalSealedEvaluationBoundaryError):
        boundary.build_test_targets(())


def test_build_test_targets_deterministic_regardless_of_input_order() -> None:
    rows_forward = (
        _row("m1", date(2024, 1, 5)),
        _row("m2", date(2024, 1, 3)),
        _row("m3", date(2024, 1, 4)),
    )
    rows_reversed = tuple(reversed(rows_forward))
    assert boundary.build_test_targets(rows_forward) == boundary.build_test_targets(
        rows_reversed
    )


def test_build_test_targets_several_matches_same_date() -> None:
    rows = (
        _row("m1", date(2024, 6, 1), player_1="Alice", player_2="Bob"),
        _row("m2", date(2024, 6, 1), player_1="Carol", player_2="Dave"),
    )
    targets = boundary.build_test_targets(rows)
    assert len(targets) == 4
    assert len({t.target_match_id for t in targets}) == 2


def test_build_test_targets_enforce_frozen_cardinalities_rejects_wrong_count() -> None:
    with pytest.raises(boundary.FinalSealedEvaluationBoundaryError):
        boundary.build_test_targets(
            (_row("m1", date(2024, 6, 1)),), enforce_frozen_cardinalities=True
        )


# --------------------------------------------------------------------- #
# D. visible_observations_for_target: calendar_day_rule y frozen         #
# --------------------------------------------------------------------- #


def test_rolling_excludes_same_day_and_future_includes_strictly_earlier() -> None:
    target = boundary.SealedTestTarget(
        "m1", "Alice", "Bob", date(2024, 3, 15), "player_1_vs_player_2", "Hard"
    )
    earlier = _observations(
        (_test_attempt(match_id="a", effective_date=date(2024, 3, 14), server="Alice", returner="X"),),
        _SCHEMA,
    )
    same_day = _observations(
        (_test_attempt(match_id="b", effective_date=date(2024, 3, 15), server="Alice", returner="Y"),),
        _SCHEMA,
    )
    future = _observations(
        (_test_attempt(match_id="c", effective_date=date(2024, 3, 16), server="Alice", returner="Z"),),
        _SCHEMA,
    )
    pool = earlier + same_day + future
    visible = boundary.visible_observations_for_target(target, pool, protocol="rolling_origin")
    assert visible == earlier


def test_frozen_never_sees_test_period_regardless_of_target_year() -> None:
    boundary_obs = _observations(
        (_test_attempt(match_id="x", effective_date=date(2023, 12, 31), server="Alice", returner="X"),),
        _SCHEMA,
    )
    over_boundary_obs = _observations(
        (_test_attempt(match_id="y", effective_date=date(2024, 1, 1), server="Alice", returner="Y"),),
        _SCHEMA,
    )
    pool = boundary_obs + over_boundary_obs
    for target_year_date in (date(2024, 1, 2), date(2025, 6, 1), date(2026, 5, 21)):
        target = boundary.SealedTestTarget(
            "m1", "Alice", "Bob", target_year_date, "player_1_vs_player_2", "Hard"
        )
        visible = boundary.visible_observations_for_target(target, pool, protocol="frozen")
        assert visible == boundary_obs


def test_visible_observations_rejects_unknown_protocol() -> None:
    target = boundary.SealedTestTarget(
        "m1", "Alice", "Bob", date(2024, 6, 1), "player_1_vs_player_2", "Hard"
    )
    with pytest.raises(boundary.FinalSealedEvaluationBoundaryError):
        boundary.visible_observations_for_target(target, (), protocol="whatever")


# --------------------------------------------------------------------- #
# E. Evaluador de poblacion: frozen/rolling, perdidas, duplicados        #
# --------------------------------------------------------------------- #


def _targets_for(when: date = date(2024, 3, 15)) -> tuple:
    return boundary.build_test_targets((_row("m1", when),))


def test_evaluate_population_frozen_never_uses_test_observations() -> None:
    dev = _dev_observations()
    earlier_test = _observations(
        (_test_attempt(match_id="earlier", effective_date=date(2024, 1, 10), server="Alice", returner="NX"),),
        _SCHEMA,
    )
    targets = _targets_for()
    result = boundary.evaluate_sealed_test_population(
        targets, dev, earlier_test, _SCHEMA, protocol="frozen"
    )
    for item in result.evaluations:
        assert isinstance(item, boundary.SealedTestTargetEvaluation)
        assert item.visible_test_observation_count == 0
        assert item.visible_max_effective_date <= boundary.FROZEN_HISTORY_CUTOFF


def test_evaluate_population_rolling_sees_only_strictly_earlier_test() -> None:
    dev = _dev_observations()
    earlier_test = _observations(
        (_test_attempt(match_id="earlier", effective_date=date(2024, 1, 10), server="Alice", returner="NX"),),
        _SCHEMA,
    )
    same_day_test = _observations(
        (_test_attempt(match_id="same-day", effective_date=date(2024, 3, 15), server="Alice", returner="NY"),),
        _SCHEMA,
    )
    future_test = _observations(
        (_test_attempt(match_id="future", effective_date=date(2024, 3, 16), server="Alice", returner="NZ"),),
        _SCHEMA,
    )
    all_test = earlier_test + same_day_test + future_test
    targets = _targets_for()
    result = boundary.evaluate_sealed_test_population(
        targets, dev, all_test, _SCHEMA, protocol="rolling_origin"
    )
    for item in result.evaluations:
        assert isinstance(item, boundary.SealedTestTargetEvaluation)
        assert item.visible_test_observation_count == 1
        assert item.visible_max_effective_date == date(2024, 1, 10)


def test_rolling_and_frozen_share_exactly_the_same_population() -> None:
    dev = _dev_observations()
    targets = boundary.build_test_targets(
        (_row("m1", date(2024, 1, 5)), _row("m2", date(2024, 6, 1)))
    )
    rolling = boundary.evaluate_sealed_test_population(
        targets, dev, (), _SCHEMA, protocol="rolling_origin"
    )
    frozen = boundary.evaluate_sealed_test_population(
        targets, dev, (), _SCHEMA, protocol="frozen"
    )
    boundary.assert_protocols_share_population(rolling, frozen)


def test_assert_protocols_share_population_detects_mismatch() -> None:
    dev = _dev_observations()
    targets_a = boundary.build_test_targets((_row("m1", date(2024, 1, 5)),))
    targets_b = boundary.build_test_targets((_row("m2", date(2024, 6, 1)),))
    rolling = boundary.evaluate_sealed_test_population(
        targets_a, dev, (), _SCHEMA, protocol="rolling_origin"
    )
    frozen = boundary.evaluate_sealed_test_population(
        targets_b, dev, (), _SCHEMA, protocol="frozen"
    )
    with pytest.raises(boundary.FinalSealedEvaluationBoundaryError):
        boundary.assert_protocols_share_population(rolling, frozen)


def test_evaluate_population_rejects_duplicate_targets() -> None:
    target = boundary.SealedTestTarget(
        "m1", "Alice", "Bob", date(2024, 6, 1), "player_1_vs_player_2", "Hard"
    )
    with pytest.raises(boundary.FinalSealedEvaluationBoundaryError):
        boundary.evaluate_sealed_test_population(
            (target, target), _dev_observations(), (), _SCHEMA, protocol="frozen"
        )


def test_evaluate_population_rejects_overlapping_development_and_test_partitions() -> None:
    """Revision de diseno: development_observations y test_observations
    deben ser particiones disjuntas; solaparlas duplicaria influencia."""
    dev = _dev_observations()
    overlapping_test = _observations(
        (_test_attempt(match_id=dev[0].match_id, effective_date=date(2024, 1, 10), server="Alice", returner="NX"),),
        _SCHEMA,
    )
    targets = _targets_for()
    with pytest.raises(boundary.FinalSealedEvaluationBoundaryError):
        boundary.evaluate_sealed_test_population(
            targets, dev, overlapping_test, _SCHEMA, protocol="rolling_origin"
        )


def test_evaluate_population_rejects_incomplete_orientation_pair() -> None:
    lone = boundary.SealedTestTarget(
        "m1", "Alice", "Bob", date(2024, 6, 1), "player_1_vs_player_2", "Hard"
    )
    with pytest.raises(boundary.FinalSealedEvaluationBoundaryError):
        boundary.evaluate_sealed_test_population(
            (lone,), _dev_observations(), (), _SCHEMA, protocol="frozen"
        )


def test_evaluate_single_target_reports_error_on_contract_violation(monkeypatch) -> None:
    target = boundary.SealedTestTarget(
        "m1", "Alice", "Bob", date(2024, 6, 1), "player_1_vs_player_2", "Hard"
    )

    def _explode(*_args, **_kwargs):
        raise boundary.TacticalMatchupContractError("boom")

    monkeypatch.setattr(boundary, "build_tactical_matchup_evidence", _explode)
    outcome = boundary.evaluate_single_test_target(
        target, _dev_observations(), _SCHEMA, protocol="frozen"
    )
    assert isinstance(outcome, boundary.SealedTestTargetEvaluationError)
    assert outcome.reason_code == "upstream_contract_violation"


def test_error_bucket_excluded_from_coverage_denominator_but_counted_separately(
    monkeypatch,
) -> None:
    dev = _dev_observations()
    targets = boundary.build_test_targets(
        (_row("m1", date(2024, 1, 5)), _row("m2", date(2024, 6, 1)))
    )
    real_build = boundary.build_tactical_matchup_evidence
    call_count = {"n": 0}

    def _flaky(observations, query, schema):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise boundary.TacticalMatchupContractError("boom")
        return real_build(observations, query, schema)

    monkeypatch.setattr(boundary, "build_tactical_matchup_evidence", _flaky)
    result = boundary.evaluate_sealed_test_population(
        targets, dev, (), _SCHEMA, protocol="frozen"
    )
    errors = boundary.aggregate_error_bucket(result)
    coverage = boundary.aggregate_orientation_coverage_by_status(result)
    assert errors["error_count"] == 1
    assert coverage["excluded_as_error"] == 1
    assert coverage["counted"] == len(result.evaluations) - 1


# --------------------------------------------------------------------- #
# F. Metricas: cobertura, patrones, reconciliacion anual, wilson, ties   #
# --------------------------------------------------------------------- #


@lru_cache(maxsize=4)
def _sample_result(protocol: str = "frozen"):
    dev = _dev_observations()
    targets = boundary.build_test_targets(
        (
            _row("m2024", date(2024, 6, 1)),
            _row("m2025", date(2025, 6, 1)),
            _row("m2026", date(2026, 5, 20)),
        )
    )
    return boundary.evaluate_sealed_test_population(targets, dev, (), _SCHEMA, protocol=protocol)


def test_orientation_coverage_denominator_matches_target_count() -> None:
    result = _sample_result()
    coverage = boundary.aggregate_orientation_coverage_by_status(result)
    assert coverage["denominator"] == len(result.evaluations) == 6
    assert sum(coverage["by_status"].values()) == coverage["counted"]


def test_pattern_coverage_scored_vs_abstained_has_all_four_patterns() -> None:
    result = _sample_result()
    pattern_coverage = boundary.aggregate_pattern_coverage_scored_vs_abstained(result)
    assert set(pattern_coverage) == {"P02", "P04", "P05", "P06"}
    for values in pattern_coverage.values():
        assert values["scored"] + values["abstained"] == values["total_candidates"]


def test_reconciliation_by_year_separates_2024_2025_and_partial_2026() -> None:
    result = _sample_result()
    by_year = boundary.aggregate_reconciliation_by_year(result)
    assert set(by_year) == {"2024", "2025", "2026_partial"}
    assert by_year["2024"]["matches"] == 1
    assert by_year["2025"]["matches"] == 1
    assert by_year["2026_partial"]["matches"] == 1


def test_wilson_interval_width_distribution_bounds() -> None:
    result = _sample_result()
    widths = boundary.aggregate_wilson_interval_width_distribution(result)
    if widths["denominator"] > 0:
        assert 0.0 <= widths["min"] <= widths["mean"] <= widths["max"] <= 1.0


def test_rank_and_tie_distribution_shapes() -> None:
    result = _sample_result()
    distribution = boundary.aggregate_rank_and_tie_distribution(result)
    assert set(distribution) == {
        "rank_position_counts", "tie_group_size_counts", "boundary_tie_expanded_rankings",
    }


def test_abstention_reason_codes_present_when_evidence_insufficient() -> None:
    result = _sample_result()
    codes = boundary.aggregate_abstention_reason_codes(result)
    assert isinstance(codes, dict)


def test_coverage_by_surface_and_period_uses_year_bucket_convention() -> None:
    result = _sample_result()
    coverage = boundary.aggregate_coverage_by_surface_and_period(result)
    assert any(key.endswith("2026_partial") for key in coverage)


def test_all_p20_denominator_rule_keys_have_a_corresponding_aggregator() -> None:
    """Autoridad independiente: cada clave congelada por P20 en
    METRIC_DENOMINATOR_RULES debe corresponder a un comportamiento
    verificable en este modulo, no a una regla documentada pero no
    implementada."""
    import src.analysis.final_sealed_evaluation as p20

    implemented = {
        "orientation_coverage_by_status",
        "pattern_coverage_scored_vs_abstained",
        "brier_score_vs_server_won_point_label",
        "log_loss_vs_server_won_point_label",
        "population_baseline_comparison",
        "abstained_options",
        "missing_label",
        "partially_available_orientations",
        "ties",
        "incompatible_or_upstream_error_results",
        "year_2026_partial",
    }
    assert set(p20.METRIC_DENOMINATOR_RULES) == implemented


# --------------------------------------------------------------------- #
# G. P02: Brier/log-loss, probabilidad 0/1, baseline, labels ausentes    #
# --------------------------------------------------------------------- #


def test_p02_brier_log_loss_empty_is_none_not_zero() -> None:
    metrics = boundary.compute_p02_brier_and_log_loss(())
    assert metrics["denominator"] == 0
    assert metrics["brier_score"] is None
    assert metrics["log_loss"] is None


def test_p02_brier_log_loss_basic_values() -> None:
    attempts = (
        boundary.P02LabeledAttempt("m1", "Alice", "wide", 0.7, True),
        boundary.P02LabeledAttempt("m1", "Alice", "wide", 0.3, False),
    )
    metrics = boundary.compute_p02_brier_and_log_loss(attempts)
    assert metrics["denominator"] == 2
    assert metrics["brier_score"] == pytest.approx(((0.7 - 1.0) ** 2 + (0.3 - 0.0) ** 2) / 2)
    assert metrics["log_loss"] > 0.0


@pytest.mark.parametrize("score,label", [(0.0, True), (1.0, False), (0.0, False), (1.0, True)])
def test_p02_log_loss_never_raises_on_probability_0_or_1(score, label) -> None:
    attempts = (boundary.P02LabeledAttempt("m1", "Alice", "wide", score, label),)
    metrics = boundary.compute_p02_brier_and_log_loss(attempts)
    assert math.isfinite(metrics["log_loss"])


def test_p02_labeled_attempt_rejects_out_of_range_score() -> None:
    with pytest.raises(boundary.FinalSealedEvaluationBoundaryError):
        boundary.P02LabeledAttempt("m1", "Alice", "wide", 1.5, True)


def test_p02_population_baseline_delta_zero_when_model_equals_population() -> None:
    attempts = (
        boundary.P02LabeledAttempt("m1", "Alice", "wide", 0.5, True),
        boundary.P02LabeledAttempt("m1", "Alice", "wide", 0.5, False),
    )
    comparison = boundary.compute_p02_population_baseline_comparison(attempts, 0.5)
    assert comparison["delta_brier_vs_population"] == pytest.approx(0.0)
    assert comparison["delta_log_loss_vs_population"] == pytest.approx(0.0)


def test_p02_population_baseline_empty_is_none() -> None:
    comparison = boundary.compute_p02_population_baseline_comparison((), 0.5)
    assert comparison["denominator"] == 0
    assert comparison["delta_brier_vs_population"] is None


def test_p02_population_baseline_rejects_out_of_range_rate() -> None:
    with pytest.raises(boundary.FinalSealedEvaluationBoundaryError):
        boundary.compute_p02_population_baseline_comparison((), 1.2)


# --------------------------------------------------------------------- #
# H. Determinismo ante orden de entrada equivalente                      #
# --------------------------------------------------------------------- #


def test_population_evaluation_deterministic_under_shuffled_observation_order() -> None:
    dev = _dev_observations()
    dev_shuffled = tuple(reversed(dev))
    targets = _targets_for()
    result_a = boundary.evaluate_sealed_test_population(
        targets, dev, (), _SCHEMA, protocol="frozen"
    )
    result_b = boundary.evaluate_sealed_test_population(
        targets, dev_shuffled, (), _SCHEMA, protocol="frozen"
    )
    for item_a, item_b in zip(result_a.evaluations, result_b.evaluations):
        assert item_a.prioritization == item_b.prioritization


# --------------------------------------------------------------------- #
# I. Autorizacion: puerta unica, sin bypass, cero I/O                    #
# --------------------------------------------------------------------- #


def test_authorization_gate_still_true_and_reused_not_redefined() -> None:
    """P23: la puerta reexportada por P21 sigue siendo LA MISMA de P20
    (True, la unica ejecucion manual autorizada), nunca una segunda
    constante independiente. Contadores en cero: la ejecucion aun no
    se ha completado, fallado ni interrumpido."""
    import src.analysis.final_sealed_evaluation as p20

    assert boundary.REAL_TEST_EVALUATION_AUTHORIZED is True
    assert boundary.REAL_TEST_EVALUATION_AUTHORIZED is p20.REAL_TEST_EVALUATION_AUTHORIZED
    assert boundary.REAL_TEST_EVALUATION_BLOCK_REASON == p20.REAL_TEST_EVALUATION_BLOCK_REASON
    assert p20.PREVIOUS_REAL_TEST_EVALUATIONS == 0
    assert p20.COMPLETED_REAL_TEST_EVALUATIONS == 0
    assert p20.INTERRUPTED_REAL_TEST_EVALUATIONS == 0
    assert p20.AUTOMATIC_RETRIES_PERFORMED == 0


def test_run_real_test_evaluation_still_aborts_before_io(monkeypatch) -> None:
    """Ejercita deliberadamente la rama con la puerta CERRADA (parcheada
    a False aqui, en el modulo P20 real que define la funcion -- no en
    ``boundary``, que solo reexporta el mismo objeto de funcion). El
    valor vigente hoy es True (P23); este test protege el camino
    POSTERIOR al cierre de esa unica ejecucion autorizada."""
    import src.analysis.final_sealed_evaluation as p20

    monkeypatch.setattr(p20, "REAL_TEST_EVALUATION_AUTHORIZED", False)

    def _forbidden(*_a, **_k):
        raise AssertionError("I/O invocado sin autorizacion.")

    monkeypatch.setattr("builtins.open", _forbidden)
    with pytest.raises(SystemExit) as exc_info:
        boundary.run_real_test_evaluation()
    assert str(exc_info.value) == boundary.REAL_TEST_EVALUATION_BLOCK_REASON


def test_no_bypass_env_var_opens_the_gate() -> None:
    """Envenena con "false" -- el OPUESTO del valor real (True tras
    P23) -- para demostrar que el entorno tampoco puede apagarla."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT)
    env["REAL_TEST_EVALUATION_AUTHORIZED"] = "false"
    env["TENNIS_FINAL_EVALUATION_FORCE"] = "0"
    completed = subprocess.run(
        [
            sys.executable, "-c",
            "import src.analysis.final_sealed_evaluation_boundary as b; "
            "print(b.REAL_TEST_EVALUATION_AUTHORIZED)",
        ],
        cwd=str(_ROOT), env=env, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "True"


def test_boundary_functions_have_no_file_or_cli_parameters() -> None:
    """Ninguna funcion publica de la frontera acepta rutas o argv: no
    hay ninguna segunda via de ejecucion real que autorizar."""
    import inspect

    for name in boundary.__all__:
        obj = getattr(boundary, name)
        if not inspect.isfunction(obj):
            continue
        signature = inspect.signature(obj)
        for parameter in signature.parameters.values():
            assert parameter.name not in {"path", "source_path", "argv", "cli_args"}


# --------------------------------------------------------------------- #
# J. Errores cerrados: sin rutas ni identidades                          #
# --------------------------------------------------------------------- #


def test_contract_error_messages_never_leak_identities_or_paths() -> None:
    try:
        boundary.build_test_targets(())
    except boundary.FinalSealedEvaluationBoundaryError as error:
        message = str(error)
        assert "Alice" not in message
        assert str(_ROOT) not in message
        assert "\\" not in message
