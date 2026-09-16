"""Tests sinteticos P20: preflight congelado de la evaluacion final del
test 2024-2026 (previa a P21). Ningun test lee Parquet/CSV reales,
abre el snapshot privado ni ejecuta P10 real: todo dato es sintetico o
proviene exclusivamente de constantes/contratos ya publicados.

Verifican: configuracion exacta frozen, dataclass inmutable, huella
independiente, puerta de autorizacion unica (estado vigente tras P23:
``True``, una unica ejecucion manual autorizada, con razon exacta y
contadores en cero; el camino "puerta cerrada" se ejercita parcheando
explicitamente a False), bloqueo antes de cualquier I/O, cero
reintentos, temporalidad exacta (cruzada contra
``chronological_validation.py`` como autoridad independiente),
denominadores de metricas (abstenciones, partially_available, labels
ausentes, empates, errores upstream, 2026 parcial), alcance de
Brier/log-loss/baseline limitado a P02 (cruzado contra
``explainable_direction_scoring.py``), reconciliacion anual, errores
cerrados sin fugas, AST sin bypass ni I/O de import, manifiesto sin
resultados reales, y compatibilidad con las constantes P10 ya
publicadas.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import src.analysis.final_sealed_evaluation as p20
import src.analysis.chronological_validation as p10_chrono
import src.analysis.tactical_recommender_pipeline as p10
import src.analysis.explainable_direction_scoring as p02_scoring


_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- #
# 1. Configuracion exacta frozen                                         #
# --------------------------------------------------------------------- #


def test_default_spec_matches_frozen_literals() -> None:
    spec = p20.default_final_sealed_evaluation_specification()
    assert spec.contract_version == p20.FINAL_SEALED_EVALUATION_CONTRACT_VERSION
    assert spec.test_start == "2024-01-01"
    assert spec.test_end == "2026-05-21"
    assert spec.expected_test_matches == 1_531
    assert spec.expected_test_orientations == 3_062
    assert spec.patterns == ("P02", "P04", "P05", "P06")
    assert spec.encoder_policy == "component_only"
    assert spec.evidence_scope == "global_only"
    assert spec.minimum_labeled_activations == 50
    assert spec.minimum_distinct_matches == 5
    assert spec.executor_weight == 0.5
    assert spec.opponent_allowed_weight == 0.5
    assert spec.requested_top_k == 3
    assert spec.ranking_scope == "independent_within_pattern"
    assert spec.global_cross_pattern_ranking is False


def test_validate_rejects_any_field_drift() -> None:
    spec = p20.default_final_sealed_evaluation_specification()
    # __post_init__ valida en cada construccion: dataclasses.replace()
    # reconstruye la instancia y por tanto revalida de inmediato.
    with pytest.raises(p20.FinalSealedEvaluationContractError):
        dataclasses.replace(spec, minimum_labeled_activations=25)
    tampered_scope = object.__new__(p20.FinalSealedEvaluationSpecification)
    for field in dataclasses.fields(spec):
        object.__setattr__(tampered_scope, field.name, getattr(spec, field.name))
    object.__setattr__(tampered_scope, "evidence_scope", "surface_then_global")
    with pytest.raises(p20.FinalSealedEvaluationContractError):
        p20.validate_final_sealed_evaluation_specification(tampered_scope)


def test_validate_rejects_non_50_50_combination() -> None:
    spec = p20.default_final_sealed_evaluation_specification()
    scope = object.__new__(p20.FinalSealedEvaluationSpecification)
    for field in dataclasses.fields(spec):
        object.__setattr__(scope, field.name, getattr(spec, field.name))
    object.__setattr__(scope, "executor_weight", 0.6)
    object.__setattr__(scope, "opponent_allowed_weight", 0.4)
    with pytest.raises(p20.FinalSealedEvaluationContractError):
        p20.validate_final_sealed_evaluation_specification(scope)


# --------------------------------------------------------------------- #
# 2. Dataclass frozen                                                    #
# --------------------------------------------------------------------- #


def test_specification_is_frozen_and_slotted() -> None:
    spec = p20.default_final_sealed_evaluation_specification()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.minimum_labeled_activations = 1
    assert not hasattr(spec, "__dict__")


# --------------------------------------------------------------------- #
# 3. Fingerprint independiente                                           #
# --------------------------------------------------------------------- #


def test_configuration_fingerprint_recomputed_independently() -> None:
    import hashlib

    spec = p20.default_final_sealed_evaluation_specification()
    payload = {
        "encoder_policy": spec.encoder_policy,
        "evidence_scope": spec.evidence_scope,
        "fallback_policy": spec.fallback_policy,
        "minimum_labeled_activations": spec.minimum_labeled_activations,
        "minimum_distinct_matches": spec.minimum_distinct_matches,
        "combination": spec.combination,
        "score_formula": spec.score_formula,
        "executor_weight": spec.executor_weight,
        "opponent_allowed_weight": spec.opponent_allowed_weight,
        "requested_top_k": spec.requested_top_k,
        "ranking_scope": spec.ranking_scope,
        "global_cross_pattern_ranking": spec.global_cross_pattern_ranking,
    }
    independent = hashlib.sha256(
        b"tennis-final-sealed-evaluation-configuration\x00"
        + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()
    assert independent == p20.compute_configuration_fingerprint(spec)


def _bypass_construct(
    spec: p20.FinalSealedEvaluationSpecification, **overrides: object
) -> p20.FinalSealedEvaluationSpecification:
    """Construye una variante SIN pasar por __post_init__ (solo para
    probar las funciones de fingerprint de forma aislada; nunca usar
    fuera de test)."""
    scope = object.__new__(p20.FinalSealedEvaluationSpecification)
    for field in dataclasses.fields(spec):
        object.__setattr__(scope, field.name, getattr(spec, field.name))
    for name, value in overrides.items():
        object.__setattr__(scope, name, value)
    return scope


def test_configuration_fingerprint_changes_only_with_configuration() -> None:
    spec = p20.default_final_sealed_evaluation_specification()
    other_dates = _bypass_construct(spec, test_start="2024-06-01")
    assert p20.compute_configuration_fingerprint(
        spec
    ) == p20.compute_configuration_fingerprint(other_dates)
    with pytest.raises(p20.FinalSealedEvaluationContractError):
        dataclasses.replace(spec, requested_top_k=5)
    other_config = _bypass_construct(spec, requested_top_k=5)
    assert p20.compute_configuration_fingerprint(
        spec
    ) != p20.compute_configuration_fingerprint(other_config)


def test_specification_fingerprint_is_deterministic() -> None:
    spec = p20.default_final_sealed_evaluation_specification()
    assert p20.compute_specification_fingerprint(
        spec
    ) == p20.compute_specification_fingerprint(spec)
    assert len(p20.compute_specification_fingerprint(spec)) == 64


# --------------------------------------------------------------------- #
# 4-5-6. Autorizacion unica, bloqueo antes de I/O, cero reintentos        #
# --------------------------------------------------------------------- #


def test_single_authorization_gate_is_true_with_closure_contract() -> None:
    """P23: autorizacion puntual para UNA UNICA ejecucion manual, tras
    auditoria completa de P20-P22 sin defectos bloqueantes restantes.
    Razon y politica de cierre son literales exactos; los cuatro
    contadores permanecen en cero hasta que esa unica ejecucion se
    complete, falle o se interrumpa -- momento en el que un COMMIT
    POSTERIOR (no este) debe devolver la puerta a False."""
    assert p20.REAL_TEST_EVALUATION_AUTHORIZED is True
    assert p20.REAL_TEST_EVALUATION_AUTHORIZATION_REASON == (
        "single_manual_final_sealed_test_evaluation_authorized_after_full_preflight"
    )
    assert p20.PREVIOUS_REAL_TEST_EVALUATIONS == 0
    assert p20.COMPLETED_REAL_TEST_EVALUATIONS == 0
    assert p20.INTERRUPTED_REAL_TEST_EVALUATIONS == 0
    assert p20.AUTOMATIC_RETRIES_PERFORMED == 0
    assert p20.AUTOMATIC_RETRY is False
    assert p20.AUTOMATIC_RETRY_POLICY == (
        "single_manual_execution_without_automatic_retry"
    )


def test_run_real_test_evaluation_raises_systemexit_before_any_io(
    monkeypatch,
) -> None:
    """Ejercita deliberadamente la rama con la puerta CERRADA (parcheada
    a False aqui): el valor real y vigente del modulo es True tras
    P23 (una unica ejecucion manual autorizada), pero esta rama de
    codigo debe seguir siendo correcta para el estado POSTERIOR al
    cierre (tras completar/fallar/interrumpir esa ejecucion, un commit
    futuro vuelve la constante a False; este test protege ese camino
    sin depender de si ya ocurrio)."""
    monkeypatch.setattr(p20, "REAL_TEST_EVALUATION_AUTHORIZED", False)

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("I/O invocado con la evaluacion sin autorizar.")

    monkeypatch.setattr("builtins.open", _forbidden)
    with pytest.raises(SystemExit) as exc_info:
        p20.run_real_test_evaluation()
    assert str(exc_info.value) == p20.REAL_TEST_EVALUATION_BLOCK_REASON
    with pytest.raises(SystemExit):
        p20.run_real_test_evaluation("cualquier", "argumento", clave="valor")


def test_run_real_test_evaluation_never_retries_and_is_idempotent(
    monkeypatch,
) -> None:
    monkeypatch.setattr(p20, "REAL_TEST_EVALUATION_AUTHORIZED", False)
    for _ in range(3):
        with pytest.raises(SystemExit) as exc_info:
            p20.run_real_test_evaluation()
        assert exc_info.value.code == p20.REAL_TEST_EVALUATION_BLOCK_REASON
    assert p20.AUTOMATIC_RETRIES_PERFORMED == 0


def test_no_bypass_env_var_or_cli_flag_exists() -> None:
    """No existe segunda puerta: ninguna variable de entorno cambia el
    valor de la constante del modulo. Se envenena con "false" -- el
    valor OPUESTO al real (True tras P23) -- para demostrar que el
    entorno tampoco puede APAGARLA, no solo que no puede encenderla."""
    monkeypatch_env = dict(os.environ)
    monkeypatch_env["REAL_TEST_EVALUATION_AUTHORIZED"] = "false"
    monkeypatch_env["TENNIS_FINAL_EVALUATION_FORCE"] = "0"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.analysis.final_sealed_evaluation as p20; "
            "print(p20.REAL_TEST_EVALUATION_AUTHORIZED)",
        ],
        cwd=str(_ROOT),
        env={**monkeypatch_env, "PYTHONPATH": str(_ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "True"


# --------------------------------------------------------------------- #
# 7. Temporalidad exacta (autoridad independiente)                       #
# --------------------------------------------------------------------- #


def test_temporal_boundaries_match_chronological_validation_authority() -> None:
    assert p20.TEST_START_DATE == p10_chrono.TEST_START.date().isoformat()
    assert p20.TEST_END_DATE == p10_chrono.TEST_END.date().isoformat()
    assert p20.EXPECTED_TEST_MATCHES == p10_chrono.EXPECTED_SPLIT_MATCHES["test"]
    assert p20.EXPECTED_TEST_ORIENTATIONS == p20.EXPECTED_TEST_MATCHES * 2


def test_primary_protocol_is_rolling_and_matches_validation_methodology() -> None:
    """El protocolo principal reutiliza literalmente el mismo
    history_update_policy que las 4 validaciones 2020-2023 ya
    auditadas, no un valor inventado para el test."""
    assert p20.PRIMARY_HISTORY_UPDATE_POLICY == "update_after_complete_date"
    assert p20.SENSITIVITY_HISTORY_UPDATE_POLICY == "frozen_at_2023-12-31"
    assert p20.PROTOCOLS_SHARE_POPULATION is True


# --------------------------------------------------------------------- #
# 8-11. Denominadores: abstenciones, partially_available, labels, empates #
# --------------------------------------------------------------------- #


def test_metric_denominator_rules_cover_required_keys() -> None:
    required = {
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
    assert required <= set(p20.METRIC_DENOMINATOR_RULES)


@pytest.mark.parametrize(
    "key",
    [
        "abstained_options",
        "partially_available_orientations",
        "missing_label",
        "ties",
    ],
)
def test_denominator_rule_never_drops_or_zero_fills(key) -> None:
    rule = p20.METRIC_DENOMINATOR_RULES[key]
    assert (
        "never" in rule
        or "excluded" in rule
        or "not_computable" in rule
        or "counted" in rule
    )


def test_incompatible_results_excluded_from_performance_denominators() -> None:
    rule = p20.METRIC_DENOMINATOR_RULES["incompatible_or_upstream_error_results"]
    assert "excluded_from_all_performance_metric_denominators" in rule


# --------------------------------------------------------------------- #
# 12-13. Brier/log-loss y baseline: alcance exclusivo P02                #
# --------------------------------------------------------------------- #


def test_brier_and_log_loss_are_scoped_to_p02_only() -> None:
    for metric in ("brier_score_vs_server_won_point_label", "log_loss_vs_server_won_point_label"):
        assert metric in p20.PRIMARY_METRICS_P02_ONLY
        assert metric not in p20.PRIMARY_METRICS_ALL_PATTERNS


def test_population_baseline_matches_existing_p02_precedent() -> None:
    """Cruce contra la autoridad independiente: el unico scorer de
    baseline poblacional que existe en el codigo es population_only en
    explainable_direction_scoring.py, exclusivo de P02."""
    assert "population_only" in p02_scoring.SCORERS
    assert "population_baseline_comparison" in p20.PRIMARY_METRICS_P02_ONLY
    assert "population_baseline_comparison" not in p20.PRIMARY_METRICS_ALL_PATTERNS


def test_p04_p05_p06_limited_to_coverage_metrics() -> None:
    """No existe pipeline de scoring/calibracion para P04/P05/P06: solo
    cobertura, calculable desde el contrato P11 sin nuevos labels."""
    coverage_only = set(p20.PRIMARY_METRICS_ALL_PATTERNS)
    assert not coverage_only & set(p20.PRIMARY_METRICS_P02_ONLY)
    assert all("brier" not in metric and "log_loss" not in metric for metric in coverage_only)


# --------------------------------------------------------------------- #
# 14-15. 2026 parcial y reconciliacion anual                             #
# --------------------------------------------------------------------- #


def test_year_2026_partial_rule_present_and_distinct() -> None:
    rule = p20.METRIC_DENOMINATOR_RULES["year_2026_partial"]
    assert "2026-05-21" in rule
    assert "never_pooled" in rule


def test_reconciliation_by_year_is_a_primary_metric_for_all_patterns() -> None:
    assert "reconciliation_by_year_within_test" in p20.PRIMARY_METRICS_ALL_PATTERNS


# --------------------------------------------------------------------- #
# 16-17. Errores cerrados y privacidad                                   #
# --------------------------------------------------------------------- #


def test_block_reason_is_closed_without_paths_or_identities() -> None:
    message = p20.REAL_TEST_EVALUATION_BLOCK_REASON
    assert "/" not in message
    assert "\\" not in message
    assert str(_ROOT) not in message
    assert "player" not in message.lower()


def test_contract_error_never_leaks_field_values() -> None:
    spec = p20.default_final_sealed_evaluation_specification()
    with pytest.raises(p20.FinalSealedEvaluationContractError) as exc_info:
        dataclasses.replace(spec, minimum_distinct_matches=999)
    assert "999" not in str(exc_info.value)


# --------------------------------------------------------------------- #
# 18. AST: sin bypass, sin I/O de import, sin pandas/parquet             #
# --------------------------------------------------------------------- #


def test_ast_imports_cerrados_sin_pandas_ni_lectores_reales() -> None:
    tree = ast.parse(Path(p20.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)
    banned = {"pandas", "pyarrow", "argparse", "subprocess", "requests", "httpx"}
    overlap = {
        name for name in imported
        if any(name == bad or name.startswith(bad + ".") for bad in banned)
    }
    assert overlap == set()


def test_ast_no_environ_no_argv_no_read_parquet_no_read_csv() -> None:
    source = Path(p20.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {
                "environ", "getenv", "argv", "read_parquet", "read_csv",
            }, f"Acceso prohibido: {node.attr}"


def test_ast_no_module_level_calls_other_than_docstring() -> None:
    tree = ast.parse(Path(p20.__file__).read_text(encoding="utf-8"))
    for statement in tree.body:
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and type(statement.value.value) is str
        ):
            continue
        assert isinstance(
            statement,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.ClassDef,
                ast.Assign,
                ast.AnnAssign,
                ast.If,
            ),
        ), f"Estado de modulo no permitido: {type(statement).__name__}"


def test_import_subprocess_sin_efectos(tmp_path) -> None:
    sentinel = tmp_path / "sentinela.txt"
    sentinel.write_text("x", encoding="utf-8")
    before = {item.name for item in tmp_path.iterdir()}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.analysis.final_sealed_evaluation as p20; "
            "print(p20.REAL_TEST_EVALUATION_AUTHORIZED)",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "True"
    assert completed.stderr == ""
    after = {item.name for item in tmp_path.iterdir()}
    assert after == before


def test_no_real_source_reader_referenced_anywhere() -> None:
    source = Path(p20.__file__).read_text(encoding="utf-8")
    for forbidden in ("read_parquet", "read_csv", "points_enriched", "builtins.open"):
        assert forbidden not in source


# --------------------------------------------------------------------- #
# 19. Manifiesto sin resultados reales                                   #
# --------------------------------------------------------------------- #


def test_manifest_status_preflight_authorized_true_zero_counts() -> None:
    """Tras P23, ``authorized`` refleja la constante vigente (True para
    la unica ejecucion manual autorizada); los contadores permanecen en
    cero porque esa ejecucion aun no se ha completado, fallado ni
    interrumpido."""
    manifest = p20.build_preflight_manifest()
    assert manifest["status"] == "preflight_only"
    assert manifest["authorized"] is True
    assert manifest["test_evaluation_runs"] == 0
    assert manifest["previous_real_test_evaluations"] == 0
    assert manifest["interrupted_real_test_evaluations"] == 0
    assert manifest["automatic_retries_performed"] == 0
    assert manifest["no_real_metrics_observed"] is True
    assert manifest["no_target_level_data"] is True
    assert manifest["no_identities"] is True


def test_manifest_is_json_serializable_and_contains_no_secrets() -> None:
    manifest = p20.build_preflight_manifest()
    blob = json.dumps(manifest, sort_keys=True)
    assert str(_ROOT) not in blob
    assert "data/processed" not in blob
    assert "points_enriched" not in blob


def test_publish_preflight_manifest_writes_atomically(tmp_path) -> None:
    destination = tmp_path / "final_evaluation_preflight.json"
    written = p20.publish_preflight_manifest(destination)
    assert written == destination
    assert destination.is_file()
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded["status"] == "preflight_only"
    assert loaded["authorized"] is True
    remaining = list(tmp_path.iterdir())
    assert remaining == [destination]


# --------------------------------------------------------------------- #
# 20. Compatibilidad P10-P19 relevante                                    #
# --------------------------------------------------------------------- #


def test_compatible_with_p10_real_execution_gate_still_closed() -> None:
    """P20 no reabre ni relaja la puerta historica de P10."""
    assert p10.REAL_EXECUTION_AUTHORIZED is False
    assert p10.FURTHER_REAL_EXECUTION_AUTHORIZED is False


def test_compatible_with_chronological_validation_split_contract() -> None:
    assert p10_chrono.EXPECTED_SPLIT_MATCHES == {
        "train": 4_188,
        "validation": 1_805,
        "test": 1_531,
    }
    assert p10_chrono.VALIDATION_END < p10_chrono.TEST_START
