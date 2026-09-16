"""Tests P22 (Bloqueo 1, revision): paridad exacta del refactor de
``construct_tactical_attempt_records`` tras extraer el helper comun
privado ``_convert_points_to_attempt_records``, y verificacion de la
nueva funcion simetrica ``construct_tactical_attempt_records_for_test_
partition``.

Este archivo NO modifica ``tests/test_tactical_recommender_pipeline.py``
(sigue intacto: unica evidencia de paridad autoritativa es que ESA
suite completa, sin cambios, pasa identico antes y despues del
refactor -- ver informe). Este archivo AÑADE cobertura especifica del
refactor, reutilizando la fixture ``_points()`` ya validada de esa
misma suite (cubre P02/P04/P05/P06, primer y segundo saque, y una fila
del periodo de test) en vez de inventar secuencias nuevas.

Ningun test lee Parquet/CSV reales: ``_points()`` es enteramente
sintetica."""

from __future__ import annotations

import ast
import inspect
from datetime import date
from pathlib import Path

import pytest

import src.analysis.tactical_recommender_pipeline as p10
from src.recommender.tactical_feature_encoder import (
    TacticalEncodingPolicy,
    build_tactical_feature_schema,
    encode_tactical_attempt,
)
from src.recommender.tactical_signal_orchestrator import (
    ORCHESTRATOR_CONTRACT_VERSION,
    AttemptSignalRequest,
    extract_tactical_signals_for_attempt,
)
from test_tactical_recommender_pipeline import _points


@pytest.fixture(scope="module")
def schema():
    return build_tactical_feature_schema(TacticalEncodingPolicy.COMPONENT_ONLY)


@pytest.fixture(scope="module")
def split_points():
    """Reutiliza _points() (test_tactical_recommender_pipeline.py) ya
    validada; la separa con las mismas fronteras autoritativas P10."""
    validated = p10.validate_source_points(_points())
    development, _population, _seal = p10.seal_development_points(validated)
    sealed = validated.loc[validated.date.ge(p10.TEST_START)].copy(deep=True)
    return development, sealed


# --------------------------------------------------------------------- #
# A. Paridad byte/objeto exacta: wrapper == helper comun invocado       #
#    directamente con el mismo limite                                    #
# --------------------------------------------------------------------- #


def test_development_wrapper_matches_shared_helper_directly(schema, split_points) -> None:
    development, _sealed = split_points
    via_wrapper = p10.construct_tactical_attempt_records(development, schema)
    via_helper = p10._convert_points_to_attempt_records(
        development, schema, max_effective_date=p10.DEVELOPMENT_CUTOFF.date()
    )
    assert via_wrapper.records == via_helper.records
    assert via_wrapper.cache_metrics == via_helper.cache_metrics
    assert via_wrapper.first_attempts == via_helper.first_attempts
    assert via_wrapper.second_attempts == via_helper.second_attempts


def test_test_partition_wrapper_matches_shared_helper_directly(schema, split_points) -> None:
    _development, sealed = split_points
    via_wrapper = p10.construct_tactical_attempt_records_for_test_partition(sealed, schema)
    via_helper = p10._convert_points_to_attempt_records(
        sealed, schema, max_effective_date=p10.EXPECTED_LAST_DATE.date()
    )
    assert via_wrapper.records == via_helper.records


def test_development_batch_deterministic_across_repeated_calls(schema, split_points) -> None:
    development, _sealed = split_points
    first = p10.construct_tactical_attempt_records(development, schema)
    second = p10.construct_tactical_attempt_records(development, schema)
    assert first.records == second.records


# --------------------------------------------------------------------- #
# A2. Paridad NO tautologica: implementacion de referencia INDEPENDIENTE #
#     (no llama a _convert_points_to_attempt_records ni a los wrappers   #
#     salvo para comparar su resultado final) y fixture dorada           #
# --------------------------------------------------------------------- #


def _is_substantive_sequence(value: object) -> bool:
    """Mismo CONTRATO documentado (nulo/vacio/solo-espacios/sustantivo),
    reimplementado de forma independiente para esta referencia."""
    if value is None:
        return False
    if type(value) is not str:
        return False
    if value == "" or value.isspace():
        return False
    return True


def _reference_construct_attempt_records(points, schema):
    """Implementacion de referencia EXCLUSIVA de test, basada solo en el
    contrato documentado de TacticalAttemptRecord (servidor/restador
    desde ``server`` 1|2, ``server_won_point = point_winner == server``,
    segundo intento solo si ``second_serve`` es sustantivo, fault
    propagado del primer al segundo intento): NUNCA llama a
    ``_convert_points_to_attempt_records`` ni a ``construct_tactical_
    attempt_records``. Reutiliza unicamente los primitivos publicos de
    extraccion/codificacion (cuya correccion tiene su propia suite
    dedicada, fuera de alcance aqui). Esta prueba FALLARIA si el helper
    compartido se alterase, aunque wrapper y helper siguiesen
    coincidiendo entre si."""
    records = []
    for row in points.itertuples(index=False):
        server_player = row.player_1 if row.server == 1 else row.player_2
        returner_player = row.player_2 if row.server == 1 else row.player_1
        server_won_point = bool(row.point_winner == row.server)
        effective_date = row.date.date() if hasattr(row.date, "date") else row.date

        first_request = AttemptSignalRequest(
            ORCHESTRATOR_CONTRACT_VERSION, row.first_serve, 1, False
        )
        first_extraction = extract_tactical_signals_for_attempt(first_request)
        first_vector = encode_tactical_attempt(first_extraction, schema)
        first_fault = any(
            "censored_service_fault" in adaptation.reason_codes
            for adaptation in first_extraction.adaptations
        )
        records.append(
            p10.TacticalAttemptRecord(
                row.match_id, int(row.point_number), 1, effective_date, row.surface,
                server_player, returner_player, row.first_serve, False, first_fault,
                server_won_point, first_extraction, first_vector,
            )
        )
        if _is_substantive_sequence(row.second_serve):
            second_request = AttemptSignalRequest(
                ORCHESTRATOR_CONTRACT_VERSION, row.second_serve, 2, first_fault
            )
            second_extraction = extract_tactical_signals_for_attempt(second_request)
            second_vector = encode_tactical_attempt(second_extraction, schema)
            records.append(
                p10.TacticalAttemptRecord(
                    row.match_id, int(row.point_number), 2, effective_date, row.surface,
                    server_player, returner_player, row.second_serve, first_fault, False,
                    server_won_point, second_extraction, second_vector,
                )
            )
    return tuple(
        sorted(
            records,
            key=lambda item: (item.effective_date, item.match_id, item.point_number, item.serve_number),
        )
    )


def test_development_batch_matches_independent_reference_implementation(
    schema, split_points
) -> None:
    """No tautologica: la referencia NUNCA llama al helper compartido ni
    al wrapper. Cubre P02/P04/P05/P06, servidor/restador,
    server_won_point True/False, primer/segundo saque, falta/doble
    falta, censura, orden determinista y multiples puntos/partidos (via
    _points())."""
    development, _sealed = split_points
    batch = p10.construct_tactical_attempt_records(development, schema)
    reference = _reference_construct_attempt_records(development, schema)
    assert batch.records == reference


def test_test_partition_batch_matches_independent_reference_implementation(
    schema, split_points
) -> None:
    _development, sealed = split_points
    batch = p10.construct_tactical_attempt_records_for_test_partition(sealed, schema)
    reference = _reference_construct_attempt_records(sealed, schema)
    assert batch.records == reference


# Fixture dorada: valores literales esperados (independientes de
# CUALQUIER implementacion, calculados a mano desde las filas de
# _points()) para los campos trivialmente derivables de cada fila --
# server/returner/outcome/identidad/fecha/secuencia/serve_number.
_GOLDEN_DEVELOPMENT_FIRST_SERVE_FIELDS = (
    # (match_id, point_number, server_player, returner_player, server_won_point, date, sequence)
    ("m1", 1, "Alice", "Dora", True, date(2020, 1, 1), "4f17"),
    ("m2", 1, "Carol", "Bob", True, date(2020, 1, 2), "4f17"),
    ("m3", 1, "Carol", "Alice", False, date(2020, 1, 3), "4f17"),
    ("m4", 1, "Bob", "Carol", False, date(2020, 1, 4), "4f17"),
    ("m5", 1, "Alice", "Dora", False, date(2023, 12, 31), "4n"),
)
_GOLDEN_DEVELOPMENT_SECOND_SERVE_FIELDS = (
    ("m3", 1, "Carol", "Alice", False, date(2020, 1, 3), "6b28"),
    ("m5", 1, "Alice", "Dora", False, date(2023, 12, 31), "6f17"),
)


def test_development_batch_matches_golden_hand_computed_values(schema, split_points) -> None:
    """Fixture dorada literal (no deriva de ninguna implementacion,
    tampoco de la referencia anterior): valores calculados a mano desde
    las 5 filas de desarrollo de _points()."""
    development, _sealed = split_points
    batch = p10.construct_tactical_attempt_records(development, schema)
    firsts = {
        (item.match_id, item.point_number): item
        for item in batch.records if item.serve_number == 1
    }
    seconds = {
        (item.match_id, item.point_number): item
        for item in batch.records if item.serve_number == 2
    }
    assert len(firsts) == len(_GOLDEN_DEVELOPMENT_FIRST_SERVE_FIELDS)
    for match_id, point_number, server, returner, won, when, sequence in (
        _GOLDEN_DEVELOPMENT_FIRST_SERVE_FIELDS
    ):
        record = firsts[(match_id, point_number)]
        assert record.server_player == server
        assert record.returner_player == returner
        assert record.server_won_point is won
        assert record.effective_date == when
        assert record.sequence_text == sequence
    assert len(seconds) == len(_GOLDEN_DEVELOPMENT_SECOND_SERVE_FIELDS)
    for match_id, point_number, server, returner, won, when, sequence in (
        _GOLDEN_DEVELOPMENT_SECOND_SERVE_FIELDS
    ):
        record = seconds[(match_id, point_number)]
        assert record.server_player == server
        assert record.returner_player == returner
        assert record.server_won_point is won
        assert record.effective_date == when
        assert record.sequence_text == sequence


# --------------------------------------------------------------------- #
# B. Cobertura P02/P04/P05/P06, primer/segundo saque, censura            #
# --------------------------------------------------------------------- #


def test_development_batch_covers_all_four_patterns(schema, split_points) -> None:
    development, _sealed = split_points
    batch = p10.construct_tactical_attempt_records(development, schema)
    applicable: set[str] = set()
    for record in batch.records:
        applicable.update(record.extraction.applicable_patterns)
    assert {"P02", "P04", "P05", "P06"} <= applicable


def test_development_batch_has_first_and_second_serve_attempts(schema, split_points) -> None:
    development, _sealed = split_points
    batch = p10.construct_tactical_attempt_records(development, schema)
    assert batch.first_attempts > 0
    assert batch.second_attempts > 0
    assert any(record.serve_number == 2 for record in batch.records)


def test_development_batch_has_server_and_returner_roles_reconciled(
    schema, split_points
) -> None:
    development, _sealed = split_points
    batch = p10.construct_tactical_attempt_records(development, schema)
    for record in batch.records:
        assert record.server_player != record.returner_player
        assert type(record.server_won_point) is bool


def test_development_batch_documents_service_fault_context(schema, split_points) -> None:
    development, _sealed = split_points
    batch = p10.construct_tactical_attempt_records(development, schema)
    seconds = [item for item in batch.records if item.serve_number == 2]
    firsts_by_key = {
        (item.match_id, item.point_number): item
        for item in batch.records
        if item.serve_number == 1
    }
    assert seconds, "La fixture debe incluir al menos un segundo intento."
    for second in seconds:
        first = firsts_by_key[(second.match_id, second.point_number)]
        assert second.previous_attempt_was_fault == first.documents_service_fault
        assert second.documents_service_fault is False
        assert first.previous_attempt_was_fault is False


# --------------------------------------------------------------------- #
# C. P10 sigue rechazando test; P22 acepta EXCLUSIVAMENTE su rango       #
# --------------------------------------------------------------------- #


def test_construct_tactical_attempt_records_still_rejects_test_dates(schema) -> None:
    mixed = p10.validate_source_points(_points())
    with pytest.raises(p10.TacticalPipelineContractError):
        p10.construct_tactical_attempt_records(mixed, schema)


def test_test_partition_wrapper_rejects_development_dates(schema, split_points) -> None:
    development, _sealed = split_points
    with pytest.raises(p10.TacticalPipelineContractError):
        p10.construct_tactical_attempt_records_for_test_partition(development, schema)


def test_test_partition_wrapper_rejects_mixed_dev_and_test_rows(schema) -> None:
    mixed = p10.validate_source_points(_points())
    with pytest.raises(p10.TacticalPipelineContractError):
        p10.construct_tactical_attempt_records_for_test_partition(mixed, schema)


def test_no_function_accepts_both_partitions_simultaneously(schema, split_points) -> None:
    """Ninguna de las dos fronteras, por separado, procesa la particion
    del otro: no existe una tercera funcion generica que acepte ambas."""
    development, sealed = split_points
    with pytest.raises(p10.TacticalPipelineContractError):
        p10.construct_tactical_attempt_records(sealed, schema)
    with pytest.raises(p10.TacticalPipelineContractError):
        p10.construct_tactical_attempt_records_for_test_partition(development, schema)


def test_wrappers_do_not_expose_max_effective_date_to_callers() -> None:
    """Bloqueo 4: max_effective_date no debe permitir que un llamador
    EXTERNO de los dos wrappers productivos debilite su gate; solo es
    visible en la firma de las funciones internas de validacion, nunca
    en la de construct_tactical_attempt_records ni en la de
    construct_tactical_attempt_records_for_test_partition."""
    dev_params = set(inspect.signature(p10.construct_tactical_attempt_records).parameters)
    test_params = set(
        inspect.signature(p10.construct_tactical_attempt_records_for_test_partition).parameters
    )
    assert "max_effective_date" not in dev_params
    assert "max_effective_date" not in test_params


def test_wrappers_call_shared_helper_with_hardcoded_literal_bounds() -> None:
    """El limite que cada wrapper pasa al helper comun es un literal
    fijo (DEVELOPMENT_CUTOFF.date() / EXPECTED_LAST_DATE.date()), nunca
    un valor derivado de un parametro de entrada del wrapper."""
    dev_source = inspect.getsource(p10.construct_tactical_attempt_records)
    test_source = inspect.getsource(p10.construct_tactical_attempt_records_for_test_partition)
    assert "max_effective_date=DEVELOPMENT_CUTOFF.date()" in dev_source
    assert "max_effective_date=EXPECTED_LAST_DATE.date()" in test_source


def test_shared_helper_is_private_not_exported() -> None:
    assert "_convert_points_to_attempt_records" not in p10.__all__
    assert "construct_tactical_attempt_records" in p10.__all__
    assert "construct_tactical_attempt_records_for_test_partition" in p10.__all__


def test_ast_no_third_generic_partition_function() -> None:
    """No existe una funcion publica que acepte un parametro de rango
    o particion arbitrario: solo los dos wrappers cerrados y simetricos."""
    tree = ast.parse(Path(p10.__file__).read_text(encoding="utf-8"))
    public_functions = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]
    suspicious = [
        name for name in public_functions
        if "partition" in name.lower() or "range" in name.lower()
    ]
    assert set(suspicious) == {"construct_tactical_attempt_records_for_test_partition"}


# --------------------------------------------------------------------- #
# D. Suite existente de P10 sin modificaciones                           #
# --------------------------------------------------------------------- #


def test_existing_p10_test_file_is_untouched_by_this_review() -> None:
    """Documenta la evidencia de paridad autoritativa: la suite
    completa test_tactical_recommender_pipeline.py, SIN modificar,
    debe seguir dando 180 passed / 22 failed (fallos CRLF preexistentes
    de este checkout Windows, documentados en el informe; ver
    test_tactical_recommender_pipeline.py en git diff -- vacio)."""
    import subprocess

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "diff", "--stat", "--", "tests/test_tactical_recommender_pipeline.py"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout.strip() == "", (
        "tests/test_tactical_recommender_pipeline.py fue modificado; "
        "no deberia haber cambios segun el bloqueo de revision."
    )
