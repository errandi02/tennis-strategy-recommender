"""Pruebas sinteticas del contrato individual P02."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import src.analysis.first_serve_direction_classification as module
from src.analysis.first_serve_direction_classification import (
    CLASSIFICATION_CONTRACT_VERSION,
    DIRECTION_DESCRIPTIONS,
    OBSERVABLE_DIRECTION_CODES,
    FirstServeDirectionClassification,
    FirstServeDirectionContractError,
    FirstServeDirectionReason,
    FirstServeDirectionState,
    classify_first_serve_direction,
    parse_and_classify_first_serve_direction,
    validate_first_serve_direction_classification,
)
from src.parsing.serve_sequence import Diagnostic, ParseResult, Span, parse_sequence


MODULE_PATH = Path("src/analysis/first_serve_direction_classification.py")


def _classification(sequence: str) -> FirstServeDirectionClassification:
    parsed = parse_sequence(sequence, 1)
    return classify_first_serve_direction(sequence, 1, parsed)


@pytest.mark.parametrize(
    ("code", "description"),
    [("4", "wide"), ("5", "body"), ("6", "down_the_t")],
)
def test_observable_direction_codes_match_documented_parser_meanings(code, description):
    result = _classification(code)
    assert result.classification_state is FirstServeDirectionState.OBSERVED
    assert result.reason_codes == (FirstServeDirectionReason.OBSERVED,)
    assert result.eligible_for_direction_signal is True
    assert result.actor == "server"
    assert result.direction_code == code
    assert result.direction_description == description
    assert result.direction_literal == code
    assert result.direction_span == (0, 1)
    assert result.service_prefix_literal == code
    assert result.service_prefix_span == (0, 1)
    assert result.service_let_count == 0
    assert result.service_terminal_kind is None
    assert OBSERVABLE_DIRECTION_CODES == frozenset({"4", "5", "6"})
    assert DIRECTION_DESCRIPTIONS[code] == description


def test_direction_zero_is_unknown_and_never_eligible():
    result = _classification("0")
    assert result.classification_state is FirstServeDirectionState.UNKNOWN
    assert result.reason_codes == (FirstServeDirectionReason.UNKNOWN_DIRECTION,)
    assert result.eligible_for_direction_signal is False
    assert result.actor == "server"
    assert result.direction_code == "0"
    assert result.direction_description == "unknown"


@pytest.mark.parametrize(
    ("sequence", "prefix", "direction_span", "let_count"),
    [("4", "4", (0, 1), 0), ("c5", "c5", (1, 2), 1), ("ccc6", "ccc6", (3, 4), 3)],
)
def test_complete_lets_are_part_of_the_exact_prefix(sequence, prefix, direction_span, let_count):
    result = _classification(sequence)
    assert result.classification_state is FirstServeDirectionState.OBSERVED
    assert result.service_prefix_literal == prefix
    assert result.service_prefix_span == (0, len(prefix))
    assert result.direction_span == direction_span
    assert result.service_let_count == let_count


@pytest.mark.parametrize(
    ("sequence", "terminal_kind", "terminal_code"),
    [("4*", "ace", "*"), ("5#", "unreturned", "#"), ("6n", "fault", "n")],
)
def test_documented_direction_remains_observed_for_service_terminals(
    sequence, terminal_kind, terminal_code
):
    result = _classification(sequence)
    assert result.classification_state is FirstServeDirectionState.OBSERVED
    assert result.eligible_for_direction_signal is True
    assert result.service_terminal_kind == terminal_kind
    assert result.service_terminal_code == terminal_code
    assert result.service_terminal_span == (1, 2)


@pytest.mark.parametrize("sequence", ["0*", "0#", "0n"])
def test_unknown_direction_remains_unknown_for_service_terminals(sequence):
    result = _classification(sequence)
    assert result.classification_state is FirstServeDirectionState.UNKNOWN
    assert result.eligible_for_direction_signal is False
    assert result.service_terminal_kind in {"ace", "unreturned", "fault"}


@pytest.mark.parametrize("sequence", ["S", "R", "P", "Q", "V"])
def test_unit_special_codes_are_censored(sequence):
    result = _classification(sequence)
    assert result.classification_state is FirstServeDirectionState.CENSORED
    assert result.reason_codes == (FirstServeDirectionReason.SPECIAL,)
    assert result.eligible_for_direction_signal is False
    assert result.actor is None
    assert result.direction_code is None


@pytest.mark.parametrize(
    ("sequence", "reason"),
    [
        ("", FirstServeDirectionReason.EMPTY),
        (" ", FirstServeDirectionReason.WHITESPACE),
        ("\t", FirstServeDirectionReason.WHITESPACE),
        ("c", FirstServeDirectionReason.UNRECOGNIZED),
        ("cc", FirstServeDirectionReason.UNRECOGNIZED),
        ("F", FirstServeDirectionReason.UNRECOGNIZED),
        ("?", FirstServeDirectionReason.UNRECOGNIZED),
        ("C", FirstServeDirectionReason.UNRECOGNIZED),
        ("s", FirstServeDirectionReason.UNRECOGNIZED),
    ],
)
def test_absent_prefix_is_unknown_first_serve_without_normalization(sequence, reason):
    result = _classification(sequence)
    assert result.sequence_text == sequence
    assert result.classification_state is FirstServeDirectionState.UNKNOWN_FIRST_SERVE
    assert result.reason_codes == (reason,)
    assert result.eligible_for_direction_signal is False
    assert result.actor is None
    assert result.direction_code is None


def test_residual_content_does_not_erase_an_initial_documented_direction():
    result = _classification("c4f17?S")
    assert result.classification_state is FirstServeDirectionState.OBSERVED
    assert result.service_prefix_literal == "c4"
    assert result.direction_literal == "4"
    assert result.residual_text == "f17?S"
    assert result.residual_spans == (Span(2, 7),)
    assert result.warnings == result.source_parse.warnings


def test_warnings_preserve_order_multiplicity_messages_and_half_open_spans():
    result = _classification("4??")
    assert len(result.warnings) == 2
    assert tuple((item.code, item.start, item.end) for item in result.warnings) == (
        ("W_UNKNOWN_CHARACTER", 1, 2),
        ("W_UNKNOWN_CHARACTER", 2, 3),
    )
    assert all(type(item) is Diagnostic and type(item.message) is str for item in result.warnings)
    assert result.residual_spans == (Span(1, 3),)
    assert result.residual_text == "??"


def test_all_literals_reconstruct_exactly_from_half_open_spans():
    result = _classification("cc5#tail")
    for literal, span in (
        (result.direction_literal, result.direction_span),
        (result.service_prefix_literal, result.service_prefix_span),
        (result.service_terminal_code, result.service_terminal_span),
    ):
        assert span is not None and literal is not None
        assert result.sequence_text[span[0] : span[1]] == literal
    assert result.service_terminal_kind == "unreturned"


@pytest.mark.parametrize(
    "bad_sequence",
    [None, True, False, 1, 1.0, [], {}, (), object()],
)
def test_sequence_text_rejects_wrong_and_coercible_types(bad_sequence):
    with pytest.raises(TypeError, match="str real"):
        parse_and_classify_first_serve_direction(bad_sequence, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_serve", [2, 0, 3, True, False, 1.0, "1", None, object()])
def test_only_real_integer_first_serve_is_accepted(bad_serve):
    with pytest.raises(FirstServeDirectionContractError, match="int real igual a 1"):
        parse_and_classify_first_serve_direction("4", bad_serve)  # type: ignore[arg-type]


def test_low_level_rejects_non_parse_result_and_crossed_parse_results():
    with pytest.raises(TypeError, match="ParseResult exacto"):
        classify_first_serve_direction("4", 1, object())  # type: ignore[arg-type]
    with pytest.raises(FirstServeDirectionContractError, match="mismo parseo"):
        classify_first_serve_direction("4", 1, parse_sequence("5", 1))
    with pytest.raises(FirstServeDirectionContractError, match="mismo parseo"):
        classify_first_serve_direction("4", 1, parse_sequence("4", 2))


def test_high_level_calls_parser_exactly_once(monkeypatch):
    calls = []
    real_parser = parse_sequence

    def counted_parser(sequence_text, serve_number):
        calls.append((sequence_text, serve_number))
        return real_parser(sequence_text, serve_number)

    monkeypatch.setattr(module, "parse_sequence", counted_parser)
    result = module.parse_and_classify_first_serve_direction("c4*", 1)
    assert result.direction_code == "4"
    assert calls == [("c4*", 1)]


@pytest.mark.parametrize(
    ("bad_sequence", "bad_serve"),
    [(None, 1), (4, 1), ("4", 2), ("4", True), ("4", 1.0)],
)
def test_high_level_does_not_call_parser_for_invalid_inputs(
    monkeypatch, bad_sequence, bad_serve
):
    def forbidden_parser(*args, **kwargs):
        raise AssertionError("El parser no debe ejecutarse con inputs invalidos.")

    monkeypatch.setattr(module, "parse_sequence", forbidden_parser)
    with pytest.raises((TypeError, FirstServeDirectionContractError)):
        module.parse_and_classify_first_serve_direction(bad_sequence, bad_serve)


def test_low_level_and_public_validator_never_reparse(monkeypatch):
    parsed = parse_sequence("4f17", 1)

    def forbidden_parser(*args, **kwargs):
        raise AssertionError("La ruta de bajo nivel no puede ejecutar el parser.")

    monkeypatch.setattr(module, "parse_sequence", forbidden_parser)
    result = classify_first_serve_direction("4f17", 1, parsed)
    validate_first_serve_direction_classification(result, parsed)


def test_classification_is_deterministic_and_deeply_immutable():
    first = _classification("cc4*n")
    second = _classification("cc4*n")
    assert first == second
    assert type(first.reason_codes) is tuple
    assert type(first.warnings) is tuple
    assert type(first.residual_spans) is tuple
    with pytest.raises(FrozenInstanceError):
        first.direction_code = "5"  # type: ignore[misc]
    with pytest.raises(TypeError):
        DIRECTION_DESCRIPTIONS["4"] = "changed"  # type: ignore[index]


def _mutations(base: FirstServeDirectionClassification):
    warning = base.warnings[0]
    return (
        {"contract_version": "9.9.9"},
        {"sequence_text": "5*n"},
        {"serve_number": 2},
        {"classification_state": FirstServeDirectionState.UNKNOWN},
        {"classification_state": FirstServeDirectionState.OBSERVED.value},
        {"reason_codes": (FirstServeDirectionReason.UNKNOWN_DIRECTION,)},
        {"reason_codes": [FirstServeDirectionReason.OBSERVED]},
        {"eligible_for_direction_signal": False},
        {"eligible_for_direction_signal": 1},
        {"actor": None},
        {"direction_code": "5"},
        {"direction_description": "body"},
        {"direction_literal": "5"},
        {"direction_span": (1, 2)},
        {"service_prefix_literal": "5"},
        {"service_prefix_span": (0, 2)},
        {"service_let_count": 1},
        {"service_terminal_kind": "fault"},
        {"service_terminal_code": "#"},
        {"service_terminal_span": (0, 1)},
        {"warnings": ()},
        {"warnings": (replace(warning, start=0),)},
        {"residual_spans": ()},
        {"residual_spans": (Span(0, 1),)},
        {"residual_text": ""},
        {"source_parse": parse_sequence("5*n", 1)},
    )


@pytest.mark.parametrize("mutation", _mutations(_classification("4*n")))
def test_reconstructive_validator_rejects_every_public_field_mutation(mutation):
    base = _classification("4*n")
    with pytest.raises((TypeError, FirstServeDirectionContractError, ValueError)):
        replace(base, **mutation)


def test_validator_rejects_explicit_parse_from_another_text_or_serve():
    result = _classification("4")
    with pytest.raises(FirstServeDirectionContractError, match="mismo parseo"):
        validate_first_serve_direction_classification(result, parse_sequence("5", 1))
    with pytest.raises(FirstServeDirectionContractError, match="mismo parseo"):
        validate_first_serve_direction_classification(result, parse_sequence("4", 2))


def test_contract_vocabulary_is_compatible_with_published_aggregate_buckets():
    assert CLASSIFICATION_CONTRACT_VERSION == "1.0.0"
    assert tuple(reason.value for reason in FirstServeDirectionReason) == (
        "actionable_direction",
        "direction_unknown_0",
        "empty_sequence",
        "whitespace_only_sequence",
        "unrecognized_or_no_initial_direction",
        "special_unit_sequence",
    )


def test_module_has_no_io_pandas_cli_or_dependencies_on_p03_to_p09():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "pandas" not in imports
    assert not any("return_" in name and "first_serve" not in name for name in imported_from)
    assert not any("feasibility" in name for name in imported_from)
    assert not any(isinstance(node, (ast.With, ast.AsyncWith)) for node in ast.walk(tree))
    forbidden_calls = {"open", "read_csv", "read_parquet", "to_csv", "to_json"}
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert forbidden_calls.isdisjoint(called)
    assert "if __name__" not in source


def test_no_product_contract_uses_assert_for_validation():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))
