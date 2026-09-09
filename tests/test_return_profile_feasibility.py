"""Revision sintetica y adversarial del contrato P09."""
from dataclasses import FrozenInstanceError, fields, replace
import inspect

import pytest

from src.analysis import return_depth_feasibility as p05
from src.analysis import return_direction_feasibility as p04
from src.analysis import return_profile_feasibility as p09
from src.analysis import return_shot_type_feasibility as p06
from src.parsing.serve_sequence import parse_sequence


EXPECTED_TYPES = {
    "f": ("derecha, excluidos slices y golpes especiales", "groundstroke"),
    "b": ("revés, excluidos slices y golpes especiales", "groundstroke"),
    "r": ("slice de derecha", "slice"),
    "s": ("slice de revés", "slice"),
    "v": ("volea de derecha", "volley"),
    "z": ("volea de revés", "volley"),
    "o": ("remate estándar", "overhead"),
    "p": ("remate de revés", "overhead"),
    "u": ("dejada de derecha", "drop_shot"),
    "y": ("dejada de revés", "drop_shot"),
    "l": ("globo de derecha", "lob"),
    "m": ("globo de revés", "lob"),
    "h": ("media volea de derecha", "half_volley"),
    "i": ("media volea de revés", "half_volley"),
    "j": ("volea liftada de derecha", "swinging_volley"),
    "k": ("volea liftada de revés", "swinging_volley"),
    "t": ("golpe especial, incluido trick shot o tweener", "special"),
}
EXPECTED_DIRECTIONS = {
    "1": "right_side_of_right_handed_opponent_or_left_of_left_handed",
    "2": "centre",
    "3": "left_side_of_right_handed_opponent_or_right_of_left_handed",
    "0": "unknown",
}
EXPECTED_DEPTHS = {
    "7": "service_boxes",
    "8": "behind_service_line_closer_to_service_line",
    "9": "closer_to_baseline",
    "0": "unknown",
}


def test_documented_catalogues_are_exact_closed_and_contextual():
    assert dict(p09.DOCUMENTED_SHOT_TYPES) == {
        code: description for code, (description, _) in EXPECTED_TYPES.items()
    }
    assert dict(p09.SHOT_TYPE_FAMILIES) == {
        code: family for code, (_, family) in EXPECTED_TYPES.items()
    }
    assert dict(p09.LATERAL_DIRECTIONS) == EXPECTED_DIRECTIONS
    assert dict(p09.RETURN_DEPTHS) == EXPECTED_DEPTHS
    assert p09.UNKNOWN_SHOT_TYPE_CODE == "q"


def test_profile_ids_are_exactly_the_153_manual_cross_product():
    expected = {
        f"{shot}|{direction}|{depth}"
        for shot in EXPECTED_TYPES
        for direction in ("1", "2", "3")
        for depth in ("7", "8", "9")
    }
    assert p09.DOCUMENTED_PROFILE_IDS == expected
    assert len(p09.DOCUMENTED_PROFILE_IDS) == 17 * 3 * 3 == 153
    assert len(set(p09.DOCUMENTED_PROFILE_IDS)) == 153
    assert all("q" not in item and "0" not in item for item in expected)


@pytest.mark.parametrize("shot", tuple(EXPECTED_TYPES))
@pytest.mark.parametrize("direction", ("1", "2", "3"))
@pytest.mark.parametrize("depth", ("7", "8", "9"))
def test_every_complete_profile_has_one_deterministic_id(shot, direction, depth):
    result = p09.parse_and_classify_initial_return_profile(
        f"6{shot}{direction}{depth}", 1
    )
    assert result.classification_state is p09.ReturnProfileState.OBSERVED
    assert result.reason_codes == (p09.ReturnProfileReason.OBSERVED,)
    assert result.profile_id == f"{shot}|{direction}|{depth}"
    assert result.eligible_for_profile_comparison is True
    assert result.return_event_localized is True
    assert result.profile_complete is True
    assert result.unknown_components == ()
    assert result.not_documented_components == ()
    assert result.actor == "returner"


@pytest.mark.parametrize(
    "sequence,prefix,marker,shot,lateral,depth,event",
    [
        ("6f27", (0, 1), None, (1, 2), (2, 3), (3, 4), (1, 4)),
        ("6+f27", (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 5)),
        ("c6f27", (0, 2), None, (2, 3), (3, 4), (4, 5), (2, 5)),
        ("cc4+b39", (0, 3), (3, 4), (4, 5), (5, 6), (6, 7), (4, 7)),
    ],
)
def test_complete_prefix_marker_component_and_event_spans(
    sequence, prefix, marker, shot, lateral, depth, event
):
    result = p09.parse_and_classify_initial_return_profile(sequence, 1)
    assert result.service_prefix_span == prefix
    assert result.service_prefix_literal == sequence[slice(*prefix)]
    assert result.service_let_count == sequence[: prefix[1] - 1].count("c")
    assert result.service_approach_marker_span == marker
    assert result.service_approach_marker_literal == (None if marker is None else "+")
    assert result.return_shot_type_span == shot
    assert result.lateral_direction_span == lateral
    assert result.return_depth_span == depth
    assert result.return_event_span == event
    expected_event = "f27" if "f27" in sequence else "b39"
    assert sequence[slice(*event)] == expected_event


@pytest.mark.parametrize(
    "sequence,unknown,missing,codes,spans",
    [
        ("6f07", ("lateral_direction",), (), ("f", "0", "7"), ((1, 2), (2, 3), (3, 4))),
        ("6f20", ("return_depth",), (), ("f", "2", "0"), ((1, 2), (2, 3), (3, 4))),
        ("6q27", ("return_shot_type",), (), ("q", "2", "7"), ((1, 2), (2, 3), (3, 4))),
        ("6f00", ("lateral_direction", "return_depth"), (), ("f", "0", "0"), ((1, 2), (2, 3), (3, 4))),
        ("6q", ("return_shot_type",), ("lateral_direction", "return_depth"), ("q", None, None), ((1, 2), None, None)),
    ],
)
def test_documented_unknowns_preserve_component_and_never_create_profile_id(
    sequence, unknown, missing, codes, spans
):
    result = p09.parse_and_classify_initial_return_profile(sequence, 1)
    assert result.classification_state is p09.ReturnProfileState.UNKNOWN
    assert result.unknown_components == unknown
    assert result.not_documented_components == missing
    assert (result.return_shot_type_code, result.lateral_direction_code, result.return_depth_code) == codes
    assert (result.return_shot_type_span, result.lateral_direction_span, result.return_depth_span) == spans
    assert result.profile_id is None
    assert result.profile_complete is False
    assert result.eligible_for_profile_comparison is False
    assert result.actor == "returner"


@pytest.mark.parametrize(
    "sequence,missing,spans,trailing",
    [
        ("6f", ("lateral_direction", "return_depth"), ((1, 2), None, None), ()),
        ("6f2", ("return_depth",), ((1, 2), (2, 3), None), ()),
        ("6f2d@", ("return_depth",), ((1, 2), (2, 3), None), (("d@", (3, 5)),)),
    ],
)
def test_optional_absence_is_not_documented_not_unknown_or_truncated(
    sequence, missing, spans, trailing
):
    result = p09.parse_and_classify_initial_return_profile(sequence, 1)
    assert result.classification_state is p09.ReturnProfileState.NOT_DOCUMENTED
    assert result.not_documented_components == missing
    assert result.unknown_components == ()
    assert (result.return_shot_type_span, result.lateral_direction_span, result.return_depth_span) == spans
    assert tuple((item.literal, item.span) for item in result.trailing_residuals) == trailing
    assert result.actor == "returner"
    assert result.profile_id is None


@pytest.mark.parametrize(
    "sequence,reason",
    [
        ("6", p09.ReturnProfileReason.TRUNCATED),
        ("6+", p09.ReturnProfileReason.TRUNCATED),
        ("6++f27", p09.ReturnProfileReason.MODIFIER),
        ("6f7", p09.ReturnProfileReason.AMBIGUOUS_COMPONENT),
        ("6f027", p09.ReturnProfileReason.AMBIGUOUS_COMPONENT),
        ("6F27", p09.ReturnProfileReason.INVALID_TYPE),
        ("6 f27", p09.ReturnProfileReason.INVALID_TYPE),
        ("+6f27", p09.ReturnProfileReason.MISSING_PREFIX),
    ],
)
def test_unknown_initial_return_does_not_skip_or_reinterpret_tokens(sequence, reason):
    result = p09.parse_and_classify_initial_return_profile(sequence, 1)
    assert result.classification_state is p09.ReturnProfileState.UNKNOWN_INITIAL
    assert result.reason_codes == (reason,)
    assert result.return_event_localized is False
    assert result.actor is None
    assert result.profile_id is None
    assert result.return_shot_type_code is None


@pytest.mark.parametrize(
    "sequence,trailing",
    [
        ("6f270", ("0", (4, 5))),
        ("6f277", ("7", (4, 5))),
        ("6f27+", ("+", (4, 5))),
        ("6f27*", ("*", (4, 5))),
        ("6f27b1*", ("b1*", (4, 7))),
    ],
)
def test_later_content_never_erases_or_changes_the_complete_local_profile(sequence, trailing):
    result = p09.parse_and_classify_initial_return_profile(sequence, 1)
    assert result.classification_state is p09.ReturnProfileState.OBSERVED
    assert result.profile_id == "f|2|7"
    assert result.return_event_span == (1, 4)
    assert tuple((item.literal, item.span) for item in result.trailing_residuals) == (trailing,)
    assert result.parser_warnings
    assert result.parser_residuals


@pytest.mark.parametrize(
    "sequence,serve_number,previous,terminal,reason",
    [
        ("5*", 1, False, "ace", p09.ReturnProfileReason.ACE),
        ("5#", 1, False, "unreturned_serve", p09.ReturnProfileReason.UNRETURNED),
        ("5n", 1, False, "service_fault", p09.ReturnProfileReason.FAULT),
        ("5n", 2, False, "service_fault", p09.ReturnProfileReason.FAULT),
        ("5n", 2, True, "double_fault", p09.ReturnProfileReason.DOUBLE_FAULT),
        ("S", 1, False, "special_event", p09.ReturnProfileReason.SPECIAL),
        ("R", 1, False, "special_event", p09.ReturnProfileReason.SPECIAL),
        ("P", 1, False, "special_event", p09.ReturnProfileReason.SPECIAL),
        ("Q", 1, False, "special_event", p09.ReturnProfileReason.SPECIAL),
        ("V", 1, False, "special_event", p09.ReturnProfileReason.SPECIAL),
        ("c", 1, False, "incomplete_let", p09.ReturnProfileReason.INCOMPLETE_LET),
        ("cc", 2, False, "incomplete_let", p09.ReturnProfileReason.INCOMPLETE_LET),
    ],
)
def test_pre_return_outcomes_are_the_only_censored_cases(
    sequence, serve_number, previous, terminal, reason
):
    result = p09.parse_and_classify_initial_return_profile(sequence, serve_number, previous)
    assert result.classification_state is p09.ReturnProfileState.CENSORED
    assert result.reason_codes == (reason,)
    assert result.terminal_serve_outcome == terminal
    assert result.actor is None
    assert result.return_event_localized is False
    assert result.profile_id is None


@pytest.mark.parametrize("direction", ("1", "2", "3"))
@pytest.mark.parametrize("depth", ("7", "8", "9"))
def test_independent_equivalence_matrix_with_p04_p05_p06(direction, depth):
    sequence = f"6f{direction}{depth}"
    parsed = parse_sequence(sequence, 1)
    profile = p09.classify_initial_return_profile(sequence, 1, parsed)
    direction_result = p04.classify_initial_return_direction(sequence, 1, parsed)
    depth_result = p05.classify_initial_return_depth(sequence, 1, parsed)
    type_result = p06.classify_initial_return_shot_type(sequence, 1, parsed)
    assert profile.return_shot_type_code == type_result.return_shot_type_code == "f"
    assert profile.return_shot_type_span == type_result.shot_type_span == (1, 2)
    assert profile.lateral_direction_code == direction_result.lateral_direction_code == direction
    assert profile.lateral_direction_span == (direction_result.lateral_direction_start, direction_result.lateral_direction_end) == (2, 3)
    assert profile.return_depth_code == depth_result.return_depth_code == depth
    assert profile.return_depth_span == depth_result.return_depth_span == (3, 4)


def test_unknown_equivalence_is_conservative_without_productive_dependency():
    lateral_unknown = p09.parse_and_classify_initial_return_profile("6f07", 1)
    p04_lateral = p04.parse_and_classify_initial_return_direction("6f07", 1)
    p05_depth = p05.parse_and_classify_initial_return_depth("6f07", 1)
    assert lateral_unknown.lateral_direction_code == p04_lateral.lateral_direction_code == "0"
    assert lateral_unknown.return_depth_code == p05_depth.return_depth_code == "7"
    depth_unknown = p09.parse_and_classify_initial_return_profile("6f20", 1)
    p04_direction = p04.parse_and_classify_initial_return_direction("6f20", 1)
    p05_unknown = p05.parse_and_classify_initial_return_depth("6f20", 1)
    assert depth_unknown.lateral_direction_code == p04_direction.lateral_direction_code == "2"
    assert depth_unknown.return_depth_code == p05_unknown.return_depth_code == "0"
    type_unknown = p09.parse_and_classify_initial_return_profile("6q27", 1)
    p06_unknown = p06.parse_and_classify_initial_return_shot_type("6q27", 1)
    assert type_unknown.return_shot_type_code == p06_unknown.return_shot_type_code == "q"
    assert type_unknown.shot_type_observed is False
    assert type_unknown.profile_id is None


def test_high_level_validates_before_one_parser_call_and_preserves_parse_result(monkeypatch):
    original = p09.parse_sequence
    calls = []
    parsed_objects = []

    def counted(text, number):
        calls.append((text, number))
        parsed = original(text, number)
        parsed_objects.append(parsed)
        return parsed

    monkeypatch.setattr(p09, "parse_sequence", counted)
    result = p09.parse_and_classify_initial_return_profile("6+f27", 1)
    assert result.profile_id == "f|2|7"
    assert calls == [("6+f27", 1)]
    assert parsed_objects[0] == original("6+f27", 1)
    calls.clear()
    with pytest.raises(TypeError):
        p09.parse_and_classify_initial_return_profile("6f27", True)
    assert calls == []


@pytest.mark.parametrize("serve_number", (True, False, 1.0, 2.0, "1", None, 0, 3))
def test_serve_number_domain_is_strict(serve_number):
    with pytest.raises((TypeError, ValueError)):
        p09.parse_and_classify_initial_return_profile("6f27", serve_number)


@pytest.mark.parametrize("previous", (0, 1, None, "false", 0.0))
def test_previous_fault_domain_is_strict(previous):
    with pytest.raises(TypeError):
        p09.parse_and_classify_initial_return_profile("6f27", 2, previous)


def test_text_context_and_crossed_parse_results_are_rejected():
    with pytest.raises(TypeError):
        p09.parse_and_classify_initial_return_profile(None, 1)
    with pytest.raises(ValueError):
        p09.parse_and_classify_initial_return_profile("6f27", 1, True)
    with pytest.raises(ValueError):
        p09.classify_initial_return_profile("6f27", 1, parse_sequence("5f27", 1))
    with pytest.raises(ValueError):
        p09.classify_initial_return_profile("6f27", 2, parse_sequence("6f27", 1))


def test_fault_context_is_reconstructed_and_cannot_be_mutated_individually():
    result = p09.parse_and_classify_initial_return_profile("5n", 2, True)
    parsed = parse_sequence("5n", 2)
    with pytest.raises(ValueError):
        p09.validate_return_profile_classification(
            replace(result, previous_attempt_was_fault=False), parsed=parsed
        )


def test_result_nested_diagnostics_and_catalogues_are_immutable():
    result = p09.parse_and_classify_initial_return_profile("6f270", 1)
    with pytest.raises(FrozenInstanceError):
        result.profile_id = "f|2|9"
    with pytest.raises(FrozenInstanceError):
        result.parser_warnings[0].literal = "x"
    with pytest.raises(TypeError):
        p09.DOCUMENTED_SHOT_TYPES["a"] = "inventado"


def test_validator_rejects_every_public_field_mutation():
    result = p09.parse_and_classify_initial_return_profile("cc6+f27b1*", 1)
    parsed = parse_sequence(result.sequence_text, 1)
    mutations = {
        "contract_version": "x", "pattern_id": "x", "sequence_text": "cc5+f27b1*",
        "serve_number": 2, "previous_attempt_was_fault": True,
        "classification_state": p09.ReturnProfileState.UNKNOWN,
        "reason_codes": (p09.ReturnProfileReason.UNKNOWN_DEPTH,),
        "eligible_for_profile_comparison": False, "return_event_localized": False,
        "profile_complete": False, "shot_type_observed": False,
        "lateral_direction_observed": False, "return_depth_observed": False,
        "unknown_components": ("return_depth",),
        "not_documented_components": ("return_depth",), "actor": None,
        "profile_id": "f|2|9", "service_prefix_literal": "cc5",
        "service_prefix_span": (0, 2), "service_let_count": 1,
        "service_direction_code": "5", "service_direction_description": "body",
        "service_approach_marker_literal": None, "service_approach_marker_span": None,
        "return_shot_type_code": "b", "return_shot_type_description": "x",
        "return_shot_type_family": "slice", "return_shot_type_span": (0, 1),
        "lateral_direction_code": "3", "lateral_direction_description": "x",
        "lateral_direction_span": (0, 1), "return_depth_code": "9",
        "return_depth_description": "x", "return_depth_span": (0, 1),
        "return_event_span": (0, 3), "terminal_serve_outcome": "ace",
        "parser_warnings": (), "parser_residuals": (), "trailing_residuals": (),
    }
    assert set(mutations) == {field.name for field in fields(result)}
    for field_name, replacement in mutations.items():
        with pytest.raises((TypeError, ValueError)):
            p09.validate_return_profile_classification(
                replace(result, **{field_name: replacement}), parsed=parsed
            )


@pytest.mark.parametrize(
    "mutation",
    [
        {"profile_id": None}, {"profile_id": "f|7|2"}, {"profile_id": "q|2|7"},
        {"profile_id": "f|0|7"}, {"profile_id": "f|2|0"},
        {"return_shot_type_code": "q"}, {"lateral_direction_code": "0"},
        {"return_depth_code": "0"}, {"actor": None},
        {"eligible_for_profile_comparison": False},
    ],
)
def test_impossible_complete_profiles_and_ids_are_rejected(mutation):
    result = p09.parse_and_classify_initial_return_profile("6f27", 1)
    with pytest.raises((TypeError, ValueError)):
        p09.validate_return_profile_classification(replace(result, **mutation))


def test_impossible_unknown_missing_and_censored_combinations_are_rejected():
    unknown = p09.parse_and_classify_initial_return_profile("6f20", 1)
    missing = p09.parse_and_classify_initial_return_profile("6f2", 1)
    censored = p09.parse_and_classify_initial_return_profile("6*", 1)
    impossible = (
        replace(unknown, profile_id="f|2|7"), replace(unknown, unknown_components=()),
        replace(missing, not_documented_components=()),
        replace(missing, classification_state=p09.ReturnProfileState.OBSERVED),
        replace(censored, actor="returner"), replace(censored, eligible_for_profile_comparison=True),
        replace(censored, return_event_localized=True),
    )
    for result in impossible:
        with pytest.raises((TypeError, ValueError)):
            p09.validate_return_profile_classification(result)


def test_shifted_crossed_missing_and_literal_mismatched_spans_are_rejected():
    result = p09.parse_and_classify_initial_return_profile("6+f27", 1)
    mutations = (
        {"return_shot_type_span": (3, 4)}, {"lateral_direction_span": (2, 3)},
        {"return_depth_span": (3, 5)}, {"return_event_span": (2, 4)},
        {"return_shot_type_span": None}, {"return_shot_type_code": None},
        {"service_approach_marker_span": (0, 1)}, {"service_prefix_span": (0, 2)},
    )
    for mutation in mutations:
        with pytest.raises((TypeError, ValueError)):
            p09.validate_return_profile_classification(replace(result, **mutation))


def test_warning_and_residual_removal_duplication_and_order_are_rejected():
    result = p09.parse_and_classify_initial_return_profile("6f270", 1)
    assert len(result.parser_warnings) >= 2
    mutations = (
        replace(result, parser_warnings=result.parser_warnings[:-1]),
        replace(result, parser_warnings=result.parser_warnings + result.parser_warnings[:1]),
        replace(result, parser_warnings=tuple(reversed(result.parser_warnings))),
        replace(result, parser_residuals=()), replace(result, parser_residuals=result.parser_residuals * 2),
        replace(result, trailing_residuals=()),
        replace(result, trailing_residuals=(replace(result.trailing_residuals[0], literal="7"),)),
    )
    for mutation in mutations:
        with pytest.raises((TypeError, ValueError)):
            p09.validate_return_profile_classification(mutation)


def test_public_states_are_exact_exhaustive_and_closed():
    assert {state.value for state in p09.ReturnProfileState} == {
        "documented_initial_return_profile", "initial_return_profile_unknown",
        "initial_return_profile_not_documented", "unknown_initial_return", "ineligible_censored",
    }
    examples = (
        p09.parse_and_classify_initial_return_profile("6f27", 1),
        p09.parse_and_classify_initial_return_profile("6f20", 1),
        p09.parse_and_classify_initial_return_profile("6f", 1),
        p09.parse_and_classify_initial_return_profile("6f7", 1),
        p09.parse_and_classify_initial_return_profile("6*", 1),
    )
    assert {item.classification_state for item in examples} == set(p09.ReturnProfileState)


def test_static_scope_is_isolated_and_has_no_io_outcomes_or_later_shot_logic():
    source = inspect.getsource(p09)
    forbidden = (
        "return_direction_feasibility", "return_depth_feasibility",
        "return_shot_type_feasibility", "return_terminal_feasibility",
        "return_approach_feasibility", "serve_and_volley_feasibility",
        "pandas", "pyarrow", "read_parquet", "read_csv", "to_csv", "argparse",
        "__main__", "data/", "reports/", "player_1", "player_2", "point_winner",
        "server_won_point", "rallyCount", "rally_count",
    )
    assert all(token not in source for token in forbidden)
