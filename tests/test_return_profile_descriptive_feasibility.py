"""Pruebas exclusivamente sintéticas de la infraestructura descriptiva P09."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import io
import inspect
import json
import math
from pathlib import Path
import shutil

import pandas as pd
import pandas.testing as pdt
import pytest

from src.analysis import return_profile_descriptive_feasibility as p09d
from src.analysis.return_profile_feasibility import (
    DOCUMENTED_SHOT_TYPES,
    ReturnProfileState,
    parse_and_classify_initial_return_profile,
)


UPSTREAM = {
    "extractor_commit": p09d.UPSTREAM_COMMIT,
    "extractor_sha256": p09d.EXTRACTOR_SHA256,
    "chronological_summary_sha256": p09d.CHRONOLOGICAL_SUMMARY_SHA256,
}
REAL_ARTIFACT_CONTRACT = {
    "summary": (4409, "EBD87F31F0483C9B618D930C8BAA2106AAA13A7183103A4B4C855E79FE9AEA93"),
    "by_state": (7194, "AF3B96B7C09C4F1AB917DA46E2F863A6370040B23E8165483EDB96785AE2BA89"),
    "by_profile": (21955, "2980E0417A7A6FBEF83EFBF2BE68A19023C74001B24DA4AAA382D67035868358"),
    "by_group": (1867, "B1FADD7AA04E76A205520BDCF9457CDCADEE99C8DFCF691B678C76CBCCA5EA97"),
    "outcomes": (200683, "14780637AAC2162B404B3C641181718FE80DA5411B43686FC591D78D441D9A17"),
}
REAL_PUBLICATION_FINGERPRINT = "A84E59A23EF99B5A855CBBC4702EB7EB0C49285BF2BA3A5090E75AC3D356D5E3"


def synthetic_points() -> pd.DataFrame:
    rows = [
        ("m0", 1, "2009-12-31", "Hard", 1, 2, "A", "B", "6f27", None),
        ("m1", 1, "2015-06-01", "Clay", 2, 2, "A", "B", "5n", "6b38"),
        ("m2", 1, "2020-02-01", "Grass", 1, 1, "A", "B", "6q27", None),
        ("m3", 1, "2021-03-01", "Hard", 2, 1, "A", "B", "6f2", None),
        ("m4", 1, "2022-04-01", "Clay", 1, 2, "A", "B", "6F27", None),
        ("m5", 1, "2023-05-01", "Grass", 2, 2, "A", "B", "5*", None),
        ("m6", 1, "2023-06-01", "Hard", 1, 2, "A", "B", "6r19", "6s39"),
        ("m7", 1, "2024-01-01", "Clay", 1, 1, "A", "B", "6t39", None),
    ]
    return pd.DataFrame(rows, columns=p09d.SOURCE_COLUMNS)


@pytest.fixture(scope="module")
def result() -> p09d.FeasibilityResult:
    return p09d.analyze_points(synthetic_points(), upstream_contract=UPSTREAM)


def artifact_paths(base: Path) -> dict[str, Path]:
    return {
        "summary_path": base / "summary.json",
        "by_state_path": base / "by_state.csv",
        "by_profile_path": base / "by_profile.csv",
        "by_group_path": base / "by_group.csv",
        "outcomes_path": base / "outcomes.csv",
    }


def resign(result: p09d.FeasibilityResult) -> p09d.FeasibilityResult:
    frames = (result.by_state, result.by_profile, result.by_group, result.outcomes)
    payloads = tuple(p09d._table_bytes(frame) for frame in frames)
    names = ("by_state", "by_profile", "by_group", "outcomes")
    summary = dict(result.summary)
    summary["artifact_payload_sha256"] = {
        name: hashlib.sha256(payload).hexdigest().upper()
        for name, payload in zip(names, payloads)
    }
    summary["artifact_payload_bytes"] = {
        name: len(payload) for name, payload in zip(names, payloads)
    }
    summary["publication_fingerprint"] = p09d._fingerprint(summary, payloads)
    return replace(result, summary=summary, attempts=pd.DataFrame())


def resign_without_production_csv_guard(result: p09d.FeasibilityResult) -> p09d.FeasibilityResult:
    frames = (result.by_state, result.by_profile, result.by_group, result.outcomes)
    payloads = []
    for frame in frames:
        buffer = io.StringIO(newline="")
        frame.to_csv(buffer, index=False, lineterminator="\n", na_rep="")
        payloads.append(buffer.getvalue().encode("utf-8"))
    names = ("by_state", "by_profile", "by_group", "outcomes")
    summary = dict(result.summary)
    summary["artifact_payload_sha256"] = {
        name: hashlib.sha256(payload).hexdigest().upper()
        for name, payload in zip(names, payloads)
    }
    summary["artifact_payload_bytes"] = {
        name: len(payload) for name, payload in zip(names, payloads)
    }
    summary["publication_fingerprint"] = p09d._fingerprint(summary, payloads)
    return replace(result, summary=summary, attempts=pd.DataFrame())


def test_source_validation_returns_new_sorted_frame_without_mutation():
    original = synthetic_points().iloc[::-1].reset_index(drop=True)
    before = original.copy(deep=True)
    actual = p09d.validate_source_points(original)
    pdt.assert_frame_equal(original, before)
    assert actual is not original
    assert list(actual.match_id) == ["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7"]


@pytest.mark.parametrize("missing", p09d.SOURCE_COLUMNS)
def test_source_rejects_each_missing_column(missing):
    with pytest.raises(p09d.FeasibilityContractError, match="Faltan columnas"):
        p09d.validate_source_points(synthetic_points().drop(columns=missing))


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("match_id", "", "match_id"),
        ("match_id", " m0", "match_id"),
        ("match_id", 1, "match_id"),
        ("player_1", "", "player_1"),
        ("player_1", True, "player_1"),
        ("player_2", "B ", "player_2"),
        ("surface", "Carpet", "Superficies"),
        ("surface", None, "surface"),
        ("server", True, "server"),
        ("server", 1.0, "server"),
        ("server", "1", "server"),
        ("server", 3, "server"),
        ("point_winner", False, "point_winner"),
        ("point_winner", 2.0, "point_winner"),
        ("point_winner", "2", "point_winner"),
        ("point_winner", 0, "point_winner"),
        ("point_number", True, "point_number"),
        ("point_number", 1.0, "point_number"),
        ("point_number", "1", "point_number"),
        ("point_number", 0, "point_number"),
        ("first_serve", "", "first_serve"),
        ("first_serve", "   ", "first_serve"),
        ("first_serve", None, "first_serve"),
        ("first_serve", 6, "first_serve"),
        ("second_serve", 6, "second_serve"),
    ],
)
def test_source_rejects_invalid_domains_independently(field, value, message):
    points = synthetic_points()
    points[field] = points[field].astype(object)
    points.loc[0, field] = value
    with pytest.raises(p09d.FeasibilityContractError, match=message):
        p09d.validate_source_points(points)


@pytest.mark.parametrize(
    "bad_date",
    [None, "01/02/2020", "2020-2-01", "2020-02-30", 20200101, True],
)
def test_source_rejects_null_invalid_or_ambiguous_dates(bad_date):
    points = synthetic_points()
    points["date"] = points["date"].astype(object)
    points.loc[0, "date"] = bad_date
    with pytest.raises(p09d.FeasibilityContractError, match="date"):
        p09d.validate_source_points(points)


def test_source_rejects_duplicate_point_key():
    points = pd.concat([synthetic_points(), synthetic_points().iloc[[0]]], ignore_index=True)
    with pytest.raises(p09d.FeasibilityContractError, match="clave unica"):
        p09d.validate_source_points(points)


@pytest.mark.parametrize("field", ["date", "surface", "player_1", "player_2"])
def test_source_rejects_inconsistent_match_metadata(field):
    points = synthetic_points()
    duplicate = points.iloc[[0]].copy()
    duplicate.loc[:, "point_number"] = 2
    values = {"date": "2008-01-01", "surface": "Clay", "player_1": "C", "player_2": "D"}
    duplicate.loc[:, field] = values[field]
    points = pd.concat([points, duplicate], ignore_index=True)
    with pytest.raises(p09d.FeasibilityContractError, match="Metadatos inconsistentes"):
        p09d.validate_source_points(points)


def test_source_rejects_equal_players_and_expected_row_mismatch():
    points = synthetic_points()
    points.loc[0, "player_2"] = "A"
    with pytest.raises(p09d.FeasibilityContractError, match="distintos"):
        p09d.validate_source_points(points)
    with pytest.raises(p09d.FeasibilityContractError, match="source_rows_read"):
        p09d.validate_source_points(synthetic_points(), expected_rows=99)


def test_seal_happens_before_extractor_and_test_is_never_classified():
    calls: list[str] = []

    def recording_extractor(text, serve_number, previous_attempt_was_fault=False):
        calls.append(text)
        return parse_and_classify_initial_return_profile(
            text, serve_number, previous_attempt_was_fault=previous_attempt_was_fault
        )

    actual = p09d.analyze_points(
        synthetic_points(), upstream_contract=UPSTREAM, extractor=recording_extractor
    )
    assert "6t39" not in calls
    assert actual.summary["population"] == {
        "source_rows_read": 8,
        "source_matches": 8,
        "source_players": 2,
        "development_point_rows": 7,
        "development_matches": 7,
        "development_servers": 2,
        "excluded_test_matches": 1,
        "first_attempts": 7,
        "second_attempts": 2,
        "attempts_total": 9,
    }
    assert all(actual.summary["test_seal"][field] == 0 for field in p09d.TEST_ZERO_FIELDS)


def test_attempt_construction_context_actor_outcome_and_cache():
    source = p09d.validate_source_points(synthetic_points())
    development, _ = p09d.split_development_before_parsing(source)
    calls: list[tuple[str, int, bool]] = []

    def extractor(text, serve_number, previous_attempt_was_fault=False):
        calls.append((text, serve_number, previous_attempt_was_fault))
        return parse_and_classify_initial_return_profile(
            text, serve_number, previous_attempt_was_fault=previous_attempt_was_fault
        )

    attempts = p09d.construct_attempts(development, extractor=extractor)
    assert len(attempts) == 9
    assert not attempts.duplicated(["match_id", "point_number", "serve_number"]).any()
    second = attempts.loc[attempts.serve_number.eq(2)].reset_index(drop=True)
    assert list(second.second_serve_context) == [
        "documented_first_service_fault",
        "second_serve_without_documented_first_fault",
    ]
    assert list(second.previous_attempt_was_fault) == [True, False]
    assert attempts.attrs["cache_entries"] == len(set(calls)) == len(calls)
    row = attempts.loc[attempts.match_id.eq("m0")].iloc[0]
    assert row.server_player == "A" and row.returner_player == "B"
    assert bool(row.returner_won_point) is True
    assert row.profile_id == "f|2|7"
    assert row.return_shot_type_code == "f"
    assert row.return_lateral_direction_code == "2"
    assert row.return_depth_code == "7"


@pytest.mark.parametrize(
    "sequence,state,eligible,profile_id,unknown,missing",
    [
        ("6f27", "documented_initial_return_profile", True, "f|2|7", (), ()),
        ("6q27", "initial_return_profile_unknown", False, None, ("return_shot_type",), ()),
        ("6f07", "initial_return_profile_unknown", False, None, ("lateral_direction",), ()),
        ("6f20", "initial_return_profile_unknown", False, None, ("return_depth",), ()),
        ("6f2", "initial_return_profile_not_documented", False, None, (), ("return_depth",)),
        ("6F27", "unknown_initial_return", False, None, (), ()),
        ("5*", "ineligible_censored", False, None, (), ()),
    ],
)
def test_adapter_preserves_p09_states_components_and_eligibility(
    sequence, state, eligible, profile_id, unknown, missing
):
    points = synthetic_points().iloc[[0]].copy()
    points.loc[:, "first_serve"] = sequence
    source = p09d.validate_source_points(points)
    attempts = p09d.construct_attempts(source)
    row = attempts.iloc[0]
    assert row.state == state
    assert bool(row.eligible) is eligible
    assert row.profile_id == profile_id or (profile_id is None and pd.isna(row.profile_id))
    assert row.unknown_components == unknown
    assert row.not_documented_components == missing


def test_construct_attempts_does_not_mutate_extractor_result_or_source():
    source = p09d.validate_source_points(synthetic_points().iloc[[0]])
    source_before = source.copy(deep=True)
    classification = parse_and_classify_initial_return_profile("6f27", 1)
    calls = 0

    def extractor(*args, **kwargs):
        nonlocal calls
        calls += 1
        return classification

    p09d.construct_attempts(source, extractor=extractor)
    pdt.assert_frame_equal(source, source_before)
    assert calls == 1
    assert classification == parse_and_classify_initial_return_profile("6f27", 1)


def test_cache_key_includes_text_serve_number_and_fault_context():
    base = synthetic_points().iloc[[0]].copy()
    base.loc[:, "second_serve"] = "6f27"
    second = base.copy()
    second.loc[:, "match_id"] = "same2"
    second.loc[:, "point_number"] = 2
    combined = pd.concat([base, second], ignore_index=True)
    source = p09d.validate_source_points(combined)
    attempts = p09d.construct_attempts(source)
    assert len(attempts) == 4
    assert attempts.attrs["cache_entries"] == 2


def test_cache_key_distinguishes_previous_fault_context_for_second_serve():
    points = synthetic_points().iloc[[0, 1]].copy()
    points.loc[:, "match_id"] = ["fault", "no_fault"]
    points.loc[:, "point_number"] = [1, 1]
    points.loc[:, "first_serve"] = ["5n", "6f27"]
    points.loc[:, "second_serve"] = ["6b38", "6b38"]
    source = p09d.validate_source_points(points)
    calls: list[tuple[str, int, bool]] = []

    def extractor(text, serve_number, previous_attempt_was_fault=False):
        calls.append((text, serve_number, previous_attempt_was_fault))
        return parse_and_classify_initial_return_profile(
            text, serve_number, previous_attempt_was_fault=previous_attempt_was_fault
        )

    p09d.construct_attempts(source, extractor=extractor)
    assert ("6b38", 2, True) in calls
    assert ("6b38", 2, False) in calls


def test_construct_attempts_rejects_nonclassification_result():
    source = p09d.validate_source_points(synthetic_points().iloc[[0]])
    with pytest.raises(p09d.FeasibilityContractError, match="tipo invalido"):
        p09d.construct_attempts(source, extractor=lambda *args, **kwargs: object())


@pytest.mark.parametrize("shot", tuple(DOCUMENTED_SHOT_TYPES))
@pytest.mark.parametrize("direction", ("1", "2", "3"))
@pytest.mark.parametrize("depth", ("7", "8", "9"))
def test_all_153_profiles_are_retained_in_canonical_order(shot, direction, depth):
    index = list(DOCUMENTED_SHOT_TYPES).index(shot) * 9 + (int(direction) - 1) * 3 + (int(depth) - 7)
    assert p09d.PROFILE_ORDER[index] == f"{shot}|{direction}|{depth}"
    assert p09d._profile_metadata(f"{shot}|{direction}|{depth}")["return_shot_type_code"] == shot


def test_aggregates_have_exact_shapes_order_and_known_counts(result):
    assert result.by_state.shape == (14 * 5, len(p09d.BY_STATE_COLUMNS))
    assert result.by_profile.shape == (153, len(p09d.BY_PROFILE_COLUMNS))
    assert result.by_group.shape == (14, len(p09d.BY_GROUP_COLUMNS))
    assert result.outcomes.shape == (14 * 153, len(p09d.OUTCOME_COLUMNS))
    assert list(result.by_profile.profile_id) == list(p09d.PROFILE_ORDER)
    assert list(result.by_group.loc[result.by_group.group_type.eq("surface"), "surface"]) == ["Hard", "Clay", "Grass"]
    total = result.by_group.iloc[0]
    assert (
        total.attempts,
        total.complete_profiles,
        total.unknown_profiles,
        total.not_documented_profiles,
        total.unknown_initial_returns,
        total.censored_attempts,
    ) == (9, 4, 1, 1, 1, 2)
    assert math.isclose(total.complete_profile_coverage, 4 / 9)
    assert math.isclose(total.abstention_share, 5 / 9)
    assert result.by_profile.complete_profile_attempts.sum() == 4
    assert math.isclose(result.by_profile.share_of_complete_profiles.sum(), 1.0)
    assert math.isclose(result.by_profile.share_of_all_attempts.sum(), 4 / 9)
    assert list(result.by_group.loc[result.by_group.group_type.eq("period"), "period"]) == list(p09d.PERIODS)
    assert list(result.by_group.loc[result.by_group.group_type.eq("validation_fold"), "validation_fold"]) == list(p09d.FOLDS)


def test_empty_contractual_groups_and_profiles_are_explicit():
    source = p09d.validate_source_points(synthetic_points().iloc[[0]])
    attempts = p09d.construct_attempts(source)
    by_state = p09d.build_by_state(attempts)
    by_group = p09d.build_by_group(attempts)
    by_profile = p09d.build_by_profile(attempts)
    outcomes = p09d.build_outcomes(attempts)
    clay = by_group.loc[by_group.group_type.eq("surface") & by_group.surface.eq("Clay")].iloc[0]
    assert clay.attempts == 0
    assert pd.isna(clay.complete_profile_coverage)
    assert pd.isna(clay.abstention_share)
    clay_states = by_state.loc[by_state.group_type.eq("surface") & by_state.surface.eq("Clay")]
    assert len(clay_states) == 5 and clay_states.state_attempts.sum() == 0
    assert by_profile.complete_profile_attempts.eq(0).sum() == 152
    clay_outcomes = outcomes.loc[outcomes.group_type.eq("surface") & outcomes.surface.eq("Clay")]
    assert len(clay_outcomes) == 153 and clay_outcomes.trials.sum() == 0


def test_outcomes_use_only_complete_profiles_and_reconcile(result):
    total = result.outcomes.loc[result.outcomes.group_type.eq("total")]
    assert total.trials.sum() == 4
    assert total.returner_wins.sum() == 3
    assert total.server_wins.sum() == 1
    assert result.outcomes.trials.eq(0).any()
    zero = result.outcomes.loc[result.outcomes.trials.eq(0)].iloc[0]
    assert pd.isna(zero.returner_win_rate)
    assert pd.isna(zero.wilson_95_lower)
    assert pd.isna(zero.wilson_95_upper)


@pytest.mark.parametrize(
    "successes,trials,expected_lower,expected_upper",
    [
        (0, 0, None, None),
        (0, 1, 0.0, 0.7934506856227626),
        (0, 3, 0.0, 0.5614970317550454),
        (1, 1, 0.20654931437723745, 1.0),
        (3, 3, 0.4385029682449546, 1.0),
        (1, 3, 0.06149194472039621, 0.7923403991979523),
        (2, 3, 0.2076596008020477, 0.9385080552796037),
    ],
)
def test_wilson_known_values(successes, trials, expected_lower, expected_upper):
    lower, upper = p09d.wilson(successes, trials)
    if trials == 0:
        assert lower is upper is None
    else:
        assert lower == pytest.approx(expected_lower, abs=1e-15)
        assert upper == pytest.approx(expected_upper, abs=1e-15)
        assert 0 <= lower <= successes / trials <= upper <= 1


@pytest.mark.parametrize(
    "successes,trials",
    [(-1, 1), (2, 1), (0, -1), (True, 1), (1, False), (1.0, 2), (1, 2.0), ("1", 2), (1, "2"), (None, 2), (1, None)],
)
def test_wilson_rejects_invalid_domains(successes, trials):
    with pytest.raises(p09d.FeasibilityContractError, match="Wilson requiere"):
        p09d.wilson(successes, trials)


@pytest.mark.parametrize("successes", range(11))
def test_wilson_symmetry(successes):
    trials = 10
    lower, upper = p09d.wilson(successes, trials)
    mirror_lower, mirror_upper = p09d.wilson(trials - successes, trials)
    assert lower == pytest.approx(1 - mirror_upper, abs=2e-15)
    assert upper == pytest.approx(1 - mirror_lower, abs=2e-15)


def test_wilson_material_deviation_is_rejected_with_safe_group_context(monkeypatch):
    monkeypatch.setattr(p09d.math, "sqrt", lambda value: 1000.0)
    with pytest.raises(p09d.FeasibilityContractError) as exc:
        p09d.wilson(
            1,
            3,
            context={
                "aggregation_level": "profile_outcome",
                "group_type": "surface",
                "group_value": "Hard",
                "profile_id": "f|2|7",
                "sequence_text": "must-not-leak",
            },
        )
    message = str(exc.value)
    assert "successes=1" in message and "trials=3" in message
    assert "group_value='Hard'" in message and "profile_id='f|2|7'" in message
    assert "must-not-leak" not in message


def test_summary_exact_synthetic_contract(result):
    summary = result.summary
    assert summary["analysis_status"] == "available_descriptive"
    assert summary["profile_catalogue"] == {
        "expected_profiles": 153,
        "observed_profiles": 4,
        "zero_observation_profiles": 149,
        "rare_observed_profiles_fewer_than_10_attempts": 4,
        "rare_definition": "0 < complete_profile_attempts < 10; descriptive_only",
    }
    assert summary["coverage"]["denominator_attempts"] == 9
    assert summary["coverage"]["complete_profile_attempts"] == 4
    assert summary["component_distributions"]["denominator_complete_profiles"] == 4
    assert summary["outcomes_summary"]["outcome_strata"] == 2142
    assert summary["outcomes_summary"]["returner_wins"] == 3
    assert summary["outcomes_summary"]["server_wins"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "status", "population", "attempts", "state", "aggregation_level", "profile_id", "component",
        "description", "family", "zero_row", "removed_row", "duplicate_row",
        "additional_row", "row_order", "column_order", "coverage", "share",
        "outcome_trials", "outcome_wins", "wilson", "rarity", "serve_number",
        "surface", "period", "future_fold", "test_status", "test_used",
        *p09d.TEST_ZERO_FIELDS,
    ],
)
def test_semantic_mutations_fail_even_when_resigned(result, mutation):
    summary = json.loads(json.dumps(result.summary))
    by_state = result.by_state.copy(deep=True)
    by_profile = result.by_profile.copy(deep=True)
    by_group = result.by_group.copy(deep=True)
    outcomes = result.outcomes.copy(deep=True)
    if mutation == "status":
        summary["analysis_status"] = "available"
    elif mutation == "population":
        summary["population"]["attempts_total"] += 1
    elif mutation == "attempts":
        summary["attempts"]["first_attempts"] += 1
    elif mutation == "state":
        by_state.loc[0, "state_attempts"] += 1
    elif mutation == "aggregation_level":
        by_state.loc[0, "aggregation_level"] = "bad"
    elif mutation == "profile_id":
        by_profile.loc[0, "profile_id"] = "bad"
    elif mutation == "component":
        by_profile.loc[0, "return_depth_code"] = "8"
    elif mutation == "description":
        by_profile.loc[0, "return_depth_description"] = "bad"
    elif mutation == "family":
        by_profile.loc[0, "return_shot_type_family"] = "bad"
    elif mutation == "zero_row":
        zero_index = by_profile.index[by_profile.complete_profile_attempts.eq(0)][0]
        by_profile.loc[zero_index, "complete_profile_attempts"] = 1
    elif mutation == "removed_row":
        by_profile = by_profile.iloc[:-1].copy()
    elif mutation == "duplicate_row":
        by_profile = pd.concat([by_profile, by_profile.iloc[[0]]], ignore_index=True)
    elif mutation == "additional_row":
        by_group = pd.concat([by_group, by_group.iloc[[0]]], ignore_index=True)
    elif mutation == "row_order":
        by_profile = by_profile.iloc[::-1].reset_index(drop=True)
    elif mutation == "column_order":
        by_profile = by_profile[list(reversed(by_profile.columns))]
    elif mutation == "coverage":
        by_group.loc[0, "complete_profile_coverage"] = 0.99
    elif mutation == "share":
        by_profile.loc[0, "share_of_all_attempts"] = 0.5
    elif mutation == "outcome_trials":
        outcomes.loc[0, "trials"] += 1
    elif mutation == "outcome_wins":
        outcomes.loc[0, "returner_wins"] = outcomes.loc[0, "trials"] + 1
    elif mutation == "wilson":
        index = outcomes.index[outcomes.trials.gt(0)][0]
        outcomes.loc[index, "wilson_95_lower"] = 0.9
    elif mutation == "rarity":
        summary["profile_catalogue"]["rare_observed_profiles_fewer_than_10_attempts"] -= 1
    elif mutation == "serve_number":
        by_group.loc[1, "serve_number"] = 3
    elif mutation == "surface":
        by_group.loc[3, "surface"] = "Carpet"
    elif mutation == "period":
        by_group.loc[6, "period"] = "future"
    elif mutation == "future_fold":
        by_group.loc[9, "validation_fold"] = "2024"
    elif mutation == "test_status":
        summary["test_seal"]["test_status"] = "opened"
    elif mutation == "test_used":
        summary["test_seal"]["used_for_method_selection"] = True
    else:
        summary["test_seal"][mutation] = 1
    mutated = p09d.FeasibilityResult(summary, by_state, by_profile, by_group, outcomes, pd.DataFrame())
    mutated = resign(mutated)
    with pytest.raises(p09d.FeasibilityContractError) as exc:
        p09d.validate_result(mutated)
    assert "publication_fingerprint no reconcilia" not in str(exc.value).lower()


@pytest.mark.parametrize(
    "mutation",
    [
        "analysis_version", "fingerprint_version", "extractor_commit",
        "extractor_hash", "chronology_hash", "first_attempts", "second_attempts",
        "total_attempts", "state_count", "coverage_denominator", "coverage_complete",
        "coverage_share", "expected_profiles", "observed_profiles", "zero_profiles",
        "rare_profiles", "outcome_strata", "zero_trial_strata", "rare_trial_strata",
        "outcome_returner_wins", "outcome_server_wins",
    ],
)
def test_each_summary_contract_mutation_is_semantically_rejected_when_resigned(result, mutation):
    summary = json.loads(json.dumps(result.summary))
    if mutation == "analysis_version":
        summary["analysis_version"] = "1.0.1"
    elif mutation == "fingerprint_version":
        summary["fingerprint_contract"]["version"] = "2"
    elif mutation == "extractor_commit":
        summary["extractor_contract"]["commit"] = "deadbee"
    elif mutation == "extractor_hash":
        summary["upstream_contracts"]["extractor_sha256"] = "0" * 64
    elif mutation == "chronology_hash":
        summary["upstream_contracts"]["chronological_summary_sha256"] = "0" * 64
    elif mutation in {"first_attempts", "second_attempts", "total_attempts"}:
        key = {"first_attempts": "first_attempts", "second_attempts": "second_attempts", "total_attempts": "attempts_total"}[mutation]
        summary["attempts"][key] += 1
    elif mutation == "state_count":
        summary["state_counts"]["documented_initial_return_profile"] += 1
    elif mutation == "coverage_denominator":
        summary["coverage"]["denominator_attempts"] += 1
    elif mutation == "coverage_complete":
        summary["coverage"]["complete_profile_attempts"] += 1
    elif mutation == "coverage_share":
        summary["coverage"]["complete_profile_share_of_all_attempts"] += 0.01
    elif mutation == "expected_profiles":
        summary["profile_catalogue"]["expected_profiles"] = 152
    elif mutation == "observed_profiles":
        summary["profile_catalogue"]["observed_profiles"] += 1
    elif mutation == "zero_profiles":
        summary["profile_catalogue"]["zero_observation_profiles"] -= 1
    elif mutation == "rare_profiles":
        summary["profile_catalogue"]["rare_observed_profiles_fewer_than_10_attempts"] -= 1
    elif mutation == "outcome_strata":
        summary["outcomes_summary"]["outcome_strata"] -= 1
    elif mutation == "zero_trial_strata":
        summary["outcomes_summary"]["outcome_strata_with_zero_trials"] -= 1
    elif mutation == "rare_trial_strata":
        summary["outcomes_summary"]["outcome_strata_with_fewer_than_10_trials"] -= 1
    elif mutation == "outcome_returner_wins":
        summary["outcomes_summary"]["returner_wins"] += 1
    else:
        summary["outcomes_summary"]["server_wins"] += 1
    mutated = resign(replace(result, summary=summary, attempts=pd.DataFrame()))
    with pytest.raises(p09d.FeasibilityContractError) as exc:
        p09d.validate_result(mutated)
    assert "publication_fingerprint no reconcilia" not in str(exc.value).lower()


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_state", "state_denominator", "state_share", "unknown_group",
        "surface_order", "state_removed", "state_duplicate", "state_extra",
        "group_removed", "group_duplicate", "group_row_order", "group_column_missing",
        "group_column_extra",
    ],
)
def test_group_and_state_mutations_are_semantically_rejected_when_resigned(result, mutation):
    by_state = result.by_state.copy(deep=True)
    by_group = result.by_group.copy(deep=True)
    if mutation == "unknown_state":
        by_state.loc[0, "state"] = "unknown_state"
    elif mutation == "state_denominator":
        by_state.loc[0, "attempts"] += 1
    elif mutation == "state_share":
        by_state.loc[0, "state_share"] = 0.123
    elif mutation == "unknown_group":
        by_group.loc[0, "group_type"] = "unknown"
    elif mutation == "surface_order":
        by_group.iloc[[3, 5]] = by_group.iloc[[5, 3]].to_numpy()
    elif mutation == "state_removed":
        by_state = by_state.iloc[:-1].copy()
    elif mutation == "state_duplicate":
        by_state = pd.concat([by_state, by_state.iloc[[0]]], ignore_index=True)
    elif mutation == "state_extra":
        by_state = pd.concat([by_state, by_state.iloc[[0]].assign(group_value="extra")], ignore_index=True)
    elif mutation == "group_removed":
        by_group = by_group.iloc[:-1].copy()
    elif mutation == "group_duplicate":
        by_group = pd.concat([by_group, by_group.iloc[[0]]], ignore_index=True)
    elif mutation == "group_row_order":
        by_group = by_group.iloc[::-1].reset_index(drop=True)
    elif mutation == "group_column_missing":
        by_group = by_group.drop(columns="abstention_share")
    else:
        by_group["extra"] = 0
    mutated = resign(replace(result, by_state=by_state, by_group=by_group, attempts=pd.DataFrame()))
    with pytest.raises(p09d.FeasibilityContractError) as exc:
        p09d.validate_result(mutated)
    assert "publication_fingerprint no reconcilia" not in str(exc.value).lower()


@pytest.mark.parametrize(
    "mutation",
    [
        "invented_id", "incompatible_id", "unknown_type", "unknown_direction",
        "unknown_depth", "null_zero_profile", "omit_combination", "profile_extra",
    ],
)
def test_profile_catalogue_mutations_are_semantically_rejected_when_resigned(result, mutation):
    by_profile = result.by_profile.copy(deep=True)
    zero = by_profile.index[by_profile.complete_profile_attempts.eq(0)][0]
    if mutation == "invented_id":
        by_profile.loc[0, "profile_id"] = "x|2|7"
    elif mutation == "incompatible_id":
        by_profile.loc[0, "profile_id"] = "f|3|7"
    elif mutation == "unknown_type":
        by_profile.loc[0, "return_shot_type_code"] = "q"
    elif mutation == "unknown_direction":
        by_profile.loc[0, "return_lateral_direction_code"] = "0"
    elif mutation == "unknown_depth":
        by_profile.loc[0, "return_depth_code"] = "0"
    elif mutation == "null_zero_profile":
        by_profile.loc[zero, "complete_profile_attempts"] = None
    elif mutation == "omit_combination":
        by_profile = by_profile.drop(index=zero).reset_index(drop=True)
    else:
        by_profile = pd.concat([by_profile, by_profile.iloc[[0]]], ignore_index=True)
    mutated = resign(replace(result, by_profile=by_profile, attempts=pd.DataFrame()))
    with pytest.raises(p09d.FeasibilityContractError) as exc:
        p09d.validate_result(mutated)
    assert "publication_fingerprint no reconcilia" not in str(exc.value).lower()


@pytest.mark.parametrize(
    "mutation",
    [
        "losses", "rate", "wilson_upper", "group", "outcome_profile_id",
        "outcome_component", "zero_rate", "positive_rate_null", "outcome_removed",
        "outcome_duplicate", "outcome_extra", "outcome_order", "outcome_future_fold",
    ],
)
def test_outcome_mutations_are_semantically_rejected_when_resigned(result, mutation):
    outcomes = result.outcomes.copy(deep=True)
    positive = outcomes.index[outcomes.trials.gt(0)][0]
    zero = outcomes.index[outcomes.trials.eq(0)][0]
    if mutation == "losses":
        outcomes.loc[positive, "server_wins"] += 1
    elif mutation == "rate":
        outcomes.loc[positive, "returner_win_rate"] = 0.123
    elif mutation == "wilson_upper":
        outcomes.loc[positive, "wilson_95_upper"] = 0.123
    elif mutation == "group":
        outcomes.loc[0, "group_type"] = "unknown"
    elif mutation == "outcome_profile_id":
        outcomes.loc[0, "profile_id"] = "x|2|7"
    elif mutation == "outcome_component":
        outcomes.loc[0, "return_depth_code"] = "0"
    elif mutation == "zero_rate":
        outcomes.loc[zero, "returner_win_rate"] = 0.0
    elif mutation == "positive_rate_null":
        outcomes.loc[positive, "returner_win_rate"] = None
    elif mutation == "outcome_removed":
        outcomes = outcomes.iloc[:-1].copy()
    elif mutation == "outcome_duplicate":
        outcomes = pd.concat([outcomes, outcomes.iloc[[0]]], ignore_index=True)
    elif mutation == "outcome_extra":
        outcomes = pd.concat([outcomes, outcomes.iloc[[0]].assign(group_value="extra")], ignore_index=True)
    elif mutation == "outcome_order":
        outcomes = outcomes.iloc[::-1].reset_index(drop=True)
    else:
        outcomes.loc[0, "validation_fold"] = "2024"
    mutated = resign(replace(result, outcomes=outcomes, attempts=pd.DataFrame()))
    with pytest.raises(p09d.FeasibilityContractError) as exc:
        p09d.validate_result(mutated)
    assert "publication_fingerprint no reconcilia" not in str(exc.value).lower()


def test_dataclass_is_frozen_and_serialization_is_deterministic(result):
    with pytest.raises(FrozenInstanceError):
        result.summary = {}
    first = p09d.serialize_artifacts(result)
    second = p09d.serialize_artifacts(result)
    assert first == second
    assert all(b"NaN" not in payload and b"Infinity" not in payload for payload in first)


def test_double_serialization_detects_non_determinism(result, monkeypatch):
    original = p09d._serialize_once
    calls = 0

    def unstable(value):
        nonlocal calls
        calls += 1
        payloads = original(value)
        if calls == 2:
            return (payloads[0] + b"x", *payloads[1:])
        return payloads

    monkeypatch.setattr(p09d, "_serialize_once", unstable)
    with pytest.raises(p09d.FeasibilityContractError, match="no determinista"):
        p09d.serialize_artifacts(result)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "NaN", "Infinity"])
def test_summary_rejects_nonfinite_values_or_literals(result, bad):
    summary = dict(result.summary)
    summary["extra"] = bad
    mutated = replace(result, summary=summary, attempts=pd.DataFrame())
    with pytest.raises(p09d.FeasibilityContractError, match="no finito"):
        p09d.validate_result(mutated)


@pytest.mark.parametrize(
    "mutation",
    ["absolute_path", "variable_timestamp", "table_nan", "table_infinity", "missing_column", "extra_column", "reordered_columns"],
)
def test_forbidden_bytes_and_schema_mutations_fail_semantically_when_resigned(result, mutation):
    summary = json.loads(json.dumps(result.summary))
    by_profile = result.by_profile.copy(deep=True)
    if mutation == "absolute_path":
        summary["methodological_limits"][0] = r"C:\Users\person\private.txt"
    elif mutation == "variable_timestamp":
        summary["generated_at"] = "2026-09-10T10:11:12Z"
    elif mutation == "table_nan":
        by_profile.loc[0, "return_depth_description"] = "NaN"
    elif mutation == "table_infinity":
        by_profile.loc[0, "return_depth_description"] = "Infinity"
    elif mutation == "missing_column":
        by_profile = by_profile.drop(columns="return_depth_description")
    elif mutation == "extra_column":
        by_profile["extra"] = 0
    else:
        by_profile = by_profile[list(reversed(by_profile.columns))]
    provisional = replace(result, summary=summary, by_profile=by_profile, attempts=pd.DataFrame())
    mutated = (
        resign_without_production_csv_guard(provisional)
        if mutation in {"table_nan", "table_infinity"}
        else resign(provisional)
    )
    with pytest.raises(p09d.FeasibilityContractError) as exc:
        p09d.validate_result(mutated)
    assert "publication_fingerprint no reconcilia" not in str(exc.value).lower()


def test_round_trip_persisted_contract_and_no_accidental_index(result, tmp_path):
    paths = artifact_paths(tmp_path)
    p09d.write_artifacts(result, **paths, expected_population=None)
    restored = p09d.verify_persisted_artifacts(**paths, expected_population=None)
    assert restored.summary == result.summary
    assert p09d.serialize_artifacts(restored) == p09d.serialize_artifacts(result)
    for path in paths.values():
        assert path.exists()
    for path in (paths["by_state_path"], paths["by_profile_path"], paths["by_group_path"], paths["outcomes_path"]):
        frame = pd.read_csv(path)
        assert not any(column.startswith("Unnamed") for column in frame.columns)


def test_persisted_semantic_mutation_with_recomputed_fingerprint_fails(result, tmp_path):
    paths = artifact_paths(tmp_path)
    p09d.write_artifacts(result, **paths, expected_population=None)
    summary = json.loads(paths["summary_path"].read_text(encoding="utf-8"))
    frame = pd.read_csv(paths["by_profile_path"])
    frame.loc[0, "return_depth_description"] = "tampered"
    payload = p09d._table_bytes(frame)
    paths["by_profile_path"].write_bytes(payload)
    summary["artifact_payload_sha256"]["by_profile"] = hashlib.sha256(payload).hexdigest().upper()
    summary["artifact_payload_bytes"]["by_profile"] = len(payload)
    csv_payloads = tuple(paths[key].read_bytes() for key in ("by_state_path", "by_profile_path", "by_group_path", "outcomes_path"))
    summary["publication_fingerprint"] = p09d._fingerprint(summary, csv_payloads)
    paths["summary_path"].write_bytes(p09d._summary_bytes(summary))
    with pytest.raises(p09d.FeasibilityContractError, match="Metadata de perfil"):
        p09d.verify_persisted_artifacts(**paths, expected_population=None)


def test_not_available_is_header_only_sanitized_and_sealed(tmp_path):
    error = RuntimeError(r"fallo en C:\Users\person\secret\file.csv")
    result = p09d.not_available_result(error, "aggregation", partial_diagnostics={"safe": 3})
    assert result.summary["analysis_status"] == "not_available"
    assert result.summary["reason_codes"] == ["execution_failed"]
    assert "C:\\Users\\" not in result.summary["failure"]["message"]
    assert all(frame.empty for frame in (result.by_state, result.by_profile, result.by_group, result.outcomes))
    paths = artifact_paths(tmp_path)
    p09d.write_artifacts(result, **paths, expected_population=None)
    for key in ("by_state_path", "by_profile_path", "by_group_path", "outcomes_path"):
        assert len(paths[key].read_text(encoding="utf-8").splitlines()) == 1
    restored = p09d.verify_persisted_artifacts(**paths, expected_population=None)
    assert restored.summary["analysis_status"] == "not_available"


def test_not_available_rejects_rows_even_if_resigned(result):
    unavailable = p09d.not_available_result(RuntimeError("safe"), "aggregation")
    mutated = replace(unavailable, by_profile=result.by_profile.iloc[[0]].copy())
    mutated = resign(mutated)
    with pytest.raises(p09d.FeasibilityContractError, match="filas parciales"):
        p09d.validate_result(mutated)


@pytest.mark.parametrize("stage", p09d.FAILURE_STAGES)
def test_not_available_accepts_every_closed_failure_stage(stage):
    result = p09d.not_available_result(RuntimeError("safe"), stage)
    assert result.summary["failure"]["stage"] == stage
    assert result.summary["analysis_status"] == "not_available"


@pytest.mark.parametrize("failure_at", range(1, 6))
@pytest.mark.parametrize("with_previous", [False, True])
def test_each_staging_failure_rolls_back_and_cleans(result, tmp_path, monkeypatch, failure_at, with_previous):
    paths = artifact_paths(tmp_path)
    if with_previous:
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"previous")
    before = {path: path.read_bytes() if path.exists() else None for path in paths.values()}
    original = p09d._stage
    calls = 0

    def failing_stage(path, payload):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise OSError("synthetic staging failure")
        return original(path, payload)

    monkeypatch.setattr(p09d, "_stage", failing_stage)
    with pytest.raises(OSError, match="staging"):
        p09d.write_artifacts(result, **paths, expected_population=None)
    assert {path: path.read_bytes() if path.exists() else None for path in paths.values()} == before
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("failure_at", range(1, 6))
@pytest.mark.parametrize("with_previous", [False, True])
def test_each_replace_failure_rolls_back_and_cleans(result, tmp_path, monkeypatch, failure_at, with_previous):
    paths = artifact_paths(tmp_path)
    if with_previous:
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"previous")
    before = {path: path.read_bytes() if path.exists() else None for path in paths.values()}
    original = p09d.os.replace
    calls = 0

    def failing_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            raise OSError("synthetic replace failure")
        return original(source, destination)

    monkeypatch.setattr(p09d.os, "replace", failing_replace)
    with pytest.raises(OSError, match="replace"):
        p09d.write_artifacts(result, **paths, expected_population=None)
    assert {path: path.read_bytes() if path.exists() else None for path in paths.values()} == before
    assert not list(tmp_path.rglob("*.tmp"))


def test_post_write_verification_failure_rolls_back(result, tmp_path, monkeypatch):
    paths = artifact_paths(tmp_path)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"previous")
    monkeypatch.setattr(p09d, "verify_persisted_artifacts", lambda **kwargs: (_ for _ in ()).throw(OSError("verify failed")))
    with pytest.raises(OSError, match="verify"):
        p09d.write_artifacts(result, **paths, expected_population=None)
    assert all(path.read_bytes() == b"previous" for path in paths.values())
    assert not list(tmp_path.rglob("*.tmp"))


def test_performance_path_rejects_repository_and_artifacts_before_run(monkeypatch):
    called = False

    def forbidden():
        nonlocal called
        called = True

    monkeypatch.setattr(p09d, "run_real_analysis", forbidden)
    with pytest.raises(p09d.FeasibilityContractError, match="fuera del repositorio"):
        p09d.main(["--performance-log", "relative-performance.json"])
    assert called is False
    with pytest.raises(p09d.FeasibilityContractError, match="artefacto"):
        p09d._performance_path(str(p09d.SUMMARY_PATH))


def test_cli_exit_zero_without_real_io(result, tmp_path, monkeypatch):
    paths = artifact_paths(tmp_path / "artifacts")
    performance = tmp_path / "performance.json"
    original_write = p09d.write_artifacts
    monkeypatch.setattr(p09d, "run_real_analysis", lambda: result)
    monkeypatch.setattr(p09d, "write_artifacts", lambda value: original_write(value, **paths, expected_population=None))
    with pytest.raises(SystemExit) as exc:
        p09d.main(["--performance-log", str(performance)])
    assert exc.value.code == 0
    payload = json.loads(performance.read_text(encoding="utf-8"))
    assert payload["analysis_status"] == "available_descriptive"


def test_cli_exit_one_and_publication_failed_do_not_mask_failure(tmp_path, monkeypatch):
    performance = tmp_path / "performance.json"
    monkeypatch.setattr(p09d, "run_real_analysis", lambda: (_ for _ in ()).throw(p09d._RunFailure("aggregation", RuntimeError("boom"))))
    monkeypatch.setattr(p09d, "write_artifacts", lambda value: (_ for _ in ()).throw(OSError("publish boom")))
    with pytest.raises(SystemExit) as exc:
        p09d.main(["--performance-log", str(performance)])
    assert exc.value.code == 1
    payload = json.loads(performance.read_text(encoding="utf-8"))
    assert payload["analysis_status"] == "publication_failed"
    assert payload["failure"]["stage"] == "publication_verification"


def test_cli_analysis_failure_publishes_not_available_and_exits_one(tmp_path, monkeypatch):
    performance = tmp_path / "performance.json"
    published = []
    monkeypatch.setattr(
        p09d,
        "run_real_analysis",
        lambda: (_ for _ in ()).throw(
            p09d._RunFailure("aggregation", RuntimeError("synthetic"), {"safe": 1})
        ),
    )
    monkeypatch.setattr(p09d, "write_artifacts", lambda value: published.append(value))
    with pytest.raises(SystemExit) as exc:
        p09d.main(["--performance-log", str(performance)])
    assert exc.value.code == 1
    assert len(published) == 1
    assert published[0].summary["analysis_status"] == "not_available"
    payload = json.loads(performance.read_text(encoding="utf-8"))
    assert payload["analysis_status"] == "not_available"


def test_static_io_and_no_import_side_effect_contract():
    source = inspect.getsource(p09d)
    assert source.count("pd.read_parquet(") == 1
    assert "for row in development.itertuples(index=False)" in source
    assert ".iterrows(" not in source
    assert "if __name__ == \"__main__\":" in source
    assert source.count("construct_attempts(development, extractor=extractor)") == 1


@pytest.mark.integration
def test_real_artifacts_are_frozen_complete_and_artifact_only(tmp_path, monkeypatch):
    paths = {
        "summary": p09d.SUMMARY_PATH,
        "by_state": p09d.BY_STATE_PATH,
        "by_profile": p09d.BY_PROFILE_PATH,
        "by_group": p09d.BY_GROUP_PATH,
        "outcomes": p09d.OUTCOMES_PATH,
    }
    missing = [str(path.relative_to(p09d.ROOT)) for path in paths.values() if not path.exists()]
    if missing:
        pytest.fail(f"Faltan artefactos P09 congelados: {missing}")

    for name, path in paths.items():
        payload = path.read_bytes()
        expected_size, expected_hash = REAL_ARTIFACT_CONTRACT[name]
        assert len(payload) == expected_size
        assert hashlib.sha256(payload).hexdigest().upper() == expected_hash

    def forbidden(*args, **kwargs):
        raise AssertionError("La integración artifact-only no puede leer Parquet, construir intentos ni extraer perfiles.")

    monkeypatch.setattr(p09d.pd, "read_parquet", forbidden)
    monkeypatch.setattr(p09d, "construct_attempts", forbidden)
    monkeypatch.setattr(p09d, "parse_and_classify_initial_return_profile", forbidden)
    restored = p09d.verify_persisted_artifacts()
    summary = restored.summary
    assert summary["publication_fingerprint"] == REAL_PUBLICATION_FINGERPRINT
    assert summary["fingerprint_contract"]["version"] == "1"
    assert summary["analysis_status"] == "available_descriptive"
    assert summary["population"] == {
        "source_rows_read": 1_280_408,
        "source_matches": 7_524,
        "source_players": 1_002,
        "development_point_rows": 1_035_760,
        "development_matches": 5_993,
        "development_servers": 870,
        "excluded_test_matches": 1_531,
        "first_attempts": 1_035_760,
        "second_attempts": 391_103,
        "attempts_total": 1_426_863,
    }
    assert summary["attempts"] == {
        "first_attempts": 1_035_760,
        "second_attempts": 391_103,
        "attempts_total": 1_426_863,
        "cache_entries": 490_548,
    }
    assert summary["state_counts"] == {
        "documented_initial_return_profile": 601_246,
        "initial_return_profile_unknown": 1_376,
        "initial_return_profile_not_documented": 198_928,
        "unknown_initial_return": 96_776,
        "ineligible_censored": 528_537,
    }
    assert summary["coverage"] == {
        "denominator_attempts": 1_426_863,
        "complete_profile_attempts": 601_246,
        "complete_profile_share_of_all_attempts": 0.4213761237063404,
        "abstained_attempts": 825_617,
    }
    assert summary["profile_catalogue"] == {
        "expected_profiles": 153,
        "observed_profiles": 79,
        "zero_observation_profiles": 74,
        "rare_observed_profiles_fewer_than_10_attempts": 19,
        "rare_definition": "0 < complete_profile_attempts < 10; descriptive_only",
    }
    assert summary["component_distributions"] == {
        "denominator_complete_profiles": 601_246,
        "return_shot_type": {
            "f": 221_735, "b": 271_819, "r": 24_593, "s": 80_891,
            "v": 14, "z": 2, "o": 4, "p": 0, "u": 66, "y": 316,
            "l": 513, "m": 1_285, "h": 1, "i": 0, "j": 0, "k": 0,
            "t": 7,
        },
        "return_lateral_direction": {"1": 103_527, "2": 312_084, "3": 185_635},
        "return_depth": {"7": 159_482, "8": 293_400, "9": 148_364},
    }
    assert summary["outcomes_summary"] == {
        "unit": "documented_complete_profile_attempt",
        "denominator_complete_profiles": 601_246,
        "returner_wins": 286_028,
        "server_wins": 315_218,
        "outcome_strata": 2_142,
        "outcome_strata_with_zero_trials": 1_199,
        "outcome_strata_with_fewer_than_10_trials": 1_406,
        "wilson_confidence_level": 0.95,
        "significance_tests_performed": False,
        "automatic_ranking_performed": False,
    }
    assert (len(restored.by_state), len(restored.by_profile), len(restored.by_group), len(restored.outcomes)) == (70, 153, 14, 2_142)
    assert list(restored.by_group.loc[restored.by_group.group_type.eq("surface"), "surface"]) == ["Hard", "Clay", "Grass"]
    assert list(restored.by_group.loc[restored.by_group.group_type.eq("period"), "period"]) == ["to_2009", "2010s", "2020s"]
    assert list(restored.by_group.loc[restored.by_group.group_type.eq("validation_fold"), "validation_fold"]) == ["pre_validation", "2020", "2021", "2022", "2023"]
    assert summary["test_seal"] == {
        "test_status": "sealed",
        "used_for_method_selection": False,
        "excluded_test_matches": 1_531,
        "test_target_rows_parsed": 0,
        "test_attempts_constructed": 0,
        "test_profiles_classified": 0,
        "test_outcomes_computed": 0,
        "test_rows_evaluated": 0,
        "test_matches_evaluated": 0,
        "test_scores_computed": 0,
        "test_evaluation_runs": 0,
        "test_recommendations_generated": 0,
    }
    assert restored.by_profile.profile_id.nunique() == 153
    assert restored.by_profile.complete_profile_attempts.sum() == 601_246
    assert math.isclose(restored.by_profile.share_of_complete_profiles.sum(), 1.0, rel_tol=1e-12, abs_tol=1e-15)
    assert not restored.by_profile.return_shot_type_code.eq("q").any()
    assert not restored.by_profile.return_lateral_direction_code.eq("0").any()
    assert not restored.by_profile.return_depth_code.eq("0").any()
    z = 1.959963984540054
    for row in restored.outcomes.itertuples(index=False):
        trials = int(row.trials)
        wins = int(row.returner_wins)
        assert trials == wins + int(row.server_wins)
        if trials == 0:
            assert row.returner_win_rate == ""
            assert row.wilson_95_lower == ""
            assert row.wilson_95_upper == ""
            continue
        rate = wins / trials
        denominator = 1 + z * z / trials
        centre = (rate + z * z / (2 * trials)) / denominator
        spread = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
        lower, upper = centre - spread, centre + spread
        if wins == 0:
            lower = 0.0
        if wins == trials:
            upper = 1.0
        assert math.isclose(float(row.returner_win_rate), rate, rel_tol=1e-12, abs_tol=1e-15)
        assert math.isclose(float(row.wilson_95_lower), lower, rel_tol=1e-12, abs_tol=1e-15)
        assert math.isclose(float(row.wilson_95_upper), upper, rel_tol=1e-12, abs_tol=1e-15)

    copied = artifact_paths(tmp_path)
    for source, destination in zip(paths.values(), copied.values()):
        shutil.copy2(source, destination)
    tampered = json.loads(copied["summary_path"].read_text(encoding="utf-8"))
    tampered["population"]["source_rows_read"] += 1
    csv_payloads = tuple(copied[key].read_bytes() for key in ("by_state_path", "by_profile_path", "by_group_path", "outcomes_path"))
    tampered["publication_fingerprint"] = p09d._fingerprint(tampered, csv_payloads)
    copied["summary_path"].write_bytes(p09d._summary_bytes(tampered))
    with pytest.raises(p09d.FeasibilityContractError, match="Poblacion persistida"):
        p09d.verify_persisted_artifacts(**copied)
