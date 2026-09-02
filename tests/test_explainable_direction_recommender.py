"""Pruebas sintéticas e integración sellada del motor de recomendaciones."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import pytest

from src.analysis import explainable_direction_recommender as module


def _row(
    direction: str,
    *,
    scope: str | None = "surface",
    eligible: bool = True,
    score: float | None = None,
    match_id: str = "m1",
    player: str = "Player A",
    opponent: str = "Player B",
    fold: int = 2020,
    surface: str = "Hard",
) -> dict[str, object]:
    values = {"wide": .70, "body": .60, "T": .50}
    rate = values[direction] if score is None else score
    payload: dict[str, object] = {
        "target_match_id": match_id, "target_date": pd.Timestamp(f"{fold}-06-01"),
        "target_player": player, "opponent": opponent, "target_surface": surface,
        "fold": fold, "direction": direction, "eligible": eligible,
        "selected_scope": scope if eligible else None,
        "score": rate if eligible else np.nan,
        "server_component": .1 if eligible else np.nan,
        "opponent_component": .1 if eligible else np.nan,
        "server_selected_rate": rate if eligible else np.nan,
        "opponent_allowed_rate": rate if eligible else np.nan,
        "server_points": 50 if eligible else 0, "server_matches": 5 if eligible else 0,
        "opponent_points": 60 if eligible else 0, "opponent_matches": 6 if eligible else 0,
        "server_wilson_low": .4 if eligible else np.nan, "server_wilson_high": .8 if eligible else np.nan,
        "opponent_wilson_low": .4 if eligible else np.nan, "opponent_wilson_high": .8 if eligible else np.nan,
    }
    return payload


def _orientation(scopes=("surface", "surface", "surface"), **kwargs) -> pd.DataFrame:
    return pd.DataFrame([_row(direction, scope=scope, **kwargs) for direction, scope in zip(module.DIRECTIONS, scopes)])


def _point_rows(rows: pd.DataFrame) -> pd.DataFrame:
    values: list[dict[str, object]] = []
    for row in rows.itertuples(index=False):
        for number, outcome in ((1, True), (2, False)):
            values.append({
                "target_match_id": row.target_match_id, "target_player": row.target_player,
                "direction": row.direction, "target_surface": row.target_surface, "fold": row.fold,
                "outcome": outcome, "point_number": f"{row.target_match_id}-{row.target_player}-{row.direction}-{number}",
            })
    return pd.DataFrame(values)


MANUAL_UPSTREAM_COLUMNS = [
    "target_match_id", "target_date", "target_player", "opponent", "target_surface", "fold",
    "direction", "selected_scope", "eligible", "evidence_status", "abstention_reason",
    "server_rate", "opponent_allowed_rate", "combined_score", "server_component", "opponent_component",
    "server_direction_raw_rate_surface", "server_direction_points_surface", "server_direction_matches_surface",
    "server_direction_wilson_low_surface", "server_direction_wilson_high_surface",
    "server_direction_raw_rate_global", "server_direction_points_global", "server_direction_matches_global",
    "server_direction_wilson_low_global", "server_direction_wilson_high_global",
    "opponent_allowed_server_raw_rate_surface", "opponent_direction_points_surface", "opponent_direction_matches_surface",
    "opponent_allowed_server_wilson_low_surface", "opponent_allowed_server_wilson_high_surface",
    "opponent_allowed_server_raw_rate_global", "opponent_direction_points_global", "opponent_direction_matches_global",
    "opponent_allowed_server_wilson_low_global", "opponent_allowed_server_wilson_high_global",
]


def _faithful_upstream_scoring_rows(scope: str = "surface") -> pd.DataFrame:
    """Fixture literal e independiente del contrato de prepare_scoring_population."""
    records: list[dict[str, object]] = []
    for direction, delta in (("wide", 0.0), ("body", -0.05), ("T", -0.10)):
        surface_server = 0.61 + delta
        global_server = 0.71 + delta
        surface_opponent = 0.43 + delta
        global_opponent = 0.53 + delta
        server_rate = surface_server if scope == "surface" else global_server
        opponent_rate = surface_opponent if scope == "surface" else global_opponent
        records.append({
            "target_match_id": "m_schema", "target_date": pd.Timestamp("2020-06-01"),
            "target_player": "Server", "opponent": "Returner", "target_surface": "Hard", "fold": 2020,
            "direction": direction, "selected_scope": scope, "eligible": True,
            "evidence_status": "eligible_surface" if scope == "surface" else "eligible_global_fallback",
            "abstention_reason": None, "server_rate": server_rate, "opponent_allowed_rate": opponent_rate,
            "combined_score": (server_rate + opponent_rate) / 2, "server_component": 0.05, "opponent_component": -0.02,
            "server_direction_raw_rate_surface": surface_server, "server_direction_points_surface": 71,
            "server_direction_matches_surface": 8, "server_direction_wilson_low_surface": 0.55,
            "server_direction_wilson_high_surface": 0.67, "server_direction_raw_rate_global": global_server,
            "server_direction_points_global": 171, "server_direction_matches_global": 18,
            "server_direction_wilson_low_global": 0.65, "server_direction_wilson_high_global": 0.77,
            "opponent_allowed_server_raw_rate_surface": surface_opponent, "opponent_direction_points_surface": 83,
            "opponent_direction_matches_surface": 9, "opponent_allowed_server_wilson_low_surface": 0.35,
            "opponent_allowed_server_wilson_high_surface": 0.51, "opponent_allowed_server_raw_rate_global": global_opponent,
            "opponent_direction_points_global": 183, "opponent_direction_matches_global": 19,
            "opponent_allowed_server_wilson_low_global": 0.45,
            "opponent_allowed_server_wilson_high_global": 0.61,
        })
    return pd.DataFrame(records, columns=MANUAL_UPSTREAM_COLUMNS)


def _audits() -> tuple[dict[str, int], dict[str, int]]:
    return (
        {"source_point_rows": 12, "source_matches": 2, "source_players": 2, "eligible_points": 12},
        {"constructed_target_matches": 2, "constructed_snapshot_rows": 4, "constructed_feature_rows": 24, "test_target_matches_excluded_before_construction": 1_531, "test_target_rows_constructed": 0, "test_target_matches_constructed": 0, "test_feature_rows_constructed": 0},
    )


def _result() -> module.RecommendationResult:
    rows = pd.concat([
        _orientation(match_id="m1", player="A", opponent="B", fold=2020, surface="Hard"),
        _orientation(("global", "global", "global"), match_id="m1", player="B", opponent="A", fold=2020, surface="Hard"),
    ], ignore_index=True)
    source, sealed = _audits()
    return module.analyze_from_direction_rows(rows, _point_rows(rows), source_audit=source, sealed_audit=sealed)


def _rekey_result(result: module.RecommendationResult, *, summary: dict[str, object] | None = None) -> module.RecommendationResult:
    payloads = tuple(module._frame_bytes(frame) for frame in (result.coverage, result.rankings, result.explanations))
    rebuilt = copy.deepcopy(result.summary if summary is None else summary)
    rebuilt["artifact_payload_sha256"], rebuilt["artifact_payload_bytes"] = module._payload_metadata(payloads)
    rebuilt["publication_fingerprint"] = module._fingerprint(rebuilt, payloads)
    return module.RecommendationResult(rebuilt, result.coverage.copy(), result.rankings.copy(), result.explanations.copy(), result.recommendations.copy(), result.details.copy(), result.point_rows.copy(), rebuilt["publication_fingerprint"])


class _MemoryPath:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read_bytes(self) -> bytes:
        return self._payload

    def read_text(self, encoding: str = "utf-8") -> str:
        return self._payload.decode(encoding)


def _rekey_persisted(summary: dict[str, object], payloads: tuple[bytes, bytes, bytes]) -> bytes:
    names = ("coverage", "rankings", "explanations")
    rebuilt = copy.deepcopy(summary)
    rebuilt["artifact_payload_sha256"] = {name: hashlib.sha256(payload).hexdigest().upper() for name, payload in zip(names, payloads)}
    rebuilt["artifact_payload_bytes"] = {name: len(payload) for name, payload in zip(names, payloads)}
    core = {key: value for key, value in rebuilt.items() if key != "publication_fingerprint"}
    rebuilt["publication_fingerprint"] = hashlib.sha256(json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest().upper()
    return (json.dumps(rebuilt, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def test_available_surface_common_ranks_formula_and_explanations():
    record, details = module.evaluate_orientation(_orientation())
    assert record["recommendation_status"] == "available_full"
    assert record["comparison_scope"] == "surface"
    assert record["selected_direction"] == "wide"
    assert [detail["direction"] for detail in details] == ["wide", "body", "T"]
    assert [detail["rank"] for detail in details] == [1, 2, 3]
    assert all(detail["score"] == (detail["server_selected_rate"] + detail["opponent_allowed_rate"]) / 2 for detail in details)
    assert all("surface_history_used" in detail["explanation_codes"] for detail in details)


@pytest.mark.parametrize("scope", ["surface", "global"])
def test_faithful_upstream_schema_maps_all_fields_and_reaches_ranking_and_explanations(scope):
    upstream = _faithful_upstream_scoring_rows(scope)
    assert len(MANUAL_UPSTREAM_COLUMNS) == 36
    assert set(MANUAL_UPSTREAM_COLUMNS) == set(module.UPSTREAM_REQUIRED_COLUMNS)
    module.validate_upstream_scoring_schema(upstream)
    standardized = module._standardize_orientation_rows(upstream)
    if scope == "surface":
        assert standardized["server_wilson_low"].eq(.55).all()
        assert standardized["server_wilson_high"].eq(.67).all()
        assert standardized["opponent_wilson_low"].eq(.35).all()
        assert standardized["opponent_wilson_high"].eq(.51).all()
        assert standardized["server_points"].eq(71).all()
        assert standardized["server_matches"].eq(8).all()
        assert standardized["opponent_points"].eq(83).all()
        assert standardized["opponent_matches"].eq(9).all()
    else:
        assert standardized["server_wilson_low"].eq(.65).all()
        assert standardized["server_wilson_high"].eq(.77).all()
        assert standardized["opponent_wilson_low"].eq(.45).all()
        assert standardized["opponent_wilson_high"].eq(.61).all()
        assert standardized["server_points"].eq(171).all()
        assert standardized["server_matches"].eq(18).all()
        assert standardized["opponent_points"].eq(183).all()
        assert standardized["opponent_matches"].eq(19).all()
    assert standardized["selected_scope"].eq(scope).all()
    assert standardized["server_component"].equals(standardized["server_selected_rate"])
    assert standardized["opponent_component"].equals(standardized["opponent_allowed_rate"])
    reciprocal = upstream.copy(deep=True)
    reciprocal["target_player"] = "Returner"
    reciprocal["opponent"] = "Server"
    standardized = pd.concat([standardized, module._standardize_orientation_rows(reciprocal)], ignore_index=True)
    source, sealed = _audits()
    result = module.analyze_from_direction_rows(standardized, _point_rows(standardized), source_audit=source, sealed_audit=sealed)
    assert result.recommendations.iloc[0]["comparison_scope"] == scope
    assert not result.rankings.empty
    assert not result.explanations.empty


@pytest.mark.parametrize("column", module.UPSTREAM_REQUIRED_COLUMNS)
def test_upstream_schema_preflight_rejects_each_required_column_before_adapter(column):
    upstream = _faithful_upstream_scoring_rows().drop(columns=column)
    with pytest.raises(module.RecommendationContractError) as raised:
        module._standardize_orientation_rows(upstream)
    payload = json.loads(str(raised.value))
    assert payload["stage"] == "scoring_schema_preflight"
    assert column in payload["missing_columns"]


def test_upstream_schema_rejects_wrong_opponent_wilson_name_and_reports_all_missing():
    upstream = _faithful_upstream_scoring_rows().drop(columns=[
        "opponent_allowed_server_wilson_low_surface",
        "opponent_allowed_server_wilson_high_surface",
    ])
    upstream["opponent_direction_wilson_low_surface"] = .30
    with pytest.raises(module.RecommendationContractError) as raised:
        module.validate_upstream_scoring_schema(upstream)
    payload = json.loads(str(raised.value))
    assert payload["missing_columns"] == [
        "opponent_allowed_server_wilson_high_surface",
        "opponent_allowed_server_wilson_low_surface",
    ]


def test_upstream_schema_rejects_duplicates_and_bad_domain_but_ignores_extra_and_order():
    upstream = _faithful_upstream_scoring_rows()
    shuffled = upstream.loc[:, list(reversed(upstream.columns))].copy()
    shuffled["unknown_extra"] = "ignored_explicitly"
    pdt = module._standardize_orientation_rows(shuffled)
    canonical = module._standardize_orientation_rows(upstream)
    pd.testing.assert_frame_equal(pdt, canonical)
    duplicate = pd.concat([upstream, upstream[["direction"]]], axis=1)
    with pytest.raises(module.RecommendationContractError, match="duplicate_columns"):
        module.validate_upstream_scoring_schema(duplicate)
    invalid = upstream.copy()
    invalid["server_direction_points_surface"] = invalid["server_direction_points_surface"].astype(float)
    invalid.loc[0, "server_direction_points_surface"] = 1.5
    with pytest.raises(module.RecommendationContractError, match="conteo inválido"):
        module.validate_upstream_scoring_schema(invalid)


@pytest.mark.parametrize("value", ["71", 71.0, True, -1, np.nan, np.inf])
def test_upstream_schema_rejects_non_integer_counts_at_preflight(value):
    upstream = _faithful_upstream_scoring_rows()
    upstream["server_direction_points_surface"] = upstream["server_direction_points_surface"].astype(object)
    upstream.loc[0, "server_direction_points_surface"] = value
    with pytest.raises(module.RecommendationContractError, match="conteo inválido"):
        module.validate_upstream_scoring_schema(upstream)


@pytest.mark.parametrize(
    "column,value",
    [
        ("evidence_status", 1),
        ("evidence_status", "unknown"),
        ("server_rate", 0.01),
        ("opponent_allowed_rate", 0.01),
        ("server_direction_wilson_low_surface", 0.90),
        ("server_direction_points_surface", 172),
        ("server_direction_matches_surface", 19),
    ],
)
def test_upstream_schema_rejects_scope_or_evidence_mismatches_at_preflight(column, value):
    upstream = _faithful_upstream_scoring_rows("surface")
    upstream[column] = upstream[column].astype(object)
    upstream.loc[0, column] = value
    with pytest.raises(module.RecommendationContractError, match="scoring_schema_preflight"):
        module.validate_upstream_scoring_schema(upstream)


@pytest.mark.parametrize(
    "scope,column,value",
    [
        ("global", "server_rate", 0.01),
        ("global", "opponent_allowed_rate", 0.01),
    ],
)
def test_upstream_schema_rejects_global_selected_rate_mismatches(scope, column, value):
    upstream = _faithful_upstream_scoring_rows(scope)
    upstream.loc[0, column] = value
    with pytest.raises(module.RecommendationContractError, match="selected_scope"):
        module.validate_upstream_scoring_schema(upstream)


def test_available_global_common():
    record, details = module.evaluate_orientation(_orientation(("global", "global", "global")))
    assert record["recommendation_status"] == "available_full"
    assert record["comparison_scope"] == "global"
    assert all("global_fallback_used" in detail["explanation_codes"] for detail in details)


@pytest.mark.parametrize("scopes", [("surface", "surface", "global"), ("global", "surface", "global")])
def test_mixed_scopes_abstain_atomically(scopes):
    record, details = module.evaluate_orientation(_orientation(scopes))
    assert record["recommendation_status"] == "abstained_incomparable_scope"
    assert record["comparison_scope"] is None
    assert record["selected_direction"] is None
    assert record["reason_codes"] == ["mixed_direction_scopes"]
    assert all(detail["rank"] is None for detail in details)
    assert all("recommendation_abstained" in detail["explanation_codes"] for detail in details)


def test_insufficient_evidence_has_priority_over_mixed_scope():
    rows = _orientation(("surface", "global", "global"))
    rows.loc[rows["direction"].eq("T"), ["eligible", "selected_scope"]] = [False, None]
    for column in ("score", "server_component", "opponent_component", "server_selected_rate", "opponent_allowed_rate", "server_wilson_low", "server_wilson_high", "opponent_wilson_low", "opponent_wilson_high"):
        rows.loc[rows["direction"].eq("T"), column] = np.nan
    record, _ = module.evaluate_orientation(rows)
    assert record["recommendation_status"] == "abstained_insufficient_evidence"
    assert record["reason_codes"] == ["direction_not_scoreable"]


def test_missing_direction_is_insufficient_evidence():
    record, details = module.evaluate_orientation(_orientation().iloc[:2])
    assert record["recommendation_status"] == "abstained_insufficient_evidence"
    assert record["reason_codes"] == ["incomplete_direction_set"]
    assert len(details) == 2


@pytest.mark.parametrize("mutation", ["duplicate", "invalid_scope", "nan_score", "out_of_range_component", "inconsistent_key"])
def test_invalid_inputs_return_not_available_invalid_input(mutation):
    rows = _orientation()
    if mutation == "duplicate":
        rows.loc[2, "direction"] = "body"
    elif mutation == "invalid_scope":
        rows.loc[1, "selected_scope"] = "Carpet"
    elif mutation == "nan_score":
        rows.loc[1, "score"] = np.nan
    elif mutation == "out_of_range_component":
        rows.loc[1, "server_component"] = 1.1
    else:
        rows.loc[1, "opponent"] = "Other"
    record, details = module.evaluate_orientation(rows)
    assert record["recommendation_status"] == "not_available_invalid_input"
    assert record["selected_direction"] is None
    assert details == []


def test_tie_break_is_deterministic_and_non_evidential():
    rows = _orientation()
    rows["score"] = .5
    rows["server_selected_rate"] = .5
    rows["opponent_allowed_rate"] = .5
    record, details = module.evaluate_orientation(rows)
    assert record["selected_direction"] == "wide"
    assert [detail["direction"] for detail in details] == ["wide", "body", "T"]
    assert all("deterministic_tie_break" in detail["explanation_codes"] for detail in details)


def test_aggregate_status_scope_counts_coverage_shares_and_no_partial_rankings():
    available_surface = _orientation(match_id="m1", player="A", opponent="B", fold=2020)
    available_global = _orientation(("global", "global", "global"), match_id="m1", player="B", opponent="A", fold=2020)
    mixed = _orientation(("surface", "surface", "global"), match_id="m2", player="C", opponent="D", fold=2022, surface="Grass")
    incomplete = _orientation(match_id="m2", player="D", opponent="C", fold=2022, surface="Grass")
    incomplete.loc[incomplete["direction"].eq("T"), ["eligible", "selected_scope"]] = [False, None]
    for column in ("score", "server_component", "opponent_component", "server_selected_rate", "opponent_allowed_rate", "server_wilson_low", "server_wilson_high", "opponent_wilson_low", "opponent_wilson_high"):
        incomplete.loc[incomplete["direction"].eq("T"), column] = np.nan
    rows = pd.concat([available_surface, available_global, mixed, incomplete], ignore_index=True)
    source, sealed = _audits()
    result = module.analyze_from_direction_rows(rows, _point_rows(rows), source_audit=source, sealed_audit=sealed)
    statuses = result.summary["recommendation_statuses"]
    assert statuses["total_orientations"] == 4
    assert statuses["available_surface_common"] == 1
    assert statuses["available_global_common"] == 1
    assert statuses["abstained_incomparable_scope_orientations"] == 1
    assert statuses["abstained_insufficient_evidence_orientations"] == 1
    assert statuses["mixed_scope_patterns"] == {"surface / surface / global": 1}
    assert set(result.rankings["direction"]) == set(module.DIRECTIONS)
    assert result.rankings["orientations"].sum() > 0
    available = result.details.loc[result.details["recommendation_status"].eq("available_full")]
    assert set(available["rank"]) == {1, 2, 3}
    assert not set(result.recommendations.loc[~result.recommendations["recommendation_status"].eq("available_full"), "target_match_id"]).intersection(set(result.rankings.get("target_match_id", [])))
    for _, group in result.rankings.loc[result.rankings["orientations"].gt(0)].groupby(["fold", "surface", "rank", "comparison_scope"], sort=False):
        assert group["direction_share"].sum() == pytest.approx(1.0)


@pytest.mark.parametrize("field,value", [("eligible", 1), ("eligible", "True"), ("eligible", None)])
def test_eligible_is_strict_boolean(field, value):
    rows = _orientation()
    rows[field] = rows[field].astype(object)
    rows.loc[0, field] = value
    record, _ = module.evaluate_orientation(rows)
    assert record["recommendation_status"] == "not_available_invalid_input"


def test_sealed_and_validation_cardinalities_are_distinct_and_reject_swaps():
    # Expectativas literales: no llaman al constructor ni al validador productivo.
    sealed_target_matches = 5_993
    sealed_snapshot_rows = 11_986
    sealed_feature_rows = 71_916
    validation_matches = 1_805
    validation_orientation_rows = 3_610
    validation_feature_rows = 21_660
    validation_direction_rows = 10_830
    assert sealed_target_matches * 2 == sealed_snapshot_rows
    assert sealed_snapshot_rows * 3 * 2 == sealed_feature_rows
    assert validation_matches * 2 == validation_orientation_rows
    assert validation_orientation_rows * 3 * 2 == validation_feature_rows
    assert validation_orientation_rows * 3 == validation_direction_rows
    assert sealed_feature_rows > validation_feature_rows
    # El conjunto sellado contiene historia anterior a validation; el subconjunto
    # de validacion conserva exclusivamente los cuatro folds autorizados.
    assert sum((168, 371, 646, 620)) == validation_matches
    population = {
        "source_point_rows": 1_280_408, "source_matches": 7_524, "source_players": 1_002,
        "eligible_historical_points": 481_190, "sealed_target_matches": sealed_target_matches,
        "excluded_test_target_matches": 1_531, "sealed_snapshot_rows": sealed_snapshot_rows,
        "sealed_feature_rows": sealed_feature_rows, "validation_matches": validation_matches,
        "validation_orientation_rows": validation_orientation_rows,
        "validation_feature_rows": validation_feature_rows,
        "validation_direction_rows": validation_direction_rows,
    }
    folds = {"2020": 168, "2021": 371, "2022": 646, "2023": 620}
    module.validate_population_cardinalities(population, folds)
    wrong_validation = dict(population, validation_feature_rows=sealed_feature_rows)
    with pytest.raises(module.RecommendationContractError):
        module.validate_population_cardinalities(wrong_validation, folds)
    wrong_sealed = dict(population, sealed_feature_rows=validation_feature_rows)
    with pytest.raises(module.RecommendationContractError):
        module.validate_population_cardinalities(wrong_sealed, folds)
    wrong_direction = dict(population, validation_direction_rows=validation_feature_rows)
    with pytest.raises(module.RecommendationContractError):
        module.validate_population_cardinalities(wrong_direction, folds)


def test_validate_result_rejects_scope_manipulations_even_with_new_fingerprint():
    result = _result()
    altered = result.details.copy(deep=True)
    altered.loc[altered["direction"].eq("T"), "selected_scope"] = "global"
    modified = module.RecommendationResult(result.summary.copy(), result.coverage.copy(), result.rankings.copy(), result.explanations.copy(), result.recommendations.copy(), altered, result.point_rows.copy(), result.publication_fingerprint)
    payloads = tuple(module._frame_bytes(frame) for frame in (modified.coverage, modified.rankings, modified.explanations))
    modified.summary["artifact_payload_sha256"], modified.summary["artifact_payload_bytes"] = module._payload_metadata(payloads)
    modified.summary["publication_fingerprint"] = module._fingerprint(modified.summary, payloads)
    modified = module.RecommendationResult(modified.summary, modified.coverage, modified.rankings, modified.explanations, modified.recommendations, modified.details, modified.point_rows, modified.summary["publication_fingerprint"])
    with pytest.raises((AssertionError, module.RecommendationContractError)):
        module.validate_result(modified)


def test_validate_result_rejects_mixed_abstention_published_as_available_with_new_fingerprint():
    rows = _orientation(("surface", "surface", "global"))
    source, sealed = _audits()
    # La API pura ya demuestra abstencion; el resultado publicado solo puede construirse
    # desde un contrato disponible, por lo que la mutacion se aplica a ese resultado.
    result = _result()
    altered_recommendations = result.recommendations.copy(deep=True)
    altered_recommendations.loc[0, ["recommendation_status", "comparison_scope", "scope_state", "selected_direction"]] = ["abstained_incomparable_scope", None, "mixed_incomparable", None]
    modified = module.RecommendationResult(result.summary.copy(), result.coverage.copy(), result.rankings.copy(), result.explanations.copy(), altered_recommendations, result.details.copy(), result.point_rows.copy(), result.publication_fingerprint)
    payloads = tuple(module._frame_bytes(frame) for frame in (modified.coverage, modified.rankings, modified.explanations))
    modified.summary["artifact_payload_sha256"], modified.summary["artifact_payload_bytes"] = module._payload_metadata(payloads)
    modified.summary["publication_fingerprint"] = module._fingerprint(modified.summary, payloads)
    modified = module.RecommendationResult(modified.summary, modified.coverage, modified.rankings, modified.explanations, modified.recommendations, modified.details, modified.point_rows, modified.summary["publication_fingerprint"])
    with pytest.raises((AssertionError, module.RecommendationContractError)):
        module.validate_result(modified)


@pytest.mark.parametrize("field", module.TEST_ZERO_FIELDS)
def test_validate_result_rejects_every_nonzero_test_counter_with_rekeyed_fingerprint(field):
    result = _result()
    summary = copy.deepcopy(result.summary)
    summary["chronological_seal"][field] = 1
    with pytest.raises(module.RecommendationContractError, match="Sellado de test"):
        module.validate_result(_rekey_result(result, summary=summary))


@pytest.mark.parametrize(
    "field,value",
    [
        ("test_status", "open"),
        ("used_for_method_selection", True),
        ("test_target_matches_excluded_before_construction", 1_530),
    ],
)
def test_validate_result_rejects_test_seal_metadata_with_rekeyed_fingerprint(field, value):
    result = _result()
    summary = copy.deepcopy(result.summary)
    summary["chronological_seal"][field] = value
    with pytest.raises(module.RecommendationContractError, match="Sellado de test"):
        module.validate_result(_rekey_result(result, summary=summary))


@pytest.mark.parametrize("mutation", ["missing", "not_available"])
def test_available_result_requires_explicit_recommendation_status(mutation):
    result = _result()
    summary = copy.deepcopy(result.summary)
    if mutation == "missing":
        del summary["recommendation_status"]
    else:
        summary["recommendation_status"] = "not_available"
    with pytest.raises(module.RecommendationContractError, match="recommendation_status"):
        module.validate_result(_rekey_result(result, summary=summary))


def test_not_available_status_cannot_be_manipulated_to_available():
    unavailable = module.not_available_result("synthetic_failure")
    summary = copy.deepcopy(unavailable.summary)
    summary["recommendation_status"] = "available"
    manipulated = module.RecommendationResult(summary, unavailable.coverage, unavailable.rankings, unavailable.explanations, unavailable.recommendations, unavailable.details, unavailable.point_rows, unavailable.publication_fingerprint)
    with pytest.raises(module.RecommendationContractError):
        module.serialize_artifacts(manipulated)


def test_serialization_is_deterministic_and_persisted_contract_is_verified(tmp_path):
    result = _result()
    first = module.serialize_artifacts(result)
    second = module.serialize_artifacts(result)
    assert first == second
    paths = [tmp_path / "summary.json", tmp_path / "coverage.csv", tmp_path / "rankings.csv", tmp_path / "explanations.csv"]
    for path, payload in zip(paths, first):
        path.write_bytes(payload)
    module.verify_persisted_artifacts(*paths)
    assert json.loads(paths[0].read_text(encoding="utf-8"))["fingerprint_contract_version"] == "2"
    assert all("Unnamed" not in path.read_text(encoding="utf-8") for path in paths[1:])


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", na_rep="").encode("utf-8")


def _verify_memory_payloads(summary: dict[str, object], payloads: tuple[bytes, bytes, bytes]) -> None:
    summary_payload = _rekey_persisted(summary, payloads)
    module.verify_persisted_artifacts(
        _MemoryPath(summary_payload),
        _MemoryPath(payloads[0]),
        _MemoryPath(payloads[1]),
        _MemoryPath(payloads[2]),
    )


@pytest.mark.parametrize(
    "target,mutation",
    [
        ("coverage", "reverse"), ("rankings", "reverse"), ("explanations", "reverse"),
        ("coverage", "duplicate"), ("coverage", "drop"), ("coverage", "key"),
        ("coverage", "coverage_rate"), ("coverage", "abstention_rate"),
        ("rankings", "rank_duplicate"), ("rankings", "rank_missing"),
        ("rankings", "direction"), ("rankings", "share"), ("rankings", "gap"),
        ("rankings", "win_rate"), ("rankings", "mixed_scope"),
        ("explanations", "code"), ("explanations", "component"),
    ],
)
def test_persisted_validator_rejects_rekeyed_csv_manipulations(target, mutation):
    result = _result()
    summary = copy.deepcopy(result.summary)
    _, coverage_payload, rankings_payload, explanations_payload = module.serialize_artifacts(result)
    frames = {
        "coverage": pd.read_csv(io.BytesIO(coverage_payload)),
        "rankings": pd.read_csv(io.BytesIO(rankings_payload)),
        "explanations": pd.read_csv(io.BytesIO(explanations_payload)),
    }
    frame = frames[target]
    if mutation == "reverse":
        frames[target] = frame.iloc[::-1].reset_index(drop=True)
    elif mutation == "duplicate":
        frames[target] = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif mutation == "drop":
        frames[target] = frame.iloc[1:].reset_index(drop=True)
    elif mutation == "key":
        frame.loc[0, "scope_state"] = "invalid_input"
    elif mutation == "coverage_rate":
        frame.loc[0, "coverage_rate"] = 0.123
    elif mutation == "abstention_rate":
        frame.loc[0, "abstention_rate"] = 0.123
    elif mutation == "rank_duplicate":
        row = frame.loc[frame["rank"].eq(2), "rank"].index[0]
        frame.loc[row, "rank"] = 1
    elif mutation == "rank_missing":
        frame.loc[0, "rank"] = 4
    elif mutation == "direction":
        frame.loc[0, "direction"] = "unknown"
    elif mutation == "share":
        frame.loc[0, "direction_share"] = 0.123
    elif mutation == "gap":
        frame.loc[0, "mean_score_gap_from_rank_1"] = 0.123
    elif mutation == "win_rate":
        row = frame["observed_points"].gt(0).idxmax()
        frame.loc[row, "observed_win_rate"] = 0.123
    elif mutation == "mixed_scope":
        frame.loc[0, "comparison_scope"] = "mixed"
    elif mutation == "code":
        frame.loc[0, "explanation_code"] = "unknown_code"
    elif mutation == "component":
        frame.loc[0, "mean_server_component"] = 1.1
    payloads = tuple(_csv_bytes(frames[name]) for name in ("coverage", "rankings", "explanations"))
    with pytest.raises(module.RecommendationContractError):
        _verify_memory_payloads(summary, payloads)


@pytest.mark.parametrize("field", [*module.TEST_ZERO_FIELDS, "test_status", "used_for_method_selection", "test_target_matches_excluded_before_construction"])
def test_persisted_validator_rejects_rekeyed_test_seal_mutations(field):
    result = _result()
    summary = copy.deepcopy(result.summary)
    summary["chronological_seal"][field] = "open" if field == "test_status" else (True if field == "used_for_method_selection" else (1_530 if field == "test_target_matches_excluded_before_construction" else 1))
    _, *payloads = module.serialize_artifacts(result)
    with pytest.raises(module.RecommendationContractError, match="Sellado de test"):
        _verify_memory_payloads(summary, tuple(payloads))


@pytest.mark.parametrize("mutation", ["missing", "wrong"])
def test_persisted_validator_rejects_rekeyed_available_status_mutations(mutation):
    result = _result()
    summary = copy.deepcopy(result.summary)
    if mutation == "missing":
        del summary["recommendation_status"]
    else:
        summary["recommendation_status"] = "not_available"
    _, *payloads = module.serialize_artifacts(result)
    with pytest.raises(module.RecommendationContractError):
        _verify_memory_payloads(summary, tuple(payloads))


def test_persisted_not_available_contract_rejects_partial_or_contradictory_payloads():
    unavailable = module.not_available_result("synthetic_failure")
    summary_payload, *payloads = module.serialize_artifacts(unavailable)
    module.verify_persisted_artifacts(*[_MemoryPath(payload) for payload in (summary_payload, *payloads)])
    partial = (b"fold,surface,recommendation_status,scope_state,orientations,matches,players,available_orientations,abstained_orientations,coverage_rate,abstention_rate,reason_code\n2020,<ALL>,available_full,surface_common,1,1,1,1,0,1,0,\n", payloads[1], payloads[2])
    with pytest.raises(module.RecommendationContractError):
        _verify_memory_payloads(copy.deepcopy(unavailable.summary), partial)


@pytest.mark.parametrize("replace_position", [1, 2, 3, 4])
@pytest.mark.parametrize("preexisting", [False, True])
def test_atomic_publication_rolls_back_all_four_positions(tmp_path, monkeypatch, replace_position, preexisting):
    result = _result()
    paths = [tmp_path / "summary.json", tmp_path / "coverage.csv", tmp_path / "rankings.csv", tmp_path / "explanations.csv"]
    originals = []
    if preexisting:
        for index, path in enumerate(paths):
            value = f"original-{index}".encode()
            path.write_bytes(value)
            originals.append(value)
    else:
        originals = [None] * 4
    real_replace = module.os.replace
    calls = {"count": 0}
    def fail_replace(source, target):
        calls["count"] += 1
        if calls["count"] == replace_position:
            raise OSError("forced replacement failure")
        return real_replace(source, target)
    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError):
        module.write_artifacts(result, *paths)
    assert [path.read_bytes() if path.exists() else None for path in paths] == originals
    assert not list(tmp_path.glob(".*.tmp"))
    monkeypatch.setattr(module.os, "replace", real_replace)
    module.write_artifacts(result, *paths)
    assert [path.read_bytes() for path in paths] == list(module.serialize_artifacts(result))


def test_not_available_serializes_headers_only(tmp_path):
    result = module.not_available_result("synthetic_failure")
    payloads = module.serialize_artifacts(result)
    assert json.loads(payloads[0])["recommendation_status"] == "not_available"
    assert all(len(payload.splitlines()) == 1 for payload in payloads[1:])


@pytest.mark.integration
def test_real_recommender_contract_if_local_parquet_exists():
    if not module.POINTS_FILE.exists():
        pytest.skip("Falta el Parquet procesado local.")
    assert not module.REPRO_DIR.exists()
    module.verify_upstream_contracts()
    summary = json.loads(module.SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["population"]["source_point_rows"] == 1_280_408
    assert summary["population"]["source_matches"] == 7_524
    assert summary["population"]["source_players"] == 1_002
    assert summary["population"]["eligible_historical_points"] == 481_190
    assert summary["population"]["sealed_target_matches"] == 5_993
    assert summary["population"]["excluded_test_target_matches"] == 1_531
    assert summary["population"]["sealed_snapshot_rows"] == 11_986
    assert summary["population"]["sealed_feature_rows"] == 71_916
    assert summary["population"]["validation_matches"] == 1_805
    assert summary["population"]["validation_orientation_rows"] == 3_610
    assert summary["population"]["validation_feature_rows"] == 21_660
    assert summary["population"]["validation_direction_rows"] == 10_830
    assert summary["chronological_seal"]["test_status"] == "sealed"
    assert summary["chronological_seal"]["test_evaluation_runs"] == 0
    module.verify_persisted_artifacts()
