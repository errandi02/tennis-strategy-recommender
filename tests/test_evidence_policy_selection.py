from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.analysis.evidence_policy_selection as module
from src.analysis.evidence_policy_selection import (
    COMPARISON_COLUMNS,
    GLOBAL_CONTROL,
    PRIMARY_POLICY,
    SHORTLIST,
    EvidencePolicySelectionResult,
    UpstreamArtifacts,
    UpstreamContractError,
    build_selection,
    decide_selection,
    evaluate_absolute_limits,
    load_upstream_artifacts,
    not_available_result,
    passes_structural_minimum,
    reconstruct_dominators,
    serialize_artifacts,
    validate_result,
    validate_upstream_artifacts,
    write_artifacts,
)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", na_rep="").encode("utf-8")


def _grid_ids() -> list[str]:
    return [
        f"p{points:03d}_m{matches:02d}_{scope}"
        for points in (25, 50, 100)
        for matches in (3, 5, 10)
        for scope in ("global_only", "surface_only", "surface_then_global")
    ]


def _split_policy_id(policy_id: str) -> tuple[int, int, str]:
    points, matches, scope = policy_id.split("_", 2)
    return int(points[1:]), int(matches[1:]), scope


SYNTHETIC_SHORTLIST_VECTORS = {
    "p025_m03_global_only": (0.80, 0.24, 0.079),
    "p025_m05_global_only": (0.75, 0.22, 0.078),
    "p025_m10_global_only": (0.70, 0.20, 0.077),
    "p050_m03_global_only": (0.68, 0.19, 0.076),
    "p050_m05_global_only": (0.65, 0.18, 0.075),
    "p050_m05_surface_then_global": (0.60, 0.17, 0.070),
    "p050_m10_global_only": (0.55, 0.15, 0.065),
    "p100_m10_global_only": (0.49, 0.10, 0.060),
}


def _synthetic_candidates() -> pd.DataFrame:
    rows = []
    for policy_id in _grid_ids():
        points, matches, scope = _split_policy_id(policy_id)
        coverage, wilson, stability = SYNTHETIC_SHORTLIST_VECTORS.get(
            policy_id, (0.40, 0.30, 0.090)
        )
        rows.append({
            "policy_id": policy_id,
            "min_points": points,
            "min_matches": matches,
            "scope_policy": scope,
            "coverage_complete_match": 0.70,
            "worst_fold_complete_match_coverage": coverage,
            "median_wilson_width": wilson / 2,
            "worst_fold_p90_wilson_width": wilson,
            "worst_fold_p90_absolute_change": stability,
            "wide_coverage": 0.71,
            "body_coverage": 0.69,
            "T_coverage": 0.68,
            "hard_coverage": 0.72,
            "clay_coverage": 0.66,
            "grass_coverage": 0.55,
            "surface_selection_proportion": 0.50 if scope == "surface_then_global" else 0.0,
            "global_selection_proportion": 0.20 if scope == "surface_then_global" else 0.70,
            "abstention_proportion": 0.30,
            "scope_coherence_reconciled": True,
        })
    return pd.DataFrame(rows)


def _independent_dominators(candidates: pd.DataFrame) -> dict[str, list[str]]:
    vectors = {
        row.policy_id: (
            row.worst_fold_complete_match_coverage,
            row.worst_fold_p90_wilson_width,
            row.worst_fold_p90_absolute_change,
        )
        for row in candidates.itertuples(index=False)
    }
    result = {}
    for policy_id, value in vectors.items():
        result[policy_id] = sorted(
            other for other, candidate in vectors.items()
            if other != policy_id
            and candidate[0] >= value[0]
            and candidate[1] <= value[1]
            and candidate[2] <= value[2]
            and (
                candidate[0] > value[0]
                or candidate[1] < value[1]
                or candidate[2] < value[2]
            )
        )
    return result


def _synthetic_pareto(candidates: pd.DataFrame) -> pd.DataFrame:
    dominators = _independent_dominators(candidates)
    rows = []
    for row in candidates.itertuples(index=False):
        ds = dominators[row.policy_id]
        rows.append({
            "policy_id": row.policy_id,
            "min_points": row.min_points,
            "min_matches": row.min_matches,
            "scope_policy": row.scope_policy,
            "pareto_status": "dominated" if ds else "non_dominated",
            "dominated_by": "|".join(ds),
            "reason_code": "dominated_on_prespecified_axes" if ds else "",
            "worst_fold_complete_match_coverage": row.worst_fold_complete_match_coverage,
            "worst_fold_p90_wilson_width": row.worst_fold_p90_wilson_width,
            "worst_fold_p90_absolute_change": row.worst_fold_p90_absolute_change,
        })
    return pd.DataFrame(rows)


def _synthetic_by_fold() -> pd.DataFrame:
    rows = []
    policy_values = {
        PRIMARY_POLICY: {
            "coverage": (0.60, 0.70, 0.75, 0.80),
            "wilson": (0.17, 0.16, 0.15, 0.14),
            "stability": (np.nan, 0.05, 0.06, 0.07),
            "surface": 8,
            "global": 8,
        },
        GLOBAL_CONTROL: {
            "coverage": (0.65, 0.70, 0.75, 0.80),
            "wilson": (0.18, 0.17, 0.16, 0.15),
            "stability": (np.nan, 0.05, 0.06, 0.075),
            "surface": 0,
            "global": 16,
        },
    }
    for policy_id, values in policy_values.items():
        points, matches, scope = _split_policy_id(policy_id)
        for fold_index, fold in enumerate((2020, 2021, 2022, 2023)):
            for direction in ("wide", "body", "T"):
                for role in ("server", "opponent"):
                    base = {
                        "policy_id": policy_id,
                        "min_points": points,
                        "min_matches": matches,
                        "scope_policy": scope,
                        "fold": fold,
                        "direction": direction,
                        "role": role,
                        "selected_scope": "<ALL>",
                        "target_matches": 100,
                        "target_orientations": 200,
                        "eligible_orientations": 160,
                        "eligible_matches": 80,
                        "complete_matches": int(values["coverage"][fold_index] * 100),
                        "abstained_orientations": 40,
                        "abstained_matches": 20,
                        "eligible_players": 10,
                        "coverage_orientation": 0.80,
                        "coverage_match": 0.80,
                        "coverage_complete_match": values["coverage"][fold_index],
                        "surface_selections": values["surface"],
                        "global_selections": values["global"],
                        "abstentions": 200 - values["surface"] - values["global"],
                        "surface_selection_proportion": values["surface"] / 200,
                        "global_selection_proportion": values["global"] / 200,
                        "abstention_proportion": (200 - values["surface"] - values["global"]) / 200,
                        "wilson_width_p90": values["wilson"][fold_index],
                        "comparable_players": 0 if fold == 2020 else 2,
                        "absolute_change_p90": values["stability"][fold_index],
                    }
                    rows.append({**base, "stratum_type": "overall", "surface": "<ALL>"})
                    for surface in ("Hard", "Clay", "Grass"):
                        rows.append({**base, "stratum_type": "surface", "surface": surface})
    return pd.DataFrame(rows)


def _upstream_fingerprint(summary: dict, candidates: pd.DataFrame, by_fold: pd.DataFrame, pareto: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for frame in (candidates, by_fold, pareto):
        digest.update(_csv_bytes(frame))
    digest.update(json.dumps(
        summary, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8"))
    return digest.hexdigest().upper()


@pytest.fixture
def synthetic_upstream() -> UpstreamArtifacts:
    candidates = _synthetic_candidates()
    pareto = _synthetic_pareto(candidates)
    by_fold = _synthetic_by_fold()
    summary = {
        "analysis_status": "available",
        "pareto_summary": {
            "automatic_selection_performed": False,
            "dominated_policies": 19,
            "non_dominated_policy_ids": list(SHORTLIST),
            "not_comparable_policy_ids": [],
        },
        "sealed_test_contract": {
            "allowed_folds": [2020, 2021, 2022, 2023],
            "latest_allowed_target_date": "2023-12-31",
            "test_evaluation_runs": 0,
            "test_matches_evaluated": 0,
            "test_rows_evaluated": 0,
            "test_status": "sealed",
            "used_for_method_selection": False,
        },
        "scope_policy_contracts": {
            "global_only": "global history only",
            "surface_only": "target-surface history only; no fallback",
            "surface_then_global": "surface only if both roles pass; otherwise global only if both pass",
            "server_and_opponent_scope_must_match": True,
        },
    }
    summary["publication_fingerprint"] = _upstream_fingerprint(
        summary, candidates, by_fold, pareto
    )
    payloads = {
        "reports/evidence_policy_validation_summary.json": json.dumps(summary).encode(),
        "reports/tables/evidence_policy_validation_candidates.csv": _csv_bytes(candidates),
        "reports/tables/evidence_policy_validation_pareto.csv": _csv_bytes(pareto),
        "reports/tables/evidence_policy_validation_by_fold.csv": _csv_bytes(by_fold),
    }
    return UpstreamArtifacts(
        summary, candidates, pareto, by_fold, payloads, dict(module.UPSTREAM_HASHES)
    )


def test_loads_only_four_synthetic_artifacts_and_checks_hashes(tmp_path):
    frames = [
        json.dumps({"ok": True}).encode(), b"a\n1\n", b"b\n2\n", b"c\n3\n",
    ]
    paths = tuple(tmp_path / name for name in ("s.json", "c.csv", "p.csv", "f.csv"))
    expected = {}
    for path, payload in zip(paths, frames):
        path.write_bytes(payload)
        expected[path.name] = hashlib.sha256(payload).hexdigest().upper()
    loaded = load_upstream_artifacts(paths, expected)
    assert loaded.summary == {"ok": True}
    assert loaded.candidates.iloc[0, 0] == 1
    assert set(loaded.payloads) == set(expected)


def test_loader_rejects_hash_mismatch(tmp_path):
    paths = tuple(tmp_path / name for name in ("s.json", "c.csv", "p.csv", "f.csv"))
    for path, payload in zip(paths, (b"{}", b"a\n1\n", b"b\n2\n", b"c\n3\n")):
        path.write_bytes(payload)
    expected = {path.name: "0" * 64 for path in paths}
    with pytest.raises(UpstreamContractError, match="SHA-256"):
        load_upstream_artifacts(paths, expected)


def test_shortlist_and_pareto_are_reconstructed_independently(synthetic_upstream):
    audit = validate_upstream_artifacts(synthetic_upstream)
    assert audit["candidate_count"] == 27
    assert audit["non_dominated_count"] == 8
    assert tuple(pid for pid, ds in audit["dominators"].items() if not ds) == SHORTLIST
    assert reconstruct_dominators(synthetic_upstream.candidates) == _independent_dominators(
        synthetic_upstream.candidates
    )


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("worst_fold_complete_match_coverage", 0.50, True),
        ("worst_fold_complete_match_coverage", np.nextafter(0.50, 0.0), False),
        ("worst_fold_p90_wilson_width", 0.25, True),
        ("worst_fold_p90_wilson_width", np.nextafter(0.25, 1.0), False),
        ("worst_fold_p90_absolute_change", 0.08, True),
        ("worst_fold_p90_absolute_change", np.nextafter(0.08, 1.0), False),
    ],
)
def test_limits_are_inclusive_and_use_full_floats(field, value, expected):
    row = {
        "worst_fold_complete_match_coverage": 0.50,
        "worst_fold_p90_wilson_width": 0.25,
        "worst_fold_p90_absolute_change": 0.08,
    }
    row[field] = value
    key = {
        "worst_fold_complete_match_coverage": "passes_coverage_limit",
        "worst_fold_p90_wilson_width": "passes_wilson_limit",
        "worst_fold_p90_absolute_change": "passes_stability_limit",
    }[field]
    assert evaluate_absolute_limits(row)[key] is expected


@pytest.mark.parametrize(
    "points,matches,expected",
    [(25, 5, False), (50, 3, False), (49, 5, False), (50, 4, False), (50, 5, True)],
)
def test_structural_minimum(points, matches, expected):
    assert passes_structural_minimum({"min_points": points, "min_matches": matches}) is expected


def _checks(**overrides):
    checks = {
        "pareto_eligible": True,
        "passes_structural_minimum": True,
        "passes_coverage_limit": True,
        "passes_wilson_limit": True,
        "passes_stability_limit": True,
        "fold_contract": {"valid": True},
        "scope_contract": {"valid": True},
    }
    checks.update(overrides)
    return checks


def test_primary_passes_all_criteria_and_is_selected():
    result = decide_selection(_checks(), _checks())
    assert result == {
        "selection_status": "selected_primary",
        "selection_path": "primary_candidate_selected",
        "selected_policy": PRIMARY_POLICY,
        "reason_codes": [],
        "primary_evaluated": True,
        "primary_passed": True,
        "control_descriptively_compared": True,
        "fallback_evaluated": False,
        "fallback_activated": False,
        "third_candidate_evaluated": False,
    }


@pytest.mark.parametrize("failure", ["passes_coverage_limit", "passes_wilson_limit", "passes_stability_limit"])
def test_each_primary_absolute_failure_uses_global_fallback(failure):
    result = decide_selection(_checks(**{failure: False}), _checks())
    assert result["selection_status"] == "selected_global_fallback"
    assert result["selection_path"] == "fallback_to_global_control"
    assert result["selected_policy"] == GLOBAL_CONTROL
    assert result["reason_codes"] == [f"primary_{failure}"]
    assert result["fallback_evaluated"] is True
    assert result["fallback_activated"] is True


def test_primary_and_global_failure_produce_not_available():
    result = decide_selection(
        _checks(passes_coverage_limit=False),
        _checks(passes_wilson_limit=False),
    )
    assert result["selection_status"] == "not_available"
    assert result["selection_path"] == "no_candidate_satisfied_limits"
    assert result["selected_policy"] is None
    assert result["reason_codes"] == [
        "global_control_passes_wilson_limit", "primary_passes_coverage_limit",
    ]


def test_no_third_candidate_is_considered_when_primary_and_control_fail():
    result = decide_selection(
        _checks(passes_structural_minimum=False),
        _checks(passes_structural_minimum=False),
    )
    assert result["selection_status"] == "not_available"
    assert result["selected_policy"] is None
    assert result["third_candidate_evaluated"] is False


def test_primary_decision_does_not_activate_or_inspect_fallback():
    class UnreadableControl(dict):
        def get(self, key, default=None):
            raise AssertionError(f"fallback consultado: {key}")

    result = decide_selection(_checks(), UnreadableControl())
    assert result["selection_status"] == "selected_primary"
    assert result["fallback_evaluated"] is False
    assert result["fallback_activated"] is False


def test_complete_synthetic_selection_uses_primary(synthetic_upstream):
    result = build_selection(synthetic_upstream)
    assert result.summary["selection_status"] == "selected_primary"
    assert result.summary["selection_path"] == "primary_candidate_selected"
    assert result.summary["selected_policy"] == PRIMARY_POLICY
    assert result.summary["candidate_checks"]["passes_all_selection_criteria"] is True
    assert result.summary["control_descriptive_checks"]["passes_all_selection_criteria"] is True
    assert result.summary["decision_execution"] == {
        "primary_evaluated": True,
        "primary_passed": True,
        "control_descriptively_compared": True,
        "fallback_evaluated": False,
        "fallback_activated": False,
        "third_candidate_evaluated": False,
    }
    assert result.summary["fallback_evaluation"] == {
        "evaluated": False,
        "activated": False,
        "control_checks": None,
    }
    assert result.comparison["selected"].sum() == 1


def test_surface_then_global_contract_is_joint_and_explicit(synthetic_upstream):
    result = build_selection(synthetic_upstream)
    contract = result.summary["selected_policy_contract"]
    assert contract["scope_policy"] == "surface_then_global"
    assert contract["joint_role_eligibility"] is True
    assert contract["surface_priority"] is True
    assert contract["joint_global_fallback"] is True
    assert contract["mixed_scopes"] is False
    assert contract["abstain_when_evidence_is_insufficient"] is True


def test_mixed_scope_contract_is_rejected(synthetic_upstream):
    upstream = replace(synthetic_upstream, summary=json.loads(json.dumps(synthetic_upstream.summary)))
    upstream.summary["scope_policy_contracts"]["server_and_opponent_scope_must_match"] = False
    # Keep fingerprint aligned so the semantic contract is the failure.
    core = {k: v for k, v in upstream.summary.items() if k != "publication_fingerprint"}
    upstream.summary["publication_fingerprint"] = _upstream_fingerprint(
        core, upstream.candidates, upstream.by_fold, upstream.pareto
    )
    with pytest.raises(UpstreamContractError, match="scope|Scope|mixed"):
        build_selection(upstream)


def test_fold_2024_is_rejected(synthetic_upstream):
    by_fold = synthetic_upstream.by_fold.copy(deep=True)
    by_fold.loc[0, "fold"] = 2024
    upstream = replace(synthetic_upstream, by_fold=by_fold)
    with pytest.raises(UpstreamContractError, match="2020--2023"):
        validate_upstream_artifacts(upstream)


def test_nonzero_test_evaluation_runs_is_rejected(synthetic_upstream):
    summary = json.loads(json.dumps(synthetic_upstream.summary))
    summary["sealed_test_contract"]["test_evaluation_runs"] = 1
    upstream = replace(synthetic_upstream, summary=summary)
    with pytest.raises(UpstreamContractError, match="sellado"):
        validate_upstream_artifacts(upstream)


@pytest.mark.parametrize("target", ["candidates", "pareto", "hashes"])
def test_upstream_candidate_pareto_or_hash_manipulation_is_rejected(synthetic_upstream, target):
    if target == "candidates":
        frame = synthetic_upstream.candidates.copy(deep=True)
        frame.loc[0, "policy_id"] = "changed"
        upstream = replace(synthetic_upstream, candidates=frame)
    elif target == "pareto":
        frame = synthetic_upstream.pareto.copy(deep=True)
        frame.loc[0, "pareto_status"] = "dominated"
        upstream = replace(synthetic_upstream, pareto=frame)
    else:
        hashes = dict(synthetic_upstream.hashes)
        hashes[next(iter(hashes))] = "0" * 64
        upstream = replace(synthetic_upstream, hashes=hashes)
    with pytest.raises(UpstreamContractError):
        validate_upstream_artifacts(upstream)


def _rebind_fingerprint(
    result: EvidencePolicySelectionResult,
    summary: dict | None = None,
    comparison: pd.DataFrame | None = None,
) -> EvidencePolicySelectionResult:
    """Recalcula el fingerprint sin reutilizar la implementacion productiva."""

    rebound_summary = json.loads(json.dumps(summary if summary is not None else result.summary))
    rebound_comparison = (
        comparison.copy(deep=True) if comparison is not None else result.comparison.copy(deep=True)
    )
    core = {
        key: value for key, value in rebound_summary.items()
        if key != "publication_fingerprint"
    }
    digest = hashlib.sha256()
    digest.update(_csv_bytes(rebound_comparison))
    digest.update(json.dumps(
        core, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8"))
    fingerprint = digest.hexdigest().upper()
    rebound_summary["publication_fingerprint"] = fingerprint
    return replace(
        result,
        summary=rebound_summary,
        comparison=rebound_comparison,
        publication_fingerprint=fingerprint,
    )


def _apply_semantic_mutation(
    case: str,
    result: EvidencePolicySelectionResult,
) -> EvidencePolicySelectionResult:
    summary = json.loads(json.dumps(result.summary))
    comparison = result.comparison.copy(deep=True)
    primary_index = comparison.index[comparison["policy_id"].eq(PRIMARY_POLICY)].item()
    control_index = comparison.index[comparison["policy_id"].eq(GLOBAL_CONTROL)].item()

    if case == "wrong_path":
        summary["selection_path"] = "fallback_to_global_control"
    elif case == "missing_primary_check":
        summary["candidate_checks"].pop("passes_coverage_limit")
    elif case == "false_primary_check":
        summary["candidate_checks"]["passes_coverage_limit"] = False
    elif case == "reverse_order":
        comparison = comparison.iloc[::-1].reset_index(drop=True)
    elif case == "duplicate_context_id":
        comparison.loc[1, "policy_id"] = comparison.loc[0, "policy_id"]
    elif case == "missing_id_repeated":
        comparison.loc[2, "policy_id"] = comparison.loc[3, "policy_id"]
    elif case == "malformed_id":
        comparison.loc[0, "policy_id"] = "p25_m3_global"
    elif case == "threshold_id_mismatch":
        comparison.loc[0, "min_points"] = 50
    elif case == "scope_id_mismatch":
        comparison.loc[0, "scope_policy"] = "surface_then_global"
    elif case == "selection_role":
        comparison.loc[0, "selection_role"] = "global_control"
    elif case == "two_selected":
        comparison.loc[control_index, "selected"] = True
    elif case == "no_selected":
        comparison.loc[primary_index, "selected"] = False
    elif case == "context_selected":
        comparison.loc[primary_index, "selected"] = False
        comparison.loc[0, "selected"] = True
    elif case == "structural_flag":
        comparison.loc[0, "passes_structural_minimum"] = True
    elif case == "coverage_flag":
        comparison.loc[7, "passes_coverage_limit"] = True
    elif case == "wilson_flag":
        comparison.loc[primary_index, "passes_wilson_limit"] = False
    elif case == "stability_flag":
        comparison.loc[primary_index, "passes_stability_limit"] = False
    elif case == "all_flag":
        comparison.loc[primary_index, "passes_all_absolute_limits"] = False
    elif case == "primary_selected_despite_limit":
        summary["candidate_checks"]["passes_coverage_limit"] = False
        summary["candidate_checks"]["passes_all_absolute_limits"] = False
        summary["candidate_checks"]["passes_all_selection_criteria"] = False
        comparison.loc[primary_index, "worst_fold_complete_match_coverage"] = 0.49
        comparison.loc[primary_index, "passes_coverage_limit"] = False
        comparison.loc[primary_index, "passes_all_absolute_limits"] = False
    elif case == "control_selected_while_primary_passes":
        summary["selection_status"] = "selected_global_fallback"
        summary["selection_path"] = "fallback_to_global_control"
        summary["selected_policy"] = GLOBAL_CONTROL
        summary["selected_policy_contract"]["policy_id"] = GLOBAL_CONTROL
        summary["reason_codes"] = ["primary_passes_coverage_limit"]
        comparison.loc[primary_index, "selected"] = False
        comparison.loc[control_index, "selected"] = True
    elif case == "fallback_without_primary_failure":
        summary["decision_execution"]["fallback_evaluated"] = True
        summary["decision_execution"]["fallback_activated"] = True
        summary["fallback_evaluation"] = {
            "evaluated": True,
            "activated": True,
            "control_checks": summary["control_descriptive_checks"],
        }
    elif case == "not_available_with_policy":
        summary["selection_status"] = "not_available"
        summary["selection_path"] = "no_candidate_satisfied_limits"
        summary["reason_codes"] = ["primary_passes_coverage_limit"]
    elif case == "incompatible_reason":
        summary["reason_codes"] = ["primary_passes_coverage_limit"]
    elif case == "partial_contract":
        summary["selected_policy_contract"].pop("automatic_recommendation")
    elif case == "test_seal":
        summary["sealed_test_contract"]["test_read"] = True
    elif case == "reconciliation_removed":
        summary["reconciliations"].pop("comparison_canonical")
    elif case == "reconciliation_extra":
        summary["reconciliations"]["unknown"] = True
    elif case == "coercible_boolean":
        summary["reconciliations"]["comparison_canonical"] = "true"
    elif case == "coercible_comparison_flag":
        comparison["selected"] = comparison["selected"].astype(object)
        comparison.loc[primary_index, "selected"] = 1
    else:  # pragma: no cover - protege el catalogo del propio test
        raise AssertionError(case)
    return _rebind_fingerprint(result, summary, comparison)


@pytest.mark.parametrize(
    "case",
    [
        "wrong_path",
        "missing_primary_check",
        "false_primary_check",
        "reverse_order",
        "duplicate_context_id",
        "missing_id_repeated",
        "malformed_id",
        "threshold_id_mismatch",
        "scope_id_mismatch",
        "selection_role",
        "two_selected",
        "no_selected",
        "context_selected",
        "structural_flag",
        "coverage_flag",
        "wilson_flag",
        "stability_flag",
        "all_flag",
        "primary_selected_despite_limit",
        "control_selected_while_primary_passes",
        "fallback_without_primary_failure",
        "not_available_with_policy",
        "incompatible_reason",
        "partial_contract",
        "test_seal",
        "reconciliation_removed",
        "reconciliation_extra",
        "coercible_boolean",
        "coercible_comparison_flag",
    ],
)
def test_semantically_invalid_result_is_rejected_with_rebound_fingerprint(
    synthetic_upstream,
    case,
):
    result = build_selection(synthetic_upstream)
    manipulated = _apply_semantic_mutation(case, result)
    assert manipulated.summary["publication_fingerprint"] == manipulated.publication_fingerprint
    with pytest.raises((ValueError, UpstreamContractError)):
        validate_result(manipulated)


def test_valid_result_uses_exact_contract_order_roles_and_rejection_reasons(synthetic_upstream):
    result = build_selection(synthetic_upstream)
    assert result.comparison["policy_id"].tolist() == list(SHORTLIST)
    assert result.comparison["selection_role"].tolist() == [
        "context_only",
        "context_only",
        "context_only",
        "context_only",
        "global_control",
        "primary_candidate",
        "context_only",
        "context_only",
    ]
    assert result.comparison["rejection_reason"].tolist() == [
        "minimum_points_not_met|minimum_matches_not_met|context_only_not_in_primary_fallback_path",
        "minimum_points_not_met|context_only_not_in_primary_fallback_path",
        "minimum_points_not_met|context_only_not_in_primary_fallback_path",
        "minimum_matches_not_met|context_only_not_in_primary_fallback_path",
        "primary_selected_fallback_not_activated",
        "",
        "context_only_not_in_primary_fallback_path",
        "coverage_limit_not_met|context_only_not_in_primary_fallback_path",
    ]
    validate_result(result)


def test_control_is_descriptively_compared_without_fallback_activation(synthetic_upstream):
    summary = build_selection(synthetic_upstream).summary
    assert summary["control_descriptive_checks"]["passes_all_selection_criteria"] is True
    assert summary["decision_execution"]["control_descriptively_compared"] is True
    assert summary["fallback_evaluation"] == {
        "evaluated": False,
        "activated": False,
        "control_checks": None,
    }


def test_comparison_manipulation_without_rebound_is_detected(synthetic_upstream):
    result = build_selection(synthetic_upstream)
    comparison = result.comparison.copy(deep=True)
    comparison.loc[0, "complete_match_coverage"] += 0.01
    with pytest.raises(ValueError):
        validate_result(replace(result, comparison=comparison))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_pareto_axes_are_rejected(synthetic_upstream, bad_value):
    candidates = synthetic_upstream.candidates.copy(deep=True)
    candidates.loc[0, "worst_fold_p90_wilson_width"] = bad_value
    with pytest.raises(UpstreamContractError):
        reconstruct_dominators(candidates)


def test_serialization_is_deterministic_ordered_utf8_and_index_free(synthetic_upstream):
    result = build_selection(synthetic_upstream)
    first = serialize_artifacts(result)
    second = serialize_artifacts(result)
    assert first == second
    summary = json.loads(first[0].decode("utf-8"))
    assert summary["publication_fingerprint"] == result.publication_fingerprint
    table = pd.read_csv(io.BytesIO(first[1]))
    assert table.columns.tolist() == COMPARISON_COLUMNS
    assert table["policy_id"].tolist() == list(SHORTLIST)
    assert not any(column.startswith("Unnamed") for column in table.columns)


def test_fingerprint_changes_with_summary_or_comparison(synthetic_upstream):
    result = build_selection(synthetic_upstream)
    summary = json.loads(json.dumps(result.summary))
    summary["publication_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="Fingerprint"):
        validate_result(replace(
            result,
            summary=summary,
            publication_fingerprint="0" * 64,
        ))


def test_atomic_publication_writes_both_artifacts(tmp_path, synthetic_upstream):
    result = build_selection(synthetic_upstream)
    summary_path = tmp_path / "summary.json"
    comparison_path = tmp_path / "comparison.csv"
    write_artifacts(result, summary_path, comparison_path)
    assert (summary_path.read_bytes(), comparison_path.read_bytes()) == serialize_artifacts(result)
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_publication_rolls_back_second_replacement(tmp_path, monkeypatch, synthetic_upstream):
    result = build_selection(synthetic_upstream)
    paths = (tmp_path / "summary.json", tmp_path / "comparison.csv")
    originals = (b"old-summary", b"old-comparison")
    for path, payload in zip(paths, originals):
        path.write_bytes(payload)
    real_replace = module.os.replace
    calls = 0

    def fail_second(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("controlled second replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_second)
    with pytest.raises(OSError, match="second replacement"):
        write_artifacts(result, *paths)
    assert tuple(path.read_bytes() for path in paths) == originals
    assert not list(tmp_path.glob("*.tmp"))


def test_not_available_has_header_only_and_no_partial_selection():
    result = not_available_result(["upstream_hash_mismatch"])
    assert result.summary["selection_status"] == "not_available"
    assert result.summary["selected_policy"] is None
    assert result.comparison.empty
    assert pd.read_csv(io.BytesIO(serialize_artifacts(result)[1])).empty


def test_not_available_rejects_partial_comparison():
    result = not_available_result(["upstream_contract_failure"])
    comparison = pd.DataFrame([{column: False for column in COMPARISON_COLUMNS}])
    with pytest.raises(ValueError, match="parcial"):
        validate_result(replace(result, comparison=comparison))


def test_source_contains_no_parquet_or_data_access_and_single_decision_call():
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "points_enriched" not in source
    assert "read_parquet" not in source
    assert "build_historical_snapshots" not in source
    assert "build_direction_features" not in source
    main_source = source.split("def main()", 1)[1]
    assert main_source.count("run_selection_once()") == 1


@pytest.mark.integration
def test_published_selection_contract_without_parquet():
    paths = (
        module.SUMMARY_PATH,
        module.COMPARISON_PATH,
        module.VALIDATION_SUMMARY_PATH,
        module.VALIDATION_CANDIDATES_PATH,
        module.VALIDATION_PARETO_PATH,
        module.VALIDATION_BY_FOLD_PATH,
    )
    if not all(path.exists() for path in paths):
        pytest.skip("Faltan artefactos publicados; no se requiere ningun Parquet.")
    for relative, expected_hash in module.UPSTREAM_HASHES.items():
        assert hashlib.sha256((module.ROOT / relative).read_bytes()).hexdigest().upper() == expected_hash
    summary = json.loads(module.SUMMARY_PATH.read_text(encoding="utf-8"))
    comparison = pd.read_csv(module.COMPARISON_PATH)
    assert summary["selection_status"] == "selected_primary"
    assert summary["selected_policy"] == PRIMARY_POLICY
    assert summary["sealed_test_contract"]["test_read"] is False
    assert summary["sealed_test_contract"]["test_evaluation_runs"] == 0
    assert comparison["selected"].sum() == 1
    assert comparison.loc[comparison["selected"], "policy_id"].item() == PRIMARY_POLICY
