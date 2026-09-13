"""Pruebas sinteticas adversariales del contrato publico de recomendaciones."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, replace
from datetime import date
from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
import re

import pytest

import src.recommender.tactical_recommendation_contract as contract
from src.recommender.tactical_recommendation_contract import (
    PUBLIC_CONTRACT_VERSION,
    PUBLIC_FINGERPRINT_DOMAIN,
    PUBLIC_LIMITATIONS,
    TacticalOptionStatus,
    TacticalRecommendationContractError,
    TacticalRecommendationResponse,
    build_public_tactical_recommendation,
    canonical_tactical_recommendation_json,
    tactical_recommendation_fingerprint,
    validate_public_evidence_component,
    validate_public_tactical_recommendation,
    validate_tactical_pattern_card,
    validate_tactical_recommendation_option,
)
from src.recommender.tactical_feature_encoder import (
    P02_FEATURE_NAMES,
    P04_FEATURE_NAMES,
    P05_FEATURE_NAMES,
    P06_FEATURE_NAMES,
    P09_FEATURE_NAMES,
    TacticalEncodingPolicy,
    build_tactical_feature_schema,
    feature_schema_fingerprint,
)
from src.recommender.tactical_matchup_evidence import (
    CATEGORY_RECONCILIATIONS,
    EVIDENCE_PROVENANCE,
    MATCHUP_EVIDENCE_CONTRACT_VERSION,
    PAIR_RECONCILIATIONS,
    PATTERN_OWNERSHIP_CATALOG,
    RESULT_RECONCILIATIONS,
    TacticalCategoryEvidence,
    TacticalEvidencePerspective,
    TacticalEvidenceScope,
    TacticalEvidenceScopeStrategy,
    TacticalMatchupCategoryEvidence,
    TacticalMatchupCategoryState,
    TacticalMatchupEvidence,
    TacticalMatchupQuery,
    _evidence_state,
    _joint_state,
    _result_state,
    tactical_matchup_query_fingerprint,
    tactical_wilson_interval,
)
from src.recommender.tactical_prioritization import (
    RESULT_FINGERPRINT_DOMAIN,
    TacticalPrioritizationContractError,
    default_tactical_scoring_policy,
    explain_tactical_candidate,
    prioritize_tactical_matchup,
    tactical_prioritization_result_fingerprint,
    validate_tactical_prioritization_result,
)


MODULE_PATH = Path(contract.__file__)
_ZERO = (0, 0, 0, 0, 0, 0, 0)
_PATTERNS = (
    ("P02", P02_FEATURE_NAMES),
    ("P04", P04_FEATURE_NAMES),
    ("P05", P05_FEATURE_NAMES),
    ("P06", P06_FEATURE_NAMES),
)
_OWNERSHIP = {item.pattern_id: item for item in PATTERN_OWNERSHIP_CATALOG}


def _pattern_features(policy):
    if policy is TacticalEncodingPolicy.PROFILE_ONLY:
        return (("P02", P02_FEATURE_NAMES), ("P09", P09_FEATURE_NAMES))
    return _PATTERNS


def _ok(labeled: int, successes: int, distinct: int = 10):
    return (200, 150, 100, labeled, labeled, successes, distinct)


def _insufficient(labeled: int = 30, successes: int = 15, distinct: int = 5):
    return (200, 150, 100, labeled, labeled, successes, distinct)


def _manual_wilson(successes: int, trials: int):
    z = 1.959963984540054
    rate = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    centre = (rate + z_squared / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            rate * (1.0 - rate) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    lower = 0.0 if successes == 0 else centre - margin
    upper = 1.0 if successes == trials else centre + margin
    return rate, lower, upper


def _category(query, pattern_id, feature_name, perspective, spec):
    category = feature_name.split(".", 2)[2]
    relevant, applicable, observed, active, labeled, successes, distinct = spec
    rate, lower, upper = tactical_wilson_interval(
        successes, labeled, feature_name=feature_name
    )
    state, reasons = _evidence_state(
        structurally_applicable=pattern_id != "P02" or 1 in query.serve_numbers,
        relevant_attempt_count=relevant,
        observed_count=observed,
        active_count=active,
        labeled_count=labeled,
        match_count=distinct,
        minimum_labeled_attempts=query.minimum_labeled_attempts,
        minimum_matches=query.minimum_matches,
    )
    return TacticalCategoryEvidence(
        MATCHUP_EVIDENCE_CONTRACT_VERSION,
        query,
        tactical_matchup_query_fingerprint(query),
        query.policy,
        feature_schema_fingerprint(
            build_tactical_feature_schema(query.policy)
        ),
        pattern_id,
        feature_name,
        category,
        perspective,
        TacticalEvidenceScope.GLOBAL,
        _OWNERSHIP[pattern_id].executor_role,
        query.serve_numbers,
        query.minimum_labeled_attempts,
        query.minimum_matches,
        relevant,
        applicable,
        observed,
        active,
        labeled,
        successes,
        labeled - successes,
        rate,
        lower,
        upper,
        distinct,
        state,
        reasons,
        EVIDENCE_PROVENANCE,
        CATEGORY_RECONCILIATIONS,
    )


def _pair(query, pattern_id, feature_name, exec_spec, opp_spec):
    executor = _category(
        query, pattern_id, feature_name, TacticalEvidencePerspective.EXECUTOR, exec_spec
    )
    opponent = _category(
        query,
        pattern_id,
        feature_name,
        TacticalEvidencePerspective.OPPONENT_ALLOWED,
        opp_spec,
    )
    state, reasons = _joint_state(executor, opponent)
    return TacticalMatchupCategoryEvidence(
        MATCHUP_EVIDENCE_CONTRACT_VERSION,
        tactical_matchup_query_fingerprint(query),
        query.policy,
        feature_schema_fingerprint(build_tactical_feature_schema(query.policy)),
        pattern_id,
        feature_name,
        feature_name.split(".", 2)[2],
        executor,
        opponent,
        state,
        reasons,
        PAIR_RECONCILIATIONS,
    )


def _evidence(
    specs,
    *,
    player="AlphaOne",
    opponent="BetaTwo",
    as_of=date(2026, 9, 1),
    serves=(1,),
    patterns=("P02", "P04", "P05", "P06"),
    minimum=(50, 5),
    policy=TacticalEncodingPolicy.COMPONENT_ONLY,
):
    schema = build_tactical_feature_schema(policy)
    query = TacticalMatchupQuery(
        MATCHUP_EVIDENCE_CONTRACT_VERSION,
        player,
        opponent,
        as_of,
        policy,
        feature_schema_fingerprint(schema),
        serves,
        None,
        minimum[0],
        minimum[1],
        patterns,
        TacticalEvidenceScopeStrategy.GLOBAL_ONLY,
    )
    categories = []
    for pattern_id, features in _pattern_features(policy):
        if pattern_id not in patterns:
            continue
        for feature_name in features:
            spec_value = specs[pattern_id][feature_name.split(".", 2)[2]]
            if len(spec_value) == 2 and isinstance(spec_value[0], tuple):
                exec_spec, opp_spec = spec_value
            else:
                exec_spec = opp_spec = spec_value
            categories.append(
                _pair(query, pattern_id, feature_name, exec_spec, opp_spec)
            )
    category_tuple = tuple(categories)
    state, reasons, candidates, comparable = _result_state(category_tuple)
    state_counts = tuple(
        (name.value, sum(item.matchup_state is name for item in category_tuple))
        for name in TacticalMatchupCategoryState
    )
    return TacticalMatchupEvidence(
        MATCHUP_EVIDENCE_CONTRACT_VERSION,
        query,
        policy,
        feature_schema_fingerprint(schema),
        state,
        reasons,
        category_tuple,
        len(category_tuple),
        candidates,
        comparable,
        candidates - comparable,
        state_counts,
        100,
        100,
        60,
        10,
        0,
        15,
        10,
        5,
        False,
        RESULT_RECONCILIATIONS,
        EVIDENCE_PROVENANCE,
    )


def _all_available_specs(patterns=("P02", "P04", "P05", "P06")):
    specs = {}
    for pattern_id, features in _PATTERNS:
        if pattern_id not in patterns:
            continue
        count = len(features)
        specs[pattern_id] = {
            feature_name.split(".", 2)[2]: (
                _ok(60, round(60 * (index + 1) / (count + 1))),
                _ok(60, round(60 * (index + 2) / (count + 2))),
            )
            for index, feature_name in enumerate(features)
        }
    return specs


def _zero_specs(patterns=("P02", "P04", "P05", "P06")):
    return {
        pattern_id: {
            feature_name.split(".", 2)[2]: _ZERO for feature_name in features
        }
        for pattern_id, features in _PATTERNS
        if pattern_id in patterns
    }


def _rich_specs():
    p06 = {
        feature_name.split(".", 2)[2]: _ZERO for feature_name in P06_FEATURE_NAMES
    }
    p06.update(
        {
            "f": (_ok(60, 54), _ok(60, 54)),
            "b": (_ok(80, 68), _ok(80, 34)),
            "r": (_ok(60, 30), _ok(60, 30)),
            "s": (_ok(60, 30), _ok(60, 30)),
        }
    )
    return {
        "P02": {
            "4": (_ok(60, 30), _ok(60, 30)),
            "5": (_ok(100, 90), _ok(80, 40)),
            "6": (_ok(50, 20), _ok(50, 10)),
        },
        "P04": {
            "1": (_ok(100, 80), _ok(100, 80)),
            "2": (_ok(100, 20), _ok(100, 60)),
            "3": (_ok(100, 60), _ok(100, 60)),
        },
        "P05": {
            "7": (_ok(60, 18), _ok(60, 18)),
            "8": (_insufficient(), _ok(60, 30)),
            "9": _ZERO,
        },
        "P06": p06,
    }


def _partial_specs():
    specs = _rich_specs()
    specs["P05"] = {"7": _ZERO, "8": _ZERO, "9": _ZERO}
    return specs


def _score_specs():
    specs = _zero_specs()
    specs["P04"] = {
        "1": (_ok(100, 80), _ok(100, 80)),
        "2": (_ok(60, 0), _ok(60, 0)),
        "3": (_ok(60, 60), _ok(60, 60)),
    }
    return specs


def _state_specs(second_serve=False):
    if second_serve:
        p02 = {"4": _ZERO, "5": _ZERO, "6": _ZERO}
    else:
        p02 = {
            "4": (_ok(60, 30), _ok(60, 30)),
            "5": (_ok(60, 30), _insufficient()),
            "6": (_insufficient(), _ok(60, 30)),
        }
    return {
        "P02": p02,
        "P04": {
            "1": (_insufficient(), _insufficient()),
            "2": (_ZERO, _ZERO),
            "3": (_ok(60, 30), _ok(60, 30)),
        },
        "P05": _zero_specs(("P05",))["P05"],
        "P06": _zero_specs(("P06",))["P06"],
    }


def _source(
    specs,
    *,
    player="AlphaOne",
    opponent="BetaTwo",
    as_of=date(2026, 9, 1),
    serves=(1,),
    patterns=("P02", "P04", "P05", "P06"),
    minimum=(50, 5),
    top_k=3,
    policy=TacticalEncodingPolicy.COMPONENT_ONLY,
):
    evidence = _evidence(
        specs,
        player=player,
        opponent=opponent,
        as_of=as_of,
        serves=serves,
        patterns=patterns,
        minimum=minimum,
        policy=policy,
    )
    return prioritize_tactical_matchup(evidence, top_k=top_k)


def _response(specs, **kwargs):
    return build_public_tactical_recommendation(_source(specs, **kwargs))


def _card(response, pattern_id):
    return next(item for item in response.cards if item.pattern_id == pattern_id)


def _option(card, category):
    return next(item for item in card.options if item.category == category)


def _unchecked_replace(value, **changes):
    """Construye una mutacion adversarial sin ejecutar __post_init__."""
    clone = object.__new__(type(value))
    for field in fields(value):
        object.__setattr__(
            clone,
            field.name,
            changes.get(field.name, getattr(value, field.name)),
        )
    return clone


@pytest.fixture(scope="module")
def rich_source():
    return _source(_rich_specs())


@pytest.fixture(scope="module")
def rich_response(rich_source):
    return build_public_tactical_recommendation(rich_source)


@pytest.fixture(scope="module")
def available_response():
    return _response(_all_available_specs())


@pytest.fixture(scope="module")
def partial_response():
    return _response(_partial_specs())


@pytest.fixture(scope="module")
def absent_response():
    return _response(_zero_specs())


def test_module_imports_and_exposes_exact_public_api():
    required = {
        "PublicEvidenceComponent",
        "TacticalRecommendationOption",
        "TacticalPatternCard",
        "TacticalRecommendationResponse",
        "build_public_tactical_recommendation",
        "validate_public_evidence_component",
        "validate_tactical_recommendation_option",
        "validate_tactical_pattern_card",
        "validate_public_tactical_recommendation",
        "canonical_tactical_recommendation_json",
        "tactical_recommendation_fingerprint",
    }
    assert required <= set(contract.__all__)
    assert type(contract.__all__) is tuple
    assert all(type(name) is str for name in contract.__all__)
    for name in contract.__all__:
        assert hasattr(contract, name)
    import src.recommender.tactical_recommendation_contract as fresh

    assert fresh is contract


def test_public_signatures_follow_contract():
    expected = {
        "build_public_tactical_recommendation": ("prioritization",),
        "validate_public_evidence_component": ("component",),
        "validate_tactical_recommendation_option": ("option",),
        "validate_tactical_pattern_card": ("card",),
        "validate_public_tactical_recommendation": ("response",),
        "canonical_tactical_recommendation_json": ("response",),
        "tactical_recommendation_fingerprint": ("response",),
    }
    for name, positional in expected.items():
        parameters = tuple(
            item.name
            for item in inspect.signature(getattr(contract, name)).parameters.values()
            if item.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
        assert parameters == positional
    source_parameter = inspect.signature(
        contract.validate_public_tactical_recommendation
    ).parameters["source"]
    assert source_parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert source_parameter.default is None


@pytest.mark.parametrize(
    ("validator", "payload"),
    [
        (validate_public_evidence_component, 1),
        (validate_public_evidence_component, TacticalRecommendationResponse),
        (validate_tactical_recommendation_option, {"option": "x"}),
        (validate_tactical_pattern_card, (1, 2)),
        (validate_public_tactical_recommendation, "response"),
        (canonical_tactical_recommendation_json, 3.5),
        (tactical_recommendation_fingerprint, []),
        (build_public_tactical_recommendation, "prioritization"),
        (build_public_tactical_recommendation, {"rankings": ()}),
    ],
)
def test_public_api_rejects_non_contractual_types(validator, payload):
    with pytest.raises((TypeError, TacticalRecommendationContractError)):
        validator(payload)


def test_response_declares_exact_frozen_policy_fields(rich_response):
    assert rich_response.contract_version == PUBLIC_CONTRACT_VERSION == "1.0.0"
    assert rich_response.methodology == "descriptive_observational"
    assert rich_response.combination == "equal_weight_executor_opponent"
    assert type(rich_response.executor_weight) is float
    assert rich_response.executor_weight == 0.5
    assert type(rich_response.opponent_weight) is float
    assert rich_response.opponent_weight == 0.5
    assert rich_response.score_formula == (
        "combined_rate = 0.5 * executor_success_rate + "
        "0.5 * opponent_allowed_success_rate"
    )
    assert rich_response.uncertainty_method == (
        "descriptive_uncertainty_envelope_from_two_wilson_intervals"
    )
    assert rich_response.encoder_policy == "component_only"
    assert rich_response.evidence_scope == "global_only"
    assert rich_response.minimum_labeled_activations == 50
    assert rich_response.minimum_distinct_matches == 5
    assert rich_response.requested_top_k == 3
    assert rich_response.ranking_scope == "independent_within_pattern"
    assert rich_response.global_cross_pattern_ranking is False
    assert rich_response.limitations == PUBLIC_LIMITATIONS == (
        "observational_not_causal",
        "historical_execution_not_prescriptive",
        "uncertainty_envelope_not_joint_confidence_interval",
        "coverage_and_annotation_may_bias_results",
        "no_global_cross_pattern_ranking",
        "sealed_test_not_evaluated",
    )


def test_response_field_contract_is_exact():
    names = {item.name for item in fields(TacticalRecommendationResponse)}
    assert {
        "contract_version",
        "status",
        "methodology",
        "combination",
        "executor_weight",
        "opponent_weight",
        "encoder_policy",
        "evidence_scope",
        "minimum_labeled_activations",
        "minimum_distinct_matches",
        "requested_top_k",
        "ranking_scope",
        "global_cross_pattern_ranking",
        "cards",
        "limitations",
        "fingerprint",
    } <= names


def test_global_state_available_when_every_pattern_has_scored_candidates(
    available_response,
):
    assert available_response.status == "available"
    assert available_response.status_reason_codes == (
        "all_requested_patterns_have_scored_candidates",
    )
    assert all(item.status == "available" for item in available_response.cards)


def test_global_state_partially_available(partial_response):
    assert partial_response.status == "partially_available"
    assert partial_response.status_reason_codes == (
        "some_requested_patterns_have_scored_candidates",
    )
    assert _card(partial_response, "P05").status == "not_available"
    assert _card(partial_response, "P02").status == "available"


def test_global_state_not_available_when_nothing_is_scored(absent_response):
    assert absent_response.status == "not_available"
    assert absent_response.status_reason_codes == (
        "no_requested_pattern_has_scored_candidates",
    )
    assert all(item.status == "not_available" for item in absent_response.cards)


def test_mixed_cards_keep_per_pattern_states(rich_response):
    statuses = [item.status for item in rich_response.cards]
    assert statuses == [
        "available",
        "available",
        "partially_available",
        "partially_available",
    ]
    assert rich_response.status == "available"


@pytest.mark.parametrize("pattern_id", ["P02", "P04", "P05", "P06"])
def test_each_pattern_available_in_isolation_is_only_partial_globally(pattern_id):
    full = _all_available_specs()
    for other in ("P02", "P04", "P05", "P06"):
        if other != pattern_id:
            full[other] = _zero_specs((other,))[other]
    response = _response(full)
    assert response.status == "partially_available"
    assert _card(response, pattern_id).status == "available"
    for other in ("P02", "P04", "P05", "P06"):
        if other != pattern_id:
            assert _card(response, other).status == "not_available"


def test_response_contains_exactly_four_cards_in_contractual_order(rich_response):
    assert type(rich_response.cards) is tuple
    assert [item.pattern_id for item in rich_response.cards] == [
        "P02",
        "P04",
        "P05",
        "P06",
    ]


def test_card_catalogs_are_complete_and_in_contractual_order(rich_response):
    assert _card(rich_response, "P02").categories == ("4", "5", "6")
    assert _card(rich_response, "P04").categories == ("1", "2", "3")
    assert _card(rich_response, "P05").categories == ("7", "8", "9")
    assert _card(rich_response, "P06").categories == tuple("fbrsvzopuylmhijkt")
    for card in rich_response.cards:
        assert len(card.categories) == len(set(card.categories))
        assert [item.category for item in card.options] == list(card.categories)
        assert card.total_options == len(card.categories)


@pytest.mark.parametrize(
    ("pattern_id", "categories"),
    [
        ("P02", ("4", "5")),
        ("P02", ("4", "5", "6", "q")),
        ("P02", ("4", "4", "6")),
        ("P02", ("5", "4", "6")),
        ("P04", ("0", "2", "3")),
        ("P05", ("unknown", "8", "9")),
        ("P05", ("censored", "8", "9")),
        ("P06", ("fbrsvzopuylmhijkt",)),
    ],
)
def test_catalog_mutation_is_rejected(rich_response, pattern_id, categories):
    card = _card(rich_response, pattern_id)
    with pytest.raises(TacticalRecommendationContractError):
        replace(card, categories=categories)


@pytest.mark.parametrize("pattern_id", ["P03", "P07", "P08", "P09"])
def test_patterns_outside_public_catalog_are_rejected(rich_response, pattern_id):
    card = _card(rich_response, "P02")
    with pytest.raises(TacticalRecommendationContractError):
        replace(card, pattern_id=pattern_id)


def test_cards_cannot_be_reordered_duplicated_or_missing(rich_response):
    cards = rich_response.cards
    with pytest.raises(TacticalRecommendationContractError):
        replace(rich_response, cards=(cards[1], cards[0], cards[2], cards[3]))
    with pytest.raises(TacticalRecommendationContractError):
        replace(rich_response, cards=(cards[0], cards[1], cards[2]))
    with pytest.raises(TacticalRecommendationContractError):
        replace(rich_response, cards=(cards[0],) * 4)


@pytest.mark.parametrize(
    ("pattern_id", "expected_actor", "expected_opportunity"),
    [
        ("P02", "server", "first_serve_direction"),
        ("P04", "returner", "initial_return_direction"),
        ("P05", "returner", "initial_return_depth"),
        ("P06", "returner", "initial_return_shot_type"),
    ],
)
def test_documented_actor_and_opportunity_per_pattern(
    rich_response, pattern_id, expected_actor, expected_opportunity
):
    card = _card(rich_response, pattern_id)
    assert card.actor == expected_actor
    assert card.tactical_opportunity == expected_opportunity
    for option in card.options:
        assert option.actor == expected_actor
        assert option.tactical_opportunity == expected_opportunity


@pytest.mark.parametrize("pattern_id", ["P02", "P04", "P05", "P06"])
def test_actor_and_opportunity_mutations_are_rejected(rich_response, pattern_id):
    card = _card(rich_response, pattern_id)
    with pytest.raises(TacticalRecommendationContractError):
        replace(card, actor="Alice")
    invalid_opportunity = (
        "initial_return_direction" if pattern_id == "P02" else "first_serve_direction"
    )
    with pytest.raises(TacticalRecommendationContractError):
        replace(card, tactical_opportunity=invalid_opportunity)
    option = _option(card, card.categories[0])
    with pytest.raises(TacticalRecommendationContractError):
        replace(option, actor="player")
    if pattern_id != "P04":
        with pytest.raises(TacticalRecommendationContractError):
            replace(card, actor="returner" if pattern_id == "P02" else "server")


def test_option_status_mapping_reasons_and_fields(rich_response):
    ranked = _option(_card(rich_response, "P02"), "4")
    assert ranked.status == TacticalOptionStatus.RANKED.value
    assert ranked.reason_codes == (
        "both_perspectives_available",
        "scored_equal_weight_evidence",
    )
    executor_insufficient = _option(_card(rich_response, "P05"), "8")
    assert executor_insufficient.status == (
        TacticalOptionStatus.ABSTAINED_INSUFFICIENT_EVIDENCE.value
    )
    assert executor_insufficient.reason_codes == ("executor_evidence_insufficient",)
    not_available = _option(_card(rich_response, "P05"), "9")
    assert not_available.status == TacticalOptionStatus.NOT_AVAILABLE.value
    assert not_available.reason_codes == ("both_perspectives_not_available",)
    assert not_available.score is None
    assert not_available.descriptive_uncertainty_envelope is None
    assert not_available.rank_position is None
    assert not_available.tie_group is None


def test_opponent_insufficient_and_both_insufficient_states():
    response = _response(_state_specs())
    opponent_insufficient = _option(_card(response, "P02"), "5")
    assert opponent_insufficient.status == (
        TacticalOptionStatus.ABSTAINED_INSUFFICIENT_EVIDENCE.value
    )
    assert opponent_insufficient.reason_codes == (
        "opponent_allowed_evidence_insufficient",
    )
    both_insufficient = _option(_card(response, "P04"), "1")
    assert both_insufficient.status == (
        TacticalOptionStatus.ABSTAINED_INSUFFICIENT_EVIDENCE.value
    )
    assert both_insufficient.reason_codes == ("both_perspectives_insufficient",)


def test_not_applicable_only_for_second_serve_p02():
    response = _response(_state_specs(second_serve=True), serves=(2,))
    p02 = _card(response, "P02")
    assert p02.status == "not_available"
    for option in p02.options:
        assert option.status == TacticalOptionStatus.NOT_APPLICABLE.value
        assert option.reason_codes == ("pattern_not_applicable_to_requested_serves",)
    assert _card(response, "P04").categories == ("1", "2", "3")


def test_p02_first_serve_stays_applicable():
    response = _response(_state_specs(second_serve=False))
    for option in _card(response, "P02").options:
        assert option.status != TacticalOptionStatus.NOT_APPLICABLE.value


@pytest.mark.parametrize(
    ("category", "reason_codes"),
    [
        ("4", ("both_perspectives_available",)),
        ("4", ("scored_equal_weight_evidence", "both_perspectives_available")),
        ("4", ("both_perspectives_available", "both_perspectives_available")),
        ("4", ("executor_evidence_insufficient",)),
        ("4", ("other_reason",)),
    ],
)
def test_ranked_reason_code_mutations_are_rejected(rich_response, category, reason_codes):
    option = _option(_card(rich_response, "P02"), category)
    with pytest.raises(TacticalRecommendationContractError):
        replace(option, reason_codes=reason_codes)


@pytest.mark.parametrize(
    ("pattern_id", "category", "reason_codes", "status"),
    [
        ("P05", "8", ("opponent_allowed_evidence_insufficient",), "abstained_insufficient_evidence"),
        ("P05", "8", (), "abstained_insufficient_evidence"),
        ("P05", "8", ("executor_evidence_insufficient", "both_perspectives_insufficient"), "abstained_insufficient_evidence"),
        ("P05", "9", ("pattern_not_applicable_to_requested_serves",), "not_available"),
        ("P05", "9", ("both_perspectives_insufficient",), "not_available"),
        ("P02", "4", ("both_perspectives_not_available",), "not_available"),
        ("P02", "4", ("executor_evidence_insufficient",), "abstained_insufficient_evidence"),
    ],
)
def test_status_reason_impossible_combinations_are_rejected(
    rich_response, pattern_id, category, reason_codes, status
):
    option = _option(_card(rich_response, pattern_id), category)
    with pytest.raises(TacticalRecommendationContractError):
        replace(option, status=status, reason_codes=reason_codes)


def test_status_rejects_alias_and_enum_instances(rich_response):
    option = _option(_card(rich_response, "P02"), "4")
    with pytest.raises(TacticalRecommendationContractError):
        replace(option, status=TacticalOptionStatus.RANKED)
    with pytest.raises(TacticalRecommendationContractError):
        replace(option, status="scored")
    with pytest.raises(TacticalRecommendationContractError):
        replace(option, category="wide")


def test_score_is_exact_equal_weight_combination(rich_response):
    option = _option(_card(rich_response, "P02"), "5")
    executor = option.executor_evidence
    opponent = option.opponent_allowed_evidence
    assert option.score == 0.5 * executor.success_rate + 0.5 * opponent.success_rate
    expected_rate, expected_lower, expected_upper = _manual_wilson(90, 100)
    assert executor.success_rate == expected_rate == 0.9
    assert executor.wilson_lower == pytest.approx(expected_lower, abs=1e-15)
    assert executor.wilson_upper == pytest.approx(expected_upper, abs=1e-15)
    opponent_rate, opponent_lower, opponent_upper = _manual_wilson(40, 80)
    assert opponent.success_rate == opponent_rate == 0.5
    assert opponent.wilson_lower == pytest.approx(opponent_lower, abs=1e-15)
    assert opponent.wilson_upper == pytest.approx(opponent_upper, abs=1e-15)
    assert option.score == 0.5 * 0.9 + 0.5 * (40 / 80)
    assert option.descriptive_uncertainty_envelope[0] == pytest.approx(
        0.5 * expected_lower + 0.5 * opponent_lower, abs=1e-15
    )
    assert option.descriptive_uncertainty_envelope[1] == pytest.approx(
        0.5 * expected_upper + 0.5 * opponent_upper, abs=1e-15
    )
    lower, upper = option.descriptive_uncertainty_envelope
    assert lower <= option.score <= upper


def test_component_reconciliations_hold_for_every_evidence(rich_response):
    for card in rich_response.cards:
        for option in card.options:
            for component in (
                option.executor_evidence,
                option.opponent_allowed_evidence,
            ):
                assert component.successes + component.failures == (
                    component.labeled_activations
                )
                assert component.distinct_matches <= component.labeled_activations
                if component.labeled_activations:
                    assert (
                        component.success_rate
                        == component.successes / component.labeled_activations
                    )


def test_wilson_limits_zero_of_n_and_n_of_n():
    response = _response(_score_specs())
    zero = _option(_card(response, "P04"), "2")
    assert zero.score == 0.0
    for component in (zero.executor_evidence, zero.opponent_allowed_evidence):
        assert component.success_rate == 0.0
        assert component.wilson_lower == 0.0
        assert component.wilson_upper == tactical_wilson_interval(
            0, 60, feature_name="p04.return_direction.2"
        )[2]
    assert zero.descriptive_uncertainty_envelope[0] == 0.0
    full = _option(_card(response, "P04"), "3")
    assert full.score == 1.0
    for component in (full.executor_evidence, full.opponent_allowed_evidence):
        assert component.success_rate == 1.0
        assert component.wilson_upper == 1.0
    assert zero.executor_evidence.wilson_upper < 1.0


def test_uncertainty_envelope_bounds_require_real_floats():
    response = _response(_score_specs())
    zero = _option(_card(response, "P04"), "2")
    full = _option(_card(response, "P04"), "3")
    assert zero.descriptive_uncertainty_envelope[0] == 0.0
    assert full.descriptive_uncertainty_envelope[1] == 1.0
    with pytest.raises(TacticalRecommendationContractError):
        replace(
            zero,
            descriptive_uncertainty_envelope=(
                False,
                zero.descriptive_uncertainty_envelope[1],
            ),
        )
    with pytest.raises(TacticalRecommendationContractError):
        replace(
            full,
            descriptive_uncertainty_envelope=(
                full.descriptive_uncertainty_envelope[0],
                True,
            ),
        )


def test_zero_trials_components_are_absent_and_none(rich_response):
    option = _option(_card(rich_response, "P05"), "9")
    for component in (option.executor_evidence, option.opponent_allowed_evidence):
        assert component.labeled_activations == 0
        assert component.successes == 0 and component.failures == 0
        assert component.distinct_matches == 0
        assert component.success_rate is None
        assert component.wilson_lower is None
        assert component.wilson_upper is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("score", True),
        ("score", 1),
        ("score", 0.71),
        ("score", float("nan")),
        ("score", float("inf")),
        ("score", -0.1),
        ("score", 1.5),
        ("score", "0.7"),
        ("descriptive_uncertainty_envelope", (0.1,)),
        ("descriptive_uncertainty_envelope", (0.1, 0.9, 0.2)),
        ("descriptive_uncertainty_envelope", [0.1, 0.9]),
        ("descriptive_uncertainty_envelope", None),
        ("rank_position", True),
        ("rank_position", 0),
        ("rank_position", -1),
        ("rank_position", 1.0),
        ("rank_position", "1"),
        ("tie_group", 0),
        ("tie_group", True),
        ("score_formula", "other"),
        ("canonical_explanation", "garantiza una mejor opcion"),
        ("canonical_explanation", "causa victoria"),
    ],
)
def test_ranked_option_type_and_semantic_mutations_are_rejected(
    rich_response, field, value
):
    option = _option(_card(rich_response, "P02"), "4")
    with pytest.raises(TacticalRecommendationContractError):
        replace(option, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("score", 0.5),
        ("descriptive_uncertainty_envelope", (0.1, 0.9)),
        ("rank_position", 2),
        ("tie_group", 2),
    ],
)
def test_abstained_option_cannot_gain_score_fields(rich_response, field, value):
    option = _option(_card(rich_response, "P05"), "8")
    with pytest.raises(TacticalRecommendationContractError):
        replace(option, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("labeled_activations", True),
        ("labeled_activations", -1),
        ("labeled_activations", 1.0),
        ("labeled_activations", "60"),
        ("successes", 0.0),
        ("successes", -3),
        ("failures", "1"),
        ("distinct_matches", 999),
        ("success_rate", "0.5"),
        ("success_rate", True),
        ("success_rate", float("nan")),
        ("wilson_lower", float("inf")),
        ("wilson_upper", -0.2),
        ("scope", "surface"),
        ("perspective", "player"),
        ("evidence_state", "bogus_state"),
    ],
)
def test_component_type_scope_and_value_mutations_are_rejected(
    rich_response, field, value
):
    option = _option(_card(rich_response, "P02"), "4")
    with pytest.raises(TacticalRecommendationContractError):
        replace(option.executor_evidence, **{field: value})


def test_public_evidence_component_is_always_valid_and_validated_in_isolation(
    rich_response,
):
    available = _option(_card(rich_response, "P02"), "4").executor_evidence
    validate_public_evidence_component(available)
    assert available.labeled_activations >= 50
    assert available.distinct_matches >= 5

    insufficient = _option(
        _card(rich_response, "P05"), "8"
    ).executor_evidence
    assert insufficient.evidence_state == "insufficient_labeled_attempts"
    with pytest.raises(TacticalRecommendationContractError, match="50/5"):
        replace(insufficient, evidence_state="available")
    with pytest.raises(
        TacticalRecommendationContractError,
        match="insufficient_labeled_attempts",
    ):
        replace(available, evidence_state="insufficient_labeled_attempts")

    insufficient_matches = replace(
        available,
        evidence_state="insufficient_matches",
        distinct_matches=4,
    )
    validate_public_evidence_component(insufficient_matches)
    with pytest.raises(
        TacticalRecommendationContractError, match="insufficient_matches"
    ):
        replace(insufficient_matches, distinct_matches=5)
    with pytest.raises(
        TacticalRecommendationContractError, match="conserva metricas"
    ):
        replace(available, evidence_state="not_available")


def test_public_evidence_threshold_boundaries_are_inclusive(rich_response):
    component = _option(_card(rich_response, "P02"), "6").executor_evidence
    assert component.labeled_activations == 50
    boundary = replace(component, distinct_matches=5)
    assert boundary.evidence_state == "available"
    validate_public_evidence_component(boundary)


def test_invalid_component_cannot_cross_any_public_response_boundary(rich_response):
    card = _card(rich_response, "P02")
    option = _option(card, "4")
    invalid_component = _unchecked_replace(
        option.executor_evidence, distinct_matches=4
    )
    invalid_option = _unchecked_replace(
        option, executor_evidence=invalid_component
    )

    def substituted(items):
        return tuple(
            invalid_option if item.category == option.category else item
            for item in items
        )

    invalid_card = _unchecked_replace(
        card,
        options=substituted(card.options),
        ranked_options=substituted(card.ranked_options),
        top_options=substituted(card.top_options),
    )
    cards = tuple(
        invalid_card if item.pattern_id == card.pattern_id else item
        for item in rich_response.cards
    )
    resigned = _unchecked_replace(
        rich_response,
        cards=cards,
        fingerprint=contract._public_fingerprint_of(
            rich_response.status, rich_response.status_reason_codes, cards
        ),
    )

    for boundary in (
        validate_public_tactical_recommendation,
        canonical_tactical_recommendation_json,
        tactical_recommendation_fingerprint,
    ):
        with pytest.raises(TacticalRecommendationContractError, match="50/5"):
            boundary(resigned)


def test_component_wilson_recomputation_detects_mutations(rich_response):
    option = _option(_card(rich_response, "P02"), "4")
    component = option.executor_evidence
    with pytest.raises(TacticalRecommendationContractError):
        replace(option, executor_evidence=replace(component, wilson_lower=0.11))
    with pytest.raises(TacticalRecommendationContractError):
        replace(option, executor_evidence=replace(component, successes=31, failures=29))


def test_perspective_swap_is_rejected(rich_response):
    option = _option(_card(rich_response, "P02"), "4")
    opponent = option.opponent_allowed_evidence
    swapped = replace(
        option.executor_evidence,
        perspective="opponent_allowed",
        labeled_activations=opponent.labeled_activations,
        successes=opponent.successes,
        failures=opponent.failures,
        distinct_matches=opponent.distinct_matches,
        success_rate=opponent.success_rate,
        wilson_lower=opponent.wilson_lower,
        wilson_upper=opponent.wilson_upper,
    )
    with pytest.raises(TacticalRecommendationContractError):
        replace(option, executor_evidence=swapped)


def test_ranking_with_zero_one_two_three_scored_options():
    for count in range(0, 4):
        specs = _zero_specs()
        categories = ("4", "5", "6")[:count]
        specs["P02"] = {
            category: (
                (_ok(60, round(30 + 5 * index)), _ok(60, round(30 + 5 * index)))
                if category in categories
                else _ZERO
            )
            for index, category in enumerate(("4", "5", "6"))
        }
        response = _response(specs)
        card = _card(response, "P02")
        assert card.scored_options == count
        assert card.abstained_options_count == 3 - count
        assert card.effective_top_k == count
        assert [item.rank_position for item in card.ranked_options] == list(
            range(1, count + 1)
        )
        assert [item.category for item in card.top_options] == [
            item.category for item in card.ranked_options
        ]
        assert card.tie_expanded is False
        assert card.tie_group_count == count
        if count == 0:
            assert card.status == "not_available"
            assert card.top_options == ()


def test_ranking_keeps_upstream_exact_order_and_positions(rich_response):
    card = _card(rich_response, "P02")
    assert [item.category for item in card.ranked_options] == ["5", "4", "6"]
    assert [
        (item.category, item.rank_position, item.tie_group)
        for item in card.ranked_options
    ] == [
        ("5", 1, 1),
        ("4", 2, 2),
        ("6", 3, 3),
    ]
    assert card.tie_group_count == 3


def test_tie_before_top_k_is_kept_without_expansion():
    specs = _zero_specs()
    specs["P02"] = {
        "4": (_ok(60, 30), _ok(60, 30)),
        "5": (_ok(60, 30), _ok(60, 30)),
        "6": (_ok(60, 15), _ok(60, 15)),
    }
    card = _card(_response(specs), "P02")
    assert [item.category for item in card.ranked_options] == ["4", "5", "6"]
    assert [item.tie_group for item in card.ranked_options] == [1, 1, 2]
    assert card.effective_top_k == 3
    assert card.tie_expanded is False


def test_tie_inside_top_k_is_kept_without_expansion():
    specs = _zero_specs()
    specs["P02"] = {
        "4": (_ok(60, 45), _ok(60, 45)),
        "5": (_ok(60, 30), _ok(60, 30)),
        "6": (_ok(60, 30), _ok(60, 30)),
    }
    card = _card(_response(specs), "P02")
    assert [item.category for item in card.ranked_options] == ["4", "5", "6"]
    assert [item.tie_group for item in card.ranked_options] == [1, 2, 2]
    assert [item.rank_position for item in card.ranked_options] == [1, 2, 3]
    assert card.effective_top_k == 3
    assert card.tie_expanded is False


def test_tie_crossing_top_k_is_expanded_and_never_truncated(rich_response):
    card = _card(rich_response, "P06")
    assert card.scored_options == 4
    assert [item.category for item in card.ranked_options] == ["f", "b", "r", "s"]
    assert [item.rank_position for item in card.ranked_options] == [1, 2, 3, 4]
    assert [item.tie_group for item in card.ranked_options] == [1, 2, 3, 3]
    assert card.effective_top_k == 4
    assert card.tie_expanded is True
    assert [item.category for item in card.top_options] == ["f", "b", "r", "s"]
    assert card.tie_group_count == 3
    tie_categories = {
        item.category for item in card.top_options if item.tie_group == 3
    }
    assert tie_categories == {"r", "s"}


def test_more_than_top_k_without_ties_keeps_scored_outside_top():
    specs = _zero_specs()
    rates = (54, 48, 42, 36, 30)
    p06 = {
        feature_name.split(".", 2)[2]: _ZERO for feature_name in P06_FEATURE_NAMES
    }
    p06.update(
        {
            category: (_ok(60, score), _ok(60, score))
            for category, score in zip(("f", "b", "r", "s", "t"), rates)
        }
    )
    specs["P06"] = p06
    card = _card(_response(specs), "P06")
    assert card.scored_options == 5
    assert card.effective_top_k == 3
    assert card.tie_expanded is False
    assert [item.category for item in card.top_options] == ["f", "b", "r"]
    outside = {
        item.category
        for item in card.ranked_options
        if item.category not in {top.category for top in card.top_options}
    }
    assert outside == {"s", "t"}
    for category in outside:
        option = _option(card, category)
        assert option.status == TacticalOptionStatus.RANKED.value
        assert option.rank_position in (4, 5)
        assert option.tie_group is not None


def test_ranking_field_mutations_are_rejected(rich_response):
    card = _card(rich_response, "P06")
    with pytest.raises(TacticalRecommendationContractError):
        replace(card, effective_top_k=3)
    with pytest.raises(TacticalRecommendationContractError):
        replace(card, tie_expanded=False)
    with pytest.raises(TacticalRecommendationContractError):
        replace(card, tie_group_count=2)
    with pytest.raises(TacticalRecommendationContractError):
        replace(card, requested_top_k=4)
    with pytest.raises(TacticalRecommendationContractError):
        replace(card, scored_options=5)
    with pytest.raises(TacticalRecommendationContractError):
        replace(card, abstained_options_count=0)
    with pytest.raises(TacticalRecommendationContractError):
        replace(card, total_options=16)
    with pytest.raises(TacticalRecommendationContractError):
        replace(card, top_options=())
    with pytest.raises(TacticalRecommendationContractError):
        replace(card, top_options=card.ranked_options[:3])
    with pytest.raises(TacticalRecommendationContractError):
        replace(card, ranked_options=tuple(reversed(card.ranked_options)))
    with pytest.raises(TacticalRecommendationContractError):
        replace(card, abstained_options=tuple(reversed(card.abstained_options)))


def test_ranking_subsets_require_exact_canonical_option_objects(rich_response):
    foreign = _response(_all_available_specs())
    card = _card(rich_response, "P02")
    foreign_card = _card(foreign, "P02")
    foreign_by_category = {item.category: item for item in foreign_card.options}
    with pytest.raises(TacticalRecommendationContractError, match="ranking"):
        replace(
            card,
            ranked_options=tuple(
                foreign_by_category[item.category] for item in card.ranked_options
            ),
        )
    with pytest.raises(TacticalRecommendationContractError, match="Top-k"):
        replace(
            card,
            top_options=tuple(
                foreign_by_category[item.category] for item in card.top_options
            ),
        )

    abstained_card = _card(rich_response, "P05")
    zero_card = _card(_response(_zero_specs()), "P05")
    zero_by_category = {item.category: item for item in zero_card.options}
    with pytest.raises(TacticalRecommendationContractError, match="abstenciones"):
        replace(
            abstained_card,
            abstained_options=tuple(
                zero_by_category[item.category]
                for item in abstained_card.abstained_options
            ),
        )


def test_position_and_tie_mutations_on_options_are_rejected(rich_response):
    card = _card(rich_response, "P02")
    options = list(card.options)
    first = options[0]
    with pytest.raises(TacticalRecommendationContractError):
        replace(
            card,
            options=tuple(
                replace(item, rank_position=9) if item is first else item
                for item in options
            ),
        )
    with pytest.raises(TacticalRecommendationContractError):
        replace(
            card,
            options=tuple(
                replace(item, tie_group=99) if item is first else item
                for item in options
            ),
        )


def test_no_cross_pattern_global_ranking_exists(rich_response):
    assert not hasattr(rich_response, "global_ranking")
    field_names = {item.name for item in fields(TacticalRecommendationResponse)}
    assert "global_ranking" not in field_names
    parsed = json.loads(canonical_tactical_recommendation_json(rich_response))

    def keys_of(value):
        if type(value) is dict:
            for key, child in value.items():
                yield key
                yield from keys_of(child)
        elif type(value) is list:
            for child in value:
                yield from keys_of(child)

    assert "global_ranking" not in set(keys_of(parsed))
    best_scores = {
        card.pattern_id: max(
            option.score for option in card.options if option.score is not None
        )
        for card in rich_response.cards
    }
    contract_order = [item.pattern_id for item in rich_response.cards]
    assert contract_order == ["P02", "P04", "P05", "P06"]
    assert [best_scores[item] for item in contract_order] != sorted(
        best_scores.values(), reverse=True
    )


def test_public_dataclasses_are_frozen():
    response = _response(_state_specs())
    objects = (
        response,
        response.cards[0],
        response.cards[0].options[0],
        response.cards[0].options[0].executor_evidence,
    )
    for item in objects:
        with pytest.raises(FrozenInstanceError):
            setattr(item, fields(item)[0].name, None)


def test_public_collections_are_deeply_immutable_types(rich_response):
    containers = []

    def walk(value):
        if type(value) is tuple:
            for item in value:
                walk(item)
        elif type(value).__module__ == "src.recommender.tactical_recommendation_contract":
            for field in fields(value):
                walk(getattr(value, field.name))
        elif type(value) is not str:
            containers.append(type(value))

    walk(rich_response)
    assert set(containers) <= {
        type(None),
        int,
        bool,
        float,
        contract.MappingProxyType,
    }
    assert type(rich_response.cards) is tuple
    assert type(rich_response.limitations) is tuple
    card = _card(rich_response, "P02")
    for field in ("categories", "options", "ranked_options", "top_options", "abstained_options", "status_reason_codes"):
        assert type(getattr(card, field)) is tuple
    option = _option(card, "4")
    assert type(option.reason_codes) is tuple
    assert type(option.descriptive_uncertainty_envelope) is tuple
    assert type(card.reconciliations) is contract.MappingProxyType
    assert type(rich_response.reconciliations) is contract.MappingProxyType


def test_mutable_collections_are_not_retained_or_accepted(rich_response):
    card = _card(rich_response, "P02")
    with pytest.raises(TacticalRecommendationContractError):
        replace(card, categories=["4", "5", "6"])
    with pytest.raises(TacticalRecommendationContractError):
        replace(card, categories={"4", "5", "6"})
    with pytest.raises(TacticalRecommendationContractError):
        replace(card, reconciliations=dict(card.reconciliations))
    with pytest.raises(TacticalRecommendationContractError):
        replace(rich_response, reconciliations=dict(rich_response.reconciliations))
    option = _option(card, "4")
    with pytest.raises(TacticalRecommendationContractError):
        replace(option, reason_codes=list(option.reason_codes))
    with pytest.raises(TacticalRecommendationContractError):
        replace(option, score_formula=123)


def test_public_payload_redacts_all_identities(rich_source, rich_response):
    payload = canonical_tactical_recommendation_json(rich_response).decode("utf-8")
    for identity in (
        "AlphaOne",
        "BetaTwo",
        "2026-09-01",
        rich_source.matchup_query.query_fingerprint,
        rich_source.schema_fingerprint,
        rich_source.source_evidence_fingerprint,
        "match_id",
        "point_number",
        "sequence_text",
    ):
        assert identity not in payload
    for ranking in rich_source.rankings:
        for candidate in ranking.candidates:
            assert candidate.query_fingerprint not in payload
    assert '"player"' not in payload and '"opponent"' not in payload


@pytest.mark.parametrize(
    "key",
    [
        "match_id",
        "match_ids",
        "point_number",
        "point_numbers",
        "player",
        "player_name",
        "opponent",
        "opponent_name",
        "server_player",
        "returner_player",
        "serve_numbers",
        "sequence",
        "sequence_text",
        "raw_sequence",
        "token",
        "tokens",
        "span",
        "spans",
        "first_serve",
        "second_serve",
        "date",
        "as_of_date",
        "target_date",
        "timestamp",
        "path",
        "individual_outcome",
        "point_winner",
        "query_fingerprint",
        "schema_fingerprint",
        "source_evidence_fingerprint",
        "candidate_fingerprint",
        "evaluation",
        "test_status",
        "sealed",
    ],
)
def test_public_tree_rejects_sensitive_keys(key):
    with pytest.raises(TacticalRecommendationContractError):
        contract._validate_public_tree({key: "value"})


@pytest.mark.parametrize(
    "value",
    [
        "file:///etc/passwd",
        "C:\\Users\\secret",
        "/etc/passwd",
        "..\\secret",
        "see ../secret",
        "bad\x01control",
        float("nan"),
        float("inf"),
        -float("inf"),
        ("tuples", "are", "not", "json", "here"),
        object(),
        3.5 + 0.0j,
    ],
)
def test_public_tree_rejects_sensitive_values_and_objects(value):
    with pytest.raises(TacticalRecommendationContractError):
        contract._validate_public_tree({"value": value})


def test_identity_values_cannot_enter_public_fields():
    option = _option(_card(_response(_state_specs()), "P04"), "3")
    with pytest.raises(TacticalRecommendationContractError):
        replace(option, category="Alice")
    with pytest.raises(TacticalRecommendationContractError):
        replace(option, actor="Alice")
    with pytest.raises(TacticalRecommendationContractError):
        replace(
            option,
            executor_evidence=replace(option.executor_evidence, perspective="Alice"),
        )
    with pytest.raises(TacticalRecommendationContractError):
        replace(option, category=object())


@pytest.mark.parametrize(
    "value",
    (
        r"failure at C:\Users\alice\secret",
        r"failure at \\server\share\secret",
        "failure at /Users/alice/secret",
        "failure at /home/alice/secret",
        "failure at ~/secret",
        "failure at vscode-file:///Users/alice/secret",
    ),
)
def test_public_tree_rejects_embedded_local_paths(value):
    with pytest.raises(TacticalRecommendationContractError):
        contract._validate_public_tree({"value": value})


def test_serialization_is_canonical_deterministic_and_compact(rich_response):
    first = canonical_tactical_recommendation_json(rich_response)
    second = canonical_tactical_recommendation_json(rich_response)
    assert first == second
    assert type(first) is bytes
    parsed = json.loads(first.decode("utf-8"))
    assert first == json.dumps(
        parsed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert b"NaN" not in first and b"Infinity" not in first


def test_serialized_keys_are_sorted_recursively(rich_response):
    parsed = json.loads(canonical_tactical_recommendation_json(rich_response).decode())

    def check(value):
        if type(value) is dict:
            keys = list(value.keys())
            assert len(set(keys)) == len(keys)
            assert keys == sorted(keys)
            for child in value.values():
                check(child)
        elif type(value) is list:
            for child in value:
                check(child)

    check(parsed)


def test_fingerprint_is_independently_recomputable(rich_response):
    parsed = json.loads(
        canonical_tactical_recommendation_json(rich_response).decode("utf-8")
    )
    assert re.fullmatch(r"[0-9A-F]{64}", parsed["fingerprint"])
    expected = parsed["fingerprint"]
    parsed.pop("fingerprint")
    core = json.dumps(
        parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert sha256(PUBLIC_FINGERPRINT_DOMAIN + core).hexdigest().upper() == expected
    assert tactical_recommendation_fingerprint(rich_response) == expected


def test_fingerprint_domain_is_separated_from_upstream_domains(
    rich_source, rich_response
):
    assert PUBLIC_FINGERPRINT_DOMAIN == b"tennis-public-tactical-recommendation\x00"
    upstream_domains = {
        RESULT_FINGERPRINT_DOMAIN,
        b"tennis-tactical-scoring-policy\x00",
        b"tennis-tactical-scored-candidate\x00",
        b"tennis-tactical-pattern-ranking\x00",
        b"tennis-tactical-candidate-explanation\x00",
        b"tennis-tactical-feature-schema\x00",
        b"tennis-tactical-matchup-evidence\x00",
    }
    assert PUBLIC_FINGERPRINT_DOMAIN not in upstream_domains
    assert rich_response.fingerprint != tactical_prioritization_result_fingerprint(
        rich_source
    )
    assert rich_response.fingerprint != sha256(
        RESULT_FINGERPRINT_DOMAIN
        + canonical_tactical_recommendation_json(rich_response)
    ).hexdigest().upper()


def test_fingerprint_is_sensitive_to_semantic_change(
    available_response, rich_response
):
    assert available_response.fingerprint != rich_response.fingerprint
    assert available_response.fingerprint != _response(_partial_specs()).fingerprint


def test_resigned_self_consistent_mutation_is_still_rejected(
    rich_source, rich_response
):
    other = _response(_all_available_specs())
    mixed_cards = (
        rich_response.cards[0],
        rich_response.cards[1],
        rich_response.cards[2],
        other.cards[3],
    )
    fingerprint = contract._public_fingerprint_of(
        rich_response.status, rich_response.status_reason_codes, mixed_cards
    )
    mutated = replace(rich_response, cards=mixed_cards, fingerprint=fingerprint)
    validate_public_tactical_recommendation(mutated)
    with pytest.raises(TacticalRecommendationContractError):
        validate_public_tactical_recommendation(mutated, source=rich_source)


def test_identity_redaction_across_sources_produces_identical_public_response():
    source_a = _source(
        _rich_specs(),
        player="UniquePlayerK7",
        opponent="UniqueRivalM3",
        as_of=date(2026, 9, 1),
    )
    source_b = _source(
        _rich_specs(),
        player="AnotherPlayerZ9",
        opponent="AnotherRivalQ4",
        as_of=date(2026, 8, 1),
    )
    assert source_a.matchup_query.query_fingerprint != (
        source_b.matchup_query.query_fingerprint
    )
    response_a = build_public_tactical_recommendation(source_a)
    response_b = build_public_tactical_recommendation(source_b)
    assert response_a == response_b
    assert canonical_tactical_recommendation_json(response_a) == (
        canonical_tactical_recommendation_json(response_b)
    )
    assert tactical_recommendation_fingerprint(response_a) == (
        tactical_recommendation_fingerprint(response_b)
    )


def test_module_architecture_is_pure_and_dependency_free():
    source_text = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert not ({"pandas", "numpy", "scipy", "sklearn"} & imports)
    assert "tactical_recommender_pipeline" not in source_text
    assert "parse_and_classify" not in source_text
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not ({"open", "print", "input", "read_csv", "read_parquet"} & calls)
    assert "__main__" not in source_text
    assert "data/" not in source_text and "reports/" not in source_text
    assert not ({"socket", "urllib", "http", "requests"} & imports)
    assert imports <= {
        "__future__",
        "dataclasses",
        "enum",
        "hashlib",
        "json",
        "math",
        "re",
        "src",
        "types",
        "typing",
    }


def test_module_has_no_mutable_module_level_collections():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            assert not isinstance(value, (ast.List, ast.Dict, ast.Set))


def test_reconstructive_validation_rejects_every_public_mutation(rich_source):
    response = _response(_rich_specs())
    card_of = {item.pattern_id: item for item in response.cards}

    def mutate_card(target_pattern, **card_changes):
        def do():
            cards = tuple(
                replace(item, **card_changes)
                if item.pattern_id == target_pattern
                else item
                for item in response.cards
            )
            return replace(response, cards=cards)

        return do

    def mutate_option(target_pattern, target_category, **option_changes):
        def do():
            original = card_of[target_pattern]
            options = tuple(
                replace(item, **option_changes)
                if item.category == target_category
                else item
                for item in original.options
            )
            cards = tuple(
                replace(original, options=options)
                if item.pattern_id == target_pattern
                else item
                for item in response.cards
            )
            return replace(response, cards=cards)

        return do

    def mutate_component(target_pattern, target_category, perspective_field, **changes):
        def do():
            original = card_of[target_pattern]
            options = []
            for item in original.options:
                if item.category == target_category:
                    component = getattr(item, perspective_field)
                    options.append(replace(item, **{perspective_field: replace(component, **changes)}))
                else:
                    options.append(item)
            cards = tuple(
                replace(original, options=tuple(options))
                if item.pattern_id == target_pattern
                else item
                for item in response.cards
            )
            return replace(response, cards=cards)

        return do

    mutations = [
        lambda: replace(response, contract_version="2.0.0"),
        lambda: replace(response, status="not_available"),
        lambda: replace(response, status_reason_codes=("other",)),
        lambda: replace(response, methodology="causal"),
        lambda: replace(response, combination="weighted_by_trials"),
        lambda: replace(response, score_formula="other"),
        lambda: replace(response, uncertainty_method="joint_confidence_interval"),
        lambda: replace(response, executor_weight=0.6),
        lambda: replace(response, executor_weight=1),
        lambda: replace(response, opponent_weight=0.4),
        lambda: replace(response, encoder_policy="profile_only"),
        lambda: replace(response, evidence_scope="global"),
        lambda: replace(response, minimum_labeled_activations=30),
        lambda: replace(response, minimum_labeled_activations=True),
        lambda: replace(response, minimum_distinct_matches=1),
        lambda: replace(response, requested_top_k=5),
        lambda: replace(response, ranking_scope="global"),
        lambda: replace(response, global_cross_pattern_ranking=True),
        lambda: replace(response, limitations=PUBLIC_LIMITATIONS[:-1]),
        lambda: replace(response, limitations=PUBLIC_LIMITATIONS + ("extra",)),
        lambda: replace(
            response,
            reconciliations=contract.MappingProxyType(
                dict(response.reconciliations, policy_frozen=False)
            ),
        ),
        lambda: replace(response, fingerprint="A" * 64),
        lambda: replace(response, cards=(response.cards[1],) + response.cards[1:]),
        lambda: replace(response, cards=response.cards[:3]),
        lambda: replace(response, cards=(response.cards[0],) * 4),
        mutate_card("P02", pattern_id="P03"),
        mutate_card("P02", actor="returner"),
        mutate_card(
            "P05",
            status="available",
            status_reason_codes=("all_categories_scored",),
        ),
        mutate_card("P02", effective_top_k=4),
        mutate_card("P06", tie_expanded=False),
        mutate_card("P02", total_options=4),
        mutate_option("P02", "4", category="5"),
        mutate_option("P02", "4", status="not_available"),
        mutate_option("P02", "4", reason_codes=("executor_evidence_insufficient",)),
        mutate_option("P02", "4", score=0.501),
        mutate_option("P02", "4", descriptive_uncertainty_envelope=(0.1, 0.9)),
        mutate_option("P02", "4", rank_position=9),
        mutate_option("P02", "4", tie_group=99),
        mutate_option("P02", "4", score_formula="other"),
        mutate_option("P02", "4", canonical_explanation="texto alterado"),
        mutate_option("P02", "4", actor="Alice"),
        mutate_component("P02", "4", "executor_evidence", success_rate=0.1),
        mutate_component("P02", "4", "opponent_allowed_evidence", wilson_upper=0.11),
    ]

    for mutation in mutations:

        def case(mutation=mutation):
            mutated = mutation()
            validate_public_tactical_recommendation(mutated, source=rich_source)

        def case_without_source(mutation=mutation):
            mutated = mutation()
            validate_public_tactical_recommendation(mutated)

        with pytest.raises(TacticalRecommendationContractError):
            case()
        with pytest.raises(
            (TacticalRecommendationContractError, TypeError)
        ):
            case_without_source()


def test_source_must_itself_be_contractually_valid(rich_source, rich_response):
    with pytest.raises(TacticalPrioritizationContractError):
        replace(rich_source, state="not_available")
    validate_public_tactical_recommendation(rich_response, source=rich_source)
    assert (
        build_public_tactical_recommendation(rich_source).fingerprint
        == rich_response.fingerprint
    )


def test_frozen_policy_gate_rejects_off_contract_sources():
    base = _rich_specs()
    with pytest.raises(TacticalRecommendationContractError):
        build_public_tactical_recommendation(_source(base, minimum=(30, 5)))
    with pytest.raises(TacticalRecommendationContractError):
        build_public_tactical_recommendation(_source(base, minimum=(50, 1)))
    with pytest.raises(TacticalRecommendationContractError):
        build_public_tactical_recommendation(_source(base, patterns=("P02", "P04")))
    with pytest.raises(TacticalRecommendationContractError):
        build_public_tactical_recommendation(_source(base, top_k=2))
    profile_specs = {
        "P02": {
            feature.split(".", 2)[2]: (_ok(60, 30), _ok(60, 30))
            for feature in P02_FEATURE_NAMES
        },
        "P09": {
            feature.split(".", 2)[2]: (_ok(60, 30), _ok(60, 30))
            for feature in P09_FEATURE_NAMES
        },
    }
    with pytest.raises(TacticalRecommendationContractError):
        build_public_tactical_recommendation(
            _source(
                profile_specs,
                patterns=("P02", "P09"),
                policy=TacticalEncodingPolicy.PROFILE_ONLY,
            )
        )


def test_validate_with_source_rejects_valid_but_different_source(rich_source, rich_response):
    other_source = _source(_all_available_specs())
    validate_tactical_prioritization_result(other_source)
    with pytest.raises(TacticalRecommendationContractError):
        validate_public_tactical_recommendation(rich_response, source=other_source)
    validate_public_tactical_recommendation(rich_response, source=rich_source)


def test_upstream_source_stays_valid_and_public_reuses_explanations(
    rich_source, rich_response
):
    validate_tactical_prioritization_result(rich_source)
    assert rich_source.policy == default_tactical_scoring_policy()
    assert rich_source.matchup_query.minimum_labeled_attempts == 50
    assert rich_source.matchup_query.minimum_matches == 5
    assert all(item.requested_top_k == 3 for item in rich_source.rankings)
    assert rich_response.fingerprint
    for ranking in rich_source.rankings:
        for candidate in ranking.candidates:
            explanation = explain_tactical_candidate(candidate)
            option = _option(
                _card(rich_response, candidate.pattern_id), candidate.category
            )
            assert option.canonical_explanation == explanation.canonical_text
            assert option.pattern_id == explanation.pattern_id
            assert option.category == explanation.category
            if candidate.state.value == "scored":
                assert option.score == explanation.combined_rate
                assert option.descriptive_uncertainty_envelope == (
                    explanation.combined_lower,
                    explanation.combined_upper,
                )
