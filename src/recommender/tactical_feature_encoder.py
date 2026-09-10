"""Codificacion sintetica y determinista de senales tacticas comunes.

Convierte un ``AttemptSignalExtraction`` ya validado en features binarias con
esquema fijo. No ejecuta parser, extractores ni adaptadores; tampoco hace E/S,
ajusta parametros o calcula scores y recomendaciones.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import re
from typing import Final

from src.recommender.tactical_signal_orchestrator import (
    AttemptSignalExtraction,
    validate_attempt_signal_extraction,
)


FEATURE_CONTRACT_VERSION: Final = "1.0.0"
PATTERN_ORDER: Final = ("P02", "P04", "P05", "P06", "P09")
P06_CODE_ORDER: Final = tuple("fbrsvzopuylmhijkt")
P09_PROFILE_ORDER: Final = tuple(
    sorted(
        f"{shot}|{direction}|{depth}"
        for shot in P06_CODE_ORDER
        for direction in ("1", "2", "3")
        for depth in ("7", "8", "9")
    )
)

P02_FEATURE_NAMES: Final = tuple(
    f"p02.first_serve_direction.{code}" for code in ("4", "5", "6")
)
P04_FEATURE_NAMES: Final = tuple(
    f"p04.return_direction.{code}" for code in ("1", "2", "3")
)
P05_FEATURE_NAMES: Final = tuple(
    f"p05.return_depth.{code}" for code in ("7", "8", "9")
)
P06_FEATURE_NAMES: Final = tuple(
    f"p06.return_shot_type.{code}" for code in P06_CODE_ORDER
)
P09_FEATURE_NAMES: Final = tuple(
    f"p09.return_profile.{profile_id}" for profile_id in P09_PROFILE_ORDER
)
MASK_FEATURE_NAMES: Final = tuple(
    feature
    for pattern in ("p02", "p04", "p05", "p06", "p09")
    for feature in (f"mask.{pattern}.applicable", f"mask.{pattern}.observed")
)
CONTEXT_FEATURE_NAMES: Final = (
    "context.serve_number.1",
    "context.serve_number.2",
    "context.previous_attempt_was_fault",
)

SCHEMA_FINGERPRINT_DOMAIN: Final = b"tennis-tactical-feature-schema\x00"
VECTOR_FINGERPRINT_DOMAIN: Final = b"tennis-tactical-feature-vector\x00"
BATCH_FINGERPRINT_DOMAIN: Final = b"tennis-tactical-feature-batch\x00"

_P02_VALUE_TO_CODE: Final = (("wide", "4"), ("body", "5"), ("T", "6"))
_VECTOR_DIAGNOSTIC_ORDER: Final = (
    "active_tactical_feature_count",
    "active_mask_count",
    "active_context_count",
    "policy_excluded_observed_signal_count",
    "redundant_representation",
)
_BATCH_DIAGNOSTIC_ORDER: Final = (
    "row_count",
    "column_count",
    "total_active_feature_count",
    "total_observed_signal_count",
    "redundant_representation",
)
_FORBIDDEN_KEYS: Final = frozenset(
    {
        "sequence_text",
        "raw_sequence",
        "parse_result",
        "tokens",
        "spans",
        "warnings",
        "residuals",
        "residual_text",
        "match_id",
        "point_number",
        "player",
        "players",
        "server",
        "point_winner",
        "returner_won_point",
        "date",
        "timestamp",
        "path",
    }
)
_FORBIDDEN_TERMS: Final = ("evaluation", "scoring", "recommendation")
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_SHA256 = re.compile(r"^[0-9A-F]{64}$")


class TacticalFeatureContractError(ValueError):
    """Error cerrado del contrato de features tacticas."""


class TacticalEncodingPolicy(str, Enum):
    COMPONENT_ONLY = "component_only"
    PROFILE_ONLY = "profile_only"
    COMPONENTS_AND_PROFILE = "components_and_profile"


class TacticalFeatureAvailability(str, Enum):
    AVAILABLE = "tactical_features_available"
    NOT_OBSERVED = "no_tactical_feature_observed"


@dataclass(frozen=True)
class TacticalFeatureSchema:
    contract_version: str
    policy: TacticalEncodingPolicy
    feature_names: tuple[str, ...]
    tactical_feature_names: tuple[str, ...]
    mask_feature_names: tuple[str, ...]
    context_feature_names: tuple[str, ...]
    name_to_index: tuple[tuple[str, int], ...]
    feature_count: int
    pattern_catalog: tuple[str, ...]
    encoded_pattern_order: tuple[str, ...]
    redundant_representation: bool
    contractual_purpose: str

    def __post_init__(self) -> None:
        validate_tactical_feature_schema(self)


@dataclass(frozen=True)
class TacticalFeatureVector:
    contract_version: str
    schema_fingerprint: str
    policy: TacticalEncodingPolicy
    values: tuple[int, ...]
    serve_number: int
    active_feature_count: int
    active_feature_names: tuple[str, ...]
    observed_signal_count: int
    applicable_pattern_count: int
    abstention_count: int
    availability_state: TacticalFeatureAvailability
    diagnostics: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        validate_tactical_feature_vector(self)


@dataclass(frozen=True)
class TacticalFeatureBatch:
    contract_version: str
    schema_fingerprint: str
    policy: TacticalEncodingPolicy
    vectors: tuple[TacticalFeatureVector, ...]
    row_count: int
    column_count: int
    diagnostics: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        validate_tactical_feature_batch(self)


def _require_policy(policy: object) -> TacticalEncodingPolicy:
    if type(policy) is not TacticalEncodingPolicy:
        raise TypeError("policy debe ser TacticalEncodingPolicy exacta.")
    return policy


def _policy_contract(
    policy: TacticalEncodingPolicy,
) -> tuple[tuple[str, ...], tuple[str, ...], bool, str]:
    if policy is TacticalEncodingPolicy.COMPONENT_ONLY:
        return (
            P02_FEATURE_NAMES + P04_FEATURE_NAMES + P05_FEATURE_NAMES + P06_FEATURE_NAMES,
            ("P02", "P04", "P05", "P06"),
            False,
            "component_features_for_future_models",
        )
    if policy is TacticalEncodingPolicy.PROFILE_ONLY:
        return (
            P02_FEATURE_NAMES + P09_FEATURE_NAMES,
            ("P02", "P09"),
            False,
            "profile_features_for_future_models",
        )
    return (
        P02_FEATURE_NAMES
        + P04_FEATURE_NAMES
        + P05_FEATURE_NAMES
        + P06_FEATURE_NAMES
        + P09_FEATURE_NAMES,
        PATTERN_ORDER,
        True,
        "audit_or_interaction_research_only",
    )


def _expected_schema(
    policy: TacticalEncodingPolicy,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[str, int], ...],
    tuple[str, ...],
    bool,
    str,
]:
    tactical_names, encoded_patterns, redundant, purpose = _policy_contract(policy)
    names = tactical_names + MASK_FEATURE_NAMES + CONTEXT_FEATURE_NAMES
    mapping = tuple((name, index) for index, name in enumerate(names))
    return names, tactical_names, mapping, encoded_patterns, redundant, purpose


def build_tactical_feature_schema(
    policy: TacticalEncodingPolicy = TacticalEncodingPolicy.COMPONENT_ONLY,
) -> TacticalFeatureSchema:
    """Construye un esquema fijo exclusivamente desde el contrato versionado."""
    selected = _require_policy(policy)
    names, tactical, mapping, encoded, redundant, purpose = _expected_schema(selected)
    return TacticalFeatureSchema(
        contract_version=FEATURE_CONTRACT_VERSION,
        policy=selected,
        feature_names=names,
        tactical_feature_names=tactical,
        mask_feature_names=MASK_FEATURE_NAMES,
        context_feature_names=CONTEXT_FEATURE_NAMES,
        name_to_index=mapping,
        feature_count=len(names),
        pattern_catalog=PATTERN_ORDER,
        encoded_pattern_order=encoded,
        redundant_representation=redundant,
        contractual_purpose=purpose,
    )


def validate_tactical_feature_schema(schema: TacticalFeatureSchema) -> None:
    if type(schema) is not TacticalFeatureSchema:
        raise TypeError("schema debe ser TacticalFeatureSchema exacto.")
    if schema.contract_version != FEATURE_CONTRACT_VERSION:
        raise TacticalFeatureContractError("contract_version del schema invalida.")
    policy = _require_policy(schema.policy)
    names, tactical, mapping, encoded, redundant, purpose = _expected_schema(policy)
    expected = {
        "feature_names": names,
        "tactical_feature_names": tactical,
        "mask_feature_names": MASK_FEATURE_NAMES,
        "context_feature_names": CONTEXT_FEATURE_NAMES,
        "name_to_index": mapping,
        "feature_count": len(names),
        "pattern_catalog": PATTERN_ORDER,
        "encoded_pattern_order": encoded,
        "redundant_representation": redundant,
        "contractual_purpose": purpose,
    }
    for field_name, expected_value in expected.items():
        if getattr(schema, field_name) != expected_value:
            raise TacticalFeatureContractError(
                f"{field_name} no coincide con el esquema canonico."
            )
    for name in (
        "feature_names",
        "tactical_feature_names",
        "mask_feature_names",
        "context_feature_names",
        "name_to_index",
        "pattern_catalog",
        "encoded_pattern_order",
    ):
        if type(getattr(schema, name)) is not tuple:
            raise TacticalFeatureContractError(f"{name} debe ser tuple inmutable.")
    if type(schema.feature_count) is not int:
        raise TacticalFeatureContractError("feature_count debe ser int real.")
    if type(schema.redundant_representation) is not bool:
        raise TacticalFeatureContractError(
            "redundant_representation debe ser bool real."
        )
    if len(set(schema.feature_names)) != len(schema.feature_names):
        raise TacticalFeatureContractError("Los nombres de feature deben ser unicos.")
    if any(".0" in name or ".q" in name for name in schema.tactical_feature_names):
        raise TacticalFeatureContractError(
            "Unknown no puede convertirse en categoria tactica."
        )


def _schema_index(schema: TacticalFeatureSchema) -> dict[str, int]:
    return {name: index for name, index in schema.name_to_index}


def _feature_name(pattern_id: str, tactical_value: str) -> str:
    if pattern_id == "P02":
        mapping = dict(_P02_VALUE_TO_CODE)
        if tactical_value not in mapping:
            raise TacticalFeatureContractError("Valor P02 no codificable.")
        return f"p02.first_serve_direction.{mapping[tactical_value]}"
    if pattern_id == "P04":
        return f"p04.return_direction.{tactical_value}"
    if pattern_id == "P05":
        return f"p05.return_depth.{tactical_value}"
    if pattern_id == "P06":
        return f"p06.return_shot_type.{tactical_value}"
    if pattern_id == "P09":
        return f"p09.return_profile.{tactical_value}"
    raise TacticalFeatureContractError("Patron no codificable.")


def _mask_name(pattern_id: str, suffix: str) -> str:
    return f"mask.{pattern_id.lower()}.{suffix}"


def _expected_vector_diagnostics(
    *,
    tactical_count: int,
    mask_count: int,
    context_count: int,
    excluded_observed_count: int,
    redundant: bool,
) -> tuple[tuple[str, int], ...]:
    return (
        ("active_tactical_feature_count", tactical_count),
        ("active_mask_count", mask_count),
        ("active_context_count", context_count),
        ("policy_excluded_observed_signal_count", excluded_observed_count),
        ("redundant_representation", int(redundant)),
    )


def encode_tactical_attempt(
    extraction: AttemptSignalExtraction,
    schema: TacticalFeatureSchema,
) -> TacticalFeatureVector:
    """Codifica una extraccion ya validada sin reparar ni recalcular senales."""
    if type(extraction) is not AttemptSignalExtraction:
        raise TypeError("extraction debe ser AttemptSignalExtraction exacta.")
    if type(schema) is not TacticalFeatureSchema:
        raise TypeError("schema debe ser TacticalFeatureSchema exacto.")
    validate_attempt_signal_extraction(extraction)
    validate_tactical_feature_schema(schema)
    index = _schema_index(schema)
    values = [0] * schema.feature_count
    adaptations = {
        result.requested_pattern_id: result for result in extraction.adaptations
    }
    encoded_patterns = frozenset(schema.encoded_pattern_order)

    for pattern_id in PATTERN_ORDER:
        applicable = pattern_id in extraction.applicable_patterns
        result = adaptations.get(pattern_id)
        observed = result is not None and result.signal is not None
        values[index[_mask_name(pattern_id, "applicable")]] = int(applicable)
        values[index[_mask_name(pattern_id, "observed")]] = int(observed)
        if observed and pattern_id in encoded_patterns:
            feature_name = _feature_name(pattern_id, result.signal.tactical_value)
            if feature_name not in index:
                raise TacticalFeatureContractError(
                    "La senal observada no pertenece al catalogo del schema."
                )
            values[index[feature_name]] = 1

    values[index[f"context.serve_number.{extraction.serve_number}"]] = 1
    values[index["context.previous_attempt_was_fault"]] = int(
        extraction.previous_attempt_was_fault
    )
    value_tuple = tuple(values)
    active_names = tuple(
        name for name, value in zip(schema.feature_names, value_tuple) if value == 1
    )
    tactical_count = sum(
        value_tuple[index[name]] for name in schema.tactical_feature_names
    )
    mask_count = sum(value_tuple[index[name]] for name in schema.mask_feature_names)
    context_count = sum(
        value_tuple[index[name]] for name in schema.context_feature_names
    )
    excluded_observed = sum(
        adaptations[pattern_id].signal is not None
        for pattern_id in extraction.applicable_patterns
        if pattern_id not in encoded_patterns
    )
    state = (
        TacticalFeatureAvailability.AVAILABLE
        if tactical_count
        else TacticalFeatureAvailability.NOT_OBSERVED
    )
    return TacticalFeatureVector(
        contract_version=FEATURE_CONTRACT_VERSION,
        schema_fingerprint=feature_schema_fingerprint(schema),
        policy=schema.policy,
        values=value_tuple,
        serve_number=extraction.serve_number,
        active_feature_count=sum(value_tuple),
        active_feature_names=active_names,
        observed_signal_count=extraction.active_signal_count,
        applicable_pattern_count=len(extraction.applicable_patterns),
        abstention_count=extraction.abstention_count,
        availability_state=state,
        diagnostics=_expected_vector_diagnostics(
            tactical_count=tactical_count,
            mask_count=mask_count,
            context_count=context_count,
            excluded_observed_count=excluded_observed,
            redundant=schema.redundant_representation,
        ),
    )


def _feature_blocks() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        ("P02", P02_FEATURE_NAMES),
        ("P04", P04_FEATURE_NAMES),
        ("P05", P05_FEATURE_NAMES),
        ("P06", P06_FEATURE_NAMES),
        ("P09", P09_FEATURE_NAMES),
    )


def validate_tactical_feature_vector(vector: TacticalFeatureVector) -> None:
    """Reconstruye schema, mascaras, conteos y coherencia semantica del vector."""
    if type(vector) is not TacticalFeatureVector:
        raise TypeError("vector debe ser TacticalFeatureVector exacto.")
    if vector.contract_version != FEATURE_CONTRACT_VERSION:
        raise TacticalFeatureContractError("contract_version del vector invalida.")
    policy = _require_policy(vector.policy)
    schema = build_tactical_feature_schema(policy)
    if (
        type(vector.schema_fingerprint) is not str
        or _SHA256.fullmatch(vector.schema_fingerprint) is None
        or vector.schema_fingerprint != feature_schema_fingerprint(schema)
    ):
        raise TacticalFeatureContractError("schema_fingerprint no reconcilia.")
    if type(vector.values) is not tuple or len(vector.values) != schema.feature_count:
        raise TacticalFeatureContractError("Longitud o tipo de values invalido.")
    if any(type(value) is not int or value not in (0, 1) for value in vector.values):
        raise TacticalFeatureContractError("values solo admite enteros reales 0/1.")
    if type(vector.serve_number) is not int or vector.serve_number not in (1, 2):
        raise TacticalFeatureContractError("serve_number debe ser int real 1/2.")
    index = _schema_index(schema)
    serve_one = vector.values[index["context.serve_number.1"]]
    serve_two = vector.values[index["context.serve_number.2"]]
    previous_fault = vector.values[index["context.previous_attempt_was_fault"]]
    if serve_one + serve_two != 1 or (serve_one, serve_two) != (
        int(vector.serve_number == 1),
        int(vector.serve_number == 2),
    ):
        raise TacticalFeatureContractError("One-hot de serve_number invalido.")
    if vector.serve_number == 1 and previous_fault != 0:
        raise TacticalFeatureContractError(
            "Un primer saque no puede activar previous_attempt_was_fault."
        )

    encoded = frozenset(schema.encoded_pattern_order)
    observed_count = 0
    applicable_count = 0
    tactical_count = 0
    excluded_observed = 0
    active_by_pattern: dict[str, str | None] = {}
    for pattern_id, names in _feature_blocks():
        applicable = vector.values[index[_mask_name(pattern_id, "applicable")]]
        observed = vector.values[index[_mask_name(pattern_id, "observed")]]
        expected_applicable = int(pattern_id != "P02" or vector.serve_number == 1)
        if applicable != expected_applicable:
            raise TacticalFeatureContractError(
                f"Mascara applicable incompatible para {pattern_id}."
            )
        if observed > applicable:
            raise TacticalFeatureContractError(
                f"Mascara observed exige aplicabilidad para {pattern_id}."
            )
        block_sum = sum(vector.values[index[name]] for name in names if name in index)
        if block_sum > 1:
            raise TacticalFeatureContractError(
                f"Mas de una categoria activa para {pattern_id}."
            )
        expected_block_sum = observed if pattern_id in encoded else 0
        if block_sum != expected_block_sum:
            raise TacticalFeatureContractError(
                f"Categoria tactica y mascara no reconcilian para {pattern_id}."
            )
        active_by_pattern[pattern_id] = next(
            (name for name in names if name in index and vector.values[index[name]]),
            None,
        )
        observed_count += observed
        applicable_count += applicable
        tactical_count += block_sum
        if observed and pattern_id not in encoded:
            excluded_observed += 1

    if policy is TacticalEncodingPolicy.COMPONENTS_AND_PROFILE and (
        active_by_pattern["P09"] is not None
    ):
        profile = active_by_pattern["P09"].removeprefix("p09.return_profile.")
        shot, direction, depth = profile.split("|")
        expected_components = {
            "P04": f"p04.return_direction.{direction}",
            "P05": f"p05.return_depth.{depth}",
            "P06": f"p06.return_shot_type.{shot}",
        }
        if any(
            active_by_pattern[pattern_id] != expected_name
            for pattern_id, expected_name in expected_components.items()
        ):
            raise TacticalFeatureContractError(
                "P09 no reconcilia con P04/P05/P06 activos."
            )

    expected_active_names = tuple(
        name for name, value in zip(schema.feature_names, vector.values) if value == 1
    )
    if type(vector.active_feature_names) is not tuple or (
        vector.active_feature_names != expected_active_names
    ):
        raise TacticalFeatureContractError("active_feature_names no reconcilia.")
    if type(vector.active_feature_count) is not int or (
        vector.active_feature_count != sum(vector.values)
    ):
        raise TacticalFeatureContractError("active_feature_count no reconcilia.")
    expected_abstentions = applicable_count - observed_count
    for field_name, actual, expected in (
        ("observed_signal_count", vector.observed_signal_count, observed_count),
        ("applicable_pattern_count", vector.applicable_pattern_count, applicable_count),
        ("abstention_count", vector.abstention_count, expected_abstentions),
    ):
        if type(actual) is not int or actual != expected:
            raise TacticalFeatureContractError(f"{field_name} no reconcilia.")
    expected_state = (
        TacticalFeatureAvailability.AVAILABLE
        if tactical_count
        else TacticalFeatureAvailability.NOT_OBSERVED
    )
    if type(vector.availability_state) is not TacticalFeatureAvailability or (
        vector.availability_state is not expected_state
    ):
        raise TacticalFeatureContractError("availability_state no reconcilia.")
    mask_count = sum(vector.values[index[name]] for name in schema.mask_feature_names)
    context_count = sum(
        vector.values[index[name]] for name in schema.context_feature_names
    )
    expected_diagnostics = _expected_vector_diagnostics(
        tactical_count=tactical_count,
        mask_count=mask_count,
        context_count=context_count,
        excluded_observed_count=excluded_observed,
        redundant=schema.redundant_representation,
    )
    _validate_diagnostics(
        vector.diagnostics,
        expected_diagnostics,
        _VECTOR_DIAGNOSTIC_ORDER,
        "vector",
    )


def _validate_diagnostics(
    diagnostics: object,
    expected: tuple[tuple[str, int], ...],
    ordered_keys: tuple[str, ...],
    owner: str,
) -> None:
    if type(diagnostics) is not tuple or diagnostics != expected:
        raise TacticalFeatureContractError(
            f"Diagnosticos de {owner} no reconciliados."
        )
    if tuple(item[0] for item in diagnostics if type(item) is tuple and len(item) == 2) != ordered_keys:
        raise TacticalFeatureContractError(
            f"Orden diagnostico de {owner} invalido."
        )
    for item in diagnostics:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not int
            or item[1] < 0
        ):
            raise TacticalFeatureContractError(
                f"Diagnosticos de {owner} deben ser conteos inmutables."
            )


def _expected_batch_diagnostics(
    vectors: tuple[TacticalFeatureVector, ...], schema: TacticalFeatureSchema
) -> tuple[tuple[str, int], ...]:
    return (
        ("row_count", len(vectors)),
        ("column_count", schema.feature_count),
        (
            "total_active_feature_count",
            sum(vector.active_feature_count for vector in vectors),
        ),
        (
            "total_observed_signal_count",
            sum(vector.observed_signal_count for vector in vectors),
        ),
        ("redundant_representation", int(schema.redundant_representation)),
    )


def encode_tactical_batch(
    vectors: tuple[TacticalFeatureVector, ...],
    schema: TacticalFeatureSchema,
) -> TacticalFeatureBatch:
    """Agrupa una tupla no vacia de vectores y conserva estrictamente su orden."""
    if type(vectors) is not tuple:
        raise TypeError("vectors debe ser tuple inmutable.")
    if not vectors:
        raise TacticalFeatureContractError("Un batch no puede estar vacio.")
    if type(schema) is not TacticalFeatureSchema:
        raise TypeError("schema debe ser TacticalFeatureSchema exacto.")
    validate_tactical_feature_schema(schema)
    for vector in vectors:
        if type(vector) is not TacticalFeatureVector:
            raise TypeError("Cada fila debe ser TacticalFeatureVector exacto.")
        validate_tactical_feature_vector(vector)
        if (
            vector.schema_fingerprint != feature_schema_fingerprint(schema)
            or vector.policy is not schema.policy
        ):
            raise TacticalFeatureContractError(
                "Todas las filas deben compartir schema y policy."
            )
    return TacticalFeatureBatch(
        contract_version=FEATURE_CONTRACT_VERSION,
        schema_fingerprint=feature_schema_fingerprint(schema),
        policy=schema.policy,
        vectors=vectors,
        row_count=len(vectors),
        column_count=schema.feature_count,
        diagnostics=_expected_batch_diagnostics(vectors, schema),
    )


def validate_tactical_feature_batch(batch: TacticalFeatureBatch) -> None:
    if type(batch) is not TacticalFeatureBatch:
        raise TypeError("batch debe ser TacticalFeatureBatch exacto.")
    if batch.contract_version != FEATURE_CONTRACT_VERSION:
        raise TacticalFeatureContractError("contract_version del batch invalida.")
    policy = _require_policy(batch.policy)
    schema = build_tactical_feature_schema(policy)
    expected_fingerprint = feature_schema_fingerprint(schema)
    if batch.schema_fingerprint != expected_fingerprint:
        raise TacticalFeatureContractError("schema_fingerprint del batch no reconcilia.")
    if type(batch.vectors) is not tuple or not batch.vectors:
        raise TacticalFeatureContractError("vectors debe ser tuple no vacia.")
    for vector in batch.vectors:
        if type(vector) is not TacticalFeatureVector:
            raise TypeError("Cada fila debe ser TacticalFeatureVector exacto.")
        validate_tactical_feature_vector(vector)
        if vector.schema_fingerprint != expected_fingerprint or vector.policy is not policy:
            raise TacticalFeatureContractError(
                "Todas las filas deben compartir schema y policy."
            )
    if type(batch.row_count) is not int or batch.row_count != len(batch.vectors):
        raise TacticalFeatureContractError("row_count no reconcilia.")
    if type(batch.column_count) is not int or batch.column_count != schema.feature_count:
        raise TacticalFeatureContractError("column_count no reconcilia.")
    _validate_diagnostics(
        batch.diagnostics,
        _expected_batch_diagnostics(batch.vectors, schema),
        _BATCH_DIAGNOSTIC_ORDER,
        "batch",
    )


def _schema_structure(schema: TacticalFeatureSchema) -> dict[str, object]:
    validate_tactical_feature_schema(schema)
    return {
        "contract_version": schema.contract_version,
        "contractual_purpose": schema.contractual_purpose,
        "context_feature_names": list(schema.context_feature_names),
        "encoded_pattern_order": list(schema.encoded_pattern_order),
        "feature_count": schema.feature_count,
        "feature_names": list(schema.feature_names),
        "mask_feature_names": list(schema.mask_feature_names),
        "name_to_index": [
            {"index": index, "name": name} for name, index in schema.name_to_index
        ],
        "pattern_catalog": list(schema.pattern_catalog),
        "policy": schema.policy.value,
        "redundant_representation": schema.redundant_representation,
        "tactical_feature_names": list(schema.tactical_feature_names),
    }


def _vector_structure(vector: TacticalFeatureVector) -> dict[str, object]:
    validate_tactical_feature_vector(vector)
    return {
        "abstention_count": vector.abstention_count,
        "active_feature_count": vector.active_feature_count,
        "active_feature_names": list(vector.active_feature_names),
        "applicable_pattern_count": vector.applicable_pattern_count,
        "availability_state": vector.availability_state.value,
        "contract_version": vector.contract_version,
        "diagnostics": {key: value for key, value in vector.diagnostics},
        "observed_signal_count": vector.observed_signal_count,
        "policy": vector.policy.value,
        "schema_fingerprint": vector.schema_fingerprint,
        "serve_number": vector.serve_number,
        "values": list(vector.values),
    }


def _batch_structure(batch: TacticalFeatureBatch) -> dict[str, object]:
    validate_tactical_feature_batch(batch)
    return {
        "column_count": batch.column_count,
        "contract_version": batch.contract_version,
        "diagnostics": {key: value for key, value in batch.diagnostics},
        "policy": batch.policy.value,
        "row_count": batch.row_count,
        "schema_fingerprint": batch.schema_fingerprint,
        "vectors": [_vector_structure(vector) for vector in batch.vectors],
    }


def _validate_public_tree(value: object, *, key: str | None = None) -> None:
    if key is not None:
        lowered = key.lower()
        if (
            lowered in _FORBIDDEN_KEYS
            or lowered.startswith("test_")
            or any(term in lowered for term in _FORBIDDEN_TERMS)
        ):
            raise TacticalFeatureContractError(
                "La serializacion contiene una clave no autorizada."
            )
    if value is None or type(value) in (bool, int):
        return
    if type(value) is str:
        lowered_value = value.lower()
        if (
            "file://" in lowered_value
            or _WINDOWS_PATH.match(value)
            or value.startswith(("/", "\\\\"))
            or "../" in value
            or "..\\" in value
            or any(ord(character) < 32 for character in value)
        ):
            raise TacticalFeatureContractError(
                "La serializacion contiene texto operativo no autorizado."
            )
        return
    if type(value) is list:
        for item in value:
            _validate_public_tree(item)
        return
    if type(value) is dict:
        for child_key, child_value in value.items():
            if type(child_key) is not str:
                raise TacticalFeatureContractError("Las claves JSON deben ser strings.")
            _validate_public_tree(child_value, key=child_key)
        return
    raise TacticalFeatureContractError(
        "La serializacion contiene un objeto no contractual."
    )


def _canonical_bytes(structure: dict[str, object]) -> bytes:
    _validate_public_tree(structure)
    return json.dumps(
        structure,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_feature_schema_json(schema: TacticalFeatureSchema) -> bytes:
    return _canonical_bytes(_schema_structure(schema))


def canonical_feature_vector_json(vector: TacticalFeatureVector) -> bytes:
    return _canonical_bytes(_vector_structure(vector))


def canonical_feature_batch_json(batch: TacticalFeatureBatch) -> bytes:
    return _canonical_bytes(_batch_structure(batch))


def feature_schema_fingerprint(schema: TacticalFeatureSchema) -> str:
    return sha256(
        SCHEMA_FINGERPRINT_DOMAIN + canonical_feature_schema_json(schema)
    ).hexdigest().upper()


def feature_vector_fingerprint(vector: TacticalFeatureVector) -> str:
    return sha256(
        VECTOR_FINGERPRINT_DOMAIN + canonical_feature_vector_json(vector)
    ).hexdigest().upper()


def feature_batch_fingerprint(batch: TacticalFeatureBatch) -> str:
    return sha256(
        BATCH_FINGERPRINT_DOMAIN + canonical_feature_batch_json(batch)
    ).hexdigest().upper()


__all__ = [
    "BATCH_FINGERPRINT_DOMAIN",
    "CONTEXT_FEATURE_NAMES",
    "FEATURE_CONTRACT_VERSION",
    "MASK_FEATURE_NAMES",
    "P02_FEATURE_NAMES",
    "P04_FEATURE_NAMES",
    "P05_FEATURE_NAMES",
    "P06_CODE_ORDER",
    "P06_FEATURE_NAMES",
    "P09_FEATURE_NAMES",
    "P09_PROFILE_ORDER",
    "PATTERN_ORDER",
    "SCHEMA_FINGERPRINT_DOMAIN",
    "VECTOR_FINGERPRINT_DOMAIN",
    "TacticalEncodingPolicy",
    "TacticalFeatureAvailability",
    "TacticalFeatureBatch",
    "TacticalFeatureContractError",
    "TacticalFeatureSchema",
    "TacticalFeatureVector",
    "build_tactical_feature_schema",
    "canonical_feature_batch_json",
    "canonical_feature_schema_json",
    "canonical_feature_vector_json",
    "encode_tactical_attempt",
    "encode_tactical_batch",
    "feature_batch_fingerprint",
    "feature_schema_fingerprint",
    "feature_vector_fingerprint",
    "validate_tactical_feature_batch",
    "validate_tactical_feature_schema",
    "validate_tactical_feature_vector",
]
