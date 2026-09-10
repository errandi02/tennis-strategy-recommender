"""Registro contractual artifact-only de los patrones tacticos P02--P09.

El modulo usa exclusivamente JSON y CSV agregados versionados. No conoce la
fuente de datos analitica ni importa los modulos que produjeron esos artefactos.
La publicacion usa rollback ante fallos ordinarios; como cualquier reemplazo de
varios archivos, no puede garantizar atomicidad frente a un apagado abrupto
entre los dos ``os.replace``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path, PurePath
import re
import subprocess
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "reports" / "tactical_pattern_registry_summary.json"
REGISTRY_CSV_PATH = ROOT / "reports" / "tables" / "tactical_pattern_registry.csv"
EXPECTED_REPOSITORY_COMMIT = "cf9395941c2cd50599bbe054cae23fbd03fed981"
REGISTRY_CONTRACT_VERSION = "1.0.0"
FINGERPRINT_CONTRACT_VERSION = "1"

PATTERN_IDS = tuple(f"P{number:02d}" for number in range(2, 10))
ANALYSIS_STATUSES = frozenset(
    {"available_descriptive", "available_descriptive_not_comparable", "not_available"}
)
EVIDENCE_LEVELS = frozenset(
    {
        "descriptive_comparable",
        "descriptive_not_comparable",
        "descriptive_conditioned",
        "operational_policy",
        "not_available",
    }
)
READINESS_LEVELS = frozenset(
    {
        "eligible_component",
        "diagnostic_only",
        "blocked_no_comparator",
        "blocked_low_identifiability",
        "not_available",
    }
)
ACTORS = frozenset(
    {"server", "returner", "server_and_opponent", "returner_profile", "mixed_documented_context"}
)
UNITS = frozenset({"point", "substantive_serve_attempt", "target_orientation_direction", "initial_return_event"})
REGISTRY_STATUSES = frozenset({"available_registry", "not_available"})
FAILURE_STAGES = frozenset(
    {
        "git_preflight",
        "upstream_discovery",
        "upstream_hash_validation",
        "upstream_semantic_validation",
        "registry_construction",
        "registry_validation",
        "serialization",
        "publication_verification",
    }
)
REASON_CODES = frozenset(
    {
        "git_preflight_failed",
        "upstream_missing",
        "upstream_hash_mismatch",
        "upstream_contract_invalid",
        "registry_contract_invalid",
        "serialization_failed",
        "publication_failed",
    }
)

CSV_COLUMNS = (
    "pattern_id",
    "pattern_key",
    "display_name",
    "source_commit",
    "source_summary_path",
    "source_summary_bytes",
    "source_summary_sha256",
    "source_publication_fingerprint",
    "analysis_status",
    "evidence_level",
    "recommendation_readiness",
    "unit",
    "actor",
    "population_scope",
    "eligible_count",
    "denominator_count",
    "coverage",
    "has_observable_comparator",
    "outcomes_available",
    "test_status",
    "test_used_for_method_selection",
    "excluded_test_matches",
    "primary_contract",
    "principal_limitation",
)
INTEGER_COLUMNS = frozenset(
    {"source_summary_bytes", "eligible_count", "denominator_count", "excluded_test_matches"}
)
FLOAT_COLUMNS = frozenset({"coverage"})
BOOLEAN_COLUMNS = frozenset(
    {"has_observable_comparator", "outcomes_available", "test_used_for_method_selection"}
)


class RegistryContractError(RuntimeError):
    """Fallo cerrado del contrato del registro."""

    def __init__(
        self,
        message: str,
        *,
        stage: str = "registry_validation",
        reason_code: str = "registry_contract_invalid",
        pattern_id: str | None = None,
        artifact_path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.reason_code = reason_code
        self.pattern_id = pattern_id
        self.artifact_path = artifact_path


@dataclass(frozen=True)
class ArtifactSpec:
    path: str
    size: int
    sha256: str
    payload_key: str


@dataclass(frozen=True)
class SourceSpec:
    pattern_id: str
    pattern_key: str
    upstream_pattern_key: str
    display_name: str
    summary_path: str
    summary_size: int
    summary_sha256: str
    publication_fingerprint: str
    source_commit: str
    analysis_status: str
    evidence_level: str
    unit: str
    actor: str
    eligible_path: tuple[str, ...]
    denominator_path: tuple[str, ...]
    coverage_path: tuple[str, ...]
    state_path: tuple[str, ...] | None
    aggregate_artifact: str
    aggregate_filters: tuple[tuple[str, str], ...]
    aggregate_eligible_column: str
    aggregate_denominator_column: str | None
    aggregate_coverage_column: str | None
    comparable: bool
    diagnostic_only: bool
    sufficient_identifiability: bool
    usable_as_explainable_input: bool
    outcomes_available: bool
    outcome_artifact: str | None
    comparator: str
    primary_contract: str
    principal_limitation: str
    artifacts: tuple[ArtifactSpec, ...]


@dataclass(frozen=True)
class VerifiedSource:
    spec: SourceSpec
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class RegistryResult:
    summary: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    sources: tuple[VerifiedSource, ...] = ()


def _artifact(path: str, size: int, sha256: str, payload_key: str) -> ArtifactSpec:
    return ArtifactSpec(path, size, sha256, payload_key)


SOURCE_SPECS = (
    SourceSpec(
        "P02", "first_serve_direction", "first_serve_direction", "Direccion documentada del primer saque",
        "reports/first_serve_direction_feasibility_summary.json", 24698,
        "6BC1498AE7C65E079428F90ACFD53665ECD815459600ABFF658576BC0CEC5F0B",
        "A5820C693AE572654185257350D207019E2630BC3275F5A938F370E11F4A111A",
        "47168feddb374f4478d9b54b0e90eb58dbc69505", "available_descriptive",
        "descriptive_comparable", "point", "server",
        ("population", "directional_points"), ("population", "development_point_rows"),
        (), ("exclusions", "counts", "actionable_direction"),
        "by_direction", (("dimension", "pooled"),), "analytical_points", None, None,
        True, False, True, True, True, "outcomes",
        "wide|body|T",
        "Direccion inicial 4/5/6 del primer saque documentado.",
        "Cobertura retrospectiva; incluye primeros saques que terminan en falta y no demuestra efecto causal.",
        (
            _artifact("reports/tables/first_serve_direction_feasibility_by_direction.csv", 6496, "F6E364002DB9C965CE50819CDE221243B53906FBF9D1CDC4FE293AA75E68A9B8", "by_direction"),
            _artifact("reports/tables/first_serve_direction_feasibility_coverage.csv", 32071, "5D64E331BFDFD5E09B94217175407A7F7B15119680794BDD12F37E9A61692911", "coverage"),
            _artifact("reports/tables/first_serve_direction_feasibility_outcomes.csv", 14115, "5A7F7CDDBA9ADAB5CEA6ED892EAB4368844FF16F9BF855EF11DB6BD8BCE74C05", "outcomes"),
        ),
    ),
    SourceSpec(
        "P03", "documented_serve_and_volley_intent", "literal_plus_immediately_after_service_prefix",
        "Intencion documentada de saque y volea",
        "reports/serve_and_volley_descriptive_feasibility_summary.json", 7504,
        "D2D5965FC5C38CC5B6341D3ADC09A4D7B938ED558FD54747BCE83A312C2917AD",
        "E9D0D50E63A3746866F62526120C417742BBB50F3E32C6A29262103EBF1A94A2",
        "43b90fa0e4dbab7cd20bd55ff2d49d03f01dccfd", "available_descriptive_not_comparable",
        "descriptive_not_comparable", "substantive_serve_attempt", "server",
        ("coverage_summary", "positive_tagged_attempts"), ("population", "attempts"),
        ("coverage_summary", "positive_tagged_rate"), ("state_counts", "positive_tagged"),
        "by_state", (("analysis_state", "positive_tagged"),), "attempts", None, None,
        False, False, True, False, False, "outcomes", "none_observable",
        "Marcador + inmediato como intencion anotada de saque-red.",
        "No existe negativo observable: unknown y censura no pueden convertirse en ausencia de intencion.",
        (
            _artifact("reports/tables/serve_and_volley_descriptive_feasibility_by_group.csv", 7960, "56FD4885DDCFAB063F8B7079CC5582AE5A39C65F582B4DA4780B220ACC652751", "by_group"),
            _artifact("reports/tables/serve_and_volley_descriptive_feasibility_by_state.csv", 1123, "616073DB6C3B574B12C515E16EE85C095E685DAAD4A10FE8F1312E249496E3CA", "by_state"),
            _artifact("reports/tables/serve_and_volley_descriptive_feasibility_outcomes.csv", 148, "6CC7BE864E4D4055CC2755A17C1906C46D309D3DACE8783EE02E9713931C67BC", "outcomes"),
        ),
    ),
    SourceSpec(
        "P04", "documented_lateral_direction_of_initial_return", "documented_lateral_direction_of_initial_return",
        "Direccion lateral documentada del primer resto",
        "reports/return_direction_descriptive_feasibility_summary.json", 6582,
        "433D7CB1801E50A9DD1080E7880B17D98015960D5D5B92816A6E8B942FA0615F",
        "8C1B1C524F3E24DEC9086AAFA36784BDBAD4BAA71447735F772906769906BC1A",
        "1f5c84b8326283e57316089f6a8bf4e4dfbee7e4", "available_descriptive",
        "descriptive_conditioned", "initial_return_event", "returner",
        ("end_to_end_directional_coverage", "numerator"), ("end_to_end_directional_coverage", "denominator"),
        ("end_to_end_directional_coverage", "proportion"), None,
        "by_group", (("dimension", "total"),), "end_to_end_numerator", "end_to_end_denominator", "end_to_end_rate",
        True, False, True, True, True, "outcomes", "lateral_codes_1|2|3",
        "Direccion lateral 1/2/3 del primer resto localizado.",
        "Condicionado a retorno observable; 1/2/3 no significa cruzado, paralelo o lado universal.",
        (
            _artifact("reports/tables/return_direction_descriptive_feasibility_by_state.csv", 2741, "078F7DDCD61FDD0C631A4DD4289AC5CA3076FEF64B7BC853CE78628793ED7B93", "by_state"),
            _artifact("reports/tables/return_direction_descriptive_feasibility_by_direction.csv", 2795, "F1AEA09D8266463D4F769BF484A92F8A4088FDE62B670BF87CD79736C3BCA2B3", "by_direction"),
            _artifact("reports/tables/return_direction_descriptive_feasibility_by_group.csv", 2722, "135D2FB5111B5B4FF65BB290AC292DD09093279BEFC00F5C9BE8E9C3EA6734E4", "by_group"),
            _artifact("reports/tables/return_direction_descriptive_feasibility_outcomes.csv", 4811, "003A1EA3D459C0DD025C5E70E874EC44751AE867592DF35ADB1E20BA66A17A05", "outcomes"),
        ),
    ),
    SourceSpec(
        "P05", "documented_initial_return_depth", "documented_initial_return_depth",
        "Profundidad documentada del primer resto",
        "reports/return_depth_descriptive_feasibility_summary.json", 5041,
        "8DA410F74B226DCEDB3649A3C4FB90B8497244B60BF7C4EE2DBB61A2C0D6B567",
        "96333493F586D2F65183A12F6443ACF7F1DC2D2999B96C58AC6C07B27DD1C06B",
        "f255c411f018294e5b407e81e6e90317760a7c0d", "available_descriptive",
        "descriptive_conditioned", "initial_return_event", "returner",
        ("coverage", "observed_depth_attempts"), ("coverage", "all_attempts"),
        ("coverage", "end_to_end_coverage"), ("state_counts", "return_depth_observed"),
        "by_group", (("group_type", "total"),), "observed_depth_numerator", "observed_depth_denominator", "observed_depth_coverage",
        True, False, True, True, True, "outcomes", "depth_codes_7|8|9",
        "Profundidad documental 7/8/9 del primer resto.",
        "Condicionado a profundidad documentada; observabilidad y censura pueden sesgar las tasas.",
        (
            _artifact("reports/tables/return_depth_descriptive_feasibility_by_state.csv", 4356, "E1850F03424CB1EBF26F25E6B297A94199D53C5AD68A0D6D1AA9D2A205F6ABA5", "by_state"),
            _artifact("reports/tables/return_depth_descriptive_feasibility_by_depth.csv", 1210, "DE07B44C99873FFEEF963CC5D46C4C3E880109748BEBBC972510BAE2507966AD", "by_depth"),
            _artifact("reports/tables/return_depth_descriptive_feasibility_by_group.csv", 1854, "8FF83030223E26AA794BB790FF8301F4CFE76CB2CBB68B719E354014A7560A30", "by_group"),
            _artifact("reports/tables/return_depth_descriptive_feasibility_outcomes.csv", 6397, "62AA53002E99FC7AF11F1C0A7A4BA9DF6D4E81706F57CAC0F3A528A2CF3E6384", "outcomes"),
        ),
    ),
    SourceSpec(
        "P06", "documented_initial_return_shot_type", "documented_initial_return_shot_type",
        "Tipo documentado del primer resto",
        "reports/return_shot_type_descriptive_feasibility_summary.json", 6470,
        "30F8B9E733791BD1DB58710CAC2B55782BB50DD9549804E55E61703A4D90A796",
        "C3F30E44949237F9CABDDD18FE161B242F088504F5E610ED480053015F38FDD7",
        "c66bc78aaff1cbbf9173e746e30a0074622909fe", "available_descriptive",
        "descriptive_conditioned", "initial_return_event", "returner",
        ("coverage", "observed_type_attempts"), ("coverage", "all_attempts"),
        ("coverage", "end_to_end_coverage"), ("state_counts", "return_shot_type_observed"),
        "by_group", (("group_type", "total"),), "observed_type_numerator", "observed_type_denominator", "observed_type_coverage",
        True, False, True, True, True, "outcomes", "documented_shot_type_codes",
        "Codigo literal documentado del primer golpe de resto.",
        "Condicionado a tipo documentado; q y los eventos desconocidos o censurados no son comparadores.",
        (
            _artifact("reports/tables/return_shot_type_descriptive_feasibility_by_state.csv", 7476, "D73A5D8CC4A5E0A60ACD389A18B7F2749631DD9AAEBE7D02E497A79F63960D4E", "by_state"),
            _artifact("reports/tables/return_shot_type_descriptive_feasibility_by_type.csv", 7740, "5DDF4DAAF0326F271EAD49997AAC7A4EF54507A39DB96572DBAF8D541016EBBB", "by_type"),
            _artifact("reports/tables/return_shot_type_descriptive_feasibility_by_group.csv", 1739, "CABBCAB5EB2CE6B34FF50010477370B18FB15B560027D640AC3244567FDF3380", "by_group"),
            _artifact("reports/tables/return_shot_type_descriptive_feasibility_outcomes.csv", 43775, "F8DE44A451852086841BC273E1DA4BE0025A327AF480BBB39C8CDF4B5B3C546B", "outcomes"),
        ),
    ),
    SourceSpec(
        "P07", "documented_immediate_first_return_terminal", "documented_immediate_first_return_terminal",
        "Terminal inmediato documentado del primer resto",
        "reports/return_terminal_descriptive_feasibility_summary.json", 6316,
        "ECD31944D7253A48C468D4C2AC6A8C1004413431F332DA7E66317CB05B34B096",
        "C96BE1AFA442B22DB0FEE51C4264FA5347A3B6DDA2295EF9B1720A58994D27C1",
        "a09b2ba2548c639971ecb5d35a0c370d3e819cee", "available_descriptive",
        "descriptive_conditioned", "initial_return_event", "returner",
        ("terminal_coverage", "eligible_terminal_attempts"), ("terminal_coverage", "all_attempts"),
        ("terminal_coverage", "rate"), None,
        "by_group", (("group_type", "total"),), "eligible_terminal_attempts", "attempts", "terminal_coverage",
        True, True, True, False, True, "consistency", "winner|forced_error|unforced_error",
        "Terminal winner/error que cierra el evento local del primer resto.",
        "La consistencia con el outcome es solo auditoria; no reconstruye el rally ni mide efectividad tactica.",
        (
            _artifact("reports/tables/return_terminal_descriptive_feasibility_by_state.csv", 44616, "1AA13A0A1F6DE890670514E11799A75FC2DDF357E366E5D8397A34428105CE77", "by_state"),
            _artifact("reports/tables/return_terminal_descriptive_feasibility_by_terminal.csv", 2925, "4FA3EEE4B46800AAF24629C8095063416F10C5C9169EE42C8A1AB2B58C1EED13", "by_terminal"),
            _artifact("reports/tables/return_terminal_descriptive_feasibility_by_group.csv", 2853, "AA42694A859E38C7D5A790B10C3F983034F76173ABF3C65399BB8B0B2EC012E2", "by_group"),
            _artifact("reports/tables/return_terminal_descriptive_feasibility_consistency.csv", 18977, "D099EEF9E110DD0D0FC8A568D28AD9B66220D473837B3BE13E9837C01CE18FD5", "consistency"),
        ),
    ),
    SourceSpec(
        "P08", "documented_initial_return_approach", "documented_initial_return_approach",
        "Aproximacion documentada tras el primer resto",
        "reports/return_approach_descriptive_feasibility_summary.json", 2819,
        "004768BCA1B9FF93413F41F641546F0A9F2A54689F72C6951C87EE43F6514DC4",
        "54CA82A1985738260FE529A54CD67C00C02A353CE2931671ABD1738A35107A1E",
        "76a55d9f0e46e5dbdb5cb328d8d6c5de8866eac8", "available_descriptive_not_comparable",
        "descriptive_not_comparable", "initial_return_event", "returner",
        ("coverage", "documented_initial_return_approach_attempts"), ("coverage", "denominator_attempts"),
        ("coverage", "documented_initial_return_approach_prevalence"), ("state_counts", "documented_initial_return_approach"),
        "by_group", (("group_type", "total"),), "positive_attempts", "attempts", "documented_initial_return_approach_prevalence",
        False, False, True, False, False, "outcomes", "none_observable",
        "Marcador + inmediato despues del tipo conocido del primer resto.",
        "No existe negativo observable y la etiqueta no prueba llegada a red ni volea fisica.",
        (
            _artifact("reports/tables/return_approach_descriptive_feasibility_by_state.csv", 18922, "588B70494A9AC78B0A29A6F3392B10D722B485F2F4F39AC4119C160AA52FB41F", "by_state"),
            _artifact("reports/tables/return_approach_descriptive_feasibility_by_group.csv", 1477, "A030BE4844550589A512807D82049788BBF10A856A62C3A5EF086EEB9CD71C48", "by_group"),
            _artifact("reports/tables/return_approach_descriptive_feasibility_marker_context.csv", 512, "CBEA3504763A6EBD1FA1C45668BF65D9C8C23BE763B2D9447DA88075C9CA9900", "marker_context"),
            _artifact("reports/tables/return_approach_descriptive_feasibility_outcomes.csv", 96, "2F0AC9AFABCA2D18AC94C1F0B79B2F43B4A2E9A317CB3B068FA552860D383F38", "outcomes"),
        ),
    ),
    SourceSpec(
        "P09", "documented_initial_return_profile", "documented_initial_return_profile",
        "Perfil compuesto documentado del primer resto",
        "reports/return_profile_descriptive_feasibility_summary.json", 4409,
        "EBD87F31F0483C9B618D930C8BAA2106AAA13A7183103A4B4C855E79FE9AEA93",
        "A84E59A23EF99B5A855CBBC4702EB7EB0C49285BF2BA3A5090E75AC3D356D5E3",
        "cf9395941c2cd50599bbe054cae23fbd03fed981", "available_descriptive",
        "descriptive_conditioned", "initial_return_event", "returner_profile",
        ("coverage", "complete_profile_attempts"), ("coverage", "denominator_attempts"),
        ("coverage", "complete_profile_share_of_all_attempts"), ("state_counts", "documented_initial_return_profile"),
        "by_group", (("aggregation_level", "coverage"), ("group_type", "total")),
        "complete_profiles", "attempts", "complete_profile_coverage",
        True, False, True, True, True, "outcomes", "complete_profiles_type_x_lateral_x_depth",
        "Perfil simultaneo tipo x lateralidad x profundidad del primer resto.",
        "Condicionado a perfil completo; muchos perfiles o estratos son raros o no observados.",
        (
            _artifact("reports/tables/return_profile_descriptive_feasibility_by_state.csv", 7194, "AF3B96B7C09C4F1AB917DA46E2F863A6370040B23E8165483EDB96785AE2BA89", "by_state"),
            _artifact("reports/tables/return_profile_descriptive_feasibility_by_profile.csv", 21955, "2980E0417A7A6FBEF83EFBF2BE68A19023C74001B24DA4AAA382D67035868358", "by_profile"),
            _artifact("reports/tables/return_profile_descriptive_feasibility_by_group.csv", 1867, "B1FADD7AA04E76A205520BDCF9457CDCADEE99C8DFCF691B678C76CBCCA5EA97", "by_group"),
            _artifact("reports/tables/return_profile_descriptive_feasibility_outcomes.csv", 200683, "14780637AAC2162B404B3C641181718FE80DA5411B43686FC591D78D441D9A17", "outcomes"),
        ),
    ),
)

SOURCE_BY_ID = MappingProxyType({spec.pattern_id: spec for spec in SOURCE_SPECS})
SUMMARY_ALLOWLIST = frozenset(spec.summary_path for spec in SOURCE_SPECS)
ARTIFACT_ALLOWLIST = frozenset(artifact.path for spec in SOURCE_SPECS for artifact in spec.artifacts)


def recommendation_readiness(
    *,
    upstream_available: bool,
    explicitly_comparable: bool,
    diagnostic_only: bool,
    sufficient_identifiability: bool,
    usable_as_explainable_input: bool,
) -> str:
    """Decide readiness solo desde propiedades contractuales, sin thresholds nuevos."""
    for name, value in {
        "upstream_available": upstream_available,
        "explicitly_comparable": explicitly_comparable,
        "diagnostic_only": diagnostic_only,
        "sufficient_identifiability": sufficient_identifiability,
        "usable_as_explainable_input": usable_as_explainable_input,
    }.items():
        if type(value) is not bool:
            raise TypeError(f"{name} debe ser bool real.")
    if not upstream_available:
        return "not_available"
    if not explicitly_comparable:
        return "blocked_no_comparator"
    if diagnostic_only:
        return "diagnostic_only"
    if not sufficient_identifiability:
        return "blocked_low_identifiability"
    if usable_as_explainable_input:
        return "eligible_component"
    return "diagnostic_only"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _nested(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = mapping
    for component in path:
        if not isinstance(value, Mapping) or component not in value:
            raise RegistryContractError(
                f"Falta el campo upstream {'.'.join(path)}.",
                stage="upstream_semantic_validation",
                reason_code="upstream_contract_invalid",
            )
        value = value[component]
    return value


def _strict_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise RegistryContractError(f"{label} debe ser entero no negativo.")
    return value


def _strict_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise RegistryContractError(f"{label} debe ser float finito.")
    return value


def _validate_json_values(value: Any) -> None:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise RegistryContractError("JSON contiene una clave no textual.")
        for item in value.values():
            _validate_json_values(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_values(item)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise RegistryContractError("JSON contiene NaN o Infinity.")
    if isinstance(value, Path):
        raise RegistryContractError("JSON contiene un objeto Path.")
    if isinstance(value, str):
        if re.search(r"(?i)(?:^[a-z]:[\\/]|^/users/|^/home/)", value):
            raise RegistryContractError("El registro contiene una ruta absoluta.")
        if re.search(r"(?<![A-Za-z])(?:NaN|Infinity)(?![A-Za-z])", value):
            raise RegistryContractError("El registro contiene un literal no finito.")


def _run_git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=root, check=False, capture_output=True, text=True, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RegistryContractError(
            f"Git no pudo verificar el contrato: {completed.stderr.strip() or 'sin detalle'}",
            stage="git_preflight",
            reason_code="git_preflight_failed",
        )
    return completed.stdout.strip()


def _git_preflight(root: Path, git_runner: Callable[..., str]) -> None:
    for reference in ("HEAD", "origin/main"):
        if git_runner(root, "merge-base", EXPECTED_REPOSITORY_COMMIT, reference) != EXPECTED_REPOSITORY_COMMIT:
            raise RegistryContractError(
                f"{reference} no desciende del commit upstream congelado.",
                stage="git_preflight", reason_code="git_preflight_failed",
            )


def _validate_tracked_clean(
    root: Path, relative_path: str, expected_commit: str, git_runner: Callable[..., str], pattern_id: str
) -> None:
    tracked = git_runner(root, "ls-files", "--error-unmatch", "--", relative_path)
    if tracked.replace("\\", "/") != relative_path:
        raise RegistryContractError(
            "El artefacto upstream no esta rastreado.", stage="upstream_hash_validation",
            reason_code="upstream_hash_mismatch", pattern_id=pattern_id, artifact_path=relative_path,
        )
    if git_runner(root, "status", "--porcelain", "--", relative_path):
        raise RegistryContractError(
            "El artefacto upstream difiere de HEAD.", stage="upstream_hash_validation",
            reason_code="upstream_hash_mismatch", pattern_id=pattern_id, artifact_path=relative_path,
        )
    commit = git_runner(root, "log", "-1", "--format=%H", "--", relative_path)
    if commit != expected_commit:
        raise RegistryContractError(
            "El commit del artefacto upstream no coincide.", stage="upstream_hash_validation",
            reason_code="upstream_hash_mismatch", pattern_id=pattern_id, artifact_path=relative_path,
        )


def _read_csv(payload: bytes, path: str) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("cabecera ausente o duplicada")
        return [dict(row) for row in reader]
    except (UnicodeDecodeError, csv.Error, ValueError) as exc:
        raise RegistryContractError(
            "CSV upstream ilegible.", stage="upstream_semantic_validation",
            reason_code="upstream_contract_invalid", artifact_path=path,
        ) from exc


def _validate_test_seal(summary: Mapping[str, Any], spec: SourceSpec) -> tuple[str, bool, int]:
    seal_name = "chronological_seal" if spec.pattern_id == "P02" else "test_seal"
    seal = _nested(summary, (seal_name,))
    if not isinstance(seal, Mapping) or seal.get("test_status") != "sealed":
        raise RegistryContractError(
            "El test upstream no esta sellado.", stage="upstream_semantic_validation",
            reason_code="upstream_contract_invalid", pattern_id=spec.pattern_id,
        )
    if seal.get("used_for_method_selection") is not False:
        raise RegistryContractError(
            "El upstream uso el test para seleccion metodologica.", stage="upstream_semantic_validation",
            reason_code="upstream_contract_invalid", pattern_id=spec.pattern_id,
        )
    excluded_key = next(
        (key for key in ("excluded_test_matches", "excluded_test_target_matches", "test_target_matches_excluded_before_construction") if key in seal),
        None,
    )
    excluded = seal.get(excluded_key) if excluded_key is not None else _nested(summary, ("population", "excluded_test_matches" if "excluded_test_matches" in summary.get("population", {}) else "excluded_test_target_matches"))
    if excluded != 1531 or type(excluded) is not int:
        raise RegistryContractError(
            "El numero de partidos de test excluidos no reconcilia.", stage="upstream_semantic_validation",
            reason_code="upstream_contract_invalid", pattern_id=spec.pattern_id,
        )
    for key, value in seal.items():
        if key.startswith("test_") and key not in {"test_status", excluded_key}:
            if type(value) is not int or value != 0:
                raise RegistryContractError(
                    "Un contador de uso del test no es un entero cero.",
                    stage="upstream_semantic_validation",
                    reason_code="upstream_contract_invalid",
                    pattern_id=spec.pattern_id,
                )
    return "sealed", False, excluded


def _verify_aggregate_row(
    spec: SourceSpec, artifacts: Mapping[str, bytes], eligible: int, denominator: int, coverage: float
) -> None:
    artifact = next(item for item in spec.artifacts if item.payload_key == spec.aggregate_artifact)
    rows = _read_csv(artifacts[artifact.path], artifact.path)
    selected = [row for row in rows if all(row.get(key) == value for key, value in spec.aggregate_filters)]
    if not selected:
        raise RegistryContractError(
            "Falta el agregado contractual upstream.", stage="upstream_semantic_validation",
            reason_code="upstream_contract_invalid", pattern_id=spec.pattern_id, artifact_path=artifact.path,
        )
    if spec.pattern_id == "P02":
        aggregate_eligible = sum(int(row[spec.aggregate_eligible_column]) for row in selected)
        aggregate_denominator = denominator
        aggregate_coverage = aggregate_eligible / aggregate_denominator
    else:
        if len(selected) != 1:
            raise RegistryContractError("El agregado contractual no es unico.")
        row = selected[0]
        aggregate_eligible = int(row[spec.aggregate_eligible_column])
        aggregate_denominator = int(row[spec.aggregate_denominator_column]) if spec.aggregate_denominator_column else denominator
        aggregate_coverage = float(row[spec.aggregate_coverage_column]) if spec.aggregate_coverage_column else aggregate_eligible / aggregate_denominator
    if aggregate_eligible != eligible or aggregate_denominator != denominator or not math.isclose(aggregate_coverage, coverage, rel_tol=0.0, abs_tol=1e-15):
        raise RegistryContractError(
            "El agregado CSV no reconcilia con el summary.", stage="upstream_semantic_validation",
            reason_code="upstream_contract_invalid", pattern_id=spec.pattern_id, artifact_path=artifact.path,
        )


def _load_source(
    spec: SourceSpec, root: Path, git_runner: Callable[..., str]
) -> VerifiedSource:
    if spec.summary_path not in SUMMARY_ALLOWLIST:
        raise RegistryContractError("Summary fuera de allowlist.")
    summary_file = root / PurePath(spec.summary_path)
    if not summary_file.is_file():
        raise RegistryContractError(
            "Falta el summary upstream.", stage="upstream_discovery", reason_code="upstream_missing",
            pattern_id=spec.pattern_id, artifact_path=spec.summary_path,
        )
    _validate_tracked_clean(root, spec.summary_path, spec.source_commit, git_runner, spec.pattern_id)
    summary_payload = summary_file.read_bytes()
    if len(summary_payload) != spec.summary_size or _sha256(summary_payload) != spec.summary_sha256:
        raise RegistryContractError(
            "Tamano o SHA-256 del summary upstream incorrecto.", stage="upstream_hash_validation",
            reason_code="upstream_hash_mismatch", pattern_id=spec.pattern_id, artifact_path=spec.summary_path,
        )
    try:
        summary = json.loads(summary_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegistryContractError(
            "Summary upstream ilegible.", stage="upstream_semantic_validation",
            reason_code="upstream_contract_invalid", pattern_id=spec.pattern_id,
        ) from exc
    artifacts: dict[str, bytes] = {}
    for artifact in spec.artifacts:
        if artifact.path not in ARTIFACT_ALLOWLIST:
            raise RegistryContractError("CSV fuera de allowlist.")
        path = root / PurePath(artifact.path)
        if not path.is_file():
            raise RegistryContractError(
                "Falta un CSV upstream.", stage="upstream_discovery", reason_code="upstream_missing",
                pattern_id=spec.pattern_id, artifact_path=artifact.path,
            )
        _validate_tracked_clean(root, artifact.path, spec.source_commit, git_runner, spec.pattern_id)
        payload = path.read_bytes()
        if len(payload) != artifact.size or _sha256(payload) != artifact.sha256:
            raise RegistryContractError(
                "Tamano o SHA-256 de CSV upstream incorrecto.", stage="upstream_hash_validation",
                reason_code="upstream_hash_mismatch", pattern_id=spec.pattern_id, artifact_path=artifact.path,
            )
        artifacts[artifact.path] = payload
    if summary.get("analysis_status") != spec.analysis_status:
        raise RegistryContractError("Status upstream inesperado.", stage="upstream_semantic_validation", reason_code="upstream_contract_invalid", pattern_id=spec.pattern_id)
    if summary.get("publication_fingerprint") != spec.publication_fingerprint:
        raise RegistryContractError("Fingerprint upstream inesperado.", stage="upstream_semantic_validation", reason_code="upstream_contract_invalid", pattern_id=spec.pattern_id)
    reported_hashes = summary.get("artifact_payload_sha256")
    reported_sizes = summary.get("artifact_payload_bytes")
    if reported_hashes is not None:
        expected_hashes = {item.payload_key: item.sha256 for item in spec.artifacts}
        if reported_hashes != expected_hashes:
            raise RegistryContractError("Hashes internos upstream incoherentes.", stage="upstream_semantic_validation", reason_code="upstream_contract_invalid", pattern_id=spec.pattern_id)
    if reported_sizes is not None:
        expected_sizes = {item.payload_key: item.size for item in spec.artifacts}
        if reported_sizes != expected_sizes:
            raise RegistryContractError("Tamanos internos upstream incoherentes.", stage="upstream_semantic_validation", reason_code="upstream_contract_invalid", pattern_id=spec.pattern_id)
    population = summary.get("population")
    if not isinstance(population, Mapping) or population.get("development_matches", population.get("development_target_matches")) != 5993:
        raise RegistryContractError("Poblacion upstream incoherente.", stage="upstream_semantic_validation", reason_code="upstream_contract_invalid", pattern_id=spec.pattern_id)
    if population.get("development_point_rows", population.get("development_target_rows")) != 1035760:
        raise RegistryContractError("Filas de desarrollo upstream incoherentes.", stage="upstream_semantic_validation", reason_code="upstream_contract_invalid", pattern_id=spec.pattern_id)
    reconciliations = summary.get("reconciliations")
    if not isinstance(reconciliations, Mapping) or not reconciliations or any(value is not True for value in reconciliations.values()):
        raise RegistryContractError("Reconciliaciones upstream incompletas.", stage="upstream_semantic_validation", reason_code="upstream_contract_invalid", pattern_id=spec.pattern_id)
    _validate_test_seal(summary, spec)
    eligible = _strict_int(_nested(summary, spec.eligible_path), "eligible_count")
    denominator = _strict_int(_nested(summary, spec.denominator_path), "denominator_count")
    if denominator <= 0 or eligible > denominator:
        raise RegistryContractError("Denominador upstream invalido.")
    coverage = eligible / denominator if not spec.coverage_path else _strict_float(_nested(summary, spec.coverage_path), "coverage")
    if not math.isclose(coverage, eligible / denominator, rel_tol=0.0, abs_tol=1e-15):
        raise RegistryContractError("Cobertura upstream no reconcilia.", stage="upstream_semantic_validation", reason_code="upstream_contract_invalid", pattern_id=spec.pattern_id)
    if spec.state_path is not None and _nested(summary, spec.state_path) != eligible:
        raise RegistryContractError("Estado elegible upstream no reconcilia.", stage="upstream_semantic_validation", reason_code="upstream_contract_invalid", pattern_id=spec.pattern_id)
    _verify_aggregate_row(spec, artifacts, eligible, denominator, coverage)
    if spec.outcome_artifact is not None:
        artifact = next(item for item in spec.artifacts if item.payload_key == spec.outcome_artifact)
        rows = _read_csv(artifacts[artifact.path], artifact.path)
        if spec.outcomes_available != bool(rows):
            raise RegistryContractError("Disponibilidad de outcomes upstream incoherente.", stage="upstream_semantic_validation", reason_code="upstream_contract_invalid", pattern_id=spec.pattern_id)
    return VerifiedSource(spec=spec, summary=_freeze(summary))


def _row_from_source(source: VerifiedSource) -> dict[str, Any]:
    spec = source.spec
    summary = source.summary
    eligible = _strict_int(_nested(summary, spec.eligible_path), "eligible_count")
    denominator = _strict_int(_nested(summary, spec.denominator_path), "denominator_count")
    coverage = eligible / denominator
    test_status, used, excluded = _validate_test_seal(summary, spec)
    readiness = recommendation_readiness(
        upstream_available=spec.analysis_status != "not_available",
        explicitly_comparable=spec.comparable,
        diagnostic_only=spec.diagnostic_only,
        sufficient_identifiability=spec.sufficient_identifiability,
        usable_as_explainable_input=spec.usable_as_explainable_input,
    )
    return {
        "pattern_id": spec.pattern_id,
        "pattern_key": spec.pattern_key,
        "display_name": spec.display_name,
        "source_commit": spec.source_commit,
        "source_summary_path": spec.summary_path,
        "source_summary_bytes": spec.summary_size,
        "source_summary_sha256": spec.summary_sha256,
        "source_publication_fingerprint": spec.publication_fingerprint,
        "analysis_status": spec.analysis_status,
        "evidence_level": spec.evidence_level,
        "recommendation_readiness": readiness,
        "unit": spec.unit,
        "actor": spec.actor,
        "population_scope": "development_through_2023_test_2024_2026_sealed",
        "eligible_count": eligible,
        "denominator_count": denominator,
        "coverage": coverage,
        "has_observable_comparator": spec.comparable,
        "outcomes_available": spec.outcomes_available,
        "test_status": test_status,
        "test_used_for_method_selection": used,
        "excluded_test_matches": excluded,
        "primary_contract": spec.primary_contract,
        "principal_limitation": spec.principal_limitation,
    }


def _csv_scalar(column: str, value: Any) -> str:
    if value is None:
        return ""
    if column in BOOLEAN_COLUMNS:
        if type(value) is not bool:
            raise RegistryContractError(f"{column} debe ser bool real.")
        return "true" if value else "false"
    if column in INTEGER_COLUMNS:
        return str(_strict_int(value, column))
    if column in FLOAT_COLUMNS:
        return repr(_strict_float(value, column))
    if type(value) is not str:
        raise RegistryContractError(f"{column} debe ser texto.")
    return value


def _registry_csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        if tuple(row.keys()) != CSV_COLUMNS:
            raise RegistryContractError("Schema u orden de columnas del registro invalido.")
        writer.writerow({column: _csv_scalar(column, row[column]) for column in CSV_COLUMNS})
    payload = buffer.getvalue().encode("utf-8")
    if b"NaN" in payload or b"Infinity" in payload:
        raise RegistryContractError("CSV contiene no finitos.")
    return payload


def _parse_registry_csv(payload: bytes) -> tuple[Mapping[str, Any], ...]:
    rows = _read_csv(payload, "registry")
    if not rows and payload.decode("utf-8") != ",".join(CSV_COLUMNS) + "\n":
        raise RegistryContractError("CSV vacio no canonico.")
    parsed: list[Mapping[str, Any]] = []
    for row in rows:
        if tuple(row.keys()) != CSV_COLUMNS:
            raise RegistryContractError("Schema persistido del CSV invalido.")
        converted: dict[str, Any] = {}
        for column in CSV_COLUMNS:
            value = row[column]
            if value == "":
                converted[column] = None
            elif column in INTEGER_COLUMNS:
                if not re.fullmatch(r"0|[1-9][0-9]*", value):
                    raise RegistryContractError(f"Entero CSV invalido en {column}.")
                converted[column] = int(value)
            elif column in FLOAT_COLUMNS:
                number = float(value)
                if not math.isfinite(number):
                    raise RegistryContractError("Float CSV no finito.")
                converted[column] = number
            elif column in BOOLEAN_COLUMNS:
                if value not in {"true", "false"}:
                    raise RegistryContractError(f"Booleano CSV invalido en {column}.")
                converted[column] = value == "true"
            else:
                converted[column] = value
        parsed.append(_freeze(converted))
    return tuple(parsed)


def _counts(rows: Sequence[Mapping[str, Any]], field: str, domain: Iterable[str]) -> dict[str, int]:
    return {value: sum(row[field] == value for row in rows) for value in sorted(domain)}


def _upstream_entry(source: VerifiedSource) -> dict[str, Any]:
    spec = source.spec
    return {
        "pattern_id": spec.pattern_id,
        "pattern_key": spec.pattern_key,
        "upstream_pattern_key": spec.upstream_pattern_key,
        "summary_path": spec.summary_path,
        "summary_bytes": spec.summary_size,
        "summary_sha256": spec.summary_sha256,
        "publication_fingerprint": spec.publication_fingerprint,
        "source_commit": spec.source_commit,
        "artifacts": [
            {"path": item.path, "bytes": item.size, "sha256": item.sha256, "payload_key": item.payload_key}
            for item in spec.artifacts
        ],
    }


def _fingerprint(summary: Mapping[str, Any], csv_payload: bytes) -> str:
    stable = _thaw(summary)
    stable.pop("publication_fingerprint", None)
    digest = hashlib.sha256(_json_bytes(stable))
    digest.update(len(csv_payload).to_bytes(8, "big"))
    digest.update(csv_payload)
    return digest.hexdigest().upper()


def _summary_core(rows: Sequence[Mapping[str, Any]], sources: Sequence[VerifiedSource], csv_payload: bytes) -> dict[str, Any]:
    return {
        "registry_status": "available_registry",
        "registry_contract_version": REGISTRY_CONTRACT_VERSION,
        "generated_from_aggregated_artifacts_only": True,
        "source_data_reads": 0,
        "parquet_reads": 0,
        "test_evaluations": 0,
        "first_pattern": "P02",
        "last_pattern": "P09",
        "total_patterns": 8,
        "counts_by_analysis_status": _counts(rows, "analysis_status", ANALYSIS_STATUSES),
        "counts_by_evidence_level": _counts(rows, "evidence_level", EVIDENCE_LEVELS),
        "counts_by_recommendation_readiness": _counts(rows, "recommendation_readiness", READINESS_LEVELS),
        "eligible_components": [row["pattern_id"] for row in rows if row["recommendation_readiness"] == "eligible_component"],
        "diagnostic_only_patterns": [row["pattern_id"] for row in rows if row["recommendation_readiness"] == "diagnostic_only"],
        "blocked_patterns": [row["pattern_id"] for row in rows if row["recommendation_readiness"].startswith("blocked_")],
        "pattern_key_correspondence": {
            source.spec.pattern_id: {
                "registry": source.spec.pattern_key,
                "upstream": source.spec.upstream_pattern_key,
            }
            for source in sources
            if source.spec.pattern_key != source.spec.upstream_pattern_key
        },
        "upstream_summaries": [_upstream_entry(source) for source in sources],
        "reconciliations": {
            "all_upstream_sources_tracked_and_clean": True,
            "all_upstream_hashes_and_sizes_reconciled": True,
            "all_upstream_fingerprints_reconciled": True,
            "all_upstream_test_seals_reconciled": True,
            "coverage_reconciled": True,
            "pattern_ids_complete_unique_ordered": True,
            "readiness_recomputed": True,
            "registry_csv_reconciled": True,
            "test_evaluations_zero": True,
        },
        "registry_csv_bytes": len(csv_payload),
        "registry_csv_sha256": _sha256(csv_payload),
        "fingerprint_contract_version": FINGERPRINT_CONTRACT_VERSION,
        "global_limitations": [
            "Registro descriptivo de contratos publicados; no demuestra causalidad ni calidad de anotacion.",
            "Cobertura no equivale a readiness ni a efectividad tactica.",
            "eligible_component identifica una entrada potencial, no una recomendacion autonoma.",
            "El test 2024-2026 permanece sellado y no se evalua.",
            "La publicacion de dos archivos no resiste un apagado abrupto entre reemplazos.",
        ],
    }


def _finalize(rows: Sequence[Mapping[str, Any]], sources: Sequence[VerifiedSource]) -> RegistryResult:
    frozen_rows = tuple(_freeze(dict(row)) for row in rows)
    csv_payload = _registry_csv_bytes(frozen_rows)
    summary = _summary_core(frozen_rows, sources, csv_payload)
    summary["publication_fingerprint"] = _fingerprint(summary, csv_payload)
    result = RegistryResult(_freeze(summary), frozen_rows, tuple(sources))
    validate_registry_result(result)
    return result


def build_registry(
    *, repository_root: Path = ROOT, git_runner: Callable[..., str] = _run_git
) -> RegistryResult:
    """Construye el registro solo desde los artefactos agregados allowlisted."""
    root = Path(repository_root)
    _git_preflight(root, git_runner)
    sources = tuple(_load_source(spec, root, git_runner) for spec in SOURCE_SPECS)
    try:
        rows = tuple(_row_from_source(source) for source in sources)
        return _finalize(rows, sources)
    except RegistryContractError:
        raise
    except Exception as exc:
        raise RegistryContractError(
            "No se pudo construir el registro.", stage="registry_construction",
            reason_code="registry_contract_invalid",
        ) from exc


def _validate_row_types_and_domains(row: Mapping[str, Any]) -> None:
    if tuple(row.keys()) != CSV_COLUMNS:
        raise RegistryContractError("Schema u orden de fila invalido.")
    for column in INTEGER_COLUMNS:
        _strict_int(row[column], column)
    for column in FLOAT_COLUMNS:
        _strict_float(row[column], column)
    for column in BOOLEAN_COLUMNS:
        if type(row[column]) is not bool:
            raise RegistryContractError(f"{column} debe ser bool real.")
    for column in set(CSV_COLUMNS) - INTEGER_COLUMNS - FLOAT_COLUMNS - BOOLEAN_COLUMNS:
        if type(row[column]) is not str or not row[column]:
            raise RegistryContractError(f"{column} debe ser texto no vacio.")
    if row["analysis_status"] not in ANALYSIS_STATUSES:
        raise RegistryContractError("analysis_status fuera de dominio.")
    if row["evidence_level"] not in EVIDENCE_LEVELS:
        raise RegistryContractError("evidence_level fuera de dominio.")
    if row["recommendation_readiness"] not in READINESS_LEVELS:
        raise RegistryContractError("readiness fuera de dominio.")
    if row["unit"] not in UNITS or row["actor"] not in ACTORS:
        raise RegistryContractError("Unidad o actor fuera de dominio.")
    if row["source_summary_path"] not in SUMMARY_ALLOWLIST or PurePath(row["source_summary_path"]).is_absolute():
        raise RegistryContractError("Ruta de summary no permitida.")
    if not re.fullmatch(r"[0-9a-f]{40}", row["source_commit"]):
        raise RegistryContractError("Commit upstream mal formado.")
    if not re.fullmatch(r"[0-9A-F]{64}", row["source_summary_sha256"]):
        raise RegistryContractError("SHA-256 upstream mal formado.")
    if not re.fullmatch(r"[0-9A-F]{64}", row["source_publication_fingerprint"]):
        raise RegistryContractError("Fingerprint upstream mal formado.")
    if row["eligible_count"] > row["denominator_count"] or row["denominator_count"] <= 0:
        raise RegistryContractError("Conteos de cobertura invalidos.")
    if not math.isclose(row["coverage"], row["eligible_count"] / row["denominator_count"], rel_tol=0.0, abs_tol=1e-15):
        raise RegistryContractError("Cobertura no reconcilia.")
    if row["test_status"] != "sealed" or row["test_used_for_method_selection"] is not False or row["excluded_test_matches"] != 1531:
        raise RegistryContractError("Sellado de test invalido.")


def _expected_rows_from_sources(sources: Sequence[VerifiedSource]) -> tuple[dict[str, Any], ...]:
    if tuple(source.spec.pattern_id for source in sources) != PATTERN_IDS:
        raise RegistryContractError("Fuentes P02--P09 incompletas o desordenadas.")
    return tuple(_row_from_source(source) for source in sources)


def validate_registry_result(result: RegistryResult) -> None:
    if not isinstance(result, RegistryResult):
        raise TypeError("RegistryResult requerido.")
    summary = result.summary
    _validate_json_values(summary)
    status = summary.get("registry_status")
    if status not in REGISTRY_STATUSES:
        raise RegistryContractError("registry_status fuera de dominio.")
    if status == "not_available":
        _validate_not_available(result)
        return
    if len(result.rows) != 8:
        raise RegistryContractError("El registro debe contener ocho filas.")
    ids = tuple(row.get("pattern_id") for row in result.rows)
    if ids != PATTERN_IDS or len(set(ids)) != 8:
        raise RegistryContractError("IDs P02--P09 incompletos, duplicados o desordenados.")
    for row in result.rows:
        _validate_row_types_and_domains(row)
    if not result.sources:
        raise RegistryContractError("Faltan fuentes upstream para validar semantica.")
    expected_rows = _expected_rows_from_sources(result.sources)
    if tuple(_thaw(row) for row in result.rows) != expected_rows:
        raise RegistryContractError("Las filas no coinciden con los contratos upstream.")
    csv_payload = _registry_csv_bytes(result.rows)
    required_summary = {
        "registry_status", "registry_contract_version", "generated_from_aggregated_artifacts_only",
        "source_data_reads", "parquet_reads", "test_evaluations", "first_pattern", "last_pattern",
        "total_patterns", "counts_by_analysis_status", "counts_by_evidence_level",
        "counts_by_recommendation_readiness", "eligible_components", "diagnostic_only_patterns",
        "blocked_patterns", "pattern_key_correspondence", "upstream_summaries", "reconciliations",
        "registry_csv_bytes", "registry_csv_sha256", "fingerprint_contract_version",
        "publication_fingerprint", "global_limitations",
    }
    if set(summary) != required_summary:
        raise RegistryContractError("Schema del summary disponible invalido.")
    if (
        summary["registry_contract_version"] != REGISTRY_CONTRACT_VERSION
        or summary["generated_from_aggregated_artifacts_only"] is not True
        or any(summary[key] != 0 for key in ("source_data_reads", "parquet_reads", "test_evaluations"))
        or summary["first_pattern"] != "P02"
        or summary["last_pattern"] != "P09"
        or summary["total_patterns"] != 8
    ):
        raise RegistryContractError("Metadatos globales del registro invalidos.")
    expected_core = _summary_core(result.rows, result.sources, csv_payload)
    for key, expected in expected_core.items():
        if _thaw(summary[key]) != expected:
            raise RegistryContractError(f"Summary no reconcilia en {key}.")
    if summary["registry_csv_bytes"] != len(csv_payload) or summary["registry_csv_sha256"] != _sha256(csv_payload):
        raise RegistryContractError("Bytes o hash del CSV no reconcilian.")
    for source in result.sources:
        spec = source.spec
        row = next(item for item in result.rows if item["pattern_id"] == spec.pattern_id)
        recalculated = recommendation_readiness(
            upstream_available=spec.analysis_status != "not_available",
            explicitly_comparable=spec.comparable,
            diagnostic_only=spec.diagnostic_only,
            sufficient_identifiability=spec.sufficient_identifiability,
            usable_as_explainable_input=spec.usable_as_explainable_input,
        )
        if row["recommendation_readiness"] != recalculated:
            raise RegistryContractError("Readiness no reconcilia.")
        if (row["has_observable_comparator"], row["outcomes_available"]) != (spec.comparable, spec.outcomes_available):
            raise RegistryContractError("Comparator u outcomes no reconcilian.")
    reconciliations = summary["reconciliations"]
    if not isinstance(reconciliations, Mapping) or any(value is not True for value in reconciliations.values()):
        raise RegistryContractError("Reconciliaciones finales incompletas.")
    if summary["fingerprint_contract_version"] != FINGERPRINT_CONTRACT_VERSION:
        raise RegistryContractError("Version de fingerprint invalida.")
    if summary["publication_fingerprint"] != _fingerprint(summary, csv_payload):
        raise RegistryContractError("Fingerprint final invalido.")


def _validate_not_available(result: RegistryResult) -> None:
    if result.rows:
        raise RegistryContractError("not_available no puede publicar filas parciales.")
    required = {
        "registry_status", "registry_contract_version", "generated_from_aggregated_artifacts_only",
        "source_data_reads", "parquet_reads", "test_evaluations", "reason_codes", "failure",
        "registry_csv_bytes", "registry_csv_sha256", "fingerprint_contract_version",
        "publication_fingerprint", "global_limitations",
    }
    if set(result.summary) != required:
        raise RegistryContractError("Schema not_available invalido.")
    if (
        result.summary["registry_contract_version"] != REGISTRY_CONTRACT_VERSION
        or
        result.summary["generated_from_aggregated_artifacts_only"] is not True
        or any(result.summary[key] != 0 for key in ("source_data_reads", "parquet_reads", "test_evaluations"))
        or result.summary["fingerprint_contract_version"] != FINGERPRINT_CONTRACT_VERSION
    ):
        raise RegistryContractError("Contadores not_available invalidos.")
    codes = result.summary["reason_codes"]
    if not isinstance(codes, tuple) or not codes or any(code not in REASON_CODES for code in codes):
        raise RegistryContractError("reason_codes not_available invalidos.")
    failure = result.summary["failure"]
    if not isinstance(failure, Mapping) or set(failure) != {"stage", "type", "message", "pattern_id", "artifact_path"}:
        raise RegistryContractError("Failure not_available invalido.")
    if failure["stage"] not in FAILURE_STAGES or not all(
        type(failure[key]) is str and failure[key]
        for key in ("type", "message", "pattern_id", "artifact_path")
    ):
        raise RegistryContractError("Detalle de fallo not_available invalido.")
    limitations = result.summary["global_limitations"]
    if not isinstance(limitations, tuple) or not limitations or any(type(item) is not str or not item for item in limitations):
        raise RegistryContractError("Limitaciones not_available invalidas.")
    csv_payload = _registry_csv_bytes(())
    if result.summary["registry_csv_bytes"] != len(csv_payload) or result.summary["registry_csv_sha256"] != _sha256(csv_payload):
        raise RegistryContractError("CSV vacio not_available no reconcilia.")
    if result.summary["publication_fingerprint"] != _fingerprint(result.summary, csv_payload):
        raise RegistryContractError("Fingerprint not_available invalido.")


def _serialize_once(result: RegistryResult) -> tuple[bytes, bytes]:
    return _json_bytes(_thaw(result.summary)), _registry_csv_bytes(result.rows)


def serialize_registry_result(result: RegistryResult) -> tuple[bytes, bytes]:
    validate_registry_result(result)
    first = _serialize_once(result)
    second = _serialize_once(result)
    if first != second:
        raise RegistryContractError(
            "Serializacion no determinista.", stage="serialization", reason_code="serialization_failed"
        )
    return first


def _sanitize(message: str) -> str:
    value = message.replace(str(ROOT), "<repository>")
    value = re.sub(r"(?i)(?<![\w])(?:[a-z]:[\\/])[^\s,;]+", "<absolute-path>", value)
    value = re.sub(r"(?i)(?<!\w)/(?:users|home)/[^\s,;]+", "<absolute-path>", value)
    return value or "Fallo sin detalle"


def not_available_result(error: Exception) -> RegistryResult:
    if isinstance(error, RegistryContractError):
        stage = error.stage
        code = error.reason_code
        pattern_id = error.pattern_id or "unknown"
        artifact_path = error.artifact_path or "unknown"
    else:
        stage = "registry_construction"
        code = "registry_contract_invalid"
        pattern_id = "unknown"
        artifact_path = "unknown"
    if stage not in FAILURE_STAGES or code not in REASON_CODES:
        raise RegistryContractError("Contrato de fallo interno invalido.")
    csv_payload = _registry_csv_bytes(())
    summary: dict[str, Any] = {
        "registry_status": "not_available",
        "registry_contract_version": REGISTRY_CONTRACT_VERSION,
        "generated_from_aggregated_artifacts_only": True,
        "source_data_reads": 0,
        "parquet_reads": 0,
        "test_evaluations": 0,
        "reason_codes": [code],
        "failure": {
            "stage": stage,
            "type": type(error).__name__,
            "message": _sanitize(str(error)),
            "pattern_id": pattern_id,
            "artifact_path": artifact_path,
        },
        "registry_csv_bytes": len(csv_payload),
        "registry_csv_sha256": _sha256(csv_payload),
        "fingerprint_contract_version": FINGERPRINT_CONTRACT_VERSION,
        "global_limitations": ["No se publica un registro parcial."],
    }
    summary["publication_fingerprint"] = _fingerprint(summary, csv_payload)
    result = RegistryResult(_freeze(summary), (), ())
    validate_registry_result(result)
    return result


def _stage_payload(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.read_bytes() != payload:
            raise OSError("El staging no conserva los bytes.")
        return temporary
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def verify_persisted_registry_artifacts(
    *,
    summary_path: Path = SUMMARY_PATH,
    registry_csv_path: Path = REGISTRY_CSV_PATH,
    repository_root: Path = ROOT,
    git_runner: Callable[..., str] = _run_git,
) -> RegistryResult:
    if not summary_path.is_file() or not registry_csv_path.is_file():
        raise RegistryContractError("Faltan artefactos del registro.", stage="publication_verification")
    summary_payload = summary_path.read_bytes()
    csv_payload = registry_csv_path.read_bytes()
    try:
        summary = json.loads(summary_payload.decode("utf-8"))
        rows = _parse_registry_csv(csv_payload)
    except (UnicodeDecodeError, json.JSONDecodeError, csv.Error, ValueError) as exc:
        raise RegistryContractError("Artefactos del registro ilegibles.", stage="publication_verification") from exc
    if summary.get("registry_status") == "available_registry":
        _git_preflight(Path(repository_root), git_runner)
        sources = tuple(_load_source(spec, Path(repository_root), git_runner) for spec in SOURCE_SPECS)
    else:
        sources = ()
    result = RegistryResult(_freeze(summary), rows, sources)
    validate_registry_result(result)
    if serialize_registry_result(result) != (summary_payload, csv_payload):
        raise RegistryContractError("Bytes persistidos no canonicos.", stage="publication_verification")
    return result


def _restore(path: Path, previous: bytes | None) -> None:
    if previous is None:
        if path.exists():
            path.unlink()
    else:
        path.write_bytes(previous)


def write_registry_artifacts(
    result: RegistryResult,
    *,
    summary_path: Path = SUMMARY_PATH,
    registry_csv_path: Path = REGISTRY_CSV_PATH,
    repository_root: Path = ROOT,
    git_runner: Callable[..., str] = _run_git,
) -> None:
    """Publica dos artefactos con rollback ante fallos ordinarios."""
    payloads = serialize_registry_result(result)
    paths = (Path(summary_path), Path(registry_csv_path))
    staged: list[Path] = []
    previous: dict[Path, bytes | None] = {}
    try:
        for path, payload in zip(paths, payloads):
            staged.append(_stage_payload(path, payload))
        previous = {path: path.read_bytes() if path.exists() else None for path in paths}
        for temporary, destination in zip(staged, paths):
            os.replace(temporary, destination)
        verify_persisted_registry_artifacts(
            summary_path=paths[0], registry_csv_path=paths[1],
            repository_root=repository_root, git_runner=git_runner,
        )
    except Exception:
        for path, payload in previous.items():
            _restore(path, payload)
        raise
    finally:
        for temporary in staged:
            if temporary.exists():
                temporary.unlink()


def main() -> int:
    try:
        result = build_registry()
    except Exception as exc:
        result = not_available_result(exc)
    try:
        write_registry_artifacts(result)
    except Exception as exc:
        print(json.dumps({"registry_status": "publication_failed", "reason": _sanitize(str(exc))}, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return 1
    print(json.dumps({"registry_status": result.summary["registry_status"], "patterns": len(result.rows), "publication_fingerprint": result.summary["publication_fingerprint"]}, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0 if result.summary["registry_status"] == "available_registry" else 1


if __name__ == "__main__":
    raise SystemExit(main())
