"""Catalogo declarativo minimo para secuencias del Match Charting Project."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


PARSER_VERSION = "0.2.0"
GRAMMAR_VERSION = "mcp-0.3.2-project-0.2"
_BASE_RULE_VERSION = "0.1.0"


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    family: str
    characters: tuple[str, ...]
    source: str
    evidence_level: str
    certainty: str
    applies_from: str
    description: str
    candidate_families: tuple[str, ...]


_RULES = (
    RuleSpec("LEX-LET", "let", ("c",), "Instructions!A61", "official_explicit", "high", _BASE_RULE_VERSION, "Let de saque, repetible al inicio.", ("let",)),
    RuleSpec("LEX-SERVE-DIRECTION", "serve_direction", ("4", "5", "6"), "Instructions!A46:A47", "official_explicit", "high", _BASE_RULE_VERSION, "Direccion documentada del saque.", ("serve_direction",)),
    RuleSpec("LEX-CONTEXTUAL-ZERO", "contextual_unknown", ("0",), "Instructions!A48,A138,A179", "official_explicit", "high", _BASE_RULE_VERSION, "Valor desconocido dependiente del contexto.", ("serve_direction", "shot_direction", "return_depth")),
    RuleSpec("LEX-SHOT-TYPE", "shot_type", tuple("fbrsvzopuy lmhijktq".replace(" ", "")), "Instructions!A94:A123", "official_explicit", "high", _BASE_RULE_VERSION, "Tipo de golpe documentado.", ("shot_type",)),
    RuleSpec("LEX-SHOT-DIRECTION", "shot_direction", ("1", "2", "3"), "Instructions!A125:A128", "official_explicit", "high", _BASE_RULE_VERSION, "Direccion documentada de golpe.", ("shot_direction",)),
    RuleSpec("LEX-RETURN-DEPTH", "return_depth", ("7", "8", "9"), "Instructions!A172:A179", "official_explicit", "high", _BASE_RULE_VERSION, "Profundidad documentada del resto.", ("return_depth",)),
    RuleSpec("LEX-FAULT-ERROR", "fault_or_error", tuple("nwdxe!"), "Instructions!A50:A57,A155:A163", "official_explicit", "high", _BASE_RULE_VERSION, "Fallo de saque o error de golpe, segun contexto.", ("serve_fault", "shot_error")),
    RuleSpec("LEX-FOOT-FAULT", "serve_fault", ("g",), "Instructions!A55", "official_explicit", "high", _BASE_RULE_VERSION, "Falta de pie en el saque.", ("serve_fault",)),
    RuleSpec("LEX-MODIFIER", "modifier", ("+", "-", "=", ";", "^"), "Instructions!A70:A72,A187:A200", "official_explicit", "high", _BASE_RULE_VERSION, "Modificador documentado dependiente del contexto.", ("modifier",)),
    RuleSpec("LEX-TERMINATOR", "terminator", ("*", "#", "@"), "Instructions!A74:A81,A150:A163", "official_explicit", "high", _BASE_RULE_VERSION, "Terminador documentado dependiente del contexto.", ("terminator",)),
    RuleSpec("LEX-CHALLENGE", "challenge", ("C",), "Instructions!A213:A216", "official_explicit", "high", _BASE_RULE_VERSION, "Challenge incorrecto que detiene el punto.", ("terminator", "special_code")),
    RuleSpec("LEX-SPECIAL-CODE", "special_code", tuple("SRPQ"), "Instructions!A204:A208", "official_explicit", "high", _BASE_RULE_VERSION, "Codigo especial unitario en primer servicio.", ("special_code",)),
    RuleSpec("LEX-TIME-VIOLATION", "time_violation", ("V",), "Instructions!A59", "official_explicit", "high", _BASE_RULE_VERSION, "Perdida del primer saque por violacion de tiempo.", ("special_code",)),
    RuleSpec("LEX-WHITESPACE", "undocumented_whitespace", (), "docs/data/serve_sequence_grammar_sources.md#8-ambiguedades-e-inconsistencias-pendientes", "project_interpretation", "high", _BASE_RULE_VERSION, "Whitespace Unicode conservado como residuo.", ()),
    RuleSpec("LEX-UNDOCUMENTED", "undocumented_character", (), "Politica del proyecto: complemento del vocabulario oficial inspeccionado en MatchChart 0.3.2, hoja Instructions", "project_interpretation", "high", _BASE_RULE_VERSION, "Caracter no incluido en el catalogo oficial inspeccionado; no se le atribuye calidad ni significado.", ()),
    RuleSpec("SYN-SERVE-ACE", "serve_ace", (), "Instructions!A76:A78", "official_explicit", "high", PARSER_VERSION, "Asterisco inmediatamente posterior al prefijo como ace de servicio.", ()),
    RuleSpec("SYN-SERVE-UNRETURNED", "serve_unreturned", (), "Instructions!A76:A79", "official_explicit", "high", PARSER_VERSION, "Almohadilla inmediatamente posterior al prefijo como saque no devuelto.", ()),
    RuleSpec("SYN-SERVE-FAULT", "serve_fault", (), "Instructions!A50:A57", "official_explicit", "high", PARSER_VERSION, "Codigo de fallo inmediatamente posterior al prefijo de servicio.", ()),
    RuleSpec("SYN-SERVE-FAULT-SINGLE-CODE", "serve_fault", (), "Instructions!A50:A57,A66:A68", "official_explicit", "high", PARSER_VERSION, "Un unico codigo atomico describe el fallo; x significa ancho y largo.", ()),
    RuleSpec("SYN-SERVE-OUTCOME-STOPS", "serve_outcome", (), "Contrato aprobado del parser 0.2.0 basado en Instructions!A50:A57,A76:A79", "project_interpretation", "high", PARSER_VERSION, "La construccion sintactica se detiene tras un resultado de servicio.", ()),
    RuleSpec("WARN-CONTENT-AFTER-SERVICE-OUTCOME", "warning", (), "Contrato aprobado del parser 0.2.0", "project_interpretation", "high", PARSER_VERSION, "Contenido residual posterior a un resultado de servicio.", ()),
    RuleSpec("WARN-UNDOCUMENTED-SERVE-FAULT-COMBINATION", "warning", (), "Instructions!A50:A57,A66:A68 y contrato aprobado del parser 0.2.0", "project_interpretation", "high", PARSER_VERSION, "Segundo codigo de fallo residual en una combinacion no documentada.", ()),
)


def _build_rule_index(rules: tuple[RuleSpec, ...]) -> Mapping[str, RuleSpec]:
    rule_ids = tuple(rule.rule_id for rule in rules)
    duplicate_ids = sorted(
        {rule_id for rule_id in rule_ids if rule_ids.count(rule_id) > 1}
    )
    if duplicate_ids:
        raise ValueError(f"rule_id duplicados: {duplicate_ids}")
    return MappingProxyType({rule.rule_id: rule for rule in rules})


def _build_character_index(rules: tuple[RuleSpec, ...]) -> Mapping[str, RuleSpec]:
    index: dict[str, RuleSpec] = {}
    for rule in rules:
        if not isinstance(rule.candidate_families, tuple):
            raise TypeError(
                f"candidate_families debe ser tuple en {rule.rule_id}."
            )
        if len(rule.candidate_families) != len(set(rule.candidate_families)):
            raise ValueError(
                f"candidate_families duplicadas en {rule.rule_id}."
            )
        for character in rule.characters:
            if character in index:
                raise ValueError(f"Caracter duplicado en el catalogo: {character!r}")
            index[character] = rule
    return MappingProxyType(index)


RULES: tuple[RuleSpec, ...] = _RULES
RULES_BY_ID: Mapping[str, RuleSpec] = _build_rule_index(RULES)
CHARACTER_RULES: Mapping[str, RuleSpec] = _build_character_index(RULES)


def get_rule(rule_id: str) -> RuleSpec:
    """Devuelve una regla existente o falla ante referencias tecnicas invalidas."""
    try:
        return RULES_BY_ID[rule_id]
    except KeyError as exc:
        raise ValueError(f"rule_id inexistente: {rule_id}") from exc
