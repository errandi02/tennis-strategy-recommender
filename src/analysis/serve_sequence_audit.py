"""Auditoria descriptiva del texto observable en las secuencias de servicio."""

from __future__ import annotations

from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
import json
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
POINTS_FILE = ROOT / "data" / "processed" / "points_enriched.parquet"
REPORTS_DIR = ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"

SEQUENCE_COLUMNS = ("first_serve", "second_serve")
REQUIRED_COLUMNS = {
    "match_id",
    "point_number",
    "first_serve",
    "second_serve",
    "date",
    "surface",
}
PRESENCE_STATES = ("null", "empty", "whitespace_only", "substantive")
GROUP_DIMENSIONS = ("overall", "surface", "year", "derived_period")
SURFACE_ORDER = ("Hard", "Clay", "Grass")
PERIOD_ORDER = ("to_2009", "2010s", "2020s")
ALLOWED_SURFACES = frozenset(SURFACE_ORDER)
PERCENTILE_METHOD = "linear"
PERCENTILES = (25, 50, 75, 90, 95, 99)
CSV_NULL_REPRESENTATION = "<NULL>"
BATCH_SIZE = 100_000

GROUP_COLUMNS = [
    "column",
    "dimension",
    "group_value",
    "total_rows",
    "null_count",
    "null_proportion",
    "empty_count",
    "empty_proportion",
    "whitespace_only_count",
    "whitespace_only_proportion",
    "substantive_count",
    "substantive_proportion",
    "non_null_count",
    "raw_length_min",
    "raw_length_mean",
    "raw_length_p25",
    "raw_length_median",
    "raw_length_p75",
    "raw_length_p90",
    "raw_length_p95",
    "raw_length_p99",
    "raw_length_max",
    "trimmed_length_min",
    "trimmed_length_mean",
    "trimmed_length_p25",
    "trimmed_length_median",
    "trimmed_length_p75",
    "trimmed_length_p90",
    "trimmed_length_p95",
    "trimmed_length_p99",
    "trimmed_length_max",
]

CHARACTER_COLUMNS = [
    "column",
    "character_display",
    "character_escaped",
    "unicode_code_point",
    "unicode_name",
    "unicode_category",
    "is_ascii",
    "is_whitespace",
    "is_control",
    "occurrence_count",
    "rows_containing_count",
    "rows_containing_denominator_non_null",
    "rows_containing_proportion_non_null",
    "initial_count",
    "initial_denominator_original_length_gt_zero",
    "initial_proportion_original_length_gt_zero",
    "final_count",
    "final_denominator_original_length_gt_zero",
    "final_proportion_original_length_gt_zero",
]

EXAMPLE_COLUMNS = [
    "selection_reason",
    "selection_rank",
    "match_id",
    "point_number",
    "column",
    "presence_state",
    "raw_length",
    "trimmed_length",
    "sequence_original",
    "sequence_escaped",
    "related_character_code_point",
]


def classify_presence(value: object) -> str:
    """Clasifica presencia sin modificar el valor recibido."""
    if value is None or value is pd.NA or (
        not isinstance(value, str) and pd.isna(value)
    ):
        return "null"
    if not isinstance(value, str):
        raise TypeError("Las secuencias no nulas deben ser cadenas.")
    if value == "":
        return "empty"
    if value.isspace():
        return "whitespace_only"
    return "substantive"


def escaped_text(value: str | None) -> str | None:
    """Representa whitespace, controles y no ASCII sin alterar el original."""
    if value is None:
        return None
    escaped = []
    for character in value:
        code_point = ord(character)
        if character == " ":
            escaped.append(r"\u0020")
        elif character == "\t":
            escaped.append(r"\t")
        elif character == "\n":
            escaped.append(r"\n")
        elif character == "\r":
            escaped.append(r"\r")
        elif character == "\\":
            escaped.append(r"\\")
        elif code_point < 32 or code_point == 127 or code_point > 126:
            width = 4 if code_point <= 0xFFFF else 8
            escaped.append(f"\\U{code_point:08X}" if width == 8 else f"\\u{code_point:04X}")
        else:
            escaped.append(character)
    return "".join(escaped)


def character_display(character: str) -> str:
    labels = {" ": "<SPACE>", "\t": "<TAB>", "\n": "<LF>", "\r": "<CR>"}
    if character in labels:
        return labels[character]
    if unicodedata.category(character) == "Cc":
        return "<CONTROL>"
    return character


def derive_period(year: int) -> str:
    if year <= 2009:
        return "to_2009"
    if year <= 2019:
        return "2010s"
    return "2020s"


def _length_distribution(values: array) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    numeric = np.frombuffer(values, dtype=np.uint32)
    return {
        "count": int(numeric.size),
        "min": int(numeric.min()),
        "mean": float(numeric.mean()),
        "p25": float(np.percentile(numeric, 25, method=PERCENTILE_METHOD)),
        "median": float(np.percentile(numeric, 50, method=PERCENTILE_METHOD)),
        "p75": float(np.percentile(numeric, 75, method=PERCENTILE_METHOD)),
        "p90": float(np.percentile(numeric, 90, method=PERCENTILE_METHOD)),
        "p95": float(np.percentile(numeric, 95, method=PERCENTILE_METHOD)),
        "p99": float(np.percentile(numeric, 99, method=PERCENTILE_METHOD)),
        "max": int(numeric.max()),
    }


def _json_scalar(value: object) -> int | float | str | bool | None:
    """Convierte escalares de numpy/pandas a tipos JSON nativos."""
    if value is None or pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _empty_group() -> dict:
    return {
        "states": Counter(),
        "raw_lengths": array("I"),
        "trimmed_lengths": array("I"),
    }


def _example_key(example: Mapping) -> tuple:
    sequence = example["sequence"]
    return (
        str(example["match_id"]),
        int(example["point_number"]),
        str(example["column"]),
        "" if sequence is None else sequence,
    )


class ServeSequenceAuditAccumulator:
    """Acumula la auditoria y mantiene unicidad exacta entre lotes."""

    def __init__(self) -> None:
        self.total_rows = 0
        self.seen_keys: set[tuple[object, object]] = set()
        self.date_min: pd.Timestamp | None = None
        self.date_max: pd.Timestamp | None = None
        self.groups = {
            column: defaultdict(_empty_group) for column in SEQUENCE_COLUMNS
        }
        self.character_occurrences = {
            column: Counter() for column in SEQUENCE_COLUMNS
        }
        self.character_rows = {column: Counter() for column in SEQUENCE_COLUMNS}
        self.initial_characters = {
            column: Counter() for column in SEQUENCE_COLUMNS
        }
        self.final_characters = {column: Counter() for column in SEQUENCE_COLUMNS}
        self.non_null_rows = Counter()
        self.non_empty_rows = Counter()
        self.space_positions = {
            column: Counter() for column in SEQUENCE_COLUMNS
        }
        self.state_examples = {column: {} for column in SEQUENCE_COLUMNS}
        self.character_examples = {column: {} for column in SEQUENCE_COLUMNS}
        self.space_examples = {column: {} for column in SEQUENCE_COLUMNS}
        self.maximum_examples = {column: None for column in SEQUENCE_COLUMNS}

    @staticmethod
    def _keep_minimum(current: dict | None, candidate: dict) -> dict:
        if current is None or _example_key(candidate) < _example_key(current):
            return candidate
        return current

    def _consume_example(self, example: dict, state: str) -> None:
        column = example["column"]
        self.state_examples[column][state] = self._keep_minimum(
            self.state_examples[column].get(state), example
        )
        value = example["sequence"]
        if value is None:
            return
        maximum = self.maximum_examples[column]
        maximum_key = (-len(value),) + _example_key(example)
        if maximum is None:
            self.maximum_examples[column] = example
        else:
            current_value = maximum["sequence"]
            current_key = (-len(current_value),) + _example_key(maximum)
            if maximum_key < current_key:
                self.maximum_examples[column] = example
        for character in set(value):
            self.character_examples[column][character] = self._keep_minimum(
                self.character_examples[column].get(character), example
            )
        positions = {
            "leading_whitespace": bool(value) and value[0].isspace(),
            "trailing_whitespace": bool(value) and value[-1].isspace(),
            "internal_whitespace": any(ch.isspace() for ch in value[1:-1]),
        }
        for reason, present in positions.items():
            if present:
                self.space_positions[column][reason] += 1
                self.space_examples[column][reason] = self._keep_minimum(
                    self.space_examples[column].get(reason), example
                )

    def consume(self, batch: Mapping[str, list]) -> None:
        missing = sorted(REQUIRED_COLUMNS - set(batch))
        if missing:
            raise ValueError(f"Faltan columnas obligatorias: {missing}")
        lengths = {len(batch[column]) for column in REQUIRED_COLUMNS}
        if len(lengths) != 1:
            raise ValueError("Las columnas del lote no tienen la misma longitud.")
        batch_rows = lengths.pop()
        unexpected_surfaces = set()
        for surface in batch["surface"]:
            if surface is None or surface is pd.NA or (
                not isinstance(surface, str) and pd.isna(surface)
            ):
                unexpected_surfaces.add("<NULL>")
            elif not isinstance(surface, str):
                unexpected_surfaces.add(
                    f"{type(surface).__name__}({surface!r})"
                )
            elif surface not in ALLOWED_SURFACES:
                unexpected_surfaces.add(repr(surface))
        if unexpected_surfaces:
            raise ValueError(
                "Superficies inesperadas: " + ", ".join(sorted(unexpected_surfaces))
            )
        for index in range(batch_rows):
            match_id = batch["match_id"][index]
            point_number = batch["point_number"][index]
            if pd.isna(match_id) or pd.isna(point_number):
                raise ValueError("La clave de punto no puede contener nulos.")
            key = (match_id, point_number)
            if key in self.seen_keys:
                raise ValueError(
                    "La clave (match_id, point_number) no es unica globalmente."
                )
            self.seen_keys.add(key)

            date = pd.to_datetime(batch["date"][index], errors="coerce")
            if pd.isna(date):
                raise ValueError("La fecha no puede ser nula ni invalida.")
            surface = batch["surface"][index]
            year = int(date.year)
            period = derive_period(year)
            self.date_min = date if self.date_min is None else min(self.date_min, date)
            self.date_max = date if self.date_max is None else max(self.date_max, date)

            for column in SEQUENCE_COLUMNS:
                value = batch[column][index]
                state = classify_presence(value)
                example = {
                    "match_id": match_id,
                    "point_number": int(point_number),
                    "column": column,
                    "sequence": None if state == "null" else value,
                    "presence_state": state,
                }
                group_keys = (
                    ("overall", "all"),
                    ("surface", surface),
                    ("year", str(year)),
                    ("derived_period", period),
                )
                for group_key in group_keys:
                    group = self.groups[column][group_key]
                    group["states"][state] += 1
                    if state != "null":
                        group["raw_lengths"].append(len(value))
                        group["trimmed_lengths"].append(len(value.strip()))
                if state != "null":
                    self.non_null_rows[column] += 1
                    if value:
                        self.non_empty_rows[column] += 1
                        self.initial_characters[column][value[0]] += 1
                        self.final_characters[column][value[-1]] += 1
                        self.character_occurrences[column].update(value)
                        self.character_rows[column].update(set(value))
                self._consume_example(example, state)
            self.total_rows += 1


def _input_batches(data: object, batch_size: int = BATCH_SIZE) -> Iterable[dict]:
    if isinstance(data, pd.DataFrame):
        missing = sorted(REQUIRED_COLUMNS - set(data.columns))
        if missing:
            raise ValueError(f"Faltan columnas obligatorias: {missing}")
        for start in range(0, len(data), batch_size):
            yield data.iloc[start : start + batch_size][sorted(REQUIRED_COLUMNS)].to_dict(
                orient="list"
            )
        return
    if isinstance(data, pa.Table):
        missing = sorted(REQUIRED_COLUMNS - set(data.column_names))
        if missing:
            raise ValueError(f"Faltan columnas obligatorias: {missing}")
        for batch in data.select(sorted(REQUIRED_COLUMNS)).to_batches(batch_size):
            yield batch.to_pydict()
        return
    if isinstance(data, pa.RecordBatch):
        missing = sorted(REQUIRED_COLUMNS - set(data.schema.names))
        if missing:
            raise ValueError(f"Faltan columnas obligatorias: {missing}")
        yield pa.Table.from_batches([data]).select(sorted(REQUIRED_COLUMNS)).to_pydict()
        return
    if isinstance(data, Iterable):
        for item in data:
            yield from _input_batches(item, batch_size=batch_size)
        return
    raise TypeError("La entrada debe ser un DataFrame, tabla Arrow o iterable.")


def _group_sort_key(row: Mapping) -> tuple:
    dimension_order = {value: index for index, value in enumerate(GROUP_DIMENSIONS)}
    dimension = row["dimension"]
    value = str(row["group_value"])
    if dimension == "surface":
        value_key = (SURFACE_ORDER.index(value),) if value in SURFACE_ORDER else (99, value)
    elif dimension == "year":
        value_key = (int(value),)
    elif dimension == "derived_period":
        value_key = (PERIOD_ORDER.index(value),) if value in PERIOD_ORDER else (99, value)
    else:
        value_key = (0,)
    return (SEQUENCE_COLUMNS.index(row["column"]), dimension_order[dimension], value_key)


def _build_group_table(accumulator: ServeSequenceAuditAccumulator) -> pd.DataFrame:
    rows = []
    for column in SEQUENCE_COLUMNS:
        for (dimension, group_value), group in accumulator.groups[column].items():
            total = sum(group["states"].values())
            raw = _length_distribution(group["raw_lengths"])
            trimmed = _length_distribution(group["trimmed_lengths"])
            row = {
                "column": column,
                "dimension": dimension,
                "group_value": group_value,
                "total_rows": total,
                "non_null_count": raw["count"],
            }
            for state in PRESENCE_STATES:
                count = int(group["states"][state])
                row[f"{state}_count"] = count
                row[f"{state}_proportion"] = count / total
            for prefix, metrics in (("raw_length", raw), ("trimmed_length", trimmed)):
                for metric in (
                    "min", "mean", "p25", "median", "p75", "p90", "p95", "p99", "max"
                ):
                    row[f"{prefix}_{metric}"] = metrics[metric]
            rows.append(row)
    rows.sort(key=_group_sort_key)
    return pd.DataFrame(rows, columns=GROUP_COLUMNS)


def _build_character_table(accumulator: ServeSequenceAuditAccumulator) -> pd.DataFrame:
    rows = []
    for column in SEQUENCE_COLUMNS:
        non_null_denominator = int(accumulator.non_null_rows[column])
        endpoint_denominator = int(accumulator.non_empty_rows[column])
        for character in sorted(
            accumulator.character_occurrences[column], key=ord
        ):
            rows_containing = int(accumulator.character_rows[column][character])
            initial_count = int(accumulator.initial_characters[column][character])
            final_count = int(accumulator.final_characters[column][character])
            rows.append(
                {
                    "column": column,
                    "character_display": character_display(character),
                    "character_escaped": escaped_text(character),
                    "unicode_code_point": f"U+{ord(character):04X}",
                    "unicode_name": unicodedata.name(character, "UNNAMED"),
                    "unicode_category": unicodedata.category(character),
                    "is_ascii": ord(character) < 128,
                    "is_whitespace": character.isspace(),
                    "is_control": unicodedata.category(character) == "Cc",
                    "occurrence_count": int(
                        accumulator.character_occurrences[column][character]
                    ),
                    "rows_containing_count": rows_containing,
                    "rows_containing_denominator_non_null": non_null_denominator,
                    "rows_containing_proportion_non_null": (
                        rows_containing / non_null_denominator
                    ),
                    "initial_count": initial_count,
                    "initial_denominator_original_length_gt_zero": endpoint_denominator,
                    "initial_proportion_original_length_gt_zero": (
                        initial_count / endpoint_denominator
                    ),
                    "final_count": final_count,
                    "final_denominator_original_length_gt_zero": endpoint_denominator,
                    "final_proportion_original_length_gt_zero": (
                        final_count / endpoint_denominator
                    ),
                }
            )
    return pd.DataFrame(rows, columns=CHARACTER_COLUMNS)


def _format_example(reason: str, rank: int, example: Mapping, code_point: str = "") -> dict:
    value = example["sequence"]
    return {
        "selection_reason": reason,
        "selection_rank": rank,
        "match_id": example["match_id"],
        "point_number": example["point_number"],
        "column": example["column"],
        "presence_state": example["presence_state"],
        "raw_length": None if value is None else len(value),
        "trimmed_length": None if value is None else len(value.strip()),
        "sequence_original": value,
        "sequence_escaped": escaped_text(value),
        "related_character_code_point": code_point,
    }


def _build_examples_table(accumulator: ServeSequenceAuditAccumulator) -> pd.DataFrame:
    rows = []
    rank = 1
    for column in SEQUENCE_COLUMNS:
        for state in PRESENCE_STATES:
            example = accumulator.state_examples[column].get(state)
            if example is not None:
                rows.append(_format_example(f"presence_{state}", rank, example))
                rank += 1
        maximum = accumulator.maximum_examples[column]
        if maximum is not None:
            rows.append(_format_example("maximum_raw_length", rank, maximum))
            rank += 1
        for reason in (
            "leading_whitespace",
            "trailing_whitespace",
            "internal_whitespace",
        ):
            example = accumulator.space_examples[column].get(reason)
            if example is not None:
                rows.append(_format_example(reason, rank, example))
                rank += 1
        rare_characters = sorted(
            accumulator.character_occurrences[column],
            key=lambda character: (
                accumulator.character_occurrences[column][character],
                ord(character),
            ),
        )[:5]
        for character in rare_characters:
            example = accumulator.character_examples[column][character]
            rows.append(
                _format_example(
                    "lowest_character_frequency",
                    rank,
                    example,
                    f"U+{ord(character):04X}",
                )
            )
            rank += 1
    return pd.DataFrame(rows, columns=EXAMPLE_COLUMNS)


def validate_reconciliations(
    summary: Mapping, groups: pd.DataFrame, characters: pd.DataFrame
) -> None:
    total_rows = int(summary["source"]["point_rows"])
    if groups.duplicated(["column", "dimension", "group_value"]).any():
        raise ValueError("Hay grupos duplicados.")
    unexpected_columns = sorted(set(groups["column"]) - set(SEQUENCE_COLUMNS))
    unexpected_dimensions = sorted(set(groups["dimension"]) - set(GROUP_DIMENSIONS))
    if unexpected_columns:
        raise ValueError(f"Columnas agrupadas inesperadas: {unexpected_columns}.")
    if unexpected_dimensions:
        raise ValueError(f"Dimensiones agrupadas inesperadas: {unexpected_dimensions}.")

    proportion_tolerance = {"rtol": 1e-12, "atol": 1e-15}
    count_columns = [
        "total_rows",
        "null_count",
        "empty_count",
        "whitespace_only_count",
        "substantive_count",
        "non_null_count",
    ]
    for index, row in groups.iterrows():
        counts = {name: int(row[name]) for name in count_columns}
        if any(value < 0 for value in counts.values()):
            raise ValueError(f"El grupo de la fila {index} contiene cuentas negativas.")
        state_sum = sum(counts[f"{state}_count"] for state in PRESENCE_STATES)
        if state_sum != counts["total_rows"]:
            raise ValueError(f"Los estados no reconcilian en la fila de grupo {index}.")
        expected_non_null = counts["total_rows"] - counts["null_count"]
        state_non_null = sum(
            counts[f"{state}_count"]
            for state in ("empty", "whitespace_only", "substantive")
        )
        if counts["non_null_count"] != expected_non_null:
            raise ValueError(f"non_null_count no reconcilia en la fila de grupo {index}.")
        if counts["non_null_count"] != state_non_null:
            raise ValueError(
                f"non_null_count no coincide con los estados no nulos en la fila {index}."
            )
        if counts["total_rows"] <= 0:
            raise ValueError(f"El grupo de la fila {index} no contiene filas.")
        for state in PRESENCE_STATES:
            proportion = float(row[f"{state}_proportion"])
            if not 0.0 <= proportion <= 1.0:
                raise ValueError(
                    f"La proporcion de {state} esta fuera de [0, 1] en la fila {index}."
                )
            expected = counts[f"{state}_count"] / counts["total_rows"]
            if not np.isclose(proportion, expected, **proportion_tolerance):
                raise ValueError(
                    f"La proporcion de {state} no reconcilia en la fila de grupo {index}."
                )

    for column in SEQUENCE_COLUMNS:
        overall = groups[
            (groups["column"] == column) & (groups["dimension"] == "overall")
        ]
        if len(overall) != 1:
            raise ValueError(f"Falta el total global de {column}.")
        row = overall.iloc[0]
        if int(row["total_rows"]) != total_rows:
            raise ValueError(f"El total global no reconcilia para {column}.")
        for dimension in ("surface", "year", "derived_period"):
            dimension_rows = groups[
                (groups["column"] == column)
                & (groups["dimension"] == dimension)
            ]
            if dimension_rows.empty:
                raise ValueError(f"Faltan grupos de {dimension} para {column}.")
            for count_column in count_columns:
                grouped_count = int(dimension_rows[count_column].sum())
                global_count = int(row[count_column])
                if grouped_count != global_count:
                    raise ValueError(
                        f"La dimension {dimension} no reconcilia {count_column} para {column}."
                    )
        character_rows = characters[characters["column"] == column]
        endpoint_denominator = int(
            summary["columns"][column][
                "original_length_greater_than_zero_sequences"
            ]
        )
        if int(character_rows["initial_count"].sum()) != endpoint_denominator:
            raise ValueError(f"El denominador inicial no reconcilia para {column}.")
        if int(character_rows["final_count"].sum()) != endpoint_denominator:
            raise ValueError(f"El denominador final no reconcilia para {column}.")
        if not (
            character_rows["initial_denominator_original_length_gt_zero"]
            == endpoint_denominator
        ).all():
            raise ValueError(f"Denominador inicial inconsistente para {column}.")
        if not (
            character_rows["final_denominator_original_length_gt_zero"]
            == endpoint_denominator
        ).all():
            raise ValueError(f"Denominador final inconsistente para {column}.")


def analyze_serve_sequences(data: object, batch_size: int = BATCH_SIZE) -> tuple:
    """Analiza entradas en memoria sin recortar ni sustituir sus secuencias."""
    accumulator = ServeSequenceAuditAccumulator()
    for batch in _input_batches(data, batch_size=batch_size):
        accumulator.consume(batch)
    if accumulator.total_rows == 0:
        raise ValueError("La entrada no puede estar vacia.")
    groups = _build_group_table(accumulator)
    characters = _build_character_table(accumulator)
    examples = _build_examples_table(accumulator)
    overall_groups = groups[groups["dimension"] == "overall"].set_index("column")
    summary = {
        "artifact_determinism": {
            "csv_index": False,
            "csv_null_representation": CSV_NULL_REPRESENTATION,
            "json_key_order": "lexicographic",
            "timestamps_included": False,
            "utf8": True,
        },
        "columns": {},
        "endpoint_denominator_definition": (
            "Number of sequences whose original, unmodified length is greater than zero."
        ),
        "global_key_uniqueness": {
            "is_unique": len(accumulator.seen_keys) == accumulator.total_rows,
            "method": "Exact in-memory set of (match_id, point_number) across all batches.",
            "space_complexity": "O(number_of_point_rows)",
        },
        "interpretation_scope": {
            "grammar_applied": False,
            "parser_applied": False,
            "statement": "Only observable text content is described.",
        },
        "length_metric_definition": {
            "denominator": "Non-null sequence values.",
            "percentile_method": PERCENTILE_METHOD,
            "percentiles": ["p25", "p50", "p75", "p90", "p95", "p99"],
            "raw_length": "len(sequence) on the original, unmodified string.",
            "trimmed_length": "len(sequence.strip()) using Unicode whitespace semantics; secondary metric only.",
        },
        "escaped_text_definition": {
            "mechanism": "Visible ASCII is preserved; backslash is doubled; space, tab, newline, carriage return, non-ASCII and control characters use explicit escape sequences.",
            "round_trip": "Decode the ASCII escaped representation with the Python unicode_escape codec.",
        },
        "period_definitions": {
            "2010s": "2010 <= year <= 2019",
            "2020s": "year >= 2020",
            "to_2009": "year <= 2009",
        },
        "presence_state_definitions": {
            "empty": "Non-null string with original length zero.",
            "null": "Null value in the input column.",
            "substantive": "Non-null, non-empty string that is not whitespace-only.",
            "whitespace_only": "Non-empty string for which every character is Unicode whitespace.",
        },
        "reconciliations": {},
        "schema_version": 1,
        "source": {
            "date_max": accumulator.date_max.date().isoformat(),
            "date_min": accumulator.date_min.date().isoformat(),
            "path": "data/processed/points_enriched.parquet",
            "point_rows": accumulator.total_rows,
        },
    }
    for column in SEQUENCE_COLUMNS:
        row = overall_groups.loc[column]
        column_characters = characters[characters["column"] == column]
        summary["columns"][column] = {
            "character_summary": {
                "ascii_characters": int(column_characters["is_ascii"].sum()),
                "control_characters": int(column_characters["is_control"].sum()),
                "non_ascii_characters": int((~column_characters["is_ascii"]).sum()),
                "unique_characters": int(len(column_characters)),
            },
            "original_length_greater_than_zero_sequences": int(
                accumulator.non_empty_rows[column]
            ),
            "presence": {
                state: {
                    "denominator": accumulator.total_rows,
                    "numerator": int(row[f"{state}_count"]),
                    "proportion": float(row[f"{state}_proportion"]),
                }
                for state in PRESENCE_STATES
            },
            "raw_length": {
                metric: _json_scalar(row[f"raw_length_{metric}"])
                for metric in (
                    "min", "mean", "p25", "median", "p75", "p90", "p95", "p99", "max"
                )
            },
            "trimmed_length": {
                **{
                    metric: _json_scalar(row[f"trimmed_length_{metric}"])
                    for metric in (
                        "min", "mean", "p25", "median", "p75", "p90", "p95", "p99", "max"
                    )
                },
            },
            "whitespace_position_rows": {
                key: int(accumulator.space_positions[column][key])
                for key in (
                    "leading_whitespace",
                    "trailing_whitespace",
                    "internal_whitespace",
                )
            },
        }
    validate_reconciliations(summary, groups, characters)
    summary["reconciliations"] = {
        "endpoint_counts_equal_original_length_gt_zero_denominator": True,
        "global_point_key_unique": True,
        "group_dimensions_equal_total_rows": True,
        "presence_states_equal_total_rows": True,
    }
    return summary, groups, characters, examples


def analyze_local_points() -> tuple:
    """Lee exclusivamente el Parquet procesado autorizado, por lotes."""
    parquet_file = pq.ParquetFile(POINTS_FILE)
    batches = parquet_file.iter_batches(
        columns=sorted(REQUIRED_COLUMNS), batch_size=BATCH_SIZE
    )
    return analyze_serve_sequences(batches, batch_size=BATCH_SIZE)


def write_audit_artifacts(
    summary: Mapping,
    groups: pd.DataFrame,
    characters: pd.DataFrame,
    examples: pd.DataFrame,
    reports_dir: Path = REPORTS_DIR,
    tables_dir: Path = TABLES_DIR,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "serve_sequence_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs = (
        (groups, "serve_sequence_audit_by_group.csv", GROUP_COLUMNS),
        (characters, "serve_sequence_character_inventory.csv", CHARACTER_COLUMNS),
        (examples, "serve_sequence_examples.csv", EXAMPLE_COLUMNS),
    )
    for frame, filename, columns in outputs:
        if list(frame.columns) != columns:
            raise ValueError(f"Esquema inesperado para {filename}.")
        frame.to_csv(
            tables_dir / filename,
            index=False,
            encoding="utf-8",
            na_rep=CSV_NULL_REPRESENTATION,
            lineterminator="\n",
        )


def main() -> None:
    summary, groups, characters, examples = analyze_local_points()
    write_audit_artifacts(summary, groups, characters, examples)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
