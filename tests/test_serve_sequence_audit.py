import json
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pyarrow as pa
import pytest

from src.analysis.serve_sequence_audit import (
    CHARACTER_COLUMNS,
    CSV_NULL_REPRESENTATION,
    EXAMPLE_COLUMNS,
    GROUP_COLUMNS,
    POINTS_FILE,
    PERCENTILE_METHOD,
    analyze_local_points,
    analyze_serve_sequences,
    classify_presence,
    escaped_text,
    validate_reconciliations,
    write_audit_artifacts,
)


def synthetic_points() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["m1", 1, "4*", None, "2009-12-31", "Hard"],
            ["m1", 2, "4w ", "", "2010-01-01", "Hard"],
            ["m2", 1, "  ", "\t", "2019-06-01", "Clay"],
            ["m2", 2, "é A", "x\x01", "2020-01-01", "Grass"],
            ["m3", 1, "6#", " a ", "2021-03-01", "Clay"],
        ],
        columns=[
            "match_id",
            "point_number",
            "first_serve",
            "second_serve",
            "date",
            "surface",
        ],
    )


def test_presence_states_are_exact_and_exhaustive():
    assert classify_presence(None) == "null"
    assert classify_presence(pd.NA) == "null"
    assert classify_presence("") == "empty"
    assert classify_presence("  ") == "whitespace_only"
    assert classify_presence("\t\n") == "whitespace_only"
    assert classify_presence(" a ") == "substantive"


def test_escaped_text_distinguishes_whitespace_control_and_non_ascii():
    assert escaped_text(" \t\n\r\x01é") == r"\u0020\t\n\r\u0001\u00E9"
    assert escaped_text("") == ""
    assert escaped_text(None) is None


def test_escaped_text_round_trip_is_unambiguous():
    original = 'space tab\tnewline\nbackslash\\quotes"\' nonasciié control\x01'
    escaped = escaped_text(original)
    assert escaped.encode("ascii").decode("unicode_escape") == original


def test_missing_required_columns_raise_error():
    with pytest.raises(ValueError, match="Faltan columnas obligatorias"):
        analyze_serve_sequences(synthetic_points().drop(columns="surface"))


def test_global_duplicate_key_across_arrow_batches_raises_error():
    first = pa.Table.from_pandas(synthetic_points().iloc[[0]], preserve_index=False)
    second = pa.Table.from_pandas(synthetic_points().iloc[[0]], preserve_index=False)
    with pytest.raises(ValueError, match="no es unica globalmente"):
        analyze_serve_sequences([first, second], batch_size=1)


def test_duplicate_key_inside_one_batch_raises_error():
    duplicated = pd.concat(
        [synthetic_points().iloc[[0]], synthetic_points().iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="no es unica globalmente"):
        analyze_serve_sequences(duplicated)


def test_invalid_surface_reports_unexpected_value():
    points = synthetic_points().iloc[[0]].copy()
    points.loc[:, "surface"] = "Carpet"
    with pytest.raises(ValueError, match=r"Superficies inesperadas: .*Carpet"):
        analyze_serve_sequences(points)


def test_null_surface_is_rejected():
    points = synthetic_points().iloc[[0]].copy()
    points.loc[:, "surface"] = None
    with pytest.raises(ValueError, match=r"Superficies inesperadas: <NULL>"):
        analyze_serve_sequences(points)


def test_non_string_surface_is_rejected():
    points = synthetic_points().iloc[[0]].copy()
    points["surface"] = pd.Series([7], dtype="object")
    with pytest.raises(ValueError, match=r"Superficies inesperadas: int\(7\)"):
        analyze_serve_sequences(points)


def test_invalid_date_is_rejected():
    points = synthetic_points().iloc[[0]].copy()
    points.loc[:, "date"] = "not-a-date"
    with pytest.raises(ValueError, match="fecha no puede ser nula ni invalida"):
        analyze_serve_sequences(points)


def test_dataframe_input_is_not_mutated():
    points = synthetic_points()
    original = points.copy(deep=True)
    analyze_serve_sequences(points, batch_size=2)
    pdt.assert_frame_equal(points, original)


def test_exact_presence_lengths_groups_and_period_boundaries():
    summary, groups, _, _ = analyze_serve_sequences(synthetic_points(), batch_size=2)
    first = summary["columns"]["first_serve"]
    second = summary["columns"]["second_serve"]
    assert first["presence"]["null"]["numerator"] == 0
    assert first["presence"]["whitespace_only"]["numerator"] == 1
    assert first["presence"]["substantive"]["numerator"] == 4
    assert first["raw_length"]["mean"] == pytest.approx(2.4)
    assert first["trimmed_length"]["mean"] == pytest.approx(1.8)
    assert second["presence"]["null"]["numerator"] == 1
    assert second["presence"]["empty"]["numerator"] == 1
    assert second["presence"]["whitespace_only"]["numerator"] == 1
    assert second["presence"]["substantive"]["numerator"] == 2
    periods = groups[
        (groups["column"] == "first_serve")
        & (groups["dimension"] == "derived_period")
    ]
    assert periods["group_value"].tolist() == ["to_2009", "2010s", "2020s"]
    assert periods["total_rows"].tolist() == [1, 2, 2]
    surfaces = groups[
        (groups["column"] == "first_serve")
        & (groups["dimension"] == "surface")
    ]
    assert surfaces["group_value"].tolist() == ["Hard", "Clay", "Grass"]
    assert surfaces["total_rows"].tolist() == [2, 2, 1]


def test_percentiles_use_explicit_linear_method_with_non_integer_result():
    points = pd.DataFrame(
        [
            ["p", 1, "a", None, "2009-01-01", "Hard"],
            ["p", 2, "aa", None, "2010-01-01", "Hard"],
            ["p", 3, "aaa", None, "2019-01-01", "Clay"],
            ["p", 4, "a" * 10, None, "2020-01-01", "Grass"],
        ],
        columns=synthetic_points().columns,
    )
    summary, _, _, _ = analyze_serve_sequences(points)
    raw = summary["columns"]["first_serve"]["raw_length"]
    assert PERCENTILE_METHOD == "linear"
    assert summary["length_metric_definition"]["percentile_method"] == "linear"
    assert raw["p25"] == pytest.approx(1.75)
    assert raw["median"] == pytest.approx(2.5)
    assert raw["p75"] == pytest.approx(4.75)


def test_character_counts_and_endpoint_denominators_are_exact():
    summary, _, characters, _ = analyze_serve_sequences(synthetic_points())
    first = characters[characters["column"] == "first_serve"].set_index(
        "unicode_code_point"
    )
    second = characters[characters["column"] == "second_serve"].set_index(
        "unicode_code_point"
    )
    assert summary["columns"]["first_serve"][
        "original_length_greater_than_zero_sequences"
    ] == 5
    assert summary["columns"]["second_serve"][
        "original_length_greater_than_zero_sequences"
    ] == 3
    assert first["initial_count"].sum() == 5
    assert first["final_count"].sum() == 5
    assert second["initial_count"].sum() == 3
    assert second["final_count"].sum() == 3
    assert (first["initial_denominator_original_length_gt_zero"] == 5).all()
    assert (second["final_denominator_original_length_gt_zero"] == 3).all()
    assert first.loc["U+0020", "character_display"] == "<SPACE>"
    assert first.loc["U+00E9", "character_escaped"] == r"\u00E9"
    assert second.loc["U+0009", "character_display"] == "<TAB>"
    assert second.loc["U+0001", "is_control"]
    assert first.loc["U+0020", "initial_count"] == 1
    assert first.loc["U+0020", "final_count"] == 2


def test_occurrences_and_rows_containing_are_different_counts():
    points = pd.DataFrame(
        [
            ["c", 1, "aaa", None, "2020-01-01", "Hard"],
            ["c", 2, "a", None, "2020-01-02", "Hard"],
        ],
        columns=synthetic_points().columns,
    )
    _, _, characters, _ = analyze_serve_sequences(points)
    character = characters[
        (characters["column"] == "first_serve")
        & (characters["unicode_code_point"] == "U+0061")
    ].iloc[0]
    assert character["occurrence_count"] == 4
    assert character["rows_containing_count"] == 2


def test_unicode_control_categories_are_distinguished():
    points = pd.DataFrame(
        [["u", 1, "\x01\u200b\ue000", None, "2020-01-01", "Hard"]],
        columns=synthetic_points().columns,
    )
    _, _, characters, _ = analyze_serve_sequences(points)
    first = characters[characters["column"] == "first_serve"].set_index(
        "unicode_code_point"
    )
    assert first.loc["U+0001", ["unicode_category", "character_display", "is_control"]].tolist() == ["Cc", "<CONTROL>", True]
    assert first.loc["U+200B", "unicode_category"] == "Cf"
    assert first.loc["U+200B", "character_escaped"] == r"\u200B"
    assert not first.loc["U+200B", "is_control"]
    assert first.loc["U+E000", "unicode_category"] == "Co"
    assert first.loc["U+E000", "character_escaped"] == r"\uE000"
    assert not first.loc["U+E000", "is_control"]


def test_examples_are_deterministic_and_preserve_original_text():
    points = synthetic_points()
    _, _, _, first_examples = analyze_serve_sequences(points)
    shuffled = points.sample(frac=1, random_state=17).reset_index(drop=True)
    _, _, _, second_examples = analyze_serve_sequences(shuffled, batch_size=1)
    pdt.assert_frame_equal(first_examples, second_examples)
    trailing = first_examples[
        (first_examples["column"] == "first_serve")
        & (first_examples["selection_reason"] == "trailing_whitespace")
    ].iloc[0]
    assert trailing["sequence_original"] == "4w "
    assert trailing["sequence_escaped"] == r"4w\u0020"
    only_whitespace = first_examples[
        (first_examples["column"] == "first_serve")
        & (first_examples["selection_reason"] == "presence_whitespace_only")
    ].iloc[0]
    assert only_whitespace["sequence_original"] == "  "
    assert only_whitespace["sequence_escaped"] == r"\u0020\u0020"


def test_reconciliation_rejects_wrong_state_total():
    summary, groups, characters, _ = analyze_serve_sequences(synthetic_points())
    broken = groups.copy(deep=True)
    index = broken[
        (broken["column"] == "first_serve")
        & (broken["dimension"] == "overall")
    ].index[0]
    broken.loc[index, "substantive_count"] += 1
    with pytest.raises(ValueError, match="estados no reconcilian"):
        validate_reconciliations(summary, broken, characters)


def test_reconciliation_rejects_wrong_endpoint_count():
    summary, groups, characters, _ = analyze_serve_sequences(synthetic_points())
    broken = characters.copy(deep=True)
    index = broken[broken["column"] == "first_serve"].index[0]
    broken.loc[index, "initial_count"] += 1
    with pytest.raises(ValueError, match="denominador inicial"):
        validate_reconciliations(summary, groups, broken)


def test_reconciliation_rejects_wrong_non_null_count():
    summary, groups, characters, _ = analyze_serve_sequences(synthetic_points())
    broken = groups.copy(deep=True)
    broken.loc[broken.index[0], "non_null_count"] += 1
    with pytest.raises(ValueError, match="non_null_count"):
        validate_reconciliations(summary, broken, characters)


def test_reconciliation_rejects_wrong_proportion():
    summary, groups, characters, _ = analyze_serve_sequences(synthetic_points())
    broken = groups.copy(deep=True)
    broken.loc[broken.index[0], "substantive_proportion"] = 0.123
    with pytest.raises(ValueError, match="proporcion de substantive"):
        validate_reconciliations(summary, broken, characters)


def test_reconciliation_rejects_dimension_aggregate_change():
    summary, groups, characters, _ = analyze_serve_sequences(synthetic_points())
    broken = groups.copy(deep=True)
    index = broken[
        (broken["column"] == "first_serve")
        & (broken["dimension"] == "surface")
        & (broken["group_value"] == "Hard")
    ].index[0]
    broken.loc[index, ["total_rows", "substantive_count", "non_null_count"]] += 1
    broken.loc[index, "substantive_proportion"] = broken.loc[index, "substantive_count"] / broken.loc[index, "total_rows"]
    with pytest.raises(ValueError, match="dimension surface no reconcilia"):
        validate_reconciliations(summary, broken, characters)


def test_reconciliation_rejects_missing_and_additional_groups():
    summary, groups, characters, _ = analyze_serve_sequences(synthetic_points())
    surface_index = groups[
        (groups["column"] == "first_serve")
        & (groups["dimension"] == "surface")
    ].index[0]
    missing = groups.drop(index=surface_index)
    with pytest.raises(ValueError, match="dimension surface no reconcilia"):
        validate_reconciliations(summary, missing, characters)
    additional = pd.concat([groups, groups.loc[[surface_index]]], ignore_index=True)
    additional.loc[additional.index[-1], "group_value"] = "Additional"
    with pytest.raises(ValueError, match="dimension surface no reconcilia"):
        validate_reconciliations(summary, additional, characters)


def test_artifacts_are_deterministic_utf8_and_have_exact_csv_schemas(tmp_path):
    outputs = analyze_serve_sequences(synthetic_points(), batch_size=2)
    summary, groups, characters, examples = outputs
    reports = tmp_path / "reports"
    tables = reports / "tables"
    write_audit_artifacts(summary, groups, characters, examples, reports, tables)
    first_bytes = {path.name: path.read_bytes() for path in reports.rglob("*") if path.is_file()}
    write_audit_artifacts(summary, groups, characters, examples, reports, tables)
    second_bytes = {path.name: path.read_bytes() for path in reports.rglob("*") if path.is_file()}
    assert first_bytes == second_bytes
    loaded_summary = json.loads(
        (reports / "serve_sequence_audit_summary.json").read_text(encoding="utf-8")
    )
    assert loaded_summary["artifact_determinism"]["csv_null_representation"] == CSV_NULL_REPRESENTATION
    expected = {
        "serve_sequence_audit_by_group.csv": GROUP_COLUMNS,
        "serve_sequence_character_inventory.csv": CHARACTER_COLUMNS,
        "serve_sequence_examples.csv": EXAMPLE_COLUMNS,
    }
    for filename, columns in expected.items():
        loaded = pd.read_csv(tables / filename, keep_default_na=False)
        assert loaded.columns.tolist() == columns
        assert not any(column.startswith("Unnamed") for column in loaded.columns)
    examples_text = (tables / "serve_sequence_examples.csv").read_text(encoding="utf-8")
    assert CSV_NULL_REPRESENTATION in examples_text
    assert "4w\\u0020" in examples_text


def test_output_schema_contains_no_sequence_assessment_fields():
    summary, groups, characters, examples = analyze_serve_sequences(synthetic_points())
    json.dumps(summary, ensure_ascii=False, sort_keys=True)
    field_names = set(groups.columns) | set(characters.columns) | set(examples.columns)
    assert not field_names.intersection(
        {"valid", "invalid", "parseable", "grammar_recognized", "tactical_meaning"}
    )
    assert summary["interpretation_scope"]["grammar_applied"] is False
    assert summary["interpretation_scope"]["parser_applied"] is False


@pytest.mark.integration
def test_real_serve_sequence_audit_matches_inspected_metrics():
    if not POINTS_FILE.exists():
        pytest.skip("No esta disponible points_enriched.parquet.")
    summary, groups, characters, examples = analyze_local_points()
    assert summary["source"]["point_rows"] == 1_280_408
    first = summary["columns"]["first_serve"]
    second = summary["columns"]["second_serve"]
    assert [first["presence"][state]["numerator"] for state in ("null", "empty", "whitespace_only", "substantive")] == [0, 0, 0, 1_280_408]
    assert [second["presence"][state]["numerator"] for state in ("null", "empty", "whitespace_only", "substantive")] == [798_865, 0, 35, 481_508]
    assert first["raw_length"]["max"] == 179
    assert second["raw_length"]["max"] == 210
    assert summary["length_metric_definition"]["percentile_method"] == "linear"
    assert [first["raw_length"][key] for key in ("median", "p95", "p99", "max")] == [4.0, 23.0, 38.0, 179]
    assert [second["raw_length"][key] for key in ("median", "p95", "p99", "max")] == [10.0, 32.0, 48.0, 210]
    assert first["original_length_greater_than_zero_sequences"] == 1_280_408
    assert second["original_length_greater_than_zero_sequences"] == 481_543
    assert first["character_summary"]["unique_characters"] == 62
    assert second["character_summary"]["unique_characters"] == 57
    assert first["character_summary"]["non_ascii_characters"] == 0
    assert second["character_summary"]["control_characters"] == 0
    assert first["whitespace_position_rows"] == {"leading_whitespace": 0, "trailing_whitespace": 298, "internal_whitespace": 0}
    assert second["whitespace_position_rows"] == {"leading_whitespace": 35, "trailing_whitespace": 43, "internal_whitespace": 4}
    assert len(groups) == 132
    assert len(characters) == 119
    assert len(examples) == 20
    first_groups = groups[groups["column"] == "first_serve"]
    surfaces = first_groups[first_groups["dimension"] == "surface"].set_index("group_value")
    assert surfaces["total_rows"].to_dict() == {"Hard": 801_089, "Clay": 311_240, "Grass": 168_079}
    periods = first_groups[first_groups["dimension"] == "derived_period"].set_index("group_value")
    assert periods["total_rows"].to_dict() == {"to_2009": 376_484, "2010s": 356_930, "2020s": 546_994}
    def selected_sequence(column, reason):
        selected = examples[
            (examples["column"] == column)
            & (examples["selection_reason"] == reason)
        ]
        assert len(selected) == 1
        return selected.iloc[0]["sequence_original"]

    assert selected_sequence("first_serve", "presence_substantive") == "R"
    assert selected_sequence("first_serve", "trailing_whitespace") == "4w "
    assert selected_sequence("second_serve", "presence_whitespace_only") == "  "
    assert pd.isna(selected_sequence("second_serve", "presence_null"))
    assert list(characters.columns) == CHARACTER_COLUMNS
    assert list(examples.columns) == EXAMPLE_COLUMNS
