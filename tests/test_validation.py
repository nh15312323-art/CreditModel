import numpy as np
import pandas as pd
from helpers import close, expect_raises

from creditlab.data.validation import (
    constant_columns,
    data_dictionary,
    duplicate_report,
    find_numeric_like_text,
    infer_kind,
    missing_report,
    n_rows_with_missing,
    quality_summary,
    range_violations,
)


def _small():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "a": [1.0, 2.0, np.nan, 4.0],
            "b": ["x", "y", "y", None],
        }
    )


def test_infer_kind():
    assert infer_kind(pd.Series([1, 2])) == "numeric"
    assert infer_kind(pd.Series([True, False])) == "boolean"
    assert infer_kind(pd.Series(["a", "b"])) == "categorical"
    assert infer_kind(pd.to_datetime(pd.Series(["2024-01-01"]))) == "datetime"


def test_missing_report_hand_computed():
    rep = missing_report(_small())
    assert set(rep["column"][:2]) == {"a", "b"}  # 결측률 높은 컬럼이 먼저
    row_a = rep[rep["column"] == "a"].iloc[0]
    assert row_a["n_missing"] == 1 and close(row_a["missing_pct"], 25.0)
    assert rep[rep["column"] == "id"].iloc[0]["n_missing"] == 0


def test_n_rows_with_missing():
    assert n_rows_with_missing(_small()) == 2


def test_data_dictionary_fields():
    dd = data_dictionary(_small(), descriptions={"a": "숫자 a"}).set_index("column")
    assert dd.loc["a", "kind"] == "numeric"
    assert dd.loc["b", "kind"] == "categorical"
    assert dd.loc["a", "min"] == 1.0 and dd.loc["a", "max"] == 4.0 and dd.loc["a", "median"] == 2.0
    assert dd.loc["a", "n_missing"] == 1 and close(dd.loc["a", "missing_pct"], 25.0)
    assert bool(dd.loc["id", "is_unique_key"]) is True
    assert bool(dd.loc["b", "is_unique_key"]) is False
    assert dd.loc["a", "description"] == "숫자 a"
    assert dd.loc["b", "description"] == ""


def test_data_dictionary_does_not_modify_input():
    df = _small()
    before = df.copy()
    data_dictionary(df)
    missing_report(df)
    pd.testing.assert_frame_equal(df, before)


def test_duplicate_report_hand_computed():
    df = pd.DataFrame({"id": [1, 1, 2, 3, 3], "v": [10, 10, 20, 30, 31]})
    rep = duplicate_report(df, "id")
    assert rep["n_full_duplicate_rows"] == 1  # (1,10) 두 번째
    assert rep["n_repeated_id_rows"] == 2  # id=1 두 번째, id=3 두 번째
    assert rep["n_conflicting_id_rows"] == 1  # id=3 은 값이 달라 충돌


def test_duplicate_report_without_id_and_bad_id():
    df = pd.DataFrame({"x": [1, 1, 2]})
    assert duplicate_report(df)["n_full_duplicate_rows"] == 1
    with expect_raises(KeyError):
        duplicate_report(df, "nope")


def test_range_violations_ignores_missing():
    df = pd.DataFrame({"age": [10, 20, 200, np.nan]})
    rep = range_violations(df, {"age": (18, 100)}).iloc[0]
    assert rep["n_below"] == 1 and rep["n_above"] == 1 and rep["n_violations"] == 2


def test_range_violations_one_sided_rule():
    df = pd.DataFrame({"income": [-5, 0, 10]})
    rep = range_violations(df, {"income": (0, None)}).iloc[0]
    assert rep["n_below"] == 1 and rep["n_above"] == 0


def test_range_violations_unknown_column():
    with expect_raises(KeyError):
        range_violations(pd.DataFrame({"a": [1]}), {"b": (0, 1)})


def test_find_numeric_like_text():
    df = pd.DataFrame(
        {
            "mostly_num": ["1", "2", "N/A", "4", "5", "6", "7", "8", "9", "10"],
            "words": list("abcdefghij"),
            "real_num": range(10),
        }
    )
    found = find_numeric_like_text(df, min_ratio=0.85)
    assert list(found["column"]) == ["mostly_num"]
    row = found.iloc[0]
    assert row["n_non_numeric"] == 1 and row["non_numeric_examples"] == ["N/A"]


def test_constant_columns():
    df = pd.DataFrame({"c": [1, 1, 1], "v": [1, 2, 3], "allnan": [np.nan] * 3})
    assert constant_columns(df) == ["c", "allnan"]


def test_quality_summary_values():
    df = pd.DataFrame({"id": [1, 1, 2], "a": [1.0, 1.0, np.nan]})
    s = quality_summary(df, "id")
    assert s["n_rows"] == 3 and s["n_columns"] == 2
    assert s["n_missing_cells"] == 1 and close(s["missing_cell_pct"], 100 / 6)
    assert s["dup_n_full_duplicate_rows"] == 1
    assert s["dup_n_repeated_id_rows"] == 1
