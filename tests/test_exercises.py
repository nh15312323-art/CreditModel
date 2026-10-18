import numpy as np
import pandas as pd

from creditlab.data.synthetic import RANGE_RULES, make_synthetic_credit_data
from creditlab.learning.exercises import (
    TARGET_DEFINITION_FIELDS,
    check_cleaning_result,
    check_target_definition,
)


def test_target_definition_reports_missing_fields():
    result = check_target_definition({"observation_unit": "대출 신청 1건", "default_definition": "TBD"})
    assert result["ok"] is False
    assert "observation_unit" not in result["missing"]
    assert "default_definition" in result["missing"]  # 'TBD' 는 빈 값으로 취급
    assert len(result["missing"]) == len(TARGET_DEFINITION_FIELDS) - 1


def test_target_definition_complete_and_none_input():
    full = {k: "내용을 채움" for k in TARGET_DEFINITION_FIELDS}
    assert check_target_definition(full) == {"ok": True, "missing": []}
    assert check_target_definition(None)["ok"] is False


def _manual_clean(raw):
    c = raw.drop_duplicates().copy()
    c["inquiries_6m"] = pd.to_numeric(c["inquiries_6m"], errors="coerce")
    c.loc[(c["age"] < 19) | (c["age"] > 100), "age"] = np.nan
    c.loc[c["card_utilization"] > 1, "card_utilization"] = np.nan
    return c


def test_cleaning_checker_accepts_correct_cleaning():
    raw = make_synthetic_credit_data(n=1000, seed=9, inject_quality_issues=True)
    result = check_cleaning_result(raw, _manual_clean(raw), "application_id", ["inquiries_6m"], RANGE_RULES)
    assert result["ok"] is True, result["problems"]


def test_cleaning_checker_catches_each_problem():
    raw = make_synthetic_credit_data(n=1000, seed=9, inject_quality_issues=True)
    assert check_cleaning_result(raw, None, "application_id", ["inquiries_6m"], RANGE_RULES)["ok"] is False
    same_object = check_cleaning_result(raw, raw, "application_id", ["inquiries_6m"], RANGE_RULES)
    assert same_object["ok"] is False and len(same_object["problems"]) >= 3
    no_dedup = _manual_clean(raw)
    no_dedup = pd.concat([no_dedup, no_dedup.head(2)])
    problems = check_cleaning_result(raw, no_dedup, "application_id", ["inquiries_6m"], RANGE_RULES)["problems"]
    assert any("중복" in p for p in problems)
