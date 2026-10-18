import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from helpers import expect_raises

from creditlab import environment_info, set_seed
from creditlab.data.synthetic import DESCRIPTIONS, RANGE_RULES, make_synthetic_credit_data
from creditlab.learning.quiz import Question, check, grade
from creditlab.reporting.quality_report import (
    build_quality_report,
    dataframe_fingerprint,
    md_table,
    save_report,
)

Q1 = Question("Q1", "부도율", "부도율은?", {"A": "부도/전체", "B": "정상/전체"}, "A", "부도율=부도/전체")
Q2 = Question("Q2", "결측", "결측은?", {"A": "항상 삭제", "B": "원인 확인"}, "B", "원인 확인이 먼저")


def test_fingerprint_changes_when_data_changes():
    df = make_synthetic_credit_data(n=300, seed=1)
    same = df.copy()
    changed = df.copy()
    changed.loc[0, "age"] = changed.loc[0, "age"] + 1
    assert dataframe_fingerprint(df) == dataframe_fingerprint(same)
    assert dataframe_fingerprint(df) != dataframe_fingerprint(changed)


def test_md_table_basic_and_empty():
    out = md_table(pd.DataFrame({"a": [1, 2], "b": ["x|y", None]}))
    lines = out.splitlines()
    assert lines[0] == "| a | b |" and lines[1] == "| --- | --- |"
    assert "x\\|y" in lines[2]  # 파이프 문자는 이스케이프
    assert md_table(pd.DataFrame()) == "_(해당 없음)_"


def test_quality_report_contains_real_numbers():
    df = make_synthetic_credit_data(n=1000, seed=2, inject_quality_issues=True)
    meta = {"dataset_id": "synthetic_loan_v1", "source": "synthetic", "id_col": "application_id"}
    text = build_quality_report(df, "default", meta, descriptions=DESCRIPTIONS, range_rules=RANGE_RULES, seed=2)
    rate = df["default"].mean()
    assert f"{rate:.2%}" in text  # 보고서의 부도율이 실제 계산값과 일치
    assert f"{len(df):,}건" in text
    assert dataframe_fingerprint(df) in text
    for heading in ["## 1. 목적", "## 3. 타깃", "## 4. 데이터 품질", "## 6. 재현 정보", "## 7. 한계"]:
        assert heading in text
    assert "교육용" in text


def test_report_does_not_modify_data():
    df = make_synthetic_credit_data(n=300, seed=4, inject_quality_issues=True)
    before = df.copy(deep=True)
    build_quality_report(df, "default", {"id_col": "application_id"})
    pd.testing.assert_frame_equal(df, before)


def test_report_fails_on_bad_target():
    df = pd.DataFrame({"y": [0, 1, 2]})
    with expect_raises(ValueError):
        build_quality_report(df, "y")


def test_save_report_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as d:
        p = save_report("# hi\n", Path(d) / "a" / "b" / "r.md")
        assert p.read_text(encoding="utf-8") == "# hi\n"


def test_quiz_check_and_grade():
    assert check(Q1, "A") is True
    assert check(Q1, "b") is False
    assert check(Q1, None) is None  # 답 미입력은 채점하지 않는다
    assert check(Q1, "Z") is None  # 보기에 없는 답
    result = grade([Q1, Q2], {"Q1": "A", "Q2": "A"})
    assert result["score"] == 1 and result["total"] == 2
    assert result["wrong"] == ["Q2"] and result["concepts_to_review"] == ["결측"]
    assert result["passed"] is False
    assert grade([Q1, Q2], {"Q1": "a", "Q2": "b"})["passed"] is True
    with expect_raises(ValueError):
        grade([], {})


def test_set_seed_makes_numpy_reproducible():
    set_seed(123)
    a = np.random.rand(3)
    set_seed(123)
    b = np.random.rand(3)
    assert (a == b).all()


def test_environment_info_has_versions_and_seed():
    info = environment_info(seed=42)
    assert info["seed"] == 42 and info["packages"]["pandas"] != "not installed"
    assert info["generated_at_utc"].count(":") == 2
