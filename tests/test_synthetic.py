import pandas as pd
from helpers import expect_raises

from creditlab.data.synthetic import RANGE_RULES, make_synthetic_credit_data
from creditlab.data.validation import duplicate_report, find_numeric_like_text, range_violations


def test_same_seed_gives_identical_data():
    a = make_synthetic_credit_data(n=500, seed=7)
    b = make_synthetic_credit_data(n=500, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_different_data():
    a = make_synthetic_credit_data(n=500, seed=7)
    b = make_synthetic_credit_data(n=500, seed=8)
    assert not a["default"].equals(b["default"])


def test_clean_data_shape_and_target():
    df = make_synthetic_credit_data(n=2000, seed=1)
    assert df.shape == (2000, 12)
    assert set(df["default"].unique()) <= {0, 1}
    assert 0.05 <= df["default"].mean() <= 0.20  # 대략 10% 안팎을 목표로 설계
    assert df.isna().sum().sum() == 0
    assert df["application_id"].is_unique
    assert df.attrs["source"] == "synthetic"


def test_clean_data_has_no_range_violations():
    df = make_synthetic_credit_data(n=2000, seed=1)
    assert range_violations(df, RANGE_RULES)["n_violations"].sum() == 0


def test_injected_issues_are_detectable():
    df = make_synthetic_credit_data(n=2000, seed=1, inject_quality_issues=True)
    assert df["income_10k"].isna().sum() > 0
    assert duplicate_report(df, "application_id")["n_full_duplicate_rows"] == 20
    assert range_violations(df, RANGE_RULES)["n_violations"].sum() >= 5
    text_like = find_numeric_like_text(df)
    assert "inquiries_6m" in list(text_like["column"])


def test_injected_missing_income_is_not_random():
    # 근속연수 1년 미만에서 소득 결측률이 훨씬 높게 설계되어 있다(MAR 성격)
    df = make_synthetic_credit_data(n=8000, seed=3, inject_quality_issues=True)
    df = df.drop_duplicates()
    short = df[df["employment_years"] < 1.0]
    long = df[df["employment_years"] >= 1.0]
    assert short["income_10k"].isna().mean() > 3 * long["income_10k"].isna().mean()


def test_small_n_rejected():
    with expect_raises(ValueError):
        make_synthetic_credit_data(n=10)
