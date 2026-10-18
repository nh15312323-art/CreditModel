import pandas as pd
from helpers import close, expect_raises

from creditlab.data.target import (
    default_rate,
    default_rate_ci,
    sample_sufficiency,
    target_distribution,
    validate_binary_target,
)


def test_default_rate_hand_computed():
    assert default_rate([0, 0, 0, 1]) == 0.25
    assert default_rate([1, 1]) == 1.0


def test_validate_accepts_bool_and_returns_int():
    s = validate_binary_target([True, False, True])
    assert list(s) == [1, 0, 1]
    assert str(s.dtype).startswith("int")


def test_validate_rejects_missing_values():
    with expect_raises(ValueError, "결측"):
        validate_binary_target([0, 1, None])


def test_validate_rejects_non_binary_values():
    with expect_raises(ValueError, "0/1"):
        validate_binary_target([0, 1, 2])
    with expect_raises(ValueError, "0/1"):
        validate_binary_target(["Y", "N"])


def test_validate_rejects_empty():
    with expect_raises(ValueError, "비어"):
        validate_binary_target([])


def test_target_distribution_counts_and_ratios():
    df = pd.DataFrame({"y": [0, 0, 0, 1]})
    dist = target_distribution(df, "y")
    assert list(dist["value"]) == [0, 1]
    assert list(dist["count"]) == [3, 1]
    assert close(dist["ratio"].sum(), 1.0)
    assert close(dist.loc[dist["value"] == 1, "ratio"].iloc[0], 0.25)


def test_target_distribution_includes_zero_count_class():
    dist = target_distribution(pd.DataFrame({"y": [0, 0, 0]}), "y")
    assert list(dist["count"]) == [3, 0]  # 부도가 0건이어도 행이 사라지면 안 된다


def test_target_distribution_missing_column():
    with expect_raises(KeyError, "타깃 컬럼"):
        target_distribution(pd.DataFrame({"a": [0, 1]}), "y")


def test_wilson_ci_matches_hand_calculation():
    # n=100, x=50 → 손계산: center=0.5, half≈0.09617 → (0.4038, 0.5962)
    y = [1] * 50 + [0] * 50
    low, high = default_rate_ci(y, 0.95)
    assert close(low, 0.4038, 1e-3)
    assert close(high, 0.5962, 1e-3)


def test_wilson_ci_zero_events_stays_in_unit_interval():
    # x=0, n=100 → 하한 0, 상한 ≈ z²/(n+z²) ≈ 0.0370
    low, high = default_rate_ci([0] * 100, 0.95)
    assert close(low, 0.0, 1e-12)
    assert close(high, 0.0370, 1e-3)


def test_wilson_ci_rejects_bad_confidence():
    with expect_raises(ValueError):
        default_rate_ci([0, 1], 1.5)


def test_sample_sufficiency_warns_on_few_events():
    result = sample_sufficiency([1] * 5 + [0] * 500, min_events=100)
    assert result["events"] == 5 and result["non_events"] == 500
    assert result["ok"] is False
    assert len(result["warnings"]) == 1


def test_sample_sufficiency_ok():
    result = sample_sufficiency([1] * 150 + [0] * 150, min_events=100, min_non_events=100)
    assert result["ok"] is True and result["warnings"] == []
