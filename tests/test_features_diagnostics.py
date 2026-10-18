"""변수 진단 테스트."""
import numpy as np
import pandas as pd
from scipy import stats

from creditlab.features import (
    VariableLog, compare_groups, condition_number, correlation_matrix, cramers_v, high_correlation_pairs,
    leakage_suspects, single_feature_auc, univariate_categorical, univariate_numeric, vif_table,
)
from helpers import close, expect_raises


def test_vif_matches_closed_form_for_two_variables():
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=5000)
    x2 = 0.8 * x1 + 0.6 * rng.normal(size=5000)
    df = pd.DataFrame({"x1": x1, "x2": x2})
    r = np.corrcoef(x1, x2)[0, 1]
    t = vif_table(df, ["x1", "x2"])
    assert close(t["vif"][0], 1 / (1 - r**2), 1e-8) and close(t["vif"][1], t["vif"][0], 1e-8)
    assert close(t["r2"][0], r**2, 1e-9)


def test_vif_independent_and_perfect_collinearity():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=4000), "b": rng.normal(size=4000)})
    assert (vif_table(df, ["a", "b"])["vif"] < 1.01).all()
    df["c"] = 2 * df["a"] - df["b"]
    t = vif_table(df, ["a", "b", "c"])
    assert np.isinf(t["vif"]).all() and (t["flag"] == "높음(>10)").all()


def test_vif_validation_and_missing_rows():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    df.loc[:9, "a"] = np.nan
    assert vif_table(df, ["a", "b"])["n_used"][0] == 190
    with expect_raises(ValueError, "2개 이상"):
        vif_table(df, ["a"])
    with expect_raises(ValueError, "하나뿐"):
        vif_table(df.assign(k=1.0), ["a", "b", "k"])
    with expect_raises(ValueError, "숫자형이 아닌"):
        vif_table(df.assign(s="x"), ["a", "s"])
    with expect_raises(ValueError, "없는 열"):
        vif_table(df, ["a", "zzz"])


def test_condition_number():
    rng = np.random.default_rng(3)
    ind = pd.DataFrame(rng.normal(size=(5000, 3)), columns=list("abc"))
    assert condition_number(ind, list("abc")) < 1.2
    col = ind.assign(d=ind["a"] + 1e-3 * rng.normal(size=5000))
    assert condition_number(col, list("abcd")) > 30


def test_correlation_and_pairs():
    x = np.arange(100, dtype=float)
    df = pd.DataFrame({"x": x, "y": x**3, "z": np.random.default_rng(0).normal(size=100)})
    sp = correlation_matrix(df, ["x", "y", "z"], "spearman")
    pe = correlation_matrix(df, ["x", "y", "z"], "pearson")
    assert close(sp.loc["x", "y"], 1.0) and pe.loc["x", "y"] < 0.97     # 단조 비선형: Spearman 1, Pearson 은 더 낮다
    pairs = high_correlation_pairs(pe, 0.9)
    assert list(pairs[["var1", "var2"]].iloc[0]) == ["x", "y"] and len(pairs) == 1
    with expect_raises(ValueError):
        correlation_matrix(df, ["x"], "kendall")
    u = pd.DataFrame({"u": np.linspace(-1, 1, 201)})
    u["v"] = u["u"] ** 2
    assert abs(correlation_matrix(u, ["u", "v"]).loc["u", "v"]) < 1e-9     # U자 관계: 상관 0 이어도 관계는 완전


def test_cramers_v():
    a = ["x"] * 50 + ["y"] * 50
    assert close(cramers_v(a, a), 1.0)
    rng = np.random.default_rng(0)
    assert cramers_v(rng.choice(list("ab"), 5000), rng.choice(list("cd"), 5000)) < 0.05
    assert cramers_v(["a", "a"], ["b", "c"]) == 0.0


def test_compare_groups_hand_example_and_scipy():
    df = pd.DataFrame({"x": [3.0, 4, 5, 1, 2, 3], "t": [1, 1, 1, 0, 0, 0]})
    r = compare_groups(df, "x", "t")
    assert close(r["diff"], 2.0) and close(r["cohens_d"], 2.0)       # 평균 4 vs 2, 표준편차 모두 1
    assert close(r["welch_p"], stats.ttest_ind([3, 4, 5], [1, 2, 3], equal_var=False).pvalue)
    assert r["ci_low"] < r["diff"] < r["ci_high"]
    with expect_raises(ValueError, "최소 2건"):
        compare_groups(pd.DataFrame({"x": [1.0, 2.0, 3.0], "t": [1, 0, 0]}), "x", "t")


def test_univariate_tables():
    df = pd.DataFrame({"x": [1.0, 2, 3, 4, np.nan, 6], "t": [0, 0, 1, 1, 0, 1], "c": ["a", "a", "b", "b", None, "b"]})
    u = univariate_numeric(df, ["x"], "t").iloc[0]
    assert u["n"] == 5 and close(u["missing_pct"], 100 / 6) and close(u["mean"], 3.2)
    assert u["n_bad"] == 3 and u["n_good"] == 2 and close(u["mean_bad"], (3 + 4 + 6) / 3) and close(u["mean_good"], 1.5)
    c = univariate_categorical(df, "c", "t", min_count=2).set_index("category")
    assert c.loc["b", "n"] == 3 and close(c.loc["b", "bad_rate"], 1.0) and c.loc["MISSING", "rare"]
    assert not c.loc["a", "rare"] and 0 <= c.loc["a", "bad_rate_lo"] <= c.loc["a", "bad_rate_hi"] <= 1
    with expect_raises(ValueError, "숫자형이 아닌"):
        univariate_numeric(df, ["c"], "t")


def test_leakage_suspects():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.2, 3000)
    df = pd.DataFrame({
        "income": rng.normal(size=3000),
        "recovery_amount": rng.normal(size=3000),          # 이름 규칙에 걸림
        "balance_after_default": y * 5 + rng.normal(0, 0.1, 3000),   # 이름 + 단일 변수 AUROC 모두
        "sneaky": y + rng.normal(0, 0.05, 3000),           # 이름은 평범하지만 거의 타깃
        "default": y,
    })
    s = leakage_suspects(df, "default")
    flagged = set(s["variable"])
    assert flagged == {"recovery_amount", "balance_after_default", "sneaky"}
    assert "income" not in flagged and "default" not in flagged
    assert close(single_feature_auc(df, "sneaky", "default"), 1.0, 0.01)
    assert single_feature_auc(df, "income", "default") < 0.55


def test_variable_log_requires_reasons():
    log = VariableLog()
    log.add("age", "keep", "업무 논리상 연령은 위험과 관련, 결측 0%", evidence="IV 0.05")
    with expect_raises(ValueError, "이유"):
        log.add("income", "drop", "   ")
    with expect_raises(ValueError, "decision"):
        log.add("income", "maybe", "x")
    with expect_raises(ValueError, "이미 기록"):
        log.add("age", "drop", "다시")
    log.add("age", "review", "연령 사용 가능 여부 법무 확인 필요", replace=True)
    log.add("loan", "keep", "신청 시점에 알 수 있음")
    f = log.to_frame()
    assert list(f["variable"]) == ["age", "loan"] and f.loc[0, "decision"] == "review"
    assert log.kept() == ["loan"] and log.undecided(["age", "loan", "zzz"]) == ["zzz"]
