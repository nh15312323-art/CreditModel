"""로지스틱 회귀(IRLS 추론, 규제) 테스트."""
import math

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

from creditlab.models import (
    fit_logistic, fit_regularized, logit, odds, regularization_path, select_C_by_cv, sigmoid, single_variable_separation,
)
from helpers import close, expect_raises


def table_2x2():
    """x=1: 부도 30/정상 70, x=0: 부도 10/정상 90."""
    x = [1] * 100 + [0] * 100
    y = [1] * 30 + [0] * 70 + [1] * 10 + [0] * 90
    return pd.DataFrame({"x": x}), pd.Series(y)


def test_sigmoid_logit_odds():
    z = np.array([-3.0, 0.0, 2.5])
    assert np.allclose(logit(sigmoid(z)), z)
    assert close(float(sigmoid(0.0)), 0.5)
    with np.errstate(over="raise", invalid="raise", divide="raise"):     # 극단값에서도 overflow 없이 안정적(0 으로의 underflow 는 무해)
        assert float(sigmoid(1000.0)) == 1.0 and float(sigmoid(-1000.0)) == 0.0
    assert close(float(odds(0.2)), 0.25)
    with expect_raises(ValueError):
        logit([0.0])
    with expect_raises(ValueError):
        odds([1.0])


def test_two_by_two_matches_closed_form():
    X, y = table_2x2()
    f = fit_logistic(X, y)
    # 절편 = ln(10/90), 기울기 = ln 오즈비, 표준오차(Woolf) = sqrt(1/30+1/70+1/10+1/90)
    assert close(f.coef[0], math.log(10 / 90), 1e-7)
    assert close(f.coef[1], math.log((30 / 70) / (10 / 90)), 1e-7)
    assert close(f.se[1], math.sqrt(1 / 30 + 1 / 70 + 1 / 10 + 1 / 90), 1e-6)
    s = f.summary().set_index("variable")
    assert close(s.loc["x", "odds_ratio"], (30 / 70) / (10 / 90), 1e-6)
    assert s.loc["x", "ci_low"] < s.loc["x", "coef"] < s.loc["x", "ci_high"]
    assert close(s.loc["x", "p_value"], 2 * stats.norm.sf(abs(f.coef[1] / f.se[1])))
    assert f.converged and f.n == 200 and f.n_events == 40 and not f.warnings


def test_matches_sklearn_unpenalized_and_loglik():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(3000, 3)), columns=list("abc"))
    y = (rng.uniform(size=3000) < 1 / (1 + np.exp(-(-1 + X["a"] * 1.0 - X["b"] * 0.5)))).astype(int)
    f = fit_logistic(X, y)
    sk = LogisticRegression(C=np.inf, max_iter=10000, tol=1e-10).fit(X, y)
    assert np.allclose(f.coef[1:], sk.coef_[0], atol=1e-4) and close(f.coef[0], sk.intercept_[0], 1e-4)
    assert np.allclose(f.predict_proba(X), sk.predict_proba(X)[:, 1], atol=1e-4)
    assert f.loglik > f.null_loglik and 0 < f.pseudo_r2 < 1
    assert abs(f.coef[1] - 1.0) < 0.2 and abs(f.coef[2] + 0.5) < 0.2        # 진짜 값 근처 복원


def test_standard_errors_from_information_matrix():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(2000, 2)), columns=["a", "b"])
    y = (rng.uniform(size=2000) < 1 / (1 + np.exp(-(0.3 + X["a"])))).astype(int)
    f = fit_logistic(X, y)
    A = np.column_stack([np.ones(2000), X.to_numpy()])
    p = 1 / (1 + np.exp(-(A @ f.coef)))
    cov = np.linalg.inv(A.T @ (A * (p * (1 - p))[:, None]))
    assert np.allclose(f.se, np.sqrt(np.diag(cov)))


def test_separation_is_detected_and_warned():
    x = np.r_[np.linspace(0, 1, 50), np.linspace(2, 3, 50)]
    y = np.r_[np.zeros(50), np.ones(50)]
    X = pd.DataFrame({"x": x})
    assert single_variable_separation(X, y) == ["x"]
    f = fit_logistic(X, y)                              # 터지지 않고 경고를 낸다
    assert any("완전히 갈립니다" in w for w in f.warnings) and abs(f.coef[1]) > 10


def test_collinearity_and_epv_warnings():
    rng = np.random.default_rng(2)
    a = rng.normal(size=500)
    X = pd.DataFrame({"a": a, "dup": a})
    y = (rng.uniform(size=500) < 0.3).astype(int)
    with expect_raises(ValueError, "공선성"):
        fit_logistic(X, y)
    X2 = pd.DataFrame(rng.normal(size=(60, 8)), columns=list("abcdefgh"))
    y2 = np.r_[np.ones(10), np.zeros(50)].astype(int)
    f = fit_logistic(X2, y2)
    assert any("변수당 이벤트 수" in w for w in f.warnings)
    X3 = pd.DataFrame({"a": a, "b": a + 0.01 * rng.normal(size=500)})
    assert any("조건수" in w for w in fit_logistic(X3, y).warnings)


def test_input_validation():
    X, y = table_2x2()
    with expect_raises(ValueError, "결측"):
        fit_logistic(X.assign(x=[np.nan] + [1.0] * 199), y)
    with expect_raises(ValueError, "숫자형이 아닌"):
        fit_logistic(X.assign(x="a"), y)
    with expect_raises(ValueError, "길이"):
        fit_logistic(X, y.iloc[:-1])
    with expect_raises(ValueError, "모두 있어야"):
        fit_logistic(X, pd.Series([0] * 200))
    with expect_raises(ValueError, "결측"):
        fit_logistic(X, pd.Series([0, 1] * 99 + [np.nan, 1]))
    f = fit_logistic(X, y)
    with expect_raises(ValueError, "필요한 열"):
        f.predict_proba(pd.DataFrame({"zzz": [1.0]}))


def test_regularization_shrinks_and_l1_zeroes():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(2000, 5)), columns=list("abcde"))
    y = (rng.uniform(size=2000) < 1 / (1 + np.exp(-(-1 + X["a"] + 0.5 * X["b"])))).astype(int)
    none = fit_regularized(X, y, "none").coef_[0]
    l2 = fit_regularized(X, y, "l2", C=0.01).coef_[0]
    assert np.linalg.norm(l2) < np.linalg.norm(none)
    l1 = fit_regularized(X, y, "l1", C=0.02).coef_[0]
    assert (np.abs(l1) < 1e-8).sum() >= 2 and abs(l1[0]) > 1e-3        # 약한 변수는 0, 강한 변수는 남음
    en = fit_regularized(X, y, "elasticnet", C=0.02, l1_ratio=0.5).coef_[0]
    assert np.linalg.norm(en) < np.linalg.norm(none)
    with expect_raises(ValueError):
        fit_regularized(X, y, "bogus")
    path = regularization_path(X, y, [10, 0.1, 0.01, 0.001], penalty="l1")
    assert path["n_nonzero"].is_monotonic_decreasing and path["n_nonzero"].iloc[-1] <= path["n_nonzero"].iloc[0]


def test_select_C_by_cv_time_order():
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(3000, 4)), columns=list("abcd"))
    y = (rng.uniform(size=3000) < 1 / (1 + np.exp(-(-1.5 + X["a"])))).astype(int)
    t = select_C_by_cv(X, y, [1e-4, 1e-2, 1.0], penalty="l2", n_splits=4)
    assert t["best"].sum() == 1 and (t["n_folds"] == 4).all()
    assert t.loc[t["C"] == 1e-4, "mean_log_loss"].iloc[0] > t["mean_log_loss"].min()      # 너무 강한 규제는 손실이 크다
