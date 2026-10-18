"""보정기·스태킹·컷오프 정책·모형 비교 테스트."""
import math

import numpy as np
import pandas as pd

from creditlab.calibration import (InterceptCalibrator, IsotonicCalibrator, PlattCalibrator, StackedModel,
                                   oof_predictions, profit_by_cutoff, theoretical_cutoff)
from creditlab.challengers import MLFeatureEncoder, fit_gbm
from creditlab.data import make_synthetic_credit_data
from creditlab.evaluation import auroc, brier_score, compare_models, paired_bootstrap_diff
from creditlab.models import fit_regularized, logit, sigmoid
from helpers import close, expect_raises


def overconfident(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.beta(1.2, 12, n)
    y = (rng.random(n) < true).astype(int)
    p = sigmoid(2.0 * logit(true) - 0.3)          # 과신(기울기 2) + 약간 높음
    return p, y


def test_platt_fixes_overconfidence_and_keeps_ranking():
    p, y = overconfident()
    cal = PlattCalibrator().fit(p[:10000], y[:10000])
    q = cal.predict(p[10000:])
    assert brier_score(y[10000:], q) < brier_score(y[10000:], p[10000:])
    assert close(auroc(y[10000:], q), auroc(y[10000:], p[10000:]), 1e-12)         # 단조 변환
    assert 0.3 < cal.a_ < 0.7                                                       # 기울기 2 의 역수 근처


def test_isotonic_calibrates_and_may_create_ties():
    p, y = overconfident()
    cal = IsotonicCalibrator().fit(p[:10000], y[:10000])
    q = cal.predict(p[10000:])
    assert brier_score(y[10000:], q) < brier_score(y[10000:], p[10000:])
    assert len(np.unique(q)) < len(q)                                               # 계단 → 동점
    assert (q >= 0).all() and (q <= 1).all()
    assert np.all(np.diff(cal.predict(np.sort(p))) >= -1e-12)                       # 단조 증가


def test_intercept_calibrator_matches_target_rate():
    p, y = overconfident()
    c = InterceptCalibrator(target_rate=0.15).fit(p)
    assert close(c.predict(p).mean(), 0.15, 1e-6)
    c2 = InterceptCalibrator().fit(p, y)
    assert close(c2.predict(p).mean(), y.mean(), 1e-6)
    with expect_raises(ValueError):
        InterceptCalibrator(target_rate=1.2).fit(p)
    with expect_raises(ValueError, "필요"):
        InterceptCalibrator().fit(p)


def test_calibrator_input_validation():
    with expect_raises(ValueError, "0~1"):
        PlattCalibrator().fit([0.1, 1.5, 0.3, 0.2], [0, 1, 0, 1])
    with expect_raises(ValueError, "결측"):
        PlattCalibrator().fit([0.1, np.nan, 0.3, 0.2], [0, 1, 0, 1])


def build_ml(n=6000, cut=4200):
    df = make_synthetic_credit_data(n=n, seed=33).sort_values("application_date").reset_index(drop=True)
    num = ["age", "card_utilization", "num_delinq_12m", "inquiries_6m", "employment_years"]
    enc = MLFeatureEncoder(num).fit(df.iloc[:cut])
    return enc.transform(df.iloc[:cut]), enc.transform(df.iloc[cut:]), df.default.iloc[:cut], df.default.iloc[cut:]


class _LR:
    def __init__(self, X, y):
        self.m = fit_regularized(X.fillna(0), y, penalty="l2", C=1.0)

    def predict_proba(self, X):
        return self.m.predict_proba(X.fillna(0))


def test_oof_predictions_are_forward_only():
    Xtr, Xte, ytr, yte = build_ml()
    oof = oof_predictions({"lr": lambda X, y: _LR(X, y)}, Xtr, ytr, n_splits=4)
    n_nan = int(oof["lr"].isna().sum())
    assert 0 < n_nan < len(oof) / 2 and oof["lr"].iloc[:n_nan].isna().all() and oof["lr"].iloc[n_nan:].notna().all()


def test_stacked_model_predicts_and_weights_are_reported():
    Xtr, Xte, ytr, yte = build_ml()
    st = StackedModel({"lr": lambda X, y: _LR(X, y), "gb": lambda X, y: fit_gbm(X, y, max_iter=60)}, n_splits=3).fit(Xtr, ytr)
    p = st.predict_proba(Xte)[:, 1]
    assert p.shape == (len(Xte),) and ((p > 0) & (p < 1)).all()
    assert set(st.meta_weights_) == {"lr", "gb"} and 0 < st.n_meta_ < len(Xtr)
    assert auroc(yte, p) > 0.62


def test_theoretical_cutoff_and_profit_curve():
    assert close(theoretical_cutoff(1, 8), 1 / 9)
    with expect_raises(ValueError):
        theoretical_cutoff(0, 8)
    rng = np.random.default_rng(0)
    p = rng.beta(1, 12, 30000)
    y = (rng.random(30000) < p).astype(int)               # 완벽히 보정된 확률
    cands = np.linspace(0.02, 0.4, 60)
    t = profit_by_cutoff(p, 1.0, 8.0, cands, y)
    best_exp = t.loc[t["expected_profit"].idxmax(), "cutoff"]
    assert abs(best_exp - 1 / 9) < 0.02                   # 이론 컷오프 근처에서 기대이익이 최대
    assert abs(t.loc[t["realized_profit"].idxmax(), "cutoff"] - 1 / 9) < 0.04
    # 과대예측(보정 깨짐)이면 최적 컷오프가 이론값에서 벗어난다
    t2 = profit_by_cutoff(np.clip(2 * p, 0, 1), 1.0, 8.0, cands, y)
    assert t2.loc[t2["expected_profit"].idxmax(), "cutoff"] > 0 and (t2["approval_rate"].diff().dropna() >= 0).all()


def test_paired_bootstrap_and_compare_models():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 800)
    s = rng.normal(size=800) + y
    a, b = s, s + rng.normal(scale=0.05, size=800)        # 거의 같은 두 모형
    r = paired_bootstrap_diff(auroc, y, a, b, n_boot=200, seed=2)
    assert r["includes_zero"] and abs(r["diff"]) < 0.02 and r["n_valid"] == 200
    c = rng.normal(size=800)                               # 무작위 모형
    r2 = paired_bootstrap_diff(auroc, y, a, c, n_boot=200, seed=2)
    assert not r2["includes_zero"] and r2["lo"] > 0
    with expect_raises(ValueError, "길이"):
        paired_bootstrap_diff(auroc, y, a[:5], b)
    p1, p2 = sigmoid(s - 2), sigmoid(0.5 * s - 2)
    tab = compare_models(y, {"A": p1, "B": p2})
    assert list(tab.index) == ["A", "B"] and {"auroc", "ece", "cal_slope"} <= set(tab.columns)
    assert close(tab.loc["A", "auroc"], tab.loc["B", "auroc"], 1e-12)
