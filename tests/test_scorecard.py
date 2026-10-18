"""점수 변환·스코어카드·등급·컷오프 테스트."""
import math

import numpy as np
import pandas as pd

from creditlab.data import make_synthetic_credit_data
from creditlab.features import fit_binners, transform_woe_frame
from creditlab.models import fit_logistic, fit_regularized
from creditlab.scorecard import (
    ScoreScaling, Scorecard, assign_grades, cutoff_analysis, grade_monotonic, grade_table, make_grade_edges,
)
from helpers import close, expect_raises


def build():
    df = make_synthetic_credit_data(n=6000, seed=3).sort_values("application_date").reset_index(drop=True)
    cut = int(len(df) * 0.7)
    dev, test = df.iloc[:cut], df.iloc[cut:]
    cols = ["age", "card_utilization", "num_delinq_12m", "housing_type"]
    bs = fit_binners(dev, cols, "default", min_events=10)
    Wd = transform_woe_frame(dev, bs)
    fit = fit_logistic(Wd, dev["default"])
    return dev, test, cols, bs, Wd, fit


def test_scaling_numbers_and_roundtrip():
    sc = ScoreScaling()                                   # 600점 = 50:1, PDO 20
    assert close(sc.factor, 20 / math.log(2)) and abs(sc.factor - 28.8539) < 1e-4 and abs(sc.offset - 487.123) < 1e-3
    assert close(float(sc.score_from_odds_good(50)), 600.0)
    assert close(float(sc.score_from_odds_good(100)), 620.0) and close(float(sc.score_from_odds_good(25)), 580.0)   # PDO: odds 2배 = +20점
    assert close(float(sc.score_from_pd(1 / 51)), 600.0)
    s = np.array([520.0, 600.0, 640.0])
    assert np.allclose(sc.score_from_pd(sc.pd_from_score(s)), s)
    assert close(float(sc.pd_from_score(600)), 1 / 51)
    assert (np.diff(sc.pd_from_score(s)) < 0).all()      # 점수가 높을수록 부도확률이 낮다
    with expect_raises(ValueError):
        ScoreScaling(pdo=0)
    with expect_raises(ValueError):
        sc.score_from_pd([0.0])
    with expect_raises(ValueError):
        sc.score_from_odds_good([-1])


def test_scorecard_sum_equals_model_exactly():
    dev, test, cols, bs, Wd, fit = build()
    card = Scorecard.from_fit(bs, fit)
    for d in (dev, test):
        v = card.verify(d)
        assert v["ok"] and v["max_abs_diff"] < 1e-8
    # 점수 ↔ PD 관계: 모형 PD 와 점수에서 되돌린 PD 가 같다
    s = card.score(test)
    assert np.allclose(card.scaling.pd_from_score(s), card.predict_pd(test), atol=1e-9)
    assert np.allclose(card.predict_pd(dev), fit.predict_proba(Wd), atol=1e-9)


def test_rounded_points_within_bound_and_riskier_bin_gets_fewer_points():
    dev, test, cols, bs, Wd, fit = build()
    card = Scorecard.from_fit(bs, fit, round_points=True)
    v = card.verify(test)
    assert v["ok"] and v["max_abs_diff"] <= 0.5 * len(cols) and (card.table_["points"] % 1 == 0).all()
    exact = Scorecard.from_fit(bs, fit).table_
    for var, g in exact.groupby("variable"):
        g = g.sort_values("woe")
        assert (np.diff(g["points"]) <= 1e-12).all()     # WOE 가 클수록(위험) 점수는 낮다 (계수 > 0)


def test_unseen_values_get_neutral_points():
    dev, test, cols, bs, Wd, fit = build()
    card = Scorecard.from_fit(bs, fit)
    row = test.iloc[:1].copy()
    row["housing_type"] = "spaceship"                    # 학습에 없던 범주
    row["age"] = np.nan                                  # 학습에 결측이 없었다
    pts = card.points_frame(row).iloc[0]
    other = card.binners["housing_type"].table_
    expected = (card.table_.query("variable == 'housing_type' and bin == 'OTHER'")["points"].iloc[0]
                if "OTHER" in other["bin"].tolist() else card._neutral_points("housing_type"))
    assert close(pts["housing_type"], expected)               # 미지 범주는 OTHER(있으면) 또는 중립 점수
    assert close(pts["age"], card._neutral_points("age"))


def test_zero_coef_excluded_and_validation():
    dev, test, cols, bs, Wd, fit = build()
    coef = {c: float(fit.slopes[f"{c}_woe"]) for c in cols}
    coef["age"] = 0.0
    card = Scorecard(bs, coef, fit.intercept)
    assert "age" not in card.variables
    assert close(card.verify(test)["max_abs_diff"], 0.0, 1e-8)       # 사용 변수만으로 만든 모형 기준으로 점수와 일치
    assert set(card.table_["variable"]) == set(cols) - {"age"}
    with expect_raises(ValueError, "0 이 아닌"):
        Scorecard(bs, {c: 0.0 for c in cols}, 0.0)
    with expect_raises(ValueError, "비닝 정보"):
        Scorecard({}, {"age": 1.0}, 0.0)


def test_from_sklearn_close_to_irls():
    dev, test, cols, bs, Wd, fit = build()
    m = fit_regularized(Wd, dev["default"], "none")
    a = Scorecard.from_fit(bs, fit).score(test)
    b = Scorecard.from_sklearn(bs, list(Wd.columns), m).score(test)
    assert np.abs(a - b).max() < 1e-3


def test_grades_boundaries_and_labels():
    edges = [500.0, 600.0]
    g = assign_grades([499.9, 500.0, 599.9, 600.0, 700.0], edges)
    assert g.tolist() == ["C", "B", "B", "A", "A"]         # 경계값은 안전한(위) 등급, 높은 점수 = A
    assert list(g.cat.categories) == ["A", "B", "C"]
    assert assign_grades([1, 10], [5], labels=["우량", "위험"]).tolist() == ["위험", "우량"]
    with expect_raises(ValueError, "오름차순"):
        assign_grades([1], [600, 500])
    with expect_raises(ValueError, "라벨"):
        assign_grades([1], [5], labels=["x"])


def test_make_grade_edges_and_table():
    rng = np.random.default_rng(0)
    s = rng.normal(550, 40, 5000)
    e = make_grade_edges(s, 5)
    assert len(e) == 4 and e == sorted(e)
    counts = assign_grades(s, e).value_counts()
    assert counts.min() >= 990 and counts.max() <= 1010   # 동일 건수
    w = make_grade_edges(s, 4, "width")
    assert np.allclose(np.diff(w), np.diff(w)[0])
    y = (rng.uniform(size=5000) < 1 / (1 + np.exp((s - 520) / 20))).astype(int)    # 점수가 낮을수록 부도
    t = grade_table(s, y, e)
    assert list(t["grade"]) == ["A", "B", "C", "D", "E"] and t["n"].sum() == 5000 and close(t["pct"].sum(), 100.0)
    assert t["bad"].sum() == y.sum() and grade_monotonic(t) == "increasing"
    assert (t["bad_rate_lo"] <= t["bad_rate"]).all() and (t["bad_rate"] <= t["bad_rate_hi"]).all()
    assert (t["score_min"] <= t["score_max"]).all() and t.loc[0, "score_min"] > t.loc[4, "score_max"]
    with expect_raises(ValueError):
        make_grade_edges(s, 1)
    with expect_raises(ValueError):
        make_grade_edges(np.array([1.0] * 100), 5)


def test_cutoff_analysis_hand_example():
    s = np.arange(1, 11, dtype=float)
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    t = cutoff_analysis(s, y, [4, 2, 11]).set_index("cutoff")
    assert close(t.loc[4, "approval_rate"], 0.7) and t.loc[4, "bad_rate_approved"] == 0.0
    assert t.loc[4, "bad_rate_rejected"] == 1.0 and close(t.loc[4, "bads_rejected_pct"], 1.0)
    assert close(t.loc[2, "bad_rate_approved"], 2 / 9)   # 점수 2~10 승인: 9건 중 부도 2건(점수 2, 3) and close(t.loc[2, "bads_rejected_pct"], 1 / 3)
    assert t.loc[11, "n_approved"] == 0 and np.isnan(t.loc[11, "bad_rate_approved"])
    with expect_raises(ValueError):
        cutoff_analysis([1, 2], [0], [1])


def test_test_period_grades_use_dev_edges():
    dev, test, cols, bs, Wd, fit = build()
    card = Scorecard.from_fit(bs, fit)
    edges = make_grade_edges(card.score(dev), 5)
    t_dev = grade_table(card.score(dev), dev["default"], edges)
    t_test = grade_table(card.score(test), test["default"], edges)
    assert grade_monotonic(t_dev) == "increasing"
    assert t_test["n"].sum() == len(test) and t_test.loc[0, "bad_rate"] < t_test.loc[4, "bad_rate"]
