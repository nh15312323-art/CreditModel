"""머신러닝 챌린저·설명 도구 테스트."""
import numpy as np
import pandas as pd

from creditlab.challengers import (
    MLFeatureEncoder, fit_forest, fit_gbm, fit_gbm_early_stop, fit_tree, gini_impurity, monotone_violation,
    partial_dependence, permutation_importance, split_gain, staged_log_loss, tune_gbm_timecv,
)
from creditlab.data import make_synthetic_credit_data
from creditlab.evaluation import auroc
from helpers import close, expect_raises

NUM = ["age", "income_10k", "loan_amount_10k", "employment_years", "card_utilization", "num_delinq_12m", "inquiries_6m"]
CAT = ["housing_type", "region"]


def build(n=6000, seed=31, cut=4200):
    df = make_synthetic_credit_data(n=n, seed=seed).sort_values("application_date").reset_index(drop=True)
    enc = MLFeatureEncoder(NUM, CAT).fit(df.iloc[:cut])
    return df, enc, enc.transform(df.iloc[:cut]), enc.transform(df.iloc[cut:]), df.default.iloc[:cut], df.default.iloc[cut:]


def test_theory_gini_example():
    assert close(gini_impurity(4, 6), 0.48) and close(gini_impurity(1, 4), 0.32) and close(gini_impurity(3, 2), 0.48)
    assert close(split_gain((1, 4), (3, 2)), 0.08)
    assert gini_impurity(0, 0) == 0.0 and gini_impurity(5, 0) == 0.0


def test_encoder_learns_categories_from_dev_only():
    df = pd.DataFrame({"n": [1.0, np.nan, 3.0], "c": ["a", "b", None]})
    enc = MLFeatureEncoder(["n"], ["c"]).fit(df)
    new = pd.DataFrame({"n": [2.0], "c": ["zzz"]})          # 개발에 없던 범주
    t = enc.transform(new)
    assert np.isnan(t["c"].iloc[0]) and t["n"].iloc[0] == 2.0
    t2 = enc.transform(df)
    assert t2["c"].tolist()[:2] == [0.0, 1.0] and np.isnan(t2["c"].iloc[2]) and np.isnan(t2["n"].iloc[1])
    assert enc.cat_mask.tolist() == [False, True] and enc.feature_names == ["n", "c"]
    with expect_raises(ValueError, "없는 열"):
        MLFeatureEncoder(["x"]).fit(df)
    with expect_raises(RuntimeError):
        MLFeatureEncoder(["n"], ["c"]).transform(df)


def test_deep_tree_overfits_and_shallow_generalizes():
    df, enc, Xtr, Xte, ytr, yte = build()
    deep = fit_tree(Xtr.fillna(-1), ytr, max_depth=None, min_samples_leaf=1)
    shallow = fit_tree(Xtr.fillna(-1), ytr, max_depth=3, min_samples_leaf=50)
    a = lambda m, X, y: auroc(y, m.predict_proba(X.fillna(-1))[:, 1])
    assert a(deep, Xtr, ytr) > 0.99 and a(deep, Xte, yte) < a(deep, Xtr, ytr) - 0.2
    assert a(shallow, Xte, yte) > a(deep, Xte, yte)


def test_gbm_and_forest_beat_chance_on_holdout():
    df, enc, Xtr, Xte, ytr, yte = build()
    g = fit_gbm(Xtr, ytr, enc.cat_mask, max_iter=100)
    f = fit_forest(Xtr.fillna(-1), ytr, n_estimators=100)
    assert auroc(yte, g.predict_proba(Xte)[:, 1]) > 0.62 and auroc(yte, f.predict_proba(Xte.fillna(-1))[:, 1]) > 0.62


def test_monotone_constraint_is_enforced_and_unconstrained_may_not():
    df, enc, Xtr, Xte, ytr, yte = build()
    mono = fit_gbm(Xtr, ytr, enc.cat_mask, monotonic={"num_delinq_12m": 1, "card_utilization": 1}, max_iter=150)
    for var in ["num_delinq_12m", "card_utilization"]:
        vals = np.unique(np.quantile(Xte[var].dropna(), np.linspace(0.05, 0.95, 12)))
        pdp = partial_dependence(mono, Xte, var, values=vals)
        assert monotone_violation(pdp["mean_pd"], 1) == 0.0
    with expect_raises(ValueError, "없습니다"):
        fit_gbm(Xtr, ytr, enc.cat_mask, monotonic={"nope": 1})
    with expect_raises(ValueError, "+1"):
        fit_gbm(Xtr, ytr, enc.cat_mask, monotonic={"age": 2})


def test_monotone_violation_function():
    assert monotone_violation([1, 2, 3], 1) == 0.0
    assert close(monotone_violation([1, 3, 2.5, 4], 1), 0.5)
    assert close(monotone_violation([3, 1, 2], -1), 1.0)


def test_early_stopping_is_time_ordered_and_picks_interior_iteration():
    df, enc, Xtr, Xte, ytr, yte = build()
    m, curve, best = fit_gbm_early_stop(Xtr, ytr, enc.cat_mask, learning_rate=0.1, max_depth=4, max_iter=300, min_samples_leaf=20)
    assert 1 <= best <= 300 and best < 300                      # 과적합이 시작되는 지점에서 멈춘다
    assert curve["train_log_loss"].iloc[-1] < curve["train_log_loss"].iloc[0]       # 학습 손실은 계속 내려가고
    assert curve.loc[curve["iteration"] == best, "valid_log_loss"].iloc[0] == curve["valid_log_loss"].min()
    assert m.n_iter_ == best
    with expect_raises(ValueError, "너무 적"):
        fit_gbm_early_stop(Xtr.iloc[:60], ytr.iloc[:60], enc.cat_mask)


def test_staged_log_loss_shape():
    df, enc, Xtr, Xte, ytr, yte = build()
    m = fit_gbm(Xtr, ytr, enc.cat_mask, max_iter=20)
    c = staged_log_loss(m, Xtr, ytr, Xte, yte)
    assert len(c) == 20 and list(c.columns) == ["iteration", "train_log_loss", "valid_log_loss"]


def test_tune_gbm_timecv_returns_sorted_grid():
    df, enc, Xtr, Xte, ytr, yte = build(n=4000, cut=3000)
    t = tune_gbm_timecv(Xtr, ytr, {"max_depth": [2, 4], "learning_rate": [0.05, 0.2]}, enc.cat_mask, n_splits=3, max_iter=60)
    assert len(t) == 4 and t["mean_auc"].is_monotonic_decreasing and (t["n_folds"] == 3).all()


def test_permutation_importance_finds_signal_and_ignores_noise():
    df, enc, Xtr, Xte, ytr, yte = build()
    m = fit_gbm(Xtr, ytr, enc.cat_mask, max_iter=150)
    imp = permutation_importance(m, Xte, yte, n_repeats=5, seed=1)
    assert imp.iloc[0]["variable"] in {"card_utilization", "num_delinq_12m"}
    assert imp.loc[imp.variable == "income_10k", "importance_mean"].iloc[0] < 0.02
    assert abs(imp.attrs["baseline_metric"] - auroc(yte, m.predict_proba(Xte)[:, 1])) < 1e-12
    # 입력 불변
    before = Xte.copy()
    permutation_importance(m, Xte, yte, n_repeats=1)
    pd.testing.assert_frame_equal(before, Xte)


def test_partial_dependence_errors():
    df, enc, Xtr, Xte, ytr, yte = build(n=3000, cut=2000)
    m = fit_gbm(Xtr, ytr, enc.cat_mask, max_iter=20)
    with expect_raises(ValueError, "없는 변수"):
        partial_dependence(m, Xte, "nope")
