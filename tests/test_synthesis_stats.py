"""합성데이터(코퓰러·TSTR·DCR)와 통계검정(순열·DeLong·다중검정) 테스트."""
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

from creditlab.data import make_synthetic_credit_data
from creditlab.stats import adjust_pvalues, delong_test, familywise_error_rate, multiple_testing_table, permutation_test
from creditlab.stats.tests import _components
from creditlab.synthesis import (GaussianCopulaSynthesizer, correlation_gap, dcr, dcr_summary, fidelity_table,
                                 target_relationship, tstr)
from helpers import close, expect_raises

COLS = ["age", "income_10k", "loan_amount_10k", "employment_years", "card_utilization", "num_delinq_12m", "inquiries_6m", "default"]


def real_data(seed=51, n=8000):
    return make_synthetic_credit_data(n=n, seed=seed).dropna()[COLS].reset_index(drop=True)


def test_copula_preserves_marginals_correlation_and_discreteness():
    real = real_data()
    tr = real.iloc[:5000]
    syn = GaussianCopulaSynthesizer(0).fit(tr).sample(5000)
    assert list(syn.columns) == COLS and len(syn) == 5000
    assert set(syn["default"].unique()) <= {0.0, 1.0} and abs(syn["default"].mean() - tr["default"].mean()) < 0.02
    assert set(syn["num_delinq_12m"].unique()) <= set(tr["num_delinq_12m"].unique())          # 이산 열은 관측된 값만
    assert correlation_gap(tr.drop(columns="default"), syn.drop(columns="default"))["max_abs"] < 0.1
    ft = fidelity_table(tr, syn)
    assert (ft["ks"] < 0.06).all() and len(ft) == len(COLS)
    # 같은 시드면 재현, 다른 시드면 다름
    s = GaussianCopulaSynthesizer(0).fit(tr)
    pd.testing.assert_frame_equal(s.sample(100, seed=1), s.sample(100, seed=1))
    assert not s.sample(100, seed=1).equals(s.sample(100, seed=2))


def test_copula_input_validation():
    with expect_raises(ValueError, "수치형"):
        GaussianCopulaSynthesizer().fit(pd.DataFrame({"a": ["x"] * 40, "b": range(40)}))
    with expect_raises(ValueError, "결측"):
        GaussianCopulaSynthesizer().fit(pd.DataFrame({"a": [1.0, np.nan] * 20, "b": range(40)}))
    with expect_raises(ValueError, "30"):
        GaussianCopulaSynthesizer().fit(pd.DataFrame({"a": [1.0, 2.0]}))
    with expect_raises(RuntimeError):
        GaussianCopulaSynthesizer().sample(5)


def test_copula_weakens_target_relationship_but_tstr_stays_close():
    real = real_data()
    tr, ho = real.iloc[:5000], real.iloc[5000:]
    syn = GaussianCopulaSynthesizer(0).fit(tr).sample(5000)
    r = tstr(tr, syn, ho, COLS[:-1], "default")
    assert r["tstr"] > 0.55 and r["tstr"] < r["trtr"] + 0.03 and close(r["gap"], r["trtr"] - r["tstr"])
    t = target_relationship(tr, syn, "card_utilization", "default")
    assert len(t) == 5 and "gap" in t.columns


def test_dcr_detects_copies_and_not_copula():
    real = real_data()
    tr, ho = real.iloc[:5000], real.iloc[5000:]
    feats = COLS[:-1]
    naive = tr.sample(500, random_state=0)
    assert dcr(tr, naive, feats).max() == 0
    syn = GaussianCopulaSynthesizer(0).fit(tr).sample(2000)
    s = dcr_summary(tr, syn, ho, feats)
    assert s["exact_copies"] == 0 and s["synth_min"] > 0
    assert dcr_summary(tr, naive, ho, feats)["exact_copies"] == 500
    assert s["share_closer_than_holdout_p5"] < 0.2          # 외운 합성이 아니라면 보류 데이터보다 훨씬 가깝지 않다


def test_oversampling_before_split_leaks():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    real = real_data(n=4000)
    feat = COLS[:-1]
    Xa, Xb, ya, yb = train_test_split(real[feat], real["default"], test_size=0.3, random_state=0, stratify=real["default"])
    honest = roc_auc_score(yb, RandomForestClassifier(100, random_state=0).fit(Xa, ya).predict_proba(Xb)[:, 1])
    dup = pd.concat([real] + [real[real["default"] == 1]] * 4, ignore_index=True)
    Xc, Xd, yc, yd = train_test_split(dup[feat], dup["default"], test_size=0.3, random_state=0, stratify=dup["default"])
    leaky = roc_auc_score(yd, RandomForestClassifier(100, random_state=0).fit(Xc, yc).predict_proba(Xd)[:, 1])
    assert leaky > honest + 0.1


def test_permutation_test_matches_welch_and_has_no_zero_p():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 60), rng.normal(0.5, 1, 60)
    r = permutation_test(a, b, n_perm=4000, seed=1)
    assert abs(r["p_value"] - sps.ttest_ind(b, a, equal_var=False).pvalue) < 0.05
    big = permutation_test(np.zeros(20), np.ones(20) * 100 + rng.normal(size=20), n_perm=200, seed=1)
    assert big["p_value"] == 1 / 201                         # 0 이 아니라 1/(n_perm+1)
    with expect_raises(ValueError):
        permutation_test([1.0], [2.0, 3.0])


def test_delong_components_and_hand_example():
    s = np.array([0.9, 0.7, 0.4, 0.8, 0.5, 0.3, 0.2, 0.1])
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    v10, v01, auc = _components(y, s)
    assert np.allclose(v10, [1.0, 0.8, 0.6]) and close(auc, 0.8) and close(v01.mean(), 0.8)


def test_delong_properties():
    rng = np.random.default_rng(3)
    y = (rng.random(3000) < 0.1).astype(int)
    base = rng.normal(size=3000) + 0.9 * y
    s1 = base + rng.normal(scale=0.5, size=3000)
    s2 = base + rng.normal(scale=1.2, size=3000)
    r = delong_test(y, s1, s2)
    assert close(r["auc1"], roc_auc_score(y, s1), 1e-12) and close(r["auc2"], roc_auc_score(y, s2), 1e-12)
    rev = delong_test(y, s2, s1)
    assert close(r["z"], -rev["z"], 1e-9) and close(r["p_value"], rev["p_value"], 1e-12)       # 대칭성
    mono = delong_test(y, s1, np.exp(s2 / 3))
    assert close(r["z"], mono["z"], 1e-9)                                                        # 단조 변환 불변
    assert r["ci_lo"] < r["diff"] < r["ci_hi"] and r["p_value"] < 0.05
    same = delong_test(y, s1, s1)
    assert same["p_value"] == 1.0 and same["diff"] == 0.0
    with expect_raises(ValueError, "2건"):
        delong_test([1, 0, 0, 0], [0.9, 0.1, 0.2, 0.3], [0.8, 0.2, 0.1, 0.3])
    with expect_raises(ValueError, "길이"):
        delong_test(y, s1[:5], s2)


def test_delong_matches_pairwise_matrix_implementation():
    rng = np.random.default_rng(4)
    y = (rng.random(300) < 0.3).astype(int)
    s = rng.normal(size=300) + y
    psi = lambda p, n: ((p[:, None] - n[None, :]) > 0) + 0.5 * ((p[:, None] - n[None, :]) == 0)
    k = psi(s[y == 1], s[y == 0])
    v10, v01, auc = _components(y, s)
    assert np.allclose(v10, k.mean(axis=1)) and np.allclose(v01, k.mean(axis=0)) and close(auc, k.mean())


def test_multiple_testing_theory_numbers():
    assert close(familywise_error_rate(20), 0.6415, 1e-4)
    pv = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    t = multiple_testing_table(pv)
    assert (int(t["raw_reject"].sum()), int(t["bonferroni_reject"].sum()), int(t["holm_reject"].sum()), int(t["bh_reject"].sum())) == (5, 1, 1, 2)
    for m in ["bonferroni", "holm", "bh"]:
        adj = adjust_pvalues(pv, m)
        assert (adj >= np.asarray(pv) - 1e-12).all() and (adj <= 1).all()
    assert (adjust_pvalues(pv, "bonferroni") >= adjust_pvalues(pv, "holm") - 1e-12).all()
    assert (adjust_pvalues(pv, "holm") >= adjust_pvalues(pv, "bh") - 1e-12).all()         # 보수성 순서
    with expect_raises(ValueError):
        adjust_pvalues([0.1, 1.5])
    with expect_raises(ValueError, "method"):
        adjust_pvalues([0.1], "x")
    with expect_raises(ValueError):
        familywise_error_rate(0)


def test_null_pvalues_false_positive_rate_and_bh_control():
    rng = np.random.default_rng(7)
    any_sig, bh_any = [], []
    for _ in range(1500):
        p = rng.uniform(size=20)
        any_sig.append((p < 0.05).any())
        bh_any.append((adjust_pvalues(p, "bh") < 0.05).any())
    assert abs(np.mean(any_sig) - 0.6415) < 0.04
    assert np.mean(bh_any) < 0.08                              # 모두 귀무일 때 BH 는 FWER 도 약 5% 근처
