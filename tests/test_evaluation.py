"""평가 지표(변별력·보정)와 안정성(PSI/CSI) 테스트."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve

from creditlab.data import make_synthetic_credit_data
from creditlab.evaluation import (
    auroc, average_precision, bootstrap_ci, brier_score, calibration_slope_intercept, calibration_table,
    cap_accuracy_ratio, confusion_at, csi_table, discrimination_summary, expected_calibration_error,
    expected_psi_noise, fixed_edges, gini, ks_statistic, log_loss_score, metrics_by_period, psi_categorical,
    psi_from_bins, psi_from_counts, psi_label, psi_numeric, psi_terms, roc_points,
)
from creditlab.features import fit_binners
from helpers import close, expect_raises

S = np.array([0.9, 0.7, 0.4, 0.8, 0.5, 0.3, 0.2, 0.1])
Y = np.array([1, 1, 1, 0, 0, 0, 0, 0])


def test_hand_example_matches_theory_page():
    assert close(auroc(Y, S), 0.8) and close(gini(Y, S), 0.6)
    assert close(ks_statistic(Y, S)["ks"], 0.6)
    assert close(brier_score(Y, S), 0.18625)
    assert abs(log_loss_score(Y, S) - 0.5458) < 1e-4
    c = confusion_at(Y, S, 0.45)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 2, 1, 3)
    assert close(c["f1"], 4 / 7)


def test_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 600)
    s = np.round(rng.normal(size=600) + y, 1)          # 동점이 많다
    assert close(auroc(y, s), roc_auc_score(y, s), 1e-12)
    assert close(average_precision(y, s), average_precision_score(y, s), 1e-12)
    fpr, tpr, _ = roc_curve(y, s)
    assert close(ks_statistic(y, s)["ks"], (tpr - fpr).max(), 1e-12)
    p = 1 / (1 + np.exp(-s))
    assert close(brier_score(y, p), brier_score_loss(y, p), 1e-12)
    assert close(log_loss_score(y, p), log_loss(y, p), 1e-9)


def test_auroc_pairwise_definition_and_monotone_invariance():
    bad, good = S[Y == 1], S[Y == 0]
    pairs = [(b > g) + 0.5 * (b == g) for b in bad for g in good]
    assert close(np.mean(pairs), auroc(Y, S))
    assert close(auroc(Y, S), auroc(Y, 2 * S)) and close(auroc(Y, S), auroc(Y, np.exp(S)))
    assert close(auroc(Y, -S), 1 - auroc(Y, S))           # 방향을 뒤집으면 1−AUROC


def test_ar_equals_gini_without_ties():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    s = rng.normal(size=400) + y                          # 연속값이라 동점 없음
    assert close(cap_accuracy_ratio(y, s), gini(y, s), 1e-12)


def test_one_class_is_nan_not_a_number():
    y = np.zeros(10, dtype=int)
    s = np.linspace(0, 1, 10)
    assert np.isnan(auroc(y, s)) and np.isnan(gini(y, s)) and np.isnan(cap_accuracy_ratio(y, s))
    assert np.isnan(discrimination_summary(y, s)["auroc"])
    with expect_raises(ValueError, "ROC"):
        roc_points(y, s)


def test_input_validation():
    with expect_raises(ValueError, "길이"):
        auroc([0, 1], [0.1])
    with expect_raises(ValueError, "결측"):
        auroc([0, 1], [0.1, np.nan])
    with expect_raises(ValueError):
        auroc([0, 2], [0.1, 0.2])
    with expect_raises(ValueError, "0~1"):
        brier_score([0, 1], [0.1, 1.5])


def test_confusion_depends_on_threshold_and_imbalance():
    y = np.r_[np.ones(1), np.zeros(99)].astype(int)       # 부도율 1%
    s = np.zeros(100)
    c = confusion_at(y, s, 1.0)                           # 모두 정상 예측
    assert close(c["accuracy"], 0.99) and close(c["recall"], 0.0) and np.isnan(c["precision"])


def test_average_precision_random_equals_base_rate_roughly():
    rng = np.random.default_rng(3)
    y = (rng.random(20000) < 0.05).astype(int)
    assert abs(average_precision(y, rng.random(20000)) - 0.05) < 0.01


def test_bootstrap_ci_contains_point_and_reports_skips():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 300)
    s = rng.normal(size=300) + y
    r = bootstrap_ci(auroc, y, s, n_boot=200, seed=1)
    assert r["lo"] < r["point"] < r["hi"] and r["n_valid"] == 200
    # 부도 1건뿐 → 부트스트랩 표본 상당수가 한 클래스. 건너뛴 횟수가 보고된다.
    y2 = np.r_[1, np.zeros(19)].astype(int)
    r2 = bootstrap_ci(auroc, y2, rng.normal(size=20), n_boot=200, seed=1)
    assert r2["skipped"] > 0 and r2["n_valid"] + r2["skipped"] == 200


def test_metrics_by_period():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 400)
    s = rng.normal(size=400) + y
    df = pd.DataFrame({"default": y})
    out = metrics_by_period(df, np.repeat(["a", "b"], 200), "default", s)
    assert list(out["period"]) == ["a", "b"] and out["n"].tolist() == [200, 200]


def test_calibration_table_and_slope():
    rng = np.random.default_rng(6)
    p = rng.beta(1, 15, 20000)
    y = (rng.random(20000) < p).astype(int)               # 완벽히 보정된 확률
    t = calibration_table(y, p, n_bins=10)
    assert t["n"].sum() == 20000 and expected_calibration_error(t) < 0.01
    cs = calibration_slope_intercept(y, p)
    assert abs(cs["slope"] - 1) < 0.1 and abs(cs["intercept"]) < 0.1
    # 확률에 2를 곱하면 AUROC 는 그대로, 보정은 깨진다
    p2 = np.clip(2 * p, 0, 1)
    assert close(auroc(y, p), auroc(y, p2), 1e-3)
    assert expected_calibration_error(calibration_table(y, p2)) > 5 * expected_calibration_error(t)
    assert calibration_slope_intercept(y, p2)["intercept"] < -0.4   # 예측이 너무 높음 → 절편이 음수
    with expect_raises(ValueError):
        calibration_table(y, p, strategy="bad")


def test_psi_hand_example_and_symmetry():
    e = np.array([0.10, 0.20, 0.40, 0.20, 0.10])
    a = np.array([0.08, 0.17, 0.38, 0.24, 0.13])
    t = psi_terms(e, a)
    assert np.allclose(np.round(t, 4), [0.0045, 0.0049, 0.0010, 0.0073, 0.0079])
    assert abs(t.sum() - 0.0255) < 5e-5
    assert close(t.sum(), psi_terms(a, e).sum())          # PSI 는 기준/비교를 바꿔도 같다
    assert close(psi_from_counts(e * 1000, a * 500), t.sum())   # 건수 규모와 무관
    assert close(psi_terms(e, e).sum(), 0.0)


def test_psi_zero_bin_floor_and_labels():
    v_small = psi_terms([0.5, 0.5], [1.0, 0.0], eps=1e-4).sum()
    v_big = psi_terms([0.5, 0.5], [1.0, 0.0], eps=1e-2).sum()
    assert np.isfinite(v_small) and v_small > v_big       # eps 선택에 따라 값이 크게 달라진다
    assert psi_label(0.05).startswith("안정") and psi_label(0.15).startswith("주의") and psi_label(0.3).startswith("큰")
    assert psi_label(float("nan")) == "계산불가"


def test_psi_numeric_fixed_edges_shift_and_noise():
    df = make_synthetic_credit_data(n=8000, seed=21).sort_values("application_date")
    dev, now = df.iloc[:4000], df.iloc[4000:]
    same = psi_numeric(dev["card_utilization"], now["card_utilization"])
    shifted = psi_numeric(dev["card_utilization"], now["card_utilization"] + 0.15)
    assert same["psi"] < 0.05 and shifted["psi"] > 0.25
    assert abs(same["table"]["expected_pct"].sum() - 1) < 1e-12 and abs(same["table"]["actual_pct"].sum() - 1) < 1e-12
    # 현재 표본으로 경계를 다시 만들면 이동이 사라진다(잘못된 방법)
    wrong = psi_numeric(now["card_utilization"] + 0.15, now["card_utilization"] + 0.15)
    assert wrong["psi"] < 1e-9
    ed = fixed_edges(dev["card_utilization"])
    assert ed[0] == -np.inf and ed[-1] == np.inf
    # 표본이 작으면 변화가 없어도 PSI 가 커진다
    rng = np.random.default_rng(5)
    vals = [psi_numeric(rng.normal(size=1000), rng.normal(size=1000))["psi"] for _ in range(200)]
    assert 0.010 < np.mean(vals) < 0.030
    assert close(expected_psi_noise(10, 1000, 1000), 0.018)


def test_psi_numeric_missing_bin():
    x = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 20)
    cur = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8, np.nan, np.nan] * 20)
    r = psi_numeric(x, cur, bins=5)
    assert "MISSING" in r["table"]["bin"].tolist() and r["psi"] > 0.1


def test_psi_categorical_new_category():
    e = pd.Series(["A"] * 50 + ["B"] * 50)
    a = pd.Series(["A"] * 40 + ["B"] * 40 + ["C"] * 20)
    r = psi_categorical(e, a)
    assert "C" in r["table"]["bin"].tolist() and r["psi"] > 0.1
    assert close(psi_categorical(e, e)["psi"], 0.0)


def test_csi_table_uses_dev_binners():
    df = make_synthetic_credit_data(n=6000, seed=8).sort_values("application_date")
    dev, now = df.iloc[:3500], df.iloc[3500:].copy()
    cols = ["age", "card_utilization", "housing_type"]
    bs = fit_binners(dev, cols, "default", min_events=10)
    base = csi_table(dev, now, bs)
    assert base["psi"].max() < 0.1
    now["card_utilization"] = now["card_utilization"] + 0.2
    moved = csi_table(dev, now, bs)
    assert moved.iloc[0]["variable"] == "card_utilization" and moved.iloc[0]["psi"] > 0.25
    assert moved["psi"].is_monotonic_decreasing
