"""비닝·WOE·IV 테스트. 손으로 계산 가능한 작은 예제와 비교한다."""
import math

import numpy as np
import pandas as pd

from creditlab.features import WoeBinner, fit_binners, iv_label, iv_summary, monotonic_direction, transform_woe_frame, woe_iv_from_counts
from helpers import close, expect_raises


def toy_category():
    """hi: 부도 8/정상 2, lo: 부도 2/정상 8 (총 부도 10, 정상 10)."""
    x = ["hi"] * 10 + ["lo"] * 10
    y = [1] * 8 + [0] * 2 + [1] * 2 + [0] * 8
    return pd.Series(x), pd.Series(y)


def test_woe_iv_hand_calculation_no_smoothing():
    # hi: %bad=0.8, %good=0.2 → WOE=ln(4)=1.386294, lo: ln(0.25)=-1.386294
    # IV = (0.8-0.2)*ln4 + (0.2-0.8)*ln(0.25) = 1.2*ln4 = 1.663553
    t = woe_iv_from_counts([8, 2], [2, 8], smoothing=0)
    assert close(t["woe"][0], math.log(4)) and close(t["woe"][1], -math.log(4))
    assert close(t["iv"].sum(), 1.2 * math.log(4))
    assert close(t["pct_bad"].sum(), 1.0) and close(t["pct_good"].sum(), 1.0)


def test_woe_positive_means_riskier():
    x, y = toy_category()
    b = WoeBinner(min_bin_pct=0.1, smoothing=0).fit(x, y)
    assert b.woe_by_bin_["hi"] > 0 > b.woe_by_bin_["lo"]
    assert close(b.iv_, 1.2 * math.log(4))


def test_smoothing_formula():
    # a=0.5, k=2: %bad_hi=(8+.5)/(10+1)=0.772727, %good_hi=(2+.5)/(10+1)=0.227273
    t = woe_iv_from_counts([8, 2], [2, 8], smoothing=0.5)
    assert close(t["pct_bad"][0], 8.5 / 11) and close(t["pct_good"][0], 2.5 / 11)
    assert close(t["woe"][0], math.log(8.5 / 2.5))


def test_zero_cell_requires_smoothing():
    with expect_raises(ValueError, "정의되지 않습니다"):
        woe_iv_from_counts([0, 10], [5, 5], smoothing=0)
    t = woe_iv_from_counts([0, 10], [5, 5], smoothing=0.5)
    assert np.isfinite(t["woe"]).all()


def test_input_validation():
    with expect_raises(ValueError):
        woe_iv_from_counts([1, 2], [1], 0.5)
    with expect_raises(ValueError, "부도와 정상이 모두"):
        woe_iv_from_counts([0, 0], [3, 3], 0.5)
    with expect_raises(ValueError):
        WoeBinner(method="bogus")
    with expect_raises(ValueError):
        WoeBinner(method="manual")
    with expect_raises(ValueError, "결측"):
        WoeBinner().fit(pd.Series([1.0, 2.0, 3.0]), pd.Series([0, 1, np.nan]))
    with expect_raises(ValueError, "모두 있어야"):
        WoeBinner().fit(pd.Series([1.0, 2.0, 3.0]), pd.Series([0, 0, 0]))
    with expect_raises(RuntimeError):
        WoeBinner().transform(pd.Series([1.0]))


def test_numeric_manual_edges_right_closed():
    x = pd.Series([1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 10], dtype=float)
    y = pd.Series([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1])
    b = WoeBinner(method="manual", edges=[3, 7], smoothing=0.5).fit(x, y)
    t = b.table_
    # (-inf,3]: 1,2,3,3 → 4건 / (3,7]: 4,5,6,7 → 4건 / (7,inf): 8,9,10,10 → 4건. 경계값 3, 7 은 아래 bin
    assert list(t["n"]) == [4, 4, 4]
    assert list(t["bad"]) == [0, 2, 4]
    assert b.transform_bins(pd.Series([3.0, 3.0001, 7.0, 7.1])).tolist() == [t["bin"][0], t["bin"][1], t["bin"][1], t["bin"][2]]
    assert close(t["pct"].sum(), 100.0) and int(t["n"].sum()) == len(x)


def test_fit_uses_train_only_and_transform_reuses_cuts():
    rng = np.random.default_rng(0)
    xtr = pd.Series(rng.normal(size=2000))
    ytr = pd.Series((rng.uniform(size=2000) < 1 / (1 + np.exp(-(xtr - 1)))).astype(int))
    b = WoeBinner(max_bins=4, min_events=1).fit(xtr, ytr)
    cuts_before = list(b.cuts_)
    out = b.transform(pd.Series([-100.0, 100.0, np.nan]))      # 범위 밖 값, 결측
    assert b.cuts_ == cuts_before                            # transform 이 경계를 바꾸지 않는다
    assert close(out.iloc[0], b.table_["woe"].iloc[0]) and close(out.iloc[1], b.table_["woe"].iloc[-1])
    assert out.iloc[2] == 0.0                                # 학습에 결측이 없었으면 중립(0)


def test_missing_bin_and_unseen_category_rules():
    x = pd.Series([1.0, 2.0, np.nan, np.nan, 3.0, 4.0, 5.0, 6.0, np.nan, 7.0] * 20)
    y = pd.Series([0, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 20)
    b = WoeBinner(max_bins=2, min_events=1).fit(x, y)
    miss = b.table_[b.table_["kind"] == "missing"]
    assert len(miss) == 1 and int(miss["n"].iloc[0]) == 60
    assert close(b.transform(pd.Series([np.nan])).iloc[0], miss["woe"].iloc[0])

    xc = pd.Series(["a"] * 50 + ["b"] * 45 + ["c"] * 3 + ["d"] * 2)
    yc = pd.Series([0, 1] * 50)
    bc = WoeBinner(min_bin_pct=0.05).fit(xc, yc)
    assert set(bc.table_["bin"]) == {"a", "b", "OTHER"}         # 희소 범주 c, d 는 OTHER 로
    other = float(bc.table_.loc[bc.table_["bin"] == "OTHER", "woe"].iloc[0])
    assert close(bc.transform(pd.Series(["zzz"])).iloc[0], other)    # 미지 범주도 OTHER 규칙
    assert bc.transform_bins(pd.Series(["c", "zzz", None])).tolist() == ["OTHER", "OTHER", "MISSING"]
    nobj = WoeBinner(min_bin_pct=0.0).fit(pd.Series(["a", "b"] * 10), pd.Series([0, 1] * 10))
    assert nobj.transform(pd.Series(["new"])).iloc[0] == 0.0     # OTHER 가 없었으면 중립


def test_min_bin_constraints_merge_small_bins():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.uniform(size=1000))
    y = pd.Series(rng.binomial(1, 0.1, 1000))
    b = WoeBinner(method="width", max_bins=10, min_bin_pct=0.2, min_events=1).fit(x, y)
    rng_rows = b.table_[b.table_["kind"] == "range"]
    assert (rng_rows["pct"] >= 20 - 1e-9).all() and len(rng_rows) <= 5


def test_monotonic_enforcement():
    # 부도율이 0.05 → 0.20 → 0.10 → 0.30 으로 단조가 아님
    parts = [(0, 200, 0.05), (1, 200, 0.20), (2, 200, 0.10), (3, 200, 0.30)]
    xs, ys = [], []
    rng = np.random.default_rng(2)
    for center, n, rate in parts:
        xs += list(center + rng.uniform(0.01, 0.99, n))
        ys += list(rng.binomial(1, rate, n))
    x, y = pd.Series(xs), pd.Series(ys)
    free = WoeBinner(method="manual", edges=[1, 2, 3]).fit(x, y)
    assert monotonic_direction(free.table_["bad_rate"]) == "none"
    mono = WoeBinner(method="width", max_bins=4, min_events=1, min_bin_pct=0.05, monotonic="increasing").fit(x, y)
    assert monotonic_direction(mono.table_["bad_rate"]) in ("increasing", "flat")
    assert len(mono.table_) < 4
    auto = WoeBinner(method="width", max_bins=4, min_events=1, monotonic="auto").fit(x, y)
    assert auto.direction_ == "increasing"


def test_monotonic_direction_helper():
    assert monotonic_direction([0.1, 0.2, 0.3]) == "increasing"
    assert monotonic_direction([0.3, 0.2, 0.2]) == "decreasing"
    assert monotonic_direction([0.2, 0.2]) == "flat"
    assert monotonic_direction([0.1, 0.3, 0.2]) == "none"
    assert monotonic_direction([0.1]) == "flat"


def test_tree_method_respects_limits():
    rng = np.random.default_rng(3)
    x = pd.Series(rng.uniform(size=3000))
    y = pd.Series((rng.uniform(size=3000) < 0.05 + 0.3 * (x > 0.7)).astype(int))
    b = WoeBinner(method="tree", max_bins=4, min_bin_pct=0.1, min_events=1).fit(x, y)
    assert 2 <= len(b.table_) <= 4 and (b.table_["pct"] >= 10 - 1e-9).all()
    assert any(abs(c - 0.7) < 0.05 for c in b.cuts_)          # 실제 변화점(0.7) 근처를 찾는다


def test_does_not_mutate_input_and_frame_helpers():
    df = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 20, "c": ["x", "y"] * 100, "t": [0, 1, 0, 0, 1, 0, 1, 1, 0, 0] * 20})
    before = df.copy()
    bs = fit_binners(df, ["a", "c"], "t", min_events=1)
    out = transform_woe_frame(df, bs)
    assert df.equals(before)
    assert list(out.columns) == ["a_woe", "c_woe"] and out.index.equals(df.index) and not out.isna().any().any()
    s = iv_summary(df, ["a", "c"], "t", min_events=1)
    assert set(s["variable"]) == {"a", "c"} and (s["iv"] >= 0).all() and s["iv"].is_monotonic_decreasing
    with expect_raises(ValueError, "manual"):
        fit_binners(df, ["a"], "t", method="manual", edges=[3])


def test_iv_is_nonnegative_and_label_boundaries():
    rng = np.random.default_rng(4)
    x = pd.Series(rng.normal(size=4000))
    y = pd.Series(rng.binomial(1, 0.1, 4000))
    assert WoeBinner().fit(x, y).iv_ >= 0
    assert WoeBinner().fit(x, y).iv_ < 0.02              # 무관한 변수는 IV 가 매우 작다
    assert iv_label(0.01).startswith("거의") and iv_label(0.05).startswith("약함")
    assert iv_label(0.2).startswith("중간") and iv_label(0.4).startswith("강함") and "누수" in iv_label(0.6)


def test_woe_sign_convention_matches_logistic_direction():
    # 위험 변수의 WOE 가 단조 증가 → 로지스틱 계수가 양(+)
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(5)
    x = pd.Series(rng.uniform(size=6000))
    y = pd.Series((rng.uniform(size=6000) < 0.03 + 0.25 * x).astype(int))
    b = WoeBinner(max_bins=5, min_events=1).fit(x, y)
    w = b.transform(x).to_numpy().reshape(-1, 1)
    coef = LogisticRegression(C=np.inf, max_iter=1000).fit(w, y).coef_[0][0]
    assert coef > 0 and abs(coef - 1) < 0.25               # WOE 입력의 계수는 이론상 1 근처
