"""단계 5: 비닝·WOE·IV.

고정된 약속 (이 프로그램 전체에서 동일)
  * 타깃: 1 = 부도(bad), 0 = 정상(good)
  * WOE = ln( %bad / %good ).  값이 클수록 위험하다 → 로지스틱 회귀 계수는 대체로 양(+)이 된다.
    (Siddiqi 교재는 반대 부호 ln(%good/%bad) 를 쓴다. IV 는 두 정의에서 같다.)
  * IV  = Σ (%bad − %good) × WOE   (항상 0 이상)
  * 스무딩: 각 bin 의 부도·정상 건수에 smoothing(기본 0.5)을 더해 %를 계산한다.
        %bad_i  = (bad_i  + a) / (총 bad  + a·k)
        %good_i = (good_i + a) / (총 good + a·k)          (k = bin 개수)
    a = 0 이면 순수한 정의 그대로이며, 0 건인 bin 이 있으면 WOE 가 정의되지 않아 오류를 낸다.

누수 방지: WoeBinner.fit 은 **학습 데이터에서만** 호출한다. 검증·테스트·운영 데이터에는
transform 만 쓰고, 경계·WOE 값은 학습 때 정해진 그대로 재사용한다.
IV 는 '인과효과'나 '미래 예측력'의 증명이 아니다. 타깃을 보고 만든 bin 은 학습 데이터에 과적합될 수 있다.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeClassifier

from ..data.target import validate_binary_target
from .diagnostics import MISSING_LABEL, _wilson

OTHER_LABEL = "OTHER"
METHODS = ("quantile", "width", "tree", "manual")


def woe_iv_from_counts(bad, good, smoothing: float = 0.5) -> pd.DataFrame:
    """bin 별 부도·정상 건수에서 WOE 와 IV 기여분을 계산한다(모듈 머리말의 정의).

    반환 열: pct_bad, pct_good(스무딩 반영 비율), woe, iv.  손으로 검산하기 쉬운 작은 함수다.
    """
    bad = np.asarray(bad, dtype=float)
    good = np.asarray(good, dtype=float)
    if bad.shape != good.shape or bad.ndim != 1 or len(bad) == 0:
        raise ValueError("bad, good 는 같은 길이의 1차원 배열이어야 합니다.")
    if (bad < 0).any() or (good < 0).any():
        raise ValueError("건수는 음수일 수 없습니다.")
    if smoothing < 0:
        raise ValueError("smoothing 은 0 이상이어야 합니다.")
    if bad.sum() == 0 or good.sum() == 0:
        raise ValueError("부도와 정상이 모두 있어야 WOE 를 계산할 수 있습니다.")
    if smoothing == 0 and ((bad == 0) | (good == 0)).any():
        raise ValueError(
            "부도 또는 정상이 0건인 bin 이 있어 WOE 가 정의되지 않습니다(ln(0)). "
            "smoothing 을 0 보다 크게 하거나, 해당 bin 을 이웃 bin 과 합치세요."
        )
    k = len(bad)
    pb = (bad + smoothing) / (bad.sum() + smoothing * k)
    pg = (good + smoothing) / (good.sum() + smoothing * k)
    woe = np.log(pb / pg)
    return pd.DataFrame({"pct_bad": pb, "pct_good": pg, "woe": woe, "iv": (pb - pg) * woe})


def monotonic_direction(rates, tol: float = 1e-12) -> str:
    """부도율 순서가 'increasing' / 'decreasing' / 'flat' / 'none'(단조 아님) 중 무엇인지."""
    r = [float(v) for v in rates if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(r) <= 1:
        return "flat"
    d = np.diff(r)
    if (np.abs(d) <= tol).all():
        return "flat"
    if (d >= -tol).all():
        return "increasing"
    if (d <= tol).all():
        return "decreasing"
    return "none"


def iv_label(iv: float) -> str:
    """IV 의 관행적 해석 라벨. **절대 규칙이 아니라 업계 관행**이며 표본 크기·bin 수에 따라 달라진다."""
    if iv < 0.02:
        return "거의 없음(<0.02)"
    if iv < 0.1:
        return "약함(0.02~0.1)"
    if iv < 0.3:
        return "중간(0.1~0.3)"
    if iv < 0.5:
        return "강함(0.3~0.5)"
    return "매우 높음(≥0.5): 누수·과적합 점검"


def _fmt(v: float) -> str:
    return f"{v:g}"


class WoeBinner:
    """변수 하나의 비닝과 WOE 변환기. fit 은 학습 데이터에서만, transform 은 어디서든.

    매개변수
      method        'quantile'(등빈도) | 'width'(등폭) | 'tree'(의사결정나무) | 'manual'(edges 직접 지정)
      max_bins      수치형 초기 구간 수의 상한(결측 bin 은 따로)
      min_bin_pct   각 bin 의 최소 비중(수치형은 결측 아닌 건 기준, 범주형은 전체 기준). 미만이면 이웃과 합침
                    (범주형은 OTHER 로 묶음)
      min_events    수치형 bin 마다 부도·정상 각각의 최소 건수. 미만이면 이웃과 합침
      smoothing     WOE 계산용 스무딩(모듈 머리말 참고)
      monotonic     None | 'auto' | 'increasing' | 'decreasing' (수치형만). 위반하는 이웃 bin 을 합쳐 부도율을 단조로 만듦
      edges         method='manual' 일 때 내부 경계값 목록. manual 에서는 병합·단조 보정을 하지 않는다
      kind          'numeric' | 'categorical' | None(자동)
    bin 은 오른쪽 닫힘 구간 (lo, hi] 이다. 경계값과 같은 값은 아래쪽 bin 에 들어간다.
    """

    def __init__(
        self,
        method: str = "quantile",
        max_bins: int = 5,
        min_bin_pct: float = 0.05,
        min_events: int = 5,
        smoothing: float = 0.5,
        monotonic: str | None = None,
        edges=None,
        kind: str | None = None,
    ):
        if method not in METHODS:
            raise ValueError(f"method 는 {METHODS} 중 하나여야 합니다: {method!r}")
        if max_bins < 2:
            raise ValueError("max_bins 는 2 이상이어야 합니다.")
        if not 0 <= min_bin_pct < 0.5:
            raise ValueError("min_bin_pct 는 0 이상 0.5 미만이어야 합니다.")
        if min_events < 0 or smoothing < 0:
            raise ValueError("min_events, smoothing 은 0 이상이어야 합니다.")
        if monotonic not in (None, "auto", "increasing", "decreasing"):
            raise ValueError("monotonic 은 None/'auto'/'increasing'/'decreasing' 중 하나여야 합니다.")
        if method == "manual" and (edges is None or len(list(edges)) == 0):
            raise ValueError("method='manual' 에는 edges(내부 경계값 목록)가 필요합니다.")
        if kind not in (None, "numeric", "categorical"):
            raise ValueError("kind 는 None/'numeric'/'categorical' 이어야 합니다.")
        self.method, self.max_bins, self.min_bin_pct = method, max_bins, min_bin_pct
        self.min_events, self.smoothing, self.monotonic = min_events, smoothing, monotonic
        self.edges, self.kind = edges, kind
        self.table_: pd.DataFrame | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, x, y) -> "WoeBinner":
        x = pd.Series(x).reset_index(drop=True)
        yv = validate_binary_target(y).reset_index(drop=True)
        if len(x) != len(yv):
            raise ValueError(f"x({len(x)})와 y({len(yv)})의 길이가 다릅니다.")
        if yv.nunique() < 2:
            raise ValueError("부도와 정상이 모두 있어야 비닝할 수 있습니다.")
        is_num = pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)
        self.kind_ = self.kind or ("numeric" if is_num else "categorical")
        if self.kind_ == "numeric":
            self._fit_numeric(x, yv.to_numpy())
        else:
            self._fit_categorical(x, yv.to_numpy())
        return self

    def _fit_numeric(self, x: pd.Series, y: np.ndarray) -> None:
        xv = x.astype(float).to_numpy()
        if np.isinf(xv).any():
            raise ValueError("무한대 값이 있습니다. 데이터 오류인지 확인하고 처리한 뒤 비닝하세요.")
        miss = np.isnan(xv)
        v, yy = xv[~miss], y[~miss]
        if len(v) == 0:
            raise ValueError("결측이 아닌 값이 없습니다.")
        cuts = self._initial_cuts(v, yy)
        if self.method != "manual":
            cuts = self._merge_small(v, yy, cuts)
            if self.monotonic is not None:
                cuts = self._enforce_monotonic(v, yy, cuts)
        self.cuts_ = [float(c) for c in cuts]
        n, bad = self._counts(v, yy, self.cuts_)
        lo = [-np.inf] + self.cuts_
        hi = self.cuts_ + [np.inf]
        bins = []
        for i in range(len(n)):
            label = f"({_fmt(lo[i])}, {_fmt(hi[i])}" + (")" if np.isinf(hi[i]) else "]")
            bins.append({"bin": label, "kind": "range", "lower": lo[i], "upper": hi[i], "n": int(n[i]), "bad": int(bad[i])})
        if miss.any():
            bins.append({"bin": MISSING_LABEL, "kind": "missing", "lower": np.nan, "upper": np.nan,
                         "n": int(miss.sum()), "bad": int(y[miss].sum())})
        self._finish(bins, total=len(x), total_bad=int(y.sum()))

    def _fit_categorical(self, x: pd.Series, y: np.ndarray) -> None:
        cat = x.astype(object).where(x.notna(), MISSING_LABEL).astype(str).to_numpy()
        total = len(cat)
        levels, counts = np.unique(cat, return_counts=True)
        kept, other_levels = [], []
        for lvl, c in zip(levels, counts):
            if lvl == MISSING_LABEL:
                continue
            (kept if c / total >= self.min_bin_pct else other_levels).append(lvl)
        self.categories_ = {lvl: lvl for lvl in kept}
        self.other_levels_ = set(other_levels)

        def stats_for(mask):
            return int(mask.sum()), int(y[mask].sum())

        rows = []
        for lvl in kept:
            n, b = stats_for(cat == lvl)
            rows.append({"bin": lvl, "kind": "category", "lower": np.nan, "upper": np.nan, "n": n, "bad": b})
        rows.sort(key=lambda r: (r["bad"] / r["n"], r["bin"]))
        if other_levels:
            n, b = stats_for(np.isin(cat, other_levels))
            rows.append({"bin": OTHER_LABEL, "kind": "other", "lower": np.nan, "upper": np.nan, "n": n, "bad": b})
        if (cat == MISSING_LABEL).any():
            n, b = stats_for(cat == MISSING_LABEL)
            rows.append({"bin": MISSING_LABEL, "kind": "missing", "lower": np.nan, "upper": np.nan, "n": n, "bad": b})
        self.cuts_ = []
        self._finish(rows, total=total, total_bad=int(y.sum()))

    def _finish(self, bins: list[dict], total: int, total_bad: int) -> None:
        t = pd.DataFrame(bins)
        t["good"] = t["n"] - t["bad"]
        if t["n"].sum() != total or t["bad"].sum() != total_bad:
            raise AssertionError("bin 건수의 합이 전체와 맞지 않습니다(내부 오류).")
        t["pct"] = 100 * t["n"] / total
        t["bad_rate"] = t["bad"] / t["n"]
        ci = [_wilson(int(b), int(n)) for b, n in zip(t["bad"], t["n"])]
        t["bad_rate_lo"] = [c[0] for c in ci]
        t["bad_rate_hi"] = [c[1] for c in ci]
        w = woe_iv_from_counts(t["bad"], t["good"], self.smoothing)
        t = pd.concat([t, w], axis=1)
        self.table_ = t[["bin", "kind", "lower", "upper", "n", "pct", "bad", "good", "bad_rate",
                         "bad_rate_lo", "bad_rate_hi", "pct_bad", "pct_good", "woe", "iv"]]
        self.iv_ = float(t["iv"].sum())
        self.woe_by_bin_ = dict(zip(t["bin"], t["woe"]))
        self.has_missing_bin_ = bool((t["kind"] == "missing").any())
        self.has_other_bin_ = bool((t["kind"] == "other").any())

    # ------------------------------------------------------------------ 경계 만들기
    @staticmethod
    def _counts(v: np.ndarray, yy: np.ndarray, cuts) -> tuple[np.ndarray, np.ndarray]:
        idx = np.searchsorted(np.asarray(cuts, dtype=float), v, side="left")
        k = len(cuts) + 1
        n = np.bincount(idx, minlength=k)
        bad = np.bincount(idx, weights=yy, minlength=k).astype(int)
        return n, bad

    def _initial_cuts(self, v: np.ndarray, yy: np.ndarray) -> list[float]:
        if self.method == "manual":
            return sorted({float(e) for e in self.edges})
        if self.method == "quantile":
            q = np.quantile(v, np.linspace(0, 1, self.max_bins + 1)[1:-1])
            cuts = np.unique(q)
        elif self.method == "width":
            cuts = np.linspace(v.min(), v.max(), self.max_bins + 1)[1:-1]
        else:  # tree
            tree = DecisionTreeClassifier(
                max_leaf_nodes=self.max_bins,
                min_samples_leaf=max(int(math.ceil(self.min_bin_pct * len(v))), 1),
                random_state=0,
            ).fit(v.reshape(-1, 1), yy)
            cuts = np.sort(tree.tree_.threshold[tree.tree_.feature >= 0])
        # 최댓값 이상의 경계는 빈 bin 을 만들므로 제거
        return [float(c) for c in cuts if c < v.max()]

    def _merge_small(self, v: np.ndarray, yy: np.ndarray, cuts: list[float]) -> list[float]:
        cuts = list(cuts)
        min_n = int(math.ceil(self.min_bin_pct * len(v)))
        while cuts:
            n, bad = self._counts(v, yy, cuts)
            good = n - bad
            viol = [i for i in range(len(n)) if n[i] < min_n or bad[i] < self.min_events or good[i] < self.min_events]
            if not viol:
                break
            i = min(viol, key=lambda j: n[j])
            nbrs = [j for j in (i - 1, i + 1) if 0 <= j < len(n)]
            j = min(nbrs, key=lambda t: n[t])
            del cuts[i - 1 if j == i - 1 else i]
        return cuts

    def _enforce_monotonic(self, v: np.ndarray, yy: np.ndarray, cuts: list[float]) -> list[float]:
        direction = self.monotonic
        if direction == "auto":
            rho = stats.spearmanr(v, yy).correlation if len(set(v)) > 1 else 0.0
            direction = "decreasing" if (not np.isnan(rho) and rho < 0) else "increasing"
        self.direction_ = direction
        cuts = list(cuts)
        while cuts:
            n, bad = self._counts(v, yy, cuts)
            rate = bad / n
            d = np.diff(rate)
            viol = np.where(d < -1e-15)[0] if direction == "increasing" else np.where(d > 1e-15)[0]
            if len(viol) == 0:
                break
            del cuts[int(viol[0])]
        return cuts

    # ------------------------------------------------------------------ transform
    def _check_fitted(self) -> None:
        if self.table_ is None:
            raise RuntimeError("fit 을 먼저 호출하세요(학습 데이터로).")

    def transform_bins(self, x) -> pd.Series:
        """각 값이 속한 bin 이름. 학습 때 정해진 경계·범주만 사용한다."""
        self._check_fitted()
        x = pd.Series(x)
        if self.kind_ == "numeric":
            xv = x.astype(float).to_numpy()
            labels = self.table_.loc[self.table_["kind"] == "range", "bin"].to_numpy()
            idx = np.searchsorted(np.asarray(self.cuts_, dtype=float), xv, side="left")
            out = np.where(np.isnan(xv), MISSING_LABEL, labels[np.minimum(idx, len(labels) - 1)])
        else:
            cat = x.astype(object).where(x.notna(), MISSING_LABEL).astype(str).to_numpy()
            known = set(self.categories_)
            out = np.array([c if (c in known or c == MISSING_LABEL) else OTHER_LABEL for c in cat], dtype=object)
        return pd.Series(out, index=x.index, name=x.name)

    def transform(self, x) -> pd.Series:
        """WOE 값으로 변환. 학습에 없던 경우의 규칙:
        * 수치형 범위 밖 값 → 맨 앞/뒤 bin     * 결측 → MISSING bin 의 WOE (학습에 결측이 없었으면 0)
        * 범주형 미지 범주 → OTHER bin 의 WOE (학습에 OTHER 가 없었으면 0)
        0 은 '정보 없음(중립)'을 뜻하므로, 이런 일이 자주 생기면 모니터링 경보로 다뤄야 한다.
        """
        bins = self.transform_bins(x)
        woe = bins.map(self.woe_by_bin_)
        return woe.fillna(0.0).astype(float).rename(pd.Series(x).name)

    def fit_transform(self, x, y) -> pd.Series:
        return self.fit(x, y).transform(x)


# ---------------------------------------------------------------------------- 여러 변수
def fit_binners(df: pd.DataFrame, columns, target: str, **binner_kwargs) -> dict:
    """여러 변수에 WoeBinner 를 학습한다. df 는 반드시 학습 데이터여야 한다."""
    if binner_kwargs.get("method") == "manual":
        raise ValueError("여러 변수에 한꺼번에 manual 을 쓸 수 없습니다. 변수별로 WoeBinner 를 만드세요.")
    return {c: WoeBinner(**binner_kwargs).fit(df[c], df[target]) for c in columns}


def transform_woe_frame(df: pd.DataFrame, binners: dict, suffix: str = "_woe") -> pd.DataFrame:
    """학습된 binner 들로 df 를 WOE 열로 변환한다. 새 DataFrame 을 돌려주며 원본은 바꾸지 않는다."""
    return pd.DataFrame({f"{c}{suffix}": b.transform(df[c]) for c, b in binners.items()}, index=df.index)


def iv_summary(df: pd.DataFrame, columns, target: str, **binner_kwargs) -> pd.DataFrame:
    """변수별 IV 순위표(학습 데이터로 계산). IV 가 높다고 곧바로 모형 변수로 채택하지 않는다."""
    binners = fit_binners(df, columns, target, **binner_kwargs)
    rows = []
    for c, b in binners.items():
        t = b.table_
        rng = t[t["kind"].isin(["range", "category"])]
        rows.append(
            {
                "variable": c,
                "kind": b.kind_,
                "n_bins": len(t),
                "iv": b.iv_,
                "iv_label": iv_label(b.iv_),
                "bad_rate_trend": monotonic_direction(rng["bad_rate"]) if b.kind_ == "numeric" else "",
                "missing_pct": float(t.loc[t["kind"] == "missing", "pct"].sum()),
                "min_bin_bad": int(t["bad"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


def plot_woe_table(table: pd.DataFrame, title: str = "", ax=None):
    """bin 별 건수(막대)와 부도율(선, 95% 구간)을 한 그림에 그린다."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(table))
    ax.bar(x, table["pct"], color="#c9d6e2", label="건수 비중(%)")
    ax.set_ylabel("share of rows (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(table["bin"], rotation=30, ha="right", fontsize=8)
    ax2 = ax.twinx()
    rate = 100 * table["bad_rate"].to_numpy()
    lo = 100 * table["bad_rate_lo"].to_numpy()
    hi = 100 * table["bad_rate_hi"].to_numpy()
    ax2.errorbar(x, rate, yerr=[rate - lo, hi - rate], fmt="o-", color="#c8553d", capsize=3, label="부도율(%)")
    ax2.set_ylabel("default rate (%)")
    ax.set_title(title)
    return ax
