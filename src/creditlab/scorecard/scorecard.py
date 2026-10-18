"""단계 7: 점수 변환, 스코어카드, 등급화, 컷오프 분석.

고정된 약속
  * 점수가 높을수록 안전하다.      Score = Offset + Factor × ln(odds_good),  odds_good = (1-PD)/PD
  * Factor = PDO / ln 2,   Offset = base_score − Factor × ln(base_odds)
    예) 600점 = 50:1(good:bad), PDO 20 → Factor 28.8539, Offset 487.123.  점수 +20 = odds 2배.
  * WOE 입력 로지스틱의 logit(PD) = β0 + Σ βj·Wj  이므로
        Score = Offset − Factor·(β0 + Σ βj·Wj) = Σ_j [ −Factor·βj·Wj + (Offset − Factor·β0)/m ]   (m = 사용 변수 수)
    → 각 변수의 구간(bin)마다 '점수'를 하나씩 정할 수 있다 = 스코어카드.
  * 점수는 예측모형의 출력이고, 컷오프(승인/거절 기준점)는 정책이다. 둘을 분리해서 다룬다.
이 모듈은 교육용이며 실제 규제 승인·IRB 적합성을 보장하지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.target import validate_binary_target
from ..features.binning import monotonic_direction
from ..features.diagnostics import _wilson
from ..models.logistic import sigmoid


@dataclass(frozen=True)
class ScoreScaling:
    base_score: float = 600.0
    base_odds: float = 50.0   # good:bad
    pdo: float = 20.0

    def __post_init__(self):
        if self.pdo <= 0:
            raise ValueError("pdo 는 0 보다 커야 합니다.")
        if self.base_odds <= 0:
            raise ValueError("base_odds 는 0 보다 커야 합니다.")

    @property
    def factor(self) -> float:
        return self.pdo / math.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * math.log(self.base_odds)

    def score_from_odds_good(self, odds_good):
        odds_good = np.asarray(odds_good, dtype=float)
        if (odds_good <= 0).any():
            raise ValueError("odds 는 0 보다 커야 합니다.")
        return self.offset + self.factor * np.log(odds_good)

    def score_from_pd(self, pd_):
        pd_ = np.asarray(pd_, dtype=float)
        if ((pd_ <= 0) | (pd_ >= 1)).any():
            raise ValueError("PD 는 0 초과 1 미만이어야 합니다.")
        return self.score_from_odds_good((1 - pd_) / pd_)

    def odds_good_from_score(self, score):
        return np.exp((np.asarray(score, dtype=float) - self.offset) / self.factor)

    def pd_from_score(self, score):
        """점수 → 부도확률. PD = 1/(1+odds_good)."""
        return 1.0 / (1.0 + self.odds_good_from_score(score))


class Scorecard:
    """WOE 로지스틱 모형 + 학습된 비닝 → 구간별 점수표.

    binners : {변수명: 학습된 WoeBinner}      coef : {변수명: 계수} (절편 제외, 0 이면 사용 안 함)
    intercept : 로지스틱 절편 β0              scaling : ScoreScaling
    round_points : True 면 구간 점수를 정수로 반올림(실무에서 흔함). 합계 오차는 최대 0.5·m 점.
    """

    def __init__(self, binners: dict, coef: dict, intercept: float, scaling: ScoreScaling | None = None, round_points: bool = False):
        self.scaling = scaling or ScoreScaling()
        self.round_points = round_points
        self.intercept = float(intercept)
        self.variables = [v for v in coef if abs(coef[v]) > 0]
        if not self.variables:
            raise ValueError("계수가 0 이 아닌 변수가 없습니다.")
        unknown = [v for v in self.variables if v not in binners]
        if unknown:
            raise ValueError(f"비닝 정보가 없는 변수: {unknown}")
        self.binners = {v: binners[v] for v in self.variables}
        self.coef = {v: float(coef[v]) for v in self.variables}
        m = len(self.variables)
        self._base_share = (self.scaling.offset - self.scaling.factor * self.intercept) / m
        rows = []
        for v in self.variables:
            t = self.binners[v].table_
            for _, r in t.iterrows():
                rows.append(
                    {
                        "variable": v,
                        "bin": r["bin"],
                        "woe": r["woe"],
                        "coef": self.coef[v],
                        "points": self._points(self.coef[v], r["woe"]),
                        "n": r["n"],
                        "pct": r["pct"],
                        "bad_rate": r["bad_rate"],
                    }
                )
        self.table_ = pd.DataFrame(rows)
        self._lookup = {(r.variable, r.bin): r.points for r in self.table_.itertuples()}

    @classmethod
    def from_fit(cls, binners: dict, fit, scaling: ScoreScaling | None = None, suffix: str = "_woe", round_points: bool = False):
        """fit_logistic 결과(변수명이 'age_woe' 형태)로 만든다."""
        coef = {name[: -len(suffix)] if name.endswith(suffix) else name: float(c) for name, c in fit.slopes.items()}
        return cls(binners, coef, fit.intercept, scaling, round_points)

    @classmethod
    def from_sklearn(cls, binners: dict, columns, model, scaling: ScoreScaling | None = None, suffix: str = "_woe", round_points: bool = False):
        coef = {(c[: -len(suffix)] if c.endswith(suffix) else c): float(v) for c, v in zip(columns, model.coef_[0])}
        return cls(binners, coef, float(model.intercept_[0]), scaling, round_points)

    def _points(self, coef: float, woe: float) -> float:
        p = -self.scaling.factor * coef * woe + self._base_share
        return float(round(p)) if self.round_points else float(p)

    def _neutral_points(self, var: str) -> float:
        return self._points(self.coef[var], 0.0)

    def points_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """각 행·변수의 구간 점수. 학습에 없던 값(미지 범주·결측)은 WOE 0(중립) 점수를 받는다."""
        out = {}
        for v in self.variables:
            bins = self.binners[v].transform_bins(df[v])
            neutral = self._neutral_points(v)
            out[v] = np.array([self._lookup.get((v, b), neutral) for b in bins], dtype=float)
        return pd.DataFrame(out, index=df.index)

    def score(self, df: pd.DataFrame) -> pd.Series:
        return self.points_frame(df).sum(axis=1).rename("score")

    def logit_pd(self, df: pd.DataFrame) -> np.ndarray:
        """모형 자체의 logit(PD) (구간 WOE 사용). 점수와 정확히 대응되는지 verify 로 검산한다."""
        z = np.full(len(df), self.intercept)
        for v in self.variables:
            z = z + self.coef[v] * self.binners[v].transform(df[v]).to_numpy()
        return z

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        return sigmoid(self.logit_pd(df))

    def verify(self, df: pd.DataFrame) -> dict:
        """점수표 합계가 모형(Offset − Factor·logit)과 얼마나 다른지 검산한다.

        round_points=False 이면 부동소수 오차 수준이어야 하고, True 이면 최대 0.5·m 점 이내여야 한다.
        """
        model_score = self.scaling.offset - self.scaling.factor * self.logit_pd(df)
        diff = np.abs(self.score(df).to_numpy() - model_score)
        bound = 0.5 * len(self.variables) if self.round_points else 1e-8
        return {"max_abs_diff": float(diff.max()), "allowed": bound, "ok": bool(diff.max() <= bound + 1e-12)}


# --------------------------------------------------------------------------- 등급화
def _labels(n: int, labels):
    if labels is None:
        if n > 26:
            raise ValueError("등급이 26개를 넘으면 labels 를 직접 지정하세요.")
        labels = [chr(ord("A") + i) for i in range(n)]
    if len(labels) != n:
        raise ValueError(f"등급 라벨 수({len(labels)})가 등급 수({n})와 다릅니다.")
    return list(labels)


def make_grade_edges(scores, n_grades: int = 5, method: str = "quantile") -> list[float]:
    """점수 구간 경계(오름차순, n_grades-1 개). 개발 구간 점수로만 만든다.

    quantile: 동일 건수   width: 동일 점수 폭.  (동일 부도율/업무 기준 구간은 edges 를 직접 지정)
    """
    s = np.asarray(scores, dtype=float)
    if n_grades < 2:
        raise ValueError("등급은 2개 이상이어야 합니다.")
    if method == "quantile":
        edges = np.unique(np.quantile(s, np.linspace(0, 1, n_grades + 1)[1:-1]))
    elif method == "width":
        edges = np.linspace(s.min(), s.max(), n_grades + 1)[1:-1]
    else:
        raise ValueError("method 는 'quantile' 또는 'width' 입니다.")
    if len(edges) != n_grades - 1:
        raise ValueError("점수 값이 겹쳐 요청한 등급 수를 만들 수 없습니다. 등급 수를 줄이세요.")
    return [float(e) for e in edges]


def assign_grades(scores, edges, labels=None) -> pd.Series:
    """점수 → 등급. 높은 점수 = 안전 = 앞쪽 라벨(A). 구간은 [경계, 다음 경계) 왼쪽 닫힘이다."""
    edges = list(edges)
    if edges != sorted(edges):
        raise ValueError("edges 는 오름차순이어야 합니다.")
    n = len(edges) + 1
    lab = _labels(n, labels)
    s = pd.Series(scores)
    idx = np.searchsorted(np.asarray(edges, dtype=float), s.to_numpy(dtype=float), side="right")  # 0 = 가장 낮은 점수
    names = np.array(lab[::-1], dtype=object)[idx]       # 낮은 점수 → 마지막 라벨(위험)
    return pd.Series(pd.Categorical(names, categories=lab, ordered=True), index=s.index, name="grade")


def grade_table(scores, y, edges, labels=None) -> pd.DataFrame:
    """등급별 건수·부도율·95% 신뢰구간·점수 범위. 행은 A(안전) → 마지막(위험) 순서."""
    yv = validate_binary_target(y).reset_index(drop=True)
    s = pd.Series(np.asarray(scores, dtype=float))
    if len(s) != len(yv):
        raise ValueError("scores 와 y 의 길이가 다릅니다.")
    g = assign_grades(s, edges, labels)
    rows = []
    for lab in g.cat.categories:
        m = (g == lab).to_numpy()
        n = int(m.sum())
        bad = int(yv[m].sum())
        lo, hi = _wilson(bad, n)
        rows.append(
            {
                "grade": lab,
                "n": n,
                "pct": 100 * n / len(s),
                "bad": bad,
                "bad_rate": bad / n if n else np.nan,
                "bad_rate_lo": lo,
                "bad_rate_hi": hi,
                "score_min": float(s[m].min()) if n else np.nan,
                "score_max": float(s[m].max()) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def grade_monotonic(table: pd.DataFrame) -> str:
    """A → 마지막 등급으로 갈 때 부도율 추세: 'increasing'(정상) / 'decreasing' / 'flat' / 'none'."""
    return monotonic_direction(table["bad_rate"].dropna().tolist())


def cutoff_analysis(scores, y, cutoffs) -> pd.DataFrame:
    """컷오프(점수 ≥ 컷오프이면 승인)별 승인율과 부도율. **정책 분석이지 모형 평가가 아니다.**"""
    yv = validate_binary_target(y).to_numpy()
    s = np.asarray(scores, dtype=float)
    if len(s) != len(yv):
        raise ValueError("scores 와 y 의 길이가 다릅니다.")
    total_bad = yv.sum()
    rows = []
    for c in cutoffs:
        ap = s >= c
        n_ap = int(ap.sum())
        rows.append(
            {
                "cutoff": c,
                "approval_rate": n_ap / len(s),
                "n_approved": n_ap,
                "bad_rate_approved": float(yv[ap].mean()) if n_ap else np.nan,
                "bad_rate_rejected": float(yv[~ap].mean()) if n_ap < len(s) else np.nan,
                "bads_rejected_pct": float(yv[~ap].sum() / total_bad) if total_bad else np.nan,
            }
        )
    return pd.DataFrame(rows)
