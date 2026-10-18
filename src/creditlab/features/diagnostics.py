"""단계 4: 단변량·다변량 분석과 변수 진단.

약속: 타깃은 1 = 부도(bad), 0 = 정상(good).
여기 있는 모든 함수는 '진단'이다. 결과가 변수를 *자동으로* 남기거나 버리지 않는다.
최종 판단은 업무 논리·안정성·데이터 품질·법적/공정성 고려와 함께 사람이 하고,
그 이유를 VariableLog 에 기록한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from ..data.target import validate_binary_target

MISSING_LABEL = "MISSING"


def _wilson(bad: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """부도율의 Wilson 신뢰구간. n=0 이면 (nan, nan)."""
    if n <= 0:
        return float("nan"), float("nan")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = bad / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _numeric_columns(df: pd.DataFrame, columns) -> list[str]:
    columns = list(columns)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"데이터에 없는 열입니다: {missing}")
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(
            f"숫자형이 아닌 열이 있습니다: {non_numeric}. 숫자인데 문자로 저장된 값('N/A' 등)이 "
            "섞였을 수 있으니 find_numeric_like_text 로 먼저 확인하세요."
        )
    return columns


# --------------------------------------------------------------------------- 단변량
def univariate_numeric(df: pd.DataFrame, columns, target: str) -> pd.DataFrame:
    """수치형 변수의 분포 요약 + 부도/정상 그룹별 평균. 행 = 변수."""
    columns = _numeric_columns(df, columns)
    y = validate_binary_target(df[target], target)
    rows = []
    for c in columns:
        s = df[c].astype(float)
        ok = s.notna()
        v = s[ok]
        bad_mask = (y == 1).to_numpy() & ok.to_numpy()
        good_mask = (y == 0).to_numpy() & ok.to_numpy()
        q = v.quantile([0.25, 0.5, 0.75]) if len(v) else pd.Series([np.nan] * 3, index=[0.25, 0.5, 0.75])
        rows.append(
            {
                "variable": c,
                "n": int(ok.sum()),
                "missing_pct": float(100 * (~ok).mean()),
                "mean": float(v.mean()) if len(v) else np.nan,
                "std": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                "min": float(v.min()) if len(v) else np.nan,
                "p25": float(q.loc[0.25]),
                "median": float(q.loc[0.5]),
                "p75": float(q.loc[0.75]),
                "max": float(v.max()) if len(v) else np.nan,
                "n_bad": int(bad_mask.sum()),
                "n_good": int(good_mask.sum()),
                "mean_bad": float(s[bad_mask].mean()) if bad_mask.any() else np.nan,
                "mean_good": float(s[good_mask].mean()) if good_mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def univariate_categorical(df: pd.DataFrame, column: str, target: str, min_count: int = 30) -> pd.DataFrame:
    """범주형 변수의 범주별 건수·부도율·95% 구간. 결측은 'MISSING' 범주로 따로 센다.

    min_count 미만인 범주는 rare=True 로 표시한다(부도율이 불안정하므로 해석 주의).
    """
    if column not in df.columns:
        raise ValueError(f"데이터에 없는 열입니다: {column}")
    y = validate_binary_target(df[target], target).to_numpy()
    cat = df[column].astype(object).where(df[column].notna(), MISSING_LABEL).astype(str).to_numpy()
    rows = []
    for level in pd.unique(cat):
        m = cat == level
        n, bad = int(m.sum()), int(y[m].sum())
        lo, hi = _wilson(bad, n)
        rows.append(
            {
                "category": level,
                "n": n,
                "pct": 100 * n / len(cat),
                "bad": bad,
                "bad_rate": bad / n,
                "bad_rate_lo": lo,
                "bad_rate_hi": hi,
                "rare": n < min_count,
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def compare_groups(df: pd.DataFrame, column: str, target: str, confidence: float = 0.95) -> dict:
    """부도 vs 정상 그룹의 평균 비교: 차이·신뢰구간·Cohen's d·Welch t p-value·Mann-Whitney p-value.

    * p-value 만 보지 말고 효과크기(d)와 구간을 함께 본다. 표본이 크면 사소한 차이도 유의하다.
    * 변수 하나씩 여러 번 검정하면 다중검정 문제가 생긴다(단계 14).
    * 이 검정은 '평균 차이'에 관한 것이며 예측모형의 우수성을 증명하지 않는다.
    """
    _numeric_columns(df, [column])
    y = validate_binary_target(df[target], target).to_numpy()
    x = df[column].astype(float).to_numpy()
    ok = ~np.isnan(x)
    bad, good = x[ok & (y == 1)], x[ok & (y == 0)]
    if len(bad) < 2 or len(good) < 2:
        raise ValueError(f"각 그룹에 최소 2건이 필요합니다 (부도 {len(bad)}건, 정상 {len(good)}건).")
    mb, mg = bad.mean(), good.mean()
    vb, vg = bad.var(ddof=1), good.var(ddof=1)
    nb, ng = len(bad), len(good)
    diff = mb - mg
    se = np.sqrt(vb / nb + vg / ng)
    dof = (vb / nb + vg / ng) ** 2 / ((vb / nb) ** 2 / (nb - 1) + (vg / ng) ** 2 / (ng - 1)) if se > 0 else np.nan
    tcrit = stats.t.ppf(0.5 + confidence / 2, dof) if se > 0 else np.nan
    pooled = np.sqrt(((nb - 1) * vb + (ng - 1) * vg) / (nb + ng - 2))
    welch_p = stats.ttest_ind(bad, good, equal_var=False).pvalue if se > 0 else np.nan
    mw_p = stats.mannwhitneyu(bad, good, alternative="two-sided").pvalue
    return {
        "variable": column,
        "n_bad": nb,
        "n_good": ng,
        "mean_bad": float(mb),
        "mean_good": float(mg),
        "diff": float(diff),
        "ci_low": float(diff - tcrit * se) if se > 0 else np.nan,
        "ci_high": float(diff + tcrit * se) if se > 0 else np.nan,
        "cohens_d": float(diff / pooled) if pooled > 0 else np.nan,
        "welch_p": float(welch_p),
        "mannwhitney_p": float(mw_p),
    }


# --------------------------------------------------------------------------- 다변량
def correlation_matrix(df: pd.DataFrame, columns, method: str = "pearson") -> pd.DataFrame:
    """상관행렬. method = 'pearson'(선형) 또는 'spearman'(순위, 단조 관계).

    상관계수는 비선형 관계(U자형 등)를 놓친다. 0 에 가깝다고 관계가 없다는 뜻이 아니다.
    """
    if method not in ("pearson", "spearman"):
        raise ValueError("method 는 'pearson' 또는 'spearman' 이어야 합니다.")
    columns = _numeric_columns(df, columns)
    return df[columns].astype(float).corr(method=method)


def high_correlation_pairs(corr: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """|상관| 이 threshold 이상인 변수 쌍(중복 없이). threshold 는 관행적 경보 기준일 뿐이다."""
    rows = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                rows.append({"var1": a, "var2": b, "corr": float(r), "abs_corr": float(abs(r))})
    out = pd.DataFrame(rows, columns=["var1", "var2", "corr", "abs_corr"])
    return out.sort_values("abs_corr", ascending=False).reset_index(drop=True)


def cramers_v(x, y) -> float:
    """두 범주형 변수의 연관성(0~1). 카이제곱 기반, 보정 없음. 방향은 알려주지 않는다."""
    tab = pd.crosstab(pd.Series(x).astype(str).to_numpy(), pd.Series(y).astype(str).to_numpy())
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(tab.to_numpy(), correction=False)[0]
    n = tab.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def vif_table(df: pd.DataFrame, columns) -> pd.DataFrame:
    """분산팽창계수(VIF). 변수 j 를 나머지 변수로 회귀한 R² 에서 VIF = 1/(1-R²).

    * 결측이 있는 행은 모든 열 기준으로 제외한다(n_used 에 표시).
    * R² 가 사실상 1 이면 VIF = inf (완전 공선성).
    * 5, 10 같은 기준은 관행이며 절대 규칙이 아니다. VIF 가 높다고 변수를 자동 삭제하지 말고
      어떤 변수들이 서로 같은 정보를 담는지 업무적으로 확인한다.
    """
    columns = _numeric_columns(df, columns)
    if len(columns) < 2:
        raise ValueError("VIF 는 변수가 2개 이상일 때 계산할 수 있습니다.")
    X = df[columns].astype(float).dropna()
    n_used = len(X)
    if n_used <= len(columns) + 1:
        raise ValueError(f"사용 가능한 행({n_used})이 변수 수({len(columns)})에 비해 너무 적습니다.")
    const = [c for c in columns if X[c].nunique() < 2]
    if const:
        raise ValueError(f"값이 하나뿐인 열이 있어 VIF 를 계산할 수 없습니다: {const}")
    A = X.to_numpy()
    rows = []
    for j, c in enumerate(columns):
        others = np.delete(A, j, axis=1)
        design = np.column_stack([np.ones(n_used), others])
        beta, *_ = np.linalg.lstsq(design, A[:, j], rcond=None)
        resid = A[:, j] - design @ beta
        sst = ((A[:, j] - A[:, j].mean()) ** 2).sum()
        r2 = 1 - (resid @ resid) / sst
        vif = float("inf") if r2 >= 1 - 1e-12 else float(1 / (1 - r2))
        flag = "높음(>10)" if vif > 10 else "주의(>5)" if vif > 5 else "보통"
        rows.append({"variable": c, "vif": vif, "r2": float(r2), "flag": flag, "n_used": n_used})
    return pd.DataFrame(rows)


def condition_number(df: pd.DataFrame, columns) -> float:
    """표준화한 설계행렬의 조건수. 클수록 계수 추정이 불안정해질 수 있다(관행: 30 이상 주의)."""
    columns = _numeric_columns(df, columns)
    X = df[columns].astype(float).dropna()
    sd = X.std(ddof=0)
    if (sd == 0).any():
        raise ValueError(f"값이 하나뿐인 열이 있습니다: {list(sd[sd == 0].index)}")
    Z = ((X - X.mean()) / sd).to_numpy()
    return float(np.linalg.cond(Z))


# --------------------------------------------------------------------------- 누수 후보
DEFAULT_LEAK_NAME_PATTERNS = (
    "after", "post_", "_post", "recover", "collection", "charge_off", "chargeoff",
    "write_off", "writeoff", "settle", "outcome", "result", "status", "사후", "회수", "대손", "결과",
)  # fmt: skip


def single_feature_auc(df: pd.DataFrame, column: str, target: str) -> float:
    """변수 하나만으로 부도를 구분하는 AUROC(방향 무관: max(auc, 1-auc)). 결측 행은 제외."""
    _numeric_columns(df, [column])
    y = validate_binary_target(df[target], target).to_numpy()
    x = df[column].astype(float).to_numpy()
    ok = ~np.isnan(x)
    if len(set(y[ok])) < 2:
        raise ValueError("한 클래스만 있어 AUROC 를 계산할 수 없습니다.")
    auc = roc_auc_score(y[ok], x[ok])
    return float(max(auc, 1 - auc))


def leakage_suspects(
    df: pd.DataFrame,
    target: str,
    columns=None,
    name_patterns=DEFAULT_LEAK_NAME_PATTERNS,
    auc_threshold: float = 0.9,
) -> pd.DataFrame:
    """규칙 기반 누수 후보 목록. **최종 판정은 사람이 한다.**

    1) 이름 규칙: 결과 이후를 암시하는 단어가 열 이름에 있음
    2) 단일 변수 AUROC 가 비정상적으로 높음 (신용 데이터에서 변수 하나로 0.9 이상이면 의심)
    규칙에 안 걸렸다고 안전한 것은 아니다. '이 변수가 예측 시점에 알려져 있는가'를 데이터 사전으로 확인한다.
    """
    if columns is None:
        columns = [c for c in df.columns if c != target]
    rows = []
    for c in columns:
        if c == target:
            continue
        name = str(c).lower()
        hit = [p for p in name_patterns if p.lower() in name]
        if hit:
            rows.append({"variable": c, "reason": "이름 규칙", "detail": f"이름에 {hit} 포함", "value": np.nan})
        if pd.api.types.is_numeric_dtype(df[c]):
            try:
                auc = single_feature_auc(df, c, target)
            except ValueError:
                continue
            if auc >= auc_threshold:
                rows.append({"variable": c, "reason": "단일 변수 AUROC", "detail": f"AUROC {auc:.3f} ≥ {auc_threshold}", "value": auc})
    return pd.DataFrame(rows, columns=["variable", "reason", "detail", "value"])


# --------------------------------------------------------------------------- 변수 결정 기록
@dataclass
class VariableLog:
    """변수 유지/제외/보류 결정을 *이유와 함께* 기록한다. 이유 없는 결정은 기록되지 않는다."""

    DECISIONS = ("keep", "drop", "review")
    _rows: dict = field(default_factory=dict)

    def add(self, variable: str, decision: str, reason: str, evidence: str = "", replace: bool = False) -> None:
        if decision not in self.DECISIONS:
            raise ValueError(f"decision 은 {self.DECISIONS} 중 하나여야 합니다: {decision!r}")
        if not str(reason).strip():
            raise ValueError("이유(reason)를 적어야 합니다. '통계적으로 유의하지 않음' 같은 한 줄이라도 근거를 남기세요.")
        if variable in self._rows and not replace:
            raise ValueError(f"{variable!r} 는 이미 기록되어 있습니다. 바꾸려면 replace=True 를 쓰세요.")
        self._rows[variable] = {"variable": variable, "decision": decision, "reason": reason.strip(), "evidence": evidence}

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self._rows.values()), columns=["variable", "decision", "reason", "evidence"])

    def kept(self) -> list[str]:
        return [v for v, r in self._rows.items() if r["decision"] == "keep"]

    def undecided(self, candidates) -> list[str]:
        """아직 결정이 기록되지 않은 후보 변수들."""
        return [c for c in candidates if c not in self._rows]
