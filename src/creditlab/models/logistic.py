"""단계 6~7: 로지스틱 회귀 (추론용 NumPy IRLS + 예측용 sklearn 규제 모형).

약속: 타깃 1 = 부도(bad). 모형은 P(부도) 를 예측한다.
  logit(PD) = ln(PD/(1-PD)) = β0 + Σ βj·xj
WOE(= ln(%bad/%good))를 입력으로 쓰면 βj 는 대체로 양(+)이다.

두 가지를 구분한다.
  * fit_logistic          : 계수의 표준오차·p-value·신뢰구간·오즈비를 구하는 '추론' 도구 (규제 없음, IRLS)
  * fit_regularized 외    : 과적합을 줄이는 '예측' 도구 (sklearn). 규제 모형의 p-value 는 표준 공식이 맞지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from ..data.target import validate_binary_target


def sigmoid(z):
    """수치적으로 안정적인 로지스틱 함수 1/(1+e^-z)."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logit(p):
    """ln(p/(1-p)). p 는 0 과 1 사이(양 끝 제외)여야 한다."""
    p = np.asarray(p, dtype=float)
    if ((p <= 0) | (p >= 1)).any():
        raise ValueError("확률은 0 초과 1 미만이어야 logit 을 계산할 수 있습니다.")
    return np.log(p / (1 - p))


def odds(p):
    """p/(1-p)."""
    p = np.asarray(p, dtype=float)
    if ((p <= 0) | (p >= 1)).any():
        raise ValueError("확률은 0 초과 1 미만이어야 odds 를 계산할 수 있습니다.")
    return p / (1 - p)


def _check_X(X) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X = X.copy()
    non_num = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_num:
        raise ValueError(f"숫자형이 아닌 열이 있습니다: {non_num}. WOE 변환 등으로 숫자로 바꾼 뒤 사용하세요.")
    if X.isna().any().any():
        bad = list(X.columns[X.isna().any()])
        raise ValueError(f"결측이 있는 열이 있습니다: {bad}. 학습 데이터로 정한 규칙(예: MISSING bin)으로 먼저 처리하세요.")
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError("무한대 값이 있습니다.")
    return X.astype(float)


def single_variable_separation(X: pd.DataFrame, y) -> list[str]:
    """변수 하나만으로 부도/정상이 완전히 갈리는 열(값 범위가 겹치지 않음). 있으면 MLE 가 존재하지 않는다."""
    X = _check_X(X)
    yv = validate_binary_target(y).to_numpy()
    out = []
    for c in X.columns:
        v = X[c].to_numpy()
        lo1, hi1, lo0, hi0 = v[yv == 1].min(), v[yv == 1].max(), v[yv == 0].min(), v[yv == 0].max()
        if hi0 < lo1 or hi1 < lo0:
            out.append(c)
    return out


@dataclass
class LogisticFit:
    """fit_logistic 의 결과. summary() 로 표를 본다."""

    names: list
    coef: np.ndarray
    se: np.ndarray
    cov: np.ndarray
    n: int
    n_events: int
    n_nonevents: int
    loglik: float
    null_loglik: float
    converged: bool
    n_iter: int
    has_intercept: bool
    warnings: list = field(default_factory=list)
    confidence: float = 0.95

    @property
    def intercept(self) -> float:
        return float(self.coef[0]) if self.has_intercept else 0.0

    @property
    def slopes(self) -> pd.Series:
        k = 1 if self.has_intercept else 0
        return pd.Series(self.coef[k:], index=self.names[k:])

    @property
    def pseudo_r2(self) -> float:
        """McFadden 의사 R² = 1 - lnL/lnL0. 선형회귀의 R² 와 같은 뜻이 아니다."""
        return float(1 - self.loglik / self.null_loglik)

    def summary(self) -> pd.DataFrame:
        z = self.coef / self.se
        p = 2 * stats.norm.sf(np.abs(z))
        zc = NormalDist().inv_cdf(0.5 + self.confidence / 2)
        lo, hi = self.coef - zc * self.se, self.coef + zc * self.se
        return pd.DataFrame(
            {
                "variable": self.names,
                "coef": self.coef,
                "std_err": self.se,
                "z": z,
                "p_value": p,
                "ci_low": lo,
                "ci_high": hi,
                "odds_ratio": np.exp(self.coef),
                "or_ci_low": np.exp(lo),
                "or_ci_high": np.exp(hi),
            }
        )

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        X = _check_X(X)
        cols = self.names[1:] if self.has_intercept else self.names
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"예측에 필요한 열이 없습니다: {missing}")
        A = X[cols].to_numpy()
        return (self.coef[0] + A @ self.coef[1:]) if self.has_intercept else A @ self.coef

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(부도). 학습 때 정한 계수를 그대로 쓴다."""
        return sigmoid(self.decision_function(X))


def fit_logistic(
    X: pd.DataFrame,
    y,
    add_intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-9,
    confidence: float = 0.95,
) -> LogisticFit:
    """규제 없는 로지스틱 회귀를 IRLS(뉴턴-랩슨)로 적합하고 표준오차·p-value 를 계산한다.

    표준오차는 정보행렬 X'WX 의 역행렬(W = p(1-p))에서 나온다. sklearn 의 C=inf(규제 없음)와
    같은 계수를 내며(테스트로 확인), 추가로 추론 정보를 준다.
    경고(fit.warnings): 수렴 실패 · 분리 의심 · 이벤트 수 부족 · 강한 공선성.
    """
    X = _check_X(X)
    yv = validate_binary_target(y).to_numpy().astype(float)
    if len(X) != len(yv):
        raise ValueError(f"X({len(X)})와 y({len(yv)})의 길이가 다릅니다.")
    if len(set(yv)) < 2:
        raise ValueError("부도와 정상이 모두 있어야 합니다.")
    names = (["intercept"] if add_intercept else []) + list(X.columns)
    A = X.to_numpy()
    if add_intercept:
        A = np.column_stack([np.ones(len(A)), A])
    n, k = A.shape
    if n <= k:
        raise ValueError(f"행({n})이 모수({k})보다 많아야 합니다.")
    sep = single_variable_separation(X, yv) if len(X.columns) else []
    beta = np.zeros(k)
    if add_intercept:
        pbar = yv.mean()
        beta[0] = np.log(pbar / (1 - pbar))
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        eta = A @ beta
        p = sigmoid(eta)
        w = np.clip(p * (1 - p), 1e-12, None)
        info = A.T @ (A * w[:, None])
        if np.linalg.cond(info) > 1e13 and sep:
            info = info + 1e-6 * np.eye(k)      # 완전 분리: MLE 가 없으므로 계산이 터지지 않게 아주 작은 값만 더하고 경고로 알린다
        elif np.linalg.cond(info) > 1e13:
            raise ValueError(
                "변수들이 거의 완전한 공선성을 가져 계수를 추정할 수 없습니다. "
                "vif_table 로 같은 정보를 담은 변수를 찾아 하나를 제거하세요."
            )
        step = np.linalg.solve(info, A.T @ (yv - p))
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    eta = A @ beta
    p = sigmoid(eta)
    w = np.clip(p * (1 - p), 1e-12, None)
    info = A.T @ (A * w[:, None])
    if sep:
        info = info + 1e-6 * np.eye(k)
    cov = np.linalg.inv(info)
    se = np.sqrt(np.diag(cov))
    eps = 1e-15
    loglik = float(np.sum(yv * np.log(np.clip(p, eps, 1)) + (1 - yv) * np.log(np.clip(1 - p, eps, 1))))
    pbar = yv.mean()
    null_ll = float(n * (pbar * np.log(pbar) + (1 - pbar) * np.log(1 - pbar)))

    warns = []
    n_events, n_non = int(yv.sum()), int(n - yv.sum())
    if not converged:
        warns.append(f"{max_iter}회 안에 수렴하지 않았습니다. 완전/준완전 분리 또는 공선성을 확인하세요.")
    if sep:
        warns.append(f"변수 하나만으로 부도/정상이 완전히 갈립니다: {sep}. 최대우도 추정치가 존재하지 않아 계수가 매우 커집니다(누수도 의심).")
    elif np.max(np.abs(beta)) > 15 or np.max(se) > 50:
        warns.append("계수 또는 표준오차가 비정상적으로 큽니다. 준완전 분리·희소 이벤트·공선성 가능성이 있습니다.")
    n_vars = k - (1 if add_intercept else 0)
    if n_vars > 0 and min(n_events, n_non) / n_vars < 10:
        warns.append(f"변수당 이벤트 수가 {min(n_events, n_non) / n_vars:.1f} 로 적습니다(관행적 기준 10). 계수가 불안정할 수 있습니다.")
    if n_vars > 1:
        Z = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
        cn = float(np.linalg.cond(Z.to_numpy()))
        if cn > 30:
            warns.append(f"표준화 설계행렬의 조건수가 {cn:.0f} 입니다(관행적 기준 30). 공선성으로 계수 해석이 불안정할 수 있습니다.")
    return LogisticFit(names, beta, se, cov, n, n_events, n_non, loglik, null_ll, converged, n_iter, add_intercept, warns, confidence)


def fit_regularized(
    X: pd.DataFrame, y, penalty: str = "l2", C: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 5000, tol: float = 1e-8
) -> LogisticRegression:
    """sklearn 규제 로지스틱. penalty = 'none' | 'l2' | 'l1' | 'elasticnet'.

    C 는 규제 강도의 역수(작을수록 강한 규제). 'none' 은 C=inf. L1/elasticnet 은 saga 해법을 쓴다.
    tol 은 수렴 허용오차다. sklearn 기본값(1e-4)은 점수 소수점 단위까지 재현하기에는 느슨해서 1e-8 로 낮췄다.
    입력 변수의 단위가 서로 다르면 StandardScaler 를 먼저 적용해야 규제가 공정하다(WOE 입력은 이미 같은 단위).
    """
    X = _check_X(X)
    yv = validate_binary_target(y)
    if penalty == "none":
        model = LogisticRegression(C=np.inf, max_iter=max_iter, tol=tol)
    elif penalty == "l2":
        model = LogisticRegression(C=C, max_iter=max_iter, tol=tol)
    elif penalty == "l1":
        model = LogisticRegression(C=C, l1_ratio=1.0, solver="saga", max_iter=max_iter, tol=tol)
    elif penalty == "elasticnet":
        model = LogisticRegression(C=C, l1_ratio=l1_ratio, solver="saga", max_iter=max_iter, tol=tol)
    else:
        raise ValueError("penalty 는 'none'/'l2'/'l1'/'elasticnet' 중 하나여야 합니다.")
    return model.fit(X, yv)


def regularization_path(X: pd.DataFrame, y, Cs, penalty: str = "l2") -> pd.DataFrame:
    """C 를 바꿔 가며 계수가 어떻게 줄어드는지 보는 표. (학습 데이터만 넣는다. 튜닝이 아니라 관찰용)"""
    X = _check_X(X)
    rows = []
    for C in Cs:
        m = fit_regularized(X, y, penalty=penalty, C=C)
        row = {"C": C, "n_nonzero": int((np.abs(m.coef_[0]) > 1e-8).sum()), "intercept": float(m.intercept_[0])}
        row.update({c: float(v) for c, v in zip(X.columns, m.coef_[0])})
        rows.append(row)
    return pd.DataFrame(rows)


def select_C_by_cv(X: pd.DataFrame, y, Cs, penalty: str = "l2", n_splits: int = 4) -> pd.DataFrame:
    """시간 순서를 지키는 교차검증(TimeSeriesSplit)으로 C 별 평균 log loss 를 계산한다.

    X, y 는 **시간순으로 정렬된 학습 데이터**여야 한다(앞=과거). 테스트 데이터는 절대 넣지 않는다.
    반환: C 별 mean_log_loss, 가장 낮은 행에 best=True.
    """
    from sklearn.model_selection import TimeSeriesSplit

    X = _check_X(X).reset_index(drop=True)
    yv = validate_binary_target(y).reset_index(drop=True)
    rows = []
    for C in Cs:
        losses = []
        for tr, va in TimeSeriesSplit(n_splits=n_splits).split(X):
            if yv.iloc[tr].nunique() < 2 or yv.iloc[va].nunique() < 2:
                continue
            m = fit_regularized(X.iloc[tr], yv.iloc[tr], penalty=penalty, C=C)
            losses.append(log_loss(yv.iloc[va], m.predict_proba(X.iloc[va])[:, 1], labels=[0, 1]))
        rows.append({"C": C, "mean_log_loss": float(np.mean(losses)) if losses else np.nan, "n_folds": len(losses)})
    out = pd.DataFrame(rows)
    if out["mean_log_loss"].notna().any():
        out["best"] = out["mean_log_loss"] == out["mean_log_loss"].min()
    else:
        out["best"] = False
    return out
