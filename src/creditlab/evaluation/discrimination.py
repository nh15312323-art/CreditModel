"""변별력·확률 품질·보정 지표.

약속: 모든 함수의 `risk` 는 **클수록 위험**한 값이다(PD, 또는 −점수).
스코어카드 점수는 높을수록 안전하므로 `risk = -score` 로 넘긴다.
한 클래스만 있어 정의되지 않는 지표는 숫자를 억지로 만들지 않고 NaN 을 돌려준다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..data.target import validate_binary_target
from ..features.diagnostics import _wilson
from ..models.logistic import fit_logistic, logit


def _prep(y, risk) -> tuple[np.ndarray, np.ndarray]:
    yv = validate_binary_target(y).to_numpy()
    r = np.asarray(risk, dtype=float)
    if len(r) != len(yv):
        raise ValueError(f"y({len(yv)})와 risk({len(r)})의 길이가 다릅니다.")
    if np.isnan(r).any() or np.isinf(r).any():
        raise ValueError("risk 에 결측/무한대가 있습니다. 예측 전에 먼저 처리하세요.")
    return yv, r


def _one_class(yv) -> bool:
    return yv.min() == yv.max()


def auroc(y, risk) -> float:
    """AUROC = P(부도의 risk > 정상의 risk) + 0.5·P(동점). 순위합(Mann–Whitney)으로 계산한다."""
    yv, r = _prep(y, risk)
    if _one_class(yv):
        return float("nan")
    n1, n0 = int(yv.sum()), int((1 - yv).sum())
    ranks = rankdata(r)  # 동점은 평균 순위
    return float((ranks[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gini(y, risk) -> float:
    """Gini = 2·AUROC − 1.  (조직마다 AR/Gini 정의가 다를 수 있으니 보고서에 산식을 적는다.)"""
    a = auroc(y, risk)
    return float("nan") if np.isnan(a) else 2 * a - 1


def roc_points(y, risk) -> pd.DataFrame:
    """임계값 이상을 부도로 예측할 때의 (FPR, TPR). 동점은 한 점으로 묶는다."""
    yv, r = _prep(y, risk)
    if _one_class(yv):
        raise ValueError("한 클래스만 있어 ROC 를 그릴 수 없습니다.")
    order = np.argsort(-r, kind="mergesort")
    rs, ys = r[order], yv[order]
    last = np.r_[np.where(np.diff(rs) != 0)[0], len(rs) - 1]
    tp = np.cumsum(ys)[last]
    fp = (last + 1) - tp
    out = pd.DataFrame(
        {"threshold": rs[last], "fpr": fp / (1 - yv).sum(), "tpr": tp / yv.sum()}
    )
    first = pd.DataFrame({"threshold": [np.inf], "fpr": [0.0], "tpr": [0.0]})
    return pd.concat([first, out], ignore_index=True)


def ks_statistic(y, risk) -> dict:
    """KS = max |F_bad(t) − F_good(t)|.  위험 점수가 높은 쪽부터 누적한 TPR−FPR 의 최대값과 그 임계값."""
    pts = roc_points(y, risk)
    diff = pts["tpr"] - pts["fpr"]
    i = int(diff.idxmax())
    return {"ks": float(diff.iloc[i]), "threshold": float(pts["threshold"].iloc[i])}


def cap_accuracy_ratio(y, risk) -> float:
    """CAP 곡선 기반 AR = (CAP 면적 − 0.5) / (완벽모형 면적 − 0.5).

    동점이 없으면 Gini 와 같다. 동점 처리 방식이 다르면 값이 조금 달라질 수 있다.
    """
    yv, r = _prep(y, risk)
    if _one_class(yv):
        return float("nan")
    order = np.argsort(-r, kind="mergesort")
    ys = yv[order]
    n, nb = len(ys), ys.sum()
    x = np.r_[0.0, np.arange(1, n + 1) / n]
    cap = np.r_[0.0, np.cumsum(ys) / nb]
    area = np.trapezoid(cap, x)
    perfect = 1 - (nb / n) / 2
    return float((area - 0.5) / (perfect - 0.5))


def confusion_at(y, risk, threshold: float) -> dict:
    """risk ≥ threshold 이면 '부도 예측'. 임계값에 따라 모든 지표가 달라진다."""
    yv, r = _prep(y, risk)
    pred = r >= threshold
    tp = int((pred & (yv == 1)).sum())
    fp = int((pred & (yv == 0)).sum())
    fn = int((~pred & (yv == 1)).sum())
    tn = int((~pred & (yv == 0)).sum())

    def div(a, b):
        return float(a / b) if b else float("nan")

    return {
        "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": div(tp + tn, len(yv)), "precision": div(tp, tp + fp),
        "recall": div(tp, tp + fn), "specificity": div(tn, tn + fp),
        "f1": div(2 * tp, 2 * tp + fp + fn),
    }


def average_precision(y, risk) -> float:
    """평균 정밀도(PR-AUC 의 계단 근사). 무작위 모형의 값은 부도율과 같다."""
    yv, r = _prep(y, risk)
    if yv.sum() == 0:
        return float("nan")
    order = np.argsort(-r, kind="mergesort")
    rs, ys = r[order], yv[order]
    last = np.r_[np.where(np.diff(rs) != 0)[0], len(rs) - 1]
    tp = np.cumsum(ys)[last]
    prec = tp / (last + 1)
    rec = tp / yv.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def brier_score(y, p) -> float:
    yv, pv = _prep(y, p)
    if (pv < 0).any() or (pv > 1).any():
        raise ValueError("확률은 0~1 이어야 합니다.")
    return float(np.mean((pv - yv) ** 2))


def log_loss_score(y, p, eps: float = 1e-15) -> float:
    yv, pv = _prep(y, p)
    if (pv < 0).any() or (pv > 1).any():
        raise ValueError("확률은 0~1 이어야 합니다.")
    pv = np.clip(pv, eps, 1 - eps)
    return float(-np.mean(yv * np.log(pv) + (1 - yv) * np.log(1 - pv)))


def bootstrap_ci(metric, y, risk, n_boot: int = 500, confidence: float = 0.95, seed: int = 0) -> dict:
    """부트스트랩 신뢰구간. 한 클래스만 뽑힌 표본은 건너뛰고 건너뛴 횟수를 알려 준다."""
    yv, r = _prep(y, risk)
    rng = np.random.default_rng(seed)
    point = metric(yv, r)
    vals, skipped = [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, len(yv), len(yv))
        if _one_class(yv[idx]):
            skipped += 1
            continue
        vals.append(metric(yv[idx], r[idx]))
    if not vals:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_valid": 0, "skipped": skipped}
    a = (1 - confidence) / 2
    lo, hi = np.quantile(vals, [a, 1 - a])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "n_valid": len(vals), "skipped": skipped}


def discrimination_summary(y, risk) -> dict:
    """AUROC·Gini·KS·AP 한 번에. 표본 수와 부도 건수도 같이 돌려준다."""
    yv, r = _prep(y, risk)
    base = {"n": len(yv), "bads": int(yv.sum()), "bad_rate": float(yv.mean())}
    if _one_class(yv):
        return {**base, "auroc": np.nan, "gini": np.nan, "ks": np.nan, "avg_precision": average_precision(yv, r)}
    return {**base, "auroc": auroc(yv, r), "gini": gini(yv, r), "ks": ks_statistic(yv, r)["ks"],
            "avg_precision": average_precision(yv, r)}


def metrics_by_period(df: pd.DataFrame, period, y_col: str, risk) -> pd.DataFrame:
    """기간별 변별력. period 는 df 와 같은 길이의 구분값(예: 분기). 부도가 적은 기간은 불안정하다."""
    d = pd.DataFrame({"period": pd.Series(period).to_numpy(), "y": df[y_col].to_numpy(), "risk": np.asarray(risk, float)})
    rows = []
    for k, g in d.groupby("period", sort=True, observed=True):
        rows.append({"period": k, **discrimination_summary(g["y"], g["risk"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 보정
def calibration_table(y, p, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    """예측확률 구간별 평균 예측 vs 실제 부도율(+Wilson 구간). strategy: quantile | width."""
    yv, pv = _prep(y, p)
    if (pv < 0).any() or (pv > 1).any():
        raise ValueError("확률은 0~1 이어야 합니다.")
    if strategy == "quantile":
        edges = np.unique(np.quantile(pv, np.linspace(0, 1, n_bins + 1)))
    elif strategy == "width":
        edges = np.linspace(0, 1, n_bins + 1)
    else:
        raise ValueError("strategy 는 'quantile' 또는 'width' 입니다.")
    idx = np.clip(np.searchsorted(edges[1:-1], pv, side="right"), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        bad = int(yv[m].sum())
        lo, hi = _wilson(bad, n)
        rows.append({"bin": b + 1, "n": n, "mean_pred": float(pv[m].mean()), "obs_rate": bad / n,
                     "obs_lo": lo, "obs_hi": hi, "gap": float(pv[m].mean() - bad / n)})
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    """ECE = Σ (n_b/N)·|평균예측 − 실제부도율|.  구간 정의에 민감하다."""
    w = table["n"] / table["n"].sum()
    return float((w * (table["mean_pred"] - table["obs_rate"]).abs()).sum())


def calibration_slope_intercept(y, p, eps: float = 1e-6) -> dict:
    """logit(p) 에 로지스틱 회귀를 다시 적합: 기울기 1·절편 0 이면 보정이 잘 된 것.

    기울기 < 1 이면 예측이 과도하게 극단적(과적합), > 1 이면 너무 보수적이다.
    절편은 전반적으로 예측이 높은지/낮은지를 보여 준다(기울기와 함께 해석).
    """
    yv, pv = _prep(y, p)
    z = logit(np.clip(pv, eps, 1 - eps))
    fit = fit_logistic(pd.DataFrame({"logit_p": z}), pd.Series(yv))
    return {"intercept": float(fit.intercept), "slope": float(fit.slopes["logit_p"])}
