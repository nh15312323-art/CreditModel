"""모형 설명 도구(순열 중요도·부분의존). 설명은 '모형이 어떻게 예측하는가'이지 '무엇이 원인인가'가 아니다."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation import auroc


def _proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def permutation_importance(model, X: pd.DataFrame, y, n_repeats: int = 5, seed: int = 0, metric=auroc) -> pd.DataFrame:
    """변수 하나의 값을 섞었을 때 성능(기본 AUROC)이 얼마나 떨어지는가. **평가용(테스트/검증) 데이터**로 계산한다.

    상관된 변수끼리는 중요도가 서로 나뉘거나 한쪽에 몰려 보일 수 있다. 인과 효과가 아니다.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    yv = pd.Series(y).reset_index(drop=True)
    base = metric(yv, _proba(model, X))
    rng = np.random.default_rng(seed)
    rows = []
    for c in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            drops.append(base - metric(yv, _proba(model, Xp)))
        rows.append({"variable": c, "importance_mean": float(np.mean(drops)), "importance_std": float(np.std(drops))})
    out = pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    out.attrs["baseline_metric"] = float(base)
    return out


def partial_dependence(model, X: pd.DataFrame, feature: str, values=None, n_points: int = 10) -> pd.DataFrame:
    """변수를 값 하나로 고정했을 때의 평균 예측확률(부분의존). 상관된 변수가 있으면 비현실적 조합이 섞인다."""
    X = pd.DataFrame(X).reset_index(drop=True)
    if feature not in X.columns:
        raise ValueError(f"데이터에 없는 변수입니다: {feature}")
    if values is None:
        x = X[feature].dropna()
        values = np.unique(np.quantile(x, np.linspace(0.05, 0.95, n_points)))
    rows = []
    for v in values:
        Xv = X.copy()
        Xv[feature] = v
        rows.append({"value": float(v), "mean_pd": float(_proba(model, Xv).mean())})
    return pd.DataFrame(rows)


def monotone_violation(curve_values, direction: int = 1, tol: float = 1e-12) -> float:
    """곡선이 direction(+1 비감소 / −1 비증가) 방향에서 벗어난 가장 큰 크기. 0 이면 단조."""
    v = np.asarray(curve_values, dtype=float)
    d = np.diff(v) * direction
    return float(max(0.0, -d.min())) if len(d) and -d.min() > tol else 0.0
