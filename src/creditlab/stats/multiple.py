"""다중검정 보정: Bonferroni · Holm · Benjamini–Hochberg."""
from __future__ import annotations

import numpy as np
import pandas as pd


def familywise_error_rate(m: int, alpha: float = 0.05) -> float:
    """귀무가설이 모두 참인 m 번의 독립 검정에서 하나라도 유의할 확률 1−(1−α)^m."""
    if m < 1 or not 0 < alpha < 1:
        raise ValueError("m≥1, 0<alpha<1 이어야 합니다.")
    return float(1 - (1 - alpha) ** m)


def _check(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or len(p) == 0 or np.isnan(p).any() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-value 는 0~1 사이의 결측 없는 1차원 목록이어야 합니다.")
    return p


def adjust_pvalues(pvals, method: str = "bh") -> np.ndarray:
    """보정된 p-value. method: 'bonferroni' | 'holm' | 'bh'. 보정 p < α 이면 기각."""
    p = _check(pvals)
    m = len(p)
    order = np.argsort(p)
    ps = p[order]
    if method == "bonferroni":
        adj = np.minimum(ps * m, 1.0)
    elif method == "holm":
        adj = np.minimum(np.maximum.accumulate((m - np.arange(m)) * ps), 1.0)
    elif method == "bh":
        adj = np.minimum.accumulate((ps * m / np.arange(1, m + 1))[::-1])[::-1]
        adj = np.minimum(adj, 1.0)
    else:
        raise ValueError("method 는 'bonferroni', 'holm', 'bh' 중 하나입니다.")
    out = np.empty(m)
    out[order] = adj
    return out


def multiple_testing_table(pvals, names=None, alpha: float = 0.05) -> pd.DataFrame:
    """개별/Bonferroni/Holm/BH 기각 여부를 한 표로."""
    p = _check(pvals)
    names = list(names) if names is not None else [f"test{i + 1}" for i in range(len(p))]
    if len(names) != len(p):
        raise ValueError("names 와 pvals 의 길이가 다릅니다.")
    t = pd.DataFrame({"test": names, "p": p})
    t["raw_reject"] = p < alpha
    for m in ["bonferroni", "holm", "bh"]:
        t[f"p_{m}"] = adjust_pvalues(p, m)
        t[f"{m}_reject"] = t[f"p_{m}"] < alpha
    return t
