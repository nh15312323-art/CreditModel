"""집단별 점검. 지표는 여러 개이고 *동시에 만족할 수 없는 경우*가 있다. 법적 판단 기준이 아니다."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.target import validate_binary_target
from ..evaluation import auroc


def approval_ratio(approved, group) -> tuple[dict, float]:
    """집단별 승인율과 (최저 승인율 / 최고 승인율). 최고 승인율이 0 이면 NaN."""
    a, g = np.asarray(approved, dtype=float), np.asarray(group)
    if len(a) != len(g):
        raise ValueError("approved 와 group 의 길이가 다릅니다.")
    rates = {k: float(a[g == k].mean()) for k in pd.unique(g)}
    hi = max(rates.values())
    return rates, (min(rates.values()) / hi if hi > 0 else float("nan"))


def group_metrics(y, pd_pred, approved, group, min_n: int = 100) -> pd.DataFrame:
    """집단별 건수·실제 부도율·평균 PD·보정 차이·승인율·AUROC. min_n 미만 집단은 small_sample=True."""
    yv = validate_binary_target(y).to_numpy()
    p, a, g = np.asarray(pd_pred, float), np.asarray(approved, float), np.asarray(group)
    if not (len(yv) == len(p) == len(a) == len(g)):
        raise ValueError("입력 길이가 모두 같아야 합니다.")
    rows = []
    for k in sorted(pd.unique(g), key=str):
        m = g == k
        rows.append({"group": k, "n": int(m.sum()), "bad_rate": float(yv[m].mean()), "mean_pd": float(p[m].mean()),
                     "calibration_gap": float(p[m].mean() - yv[m].mean()), "approval_rate": float(a[m].mean()),
                     "auroc": auroc(yv[m], p[m]), "small_sample": bool(m.sum() < min_n)})
    return pd.DataFrame(rows)
