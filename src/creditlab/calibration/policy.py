"""컷오프 정책: 보정된 확률이 있을 때 기대이익 기준 이론 컷오프 m/(m+L)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.target import validate_binary_target


def theoretical_cutoff(margin: float, loss: float) -> float:
    """정상 1건의 이익 m, 부도 1건의 손실 L 일 때 (1−p)m − pL > 0 ⇔ p < m/(m+L). 확률이 보정되어 있을 때만 성립."""
    if margin <= 0 or loss <= 0:
        raise ValueError("margin 과 loss 는 양수여야 합니다.")
    return float(margin / (margin + loss))


def profit_by_cutoff(p, margin: float, loss: float, cutoffs, y=None) -> pd.DataFrame:
    """각 컷오프(p < cutoff 승인)의 승인율·기대이익. y 를 주면 *실현* 이익도 계산한다(평가 구간에서만)."""
    pv = np.asarray(p, dtype=float)
    yv = validate_binary_target(y).to_numpy() if y is not None else None
    rows = []
    for t in cutoffs:
        ap = pv < t
        row = {"cutoff": float(t), "approval_rate": float(ap.mean()),
               "expected_profit": float(((1 - pv[ap]) * margin - pv[ap] * loss).sum())}
        if yv is not None:
            row["realized_profit"] = float(((1 - yv[ap]) * margin - yv[ap] * loss).sum())
        rows.append(row)
    return pd.DataFrame(rows)
