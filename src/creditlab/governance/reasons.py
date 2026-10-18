"""스코어카드 기반 불이익 사유 후보. 사용자에게 보이는 문구·개수·매핑은 법무/소비자 보호 기준을 따른다."""
from __future__ import annotations

import numpy as np
import pandas as pd


def reason_codes(card, applicants: pd.DataFrame, reference: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """변수별 점수가 기준집단(reference) 평균보다 *낮은 정도(shortfall)* 가 큰 순서로 사유 후보를 돌려준다.

    additivity: 한 신청자의 (점수 − 기준집단 평균 점수) = Σ_변수 (변수 점수 − 변수별 기준 평균 점수).
    상관된 변수가 있으면 기여가 나뉘어 사유가 불안정할 수 있다. 사유 산출 방법은 고정·검증한다.
    """
    if top_n < 1:
        raise ValueError("top_n 은 1 이상이어야 합니다.")
    ref_mean = card.points_frame(reference).mean()
    pts = card.points_frame(applicants)
    rows = []
    for idx in pts.index:
        diff = pts.loc[idx] - ref_mean
        worst = diff.sort_values().head(top_n)
        for rank, (var, d) in enumerate(worst.items(), start=1):
            rows.append({"applicant": idx, "rank": rank, "variable": var, "points": float(pts.loc[idx, var]),
                         "reference_points": float(ref_mean[var]), "shortfall": float(-d), "is_adverse": bool(d < 0)})
    return pd.DataFrame(rows)
