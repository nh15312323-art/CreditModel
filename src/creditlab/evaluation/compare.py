"""모형 비교: 같은 표본에서의 지표 표와 **쌍체 부트스트랩** 차이 구간."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.target import validate_binary_target
from .discrimination import (auroc, brier_score, calibration_slope_intercept, calibration_table,
                             expected_calibration_error, ks_statistic, log_loss_score)


def paired_bootstrap_diff(metric, y, p_a, p_b, n_boot: int = 500, confidence: float = 0.95, seed: int = 0) -> dict:
    """같은 부트스트랩 표본으로 metric(a) − metric(b) 의 구간을 만든다(쌍체). 구간이 0 을 포함하면
    차이가 우연일 수 있다는 뜻이다. 한 클래스만 뽑힌 표본은 건너뛴다."""
    yv = validate_binary_target(y).to_numpy()
    a, b = np.asarray(p_a, float), np.asarray(p_b, float)
    if not (len(yv) == len(a) == len(b)):
        raise ValueError("y, p_a, p_b 의 길이가 같아야 합니다.")
    rng = np.random.default_rng(seed)
    diffs, skipped = [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, len(yv), len(yv))
        if yv[idx].min() == yv[idx].max():
            skipped += 1
            continue
        diffs.append(metric(yv[idx], a[idx]) - metric(yv[idx], b[idx]))
    point = metric(yv, a) - metric(yv, b)
    if not diffs:
        return {"diff": float(point), "lo": np.nan, "hi": np.nan, "includes_zero": None, "n_valid": 0, "skipped": skipped}
    al = (1 - confidence) / 2
    lo, hi = np.quantile(diffs, [al, 1 - al])
    return {"diff": float(point), "lo": float(lo), "hi": float(hi), "includes_zero": bool(lo <= 0 <= hi),
            "n_valid": len(diffs), "skipped": skipped}


def compare_models(y, preds: dict, n_bins: int = 5) -> pd.DataFrame:
    """모형별 AUROC·KS·Brier·log loss·ECE·보정 기울기/절편. preds = {이름: 예측 PD}."""
    rows = []
    for name, p in preds.items():
        cs = calibration_slope_intercept(y, p)
        rows.append({"model": name, "auroc": auroc(y, p), "ks": ks_statistic(y, p)["ks"], "brier": brier_score(y, p),
                     "log_loss": log_loss_score(y, p), "ece": expected_calibration_error(calibration_table(y, p, n_bins)),
                     "cal_slope": cs["slope"], "cal_intercept": cs["intercept"], "mean_pd": float(np.mean(p))})
    return pd.DataFrame(rows).set_index("model")
