"""기대신용손실(ECL) = PD × LGD × EAD. 보정된 PD 가 전제다."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _check_unit(x, name: str):
    a = np.asarray(x, dtype=float)
    if np.isnan(a).any() or ((a < 0) | (a > 1)).any():
        raise ValueError(f"{name} 는 결측 없이 0~1 사이여야 합니다.")
    return a


def expected_credit_loss(pd_, lgd, ead):
    """ECL = PD × LGD × EAD (스칼라 또는 배열)."""
    p, l = _check_unit(pd_, "PD"), _check_unit(lgd, "LGD")
    e = np.asarray(ead, dtype=float)
    if np.isnan(e).any() or (e < 0).any():
        raise ValueError("EAD 는 결측 없이 0 이상이어야 합니다.")
    out = p * l * e
    return float(out) if np.ndim(out) == 0 else out


def portfolio_ecl(table: pd.DataFrame, count="n", pd_col="pd", lgd_col="lgd", ead_col="ead") -> pd.DataFrame:
    """등급(행)별 ECL 과 합계. 각 행: 고객 수, PD, LGD, EAD(1건당)."""
    t = table.copy()
    t["ecl"] = [t[count].iloc[i] * expected_credit_loss(t[pd_col].iloc[i], t[lgd_col].iloc[i], t[ead_col].iloc[i]) for i in range(len(t))]
    t.attrs["total_ecl"] = float(t["ecl"].sum())
    return t


def ecl_pd_bias_impact(table: pd.DataFrame, bias_factor: float, **cols) -> dict:
    """PD 를 bias_factor 배로 과소/과대 추정했을 때 충당금 변화(보정 오류가 회계 숫자가 되는 경로)."""
    base = portfolio_ecl(table, **cols).attrs["total_ecl"]
    biased = table.copy()
    pcol = cols.get("pd_col", "pd")
    biased[pcol] = np.clip(biased[pcol] * bias_factor, 0, 1)
    b = portfolio_ecl(biased, **cols).attrs["total_ecl"]
    return {"base_ecl": base, "biased_ecl": b, "change_pct": float((b / base - 1) * 100) if base else float("nan")}
