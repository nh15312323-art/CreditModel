"""분포 안정성: PSI(변수는 CSI 라고도 부른다).

핵심 규칙: 구간 경계는 **기준(개발) 구간에서 한 번 정해 고정**하고 현재 표본에 그대로 적용한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_LABEL = "MISSING"
PSI_STABLE, PSI_WARN = 0.10, 0.25   # 업계 관행. 통계 법칙이 아니므로 조직이 정하고 문서화한다.


def psi_label(psi: float) -> str:
    if np.isnan(psi):
        return "계산불가"
    return "안정(<0.10)" if psi < PSI_STABLE else ("주의(0.10~0.25)" if psi < PSI_WARN else "큰 변화(≥0.25)")


def psi_terms(expected_pct, actual_pct, eps: float = 1e-4) -> np.ndarray:
    """구간별 기여분 (a−e)·ln(a/e). 0 비율은 eps 로 바닥을 둔다(방법을 문서화할 것)."""
    e = np.clip(np.asarray(expected_pct, dtype=float), eps, None)
    a = np.clip(np.asarray(actual_pct, dtype=float), eps, None)
    if e.shape != a.shape:
        raise ValueError("expected 와 actual 의 구간 수가 다릅니다.")
    return (a - e) * np.log(a / e)


def psi_from_counts(expected_counts, actual_counts, eps: float = 1e-4) -> float:
    ec, ac = np.asarray(expected_counts, float), np.asarray(actual_counts, float)
    if ec.sum() <= 0 or ac.sum() <= 0:
        return float("nan")
    return float(psi_terms(ec / ec.sum(), ac / ac.sum(), eps).sum())


def fixed_edges(expected, bins: int = 10) -> np.ndarray:
    """기준 표본의 분위수 경계. 양 끝은 −∞/+∞ 로 열어 새 극단값도 들어가게 한다."""
    x = pd.Series(expected).dropna().to_numpy(dtype=float)
    if len(x) == 0:
        raise ValueError("기준 표본이 비었습니다.")
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        raise ValueError("서로 다른 값이 너무 적어 구간을 만들 수 없습니다. 범주형으로 다루세요.")
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _bin_counts(x, edges) -> np.ndarray:
    s = pd.Series(x).astype(float)
    cnt = pd.cut(s.dropna(), edges).value_counts(sort=False).to_numpy().astype(float)
    return np.r_[cnt, float(s.isna().sum())]   # 마지막 칸 = 결측


def psi_numeric(expected, actual, bins: int = 10, edges=None, eps: float = 1e-4) -> dict:
    """수치형 PSI. edges 를 주지 않으면 expected 분위수로 만든다. 결측은 별도 구간."""
    ed = fixed_edges(expected, bins) if edges is None else np.asarray(edges, float)
    ec, ac = _bin_counts(expected, ed), _bin_counts(actual, ed)
    if ec[-1] == 0 and ac[-1] == 0:               # 양쪽 다 결측 없음 → 결측 칸 제거
        ec, ac = ec[:-1], ac[:-1]
    ep, ap = ec / ec.sum(), ac / ac.sum()
    terms = psi_terms(ep, ap, eps)
    labels = [f"({ed[i]:.4g}, {ed[i+1]:.4g}]" for i in range(len(ed) - 1)] + ([MISSING_LABEL] if len(ec) > len(ed) - 1 else [])
    table = pd.DataFrame({"bin": labels, "expected_pct": ep, "actual_pct": ap, "psi_term": terms})
    return {"psi": float(terms.sum()), "label": psi_label(float(terms.sum())), "table": table, "edges": ed}


def psi_categorical(expected, actual, eps: float = 1e-4) -> dict:
    """범주형 PSI. 기준에 없던 범주는 별도 구간(기준 비율 0 → eps)으로 드러난다."""
    e = pd.Series(expected).astype(object).where(pd.Series(expected).notna(), MISSING_LABEL).astype(str)
    a = pd.Series(actual).astype(object).where(pd.Series(actual).notna(), MISSING_LABEL).astype(str)
    cats = sorted(set(e) | set(a))
    ep = e.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    ap = a.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    terms = psi_terms(ep, ap, eps)
    table = pd.DataFrame({"bin": cats, "expected_pct": ep, "actual_pct": ap, "psi_term": terms})
    return {"psi": float(terms.sum()), "label": psi_label(float(terms.sum())), "table": table}


def psi_from_bins(expected_bins, actual_bins, order=None, eps: float = 1e-4) -> dict:
    """이미 구간 이름이 붙은 두 Series(예: WoeBinner.transform_bins 결과, 등급)의 PSI."""
    e, a = pd.Series(expected_bins).astype(str), pd.Series(actual_bins).astype(str)
    cats = list(order) if order is not None else sorted(set(e) | set(a))
    ep = e.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    ap = a.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    terms = psi_terms(ep, ap, eps)
    table = pd.DataFrame({"bin": cats, "expected_pct": ep, "actual_pct": ap, "psi_term": terms})
    return {"psi": float(terms.sum()), "label": psi_label(float(terms.sum())), "table": table}


def csi_table(dev: pd.DataFrame, current: pd.DataFrame, binners: dict, eps: float = 1e-4) -> pd.DataFrame:
    """변수별 CSI. 모형에 쓰인 변수의 **개발 구간 비닝 경계**를 현재 표본에 그대로 적용한다."""
    rows = []
    for name, b in binners.items():
        r = psi_from_bins(b.transform_bins(dev[name]), b.transform_bins(current[name]),
                          order=list(b.table_["bin"]), eps=eps)
        top = r["table"].loc[r["table"]["psi_term"].idxmax()]
        rows.append({"variable": name, "psi": r["psi"], "label": r["label"], "largest_bin": top["bin"],
                     "largest_term": float(top["psi_term"])})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def expected_psi_noise(k: int, n_expected: int, n_actual: int) -> float:
    """분포 변화가 전혀 없을 때 우연으로 생기는 PSI 의 대략적 기댓값 (k−1)(1/n_e + 1/n_a)."""
    return float((k - 1) * (1 / n_expected + 1 / n_actual))
