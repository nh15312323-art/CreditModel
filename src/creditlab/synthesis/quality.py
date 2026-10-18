"""합성데이터 품질 3축: 충실도(닮았나) · 유용성(TSTR) · 프라이버시(DCR)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..evaluation import auroc


def _check_cols(real: pd.DataFrame, synth: pd.DataFrame, cols) -> list[str]:
    cols = list(cols)
    miss = [c for c in cols if c not in real.columns or c not in synth.columns]
    if miss:
        raise ValueError(f"두 데이터에 모두 있어야 하는 열이 없습니다: {miss}")
    return cols


def fidelity_table(real: pd.DataFrame, synth: pd.DataFrame, columns=None) -> pd.DataFrame:
    """열별 평균·표준편차·분위수·KS 통계량(분포 차이, 0 이 같음)."""
    cols = _check_cols(real, synth, real.columns if columns is None else columns)
    rows = []
    for c in cols:
        r, s = real[c].astype(float), synth[c].astype(float)
        rows.append({"column": c, "real_mean": r.mean(), "synth_mean": s.mean(), "real_std": r.std(), "synth_std": s.std(),
                     "q05_diff": s.quantile(0.05) - r.quantile(0.05), "q50_diff": s.quantile(0.5) - r.quantile(0.5),
                     "q95_diff": s.quantile(0.95) - r.quantile(0.95), "ks": float(ks_2samp(r, s).statistic)})
    return pd.DataFrame(rows)


def correlation_gap(real: pd.DataFrame, synth: pd.DataFrame, columns=None) -> dict:
    """상관행렬의 원소별 절대 차이(최대·평균)."""
    cols = _check_cols(real, synth, real.columns if columns is None else columns)
    diff = (real[cols].corr().to_numpy() - synth[cols].corr().to_numpy())
    iu = np.triu_indices_from(diff, k=1)
    return {"max_abs": float(np.abs(diff[iu]).max()), "mean_abs": float(np.abs(diff[iu]).mean()), "matrix": pd.DataFrame(diff, index=cols, columns=cols)}


def target_relationship(real: pd.DataFrame, synth: pd.DataFrame, feature: str, target: str, bins: int = 5) -> pd.DataFrame:
    """실제 표본의 분위수 구간별 부도율을 합성과 비교한다. 코퓰러가 P(Y|X) 를 약하게 만드는지 확인."""
    _check_cols(real, synth, [feature, target])
    edges = np.unique(np.quantile(real[feature], np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    def rate(d):
        g = pd.cut(d[feature], edges)
        return d.groupby(g, observed=False)[target].mean()
    out = pd.DataFrame({"real_rate": rate(real), "synth_rate": rate(synth)})
    out["gap"] = out["synth_rate"] - out["real_rate"]
    return out.reset_index(names="bin")


def _lr(features):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def tstr(real_train: pd.DataFrame, synth_train: pd.DataFrame, real_hold: pd.DataFrame, features, target: str, model_factory=None) -> dict:
    """TRTR(실로 학습·실로 평가)와 TSTR(합성으로 학습·실로 평가)의 AUROC. 평가는 항상 *실제 보류 데이터*."""
    features = list(features)
    make = model_factory or (lambda: _lr(features))
    out = {}
    for name, d in [("trtr", real_train), ("tstr", synth_train)]:
        m = make().fit(d[features], d[target])
        out[name] = auroc(real_hold[target], m.predict_proba(real_hold[features])[:, 1])
    out["gap"] = out["trtr"] - out["tstr"]
    return out


def dcr(reference: pd.DataFrame, candidate: pd.DataFrame, features) -> np.ndarray:
    """각 candidate 행에서 가장 가까운 reference 행까지의 거리(reference 표준편차로 표준화)."""
    features = list(features)
    scale = reference[features].std().replace(0, 1)
    tree = cKDTree((reference[features] / scale).to_numpy(dtype=float))
    d, _ = tree.query((candidate[features] / scale).to_numpy(dtype=float), k=1)
    return d


def dcr_summary(train: pd.DataFrame, synth: pd.DataFrame, holdout: pd.DataFrame, features) -> dict:
    """합성→학습 거리를 '보류 데이터→학습 거리'와 비교한다.

    합성이 학습 행을 외웠다면 합성→학습 거리가 보류→학습 거리보다 눈에 띄게 작다.
    exact_copies>0 이면 복제. DCR>0 이 안전하다는 뜻은 아니다(희귀 행 재식별, 멤버십 추론은 별도 평가).
    """
    ds, dh = dcr(train, synth, features), dcr(train, holdout, features)
    p5_hold = float(np.quantile(dh, 0.05))
    return {"synth_min": float(ds.min()), "synth_p5": float(np.quantile(ds, 0.05)), "synth_median": float(np.median(ds)),
            "holdout_p5": p5_hold, "holdout_median": float(np.median(dh)),
            "exact_copies": int((ds == 0).sum()), "share_closer_than_holdout_p5": float((ds < p5_hold).mean())}
