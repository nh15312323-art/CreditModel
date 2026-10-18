"""가우시안 코퓰러 합성기(수치·이진 열). 주변분포와 *선형적* 상관만 보존한다.

이 합성기는 **교육용**이다. 비선형·상호작용(P(Y|X))을 놓치기 쉽고, 합성데이터는 자동 익명화가 아니다.
(규칙으로 만든 `creditlab.data.make_synthetic_credit_data` 와는 다르다: 이 합성기는 *주어진 실제 표본을 학습*한다.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


class GaussianCopulaSynthesizer:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, df: pd.DataFrame) -> "GaussianCopulaSynthesizer":
        if df.shape[1] == 0 or len(df) < 30:
            raise ValueError("열이 있어야 하고 행이 30개 이상이어야 합니다.")
        bad = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        if bad:
            raise ValueError(f"수치형이 아닌 열이 있습니다(범주형은 먼저 인코딩): {bad}")
        if df.isna().any().any():
            raise ValueError("결측이 있습니다. 합성 전에 먼저 처리하세요(처리 방법도 기록).")
        self.columns_ = list(df.columns)
        self.sorted_ = {c: np.sort(df[c].to_numpy(dtype=float)) for c in self.columns_}
        self.discrete_ = {c: bool(df[c].nunique() <= 2 or np.allclose(df[c], np.round(df[c]))) for c in self.columns_}
        self.is_int_ = {c: bool(np.allclose(df[c], np.round(df[c]))) for c in self.columns_}
        n = len(df)
        z = np.column_stack([norm.ppf(rankdata(df[c]) / (n + 1)) for c in self.columns_])
        corr = np.corrcoef(z, rowvar=False)
        corr = np.atleast_2d(corr)
        w, v = np.linalg.eigh(corr)                       # 수치 오차로 양의 준정부호가 깨지지 않게 보정
        w = np.clip(w, 1e-8, None)
        corr = (v * w) @ v.T
        d = np.sqrt(np.diag(corr))
        self.corr_ = corr / np.outer(d, d)
        self.n_train_ = n
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if not hasattr(self, "corr_"):
            raise RuntimeError("fit 을 먼저 호출하세요.")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        z = rng.multivariate_normal(np.zeros(len(self.columns_)), self.corr_, size=n)
        u = norm.cdf(z)
        out = {}
        for j, c in enumerate(self.columns_):
            method = "inverted_cdf" if self.discrete_[c] else "linear"     # 이산 열은 관측된 값만 내보낸다
            out[c] = np.quantile(self.sorted_[c], u[:, j], method=method)
        return pd.DataFrame(out)
