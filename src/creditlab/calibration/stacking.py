"""스태킹: 기본 모형의 *표본 외(OOF) 예측*으로 메타 모형을 학습한다(시간순 분할)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ..data.target import validate_binary_target
from ..models.logistic import fit_logistic, logit, sigmoid


def oof_predictions(factories: dict, X: pd.DataFrame, y, n_splits: int = 4) -> pd.DataFrame:
    """시간순 OOF 예측. factories = {이름: f(X_train, y_train) -> predict_proba 를 가진 모형}.

    TimeSeriesSplit 은 **가장 앞 구간에 대한 OOF 가 없으므로** 그 행은 NaN 이다(과거 데이터가 없으니 정상).
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    yv = validate_binary_target(y).reset_index(drop=True)
    out = pd.DataFrame(np.nan, index=X.index, columns=list(factories))
    for tr, va in TimeSeriesSplit(n_splits=n_splits).split(X):
        if yv.iloc[tr].nunique() < 2:
            continue
        for name, make in factories.items():
            m = make(X.iloc[tr], yv.iloc[tr])
            out.loc[out.index[va], name] = m.predict_proba(X.iloc[va])[:, 1]
    return out


class StackedModel:
    """기본 모형들의 OOF 로짓 → 로지스틱 메타 모형. 기본 모형은 마지막에 개발 전체로 다시 적합한다.

    메타 모형은 *기본 모형이 보지 못한* 예측으로 학습되므로 누수가 없다. 단, 최종 기본 모형은 OOF 때보다
    더 많은 데이터로 학습되어 분포가 약간 다르다(알려진 근사).
    """

    def __init__(self, factories: dict, n_splits: int = 4):
        self.factories, self.n_splits = factories, n_splits

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        yv = validate_binary_target(y).reset_index(drop=True)
        oof = oof_predictions(self.factories, X, yv, self.n_splits)
        ok = oof.notna().all(axis=1)
        self.n_meta_ = int(ok.sum())
        Z = oof.loc[ok].apply(lambda c: logit(np.clip(c, 1e-6, 1 - 1e-6)))
        f = fit_logistic(Z, yv[ok].reset_index(drop=True))
        self.meta_intercept_ = float(f.intercept)
        self.meta_weights_ = {k: float(v) for k, v in f.slopes.items()}
        self.base_ = {name: make(X, yv) for name, make in self.factories.items()}
        self.oof_ = oof
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = pd.DataFrame(X)
        z = self.meta_intercept_ + sum(
            w * logit(np.clip(self.base_[n].predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)) for n, w in self.meta_weights_.items())
        p = sigmoid(z)
        return np.column_stack([1 - p, p])
