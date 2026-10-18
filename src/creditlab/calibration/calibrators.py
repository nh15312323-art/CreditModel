"""확률 보정기. 보정은 **학습에 쓰지 않은 보정 전용 구간**에서 적합한다.

Platt·절편 보정은 단조 변환이라 AUROC 를 바꾸지 않는다. Isotonic 은 계단(동점)을 만들어 순위가 조금 변할 수 있다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.isotonic import IsotonicRegression

from ..data.target import validate_binary_target
from ..models.logistic import fit_logistic, logit, sigmoid

_EPS = 1e-6


def _clip(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if np.isnan(p).any():
        raise ValueError("확률에 결측이 있습니다.")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("확률은 0~1 이어야 합니다.")
    return np.clip(p, _EPS, 1 - _EPS)


class PlattCalibrator:
    """p' = σ(a·logit(p) + b). 단순·안정. 왜곡이 시그모이드형일 때 적합."""

    def fit(self, p, y):
        z = pd.DataFrame({"z": logit(_clip(p))})
        f = fit_logistic(z, validate_binary_target(y).reset_index(drop=True))
        self.a_, self.b_ = float(f.slopes["z"]), float(f.intercept)
        return self

    def predict(self, p) -> np.ndarray:
        return sigmoid(self.a_ * logit(_clip(p)) + self.b_)


class IsotonicCalibrator:
    """단조 증가 계단함수. 유연하지만 보정 표본이 적으면 과적합한다."""

    def fit(self, p, y):
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
            np.asarray(p, float), validate_binary_target(y).to_numpy())
        return self

    def predict(self, p) -> np.ndarray:
        return np.asarray(self.iso_.predict(np.asarray(p, float)))


class InterceptCalibrator:
    """로그오즈에 상수 b 를 더해 평균 예측을 목표 부도율에 맞춘다(전체 수준 이동 = prior shift 보정)."""

    def __init__(self, target_rate: float | None = None):
        self.target_rate = target_rate

    def fit(self, p, y=None):
        target = self.target_rate
        if target is None:
            if y is None:
                raise ValueError("target_rate 또는 y 중 하나가 필요합니다.")
            target = float(validate_binary_target(y).mean())
        if not 0 < target < 1:
            raise ValueError("목표 부도율은 0~1 사이여야 합니다.")
        z = logit(_clip(p))
        self.b_ = float(brentq(lambda b: sigmoid(z + b).mean() - target, -20, 20))
        self.target_ = float(target)
        return self

    def predict(self, p) -> np.ndarray:
        return sigmoid(logit(_clip(p)) + self.b_)
