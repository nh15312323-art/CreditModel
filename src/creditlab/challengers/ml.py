"""머신러닝 챌린저(트리·포레스트·그래디언트 부스팅) 도구 — scikit-learn 만 사용한다.

약속: 입력 데이터는 **시간순(앞=과거)** 으로 정렬되어 있어야 하고, 인코더·조기 종료·튜닝은
모두 **개발 구간 안에서만** 한다. 테스트 구간은 마지막에 한 번 `transform`/`predict` 만 한다.
(XGBoost/CatBoost 는 같은 아이디어의 다른 구현체이며 이 환경에서는 실행하지 못했다.)
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier

from ..data.target import validate_binary_target


def gini_impurity(bad: float, good: float) -> float:
    """노드의 지니 불순도 2p(1−p)."""
    n = bad + good
    if n <= 0:
        return 0.0
    p = bad / n
    return 2 * p * (1 - p)


def split_gain(left: tuple[float, float], right: tuple[float, float]) -> float:
    """(bad, good) 쌍 두 개로 나눴을 때의 불순도 감소(분할 이득)."""
    nl, nr = sum(left), sum(right)
    parent = gini_impurity(left[0] + right[0], left[1] + right[1])
    child = (nl * gini_impurity(*left) + nr * gini_impurity(*right)) / (nl + nr)
    return parent - child


class MLFeatureEncoder:
    """개발 구간에서만 범주 목록을 배우는 인코더. 수치형 결측은 NaN 으로 둔다(HGB 가 처리).

    범주형은 개발 구간 범주 → 0,1,2… 코드. **개발에 없던 범주·결측은 NaN** 이 된다(모형은 이를 '결측 방향'으로 처리).
    """

    def __init__(self, numeric, categorical=()):
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.categories_: dict[str, list] = {}

    @property
    def feature_names(self) -> list[str]:
        return self.numeric + self.categorical

    @property
    def cat_mask(self) -> np.ndarray:
        return np.array([False] * len(self.numeric) + [True] * len(self.categorical))

    def fit(self, df: pd.DataFrame) -> "MLFeatureEncoder":
        miss = [c for c in self.feature_names if c not in df.columns]
        if miss:
            raise ValueError(f"데이터에 없는 열입니다: {miss}")
        self.categories_ = {c: sorted(df[c].dropna().astype(str).unique()) for c in self.categorical}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.categories_ and self.categorical:
            raise RuntimeError("fit 을 먼저 호출하세요.")
        out = pd.DataFrame(index=df.index)
        for c in self.numeric:
            out[c] = df[c].astype(float)
        for c in self.categorical:
            mapping = {v: float(i) for i, v in enumerate(self.categories_[c])}
            out[c] = df[c].astype("object").map(lambda v, m=mapping: m.get(str(v), np.nan) if pd.notna(v) else np.nan).astype(float)
        return out


def fit_tree(X, y, max_depth=3, min_samples_leaf=50, seed=0) -> DecisionTreeClassifier:
    """단일 결정트리. NaN 은 지원 버전에서 그대로 처리된다."""
    return DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed).fit(X, validate_binary_target(y))


def fit_forest(X, y, n_estimators=300, min_samples_leaf=30, max_features="sqrt", seed=0) -> RandomForestClassifier:
    """랜덤포레스트(배깅). 깊은 트리 여러 개의 평균."""
    return RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features,
                                  n_jobs=1, random_state=seed).fit(X, validate_binary_target(y))


def fit_gbm(X, y, cat_mask=None, monotonic: dict | None = None, max_depth=3, learning_rate=0.05, max_iter=300,
            min_samples_leaf=50, l2=1.0, seed=0) -> HistGradientBoostingClassifier:
    """그래디언트 부스팅(HistGradientBoosting). monotonic={'변수': +1/−1} 은 그 변수가 커질 때 예측이
    비감소/비증가하도록 제약한다(범주형 변수에는 쓸 수 없다). 내장 조기 종료는 무작위 검증 분할이라 끈다.
    """
    X = pd.DataFrame(X)
    cst = None
    if monotonic:
        unknown = [k for k in monotonic if k not in X.columns]
        if unknown:
            raise ValueError(f"단조 제약 변수가 데이터에 없습니다: {unknown}")
        if any(v not in (-1, 1, 0) for v in monotonic.values()):
            raise ValueError("단조 제약 값은 +1, −1, 0 이어야 합니다.")
        cst = {k: int(v) for k, v in monotonic.items()}
    m = HistGradientBoostingClassifier(
        max_depth=max_depth, learning_rate=learning_rate, max_iter=max_iter, min_samples_leaf=min_samples_leaf,
        l2_regularization=l2, early_stopping=False, monotonic_cst=cst, categorical_features=cat_mask, random_state=seed)
    return m.fit(X, validate_binary_target(y))


def staged_log_loss(model, X_tr, y_tr, X_va, y_va) -> pd.DataFrame:
    """부스팅 반복 횟수별 학습/검증 log loss. 검증 곡선의 최저점이 적정 반복 횟수 후보."""
    ytr, yva = validate_binary_target(y_tr).to_numpy(), validate_binary_target(y_va).to_numpy()
    rows = []
    for i, (ptr, pva) in enumerate(zip(model.staged_predict_proba(X_tr), model.staged_predict_proba(X_va)), start=1):
        rows.append({"iteration": i, "train_log_loss": log_loss(ytr, ptr[:, 1], labels=[0, 1]),
                     "valid_log_loss": log_loss(yva, pva[:, 1], labels=[0, 1])})
    return pd.DataFrame(rows)


def fit_gbm_early_stop(X, y, cat_mask=None, monotonic=None, valid_frac=0.2, max_iter=600, **params):
    """시간순 조기 종료: 앞부분으로 학습·뒷부분(미래)으로 검증해 반복 횟수를 고른 뒤, 개발 전체로 다시 적합.

    반환: (최종 모형, 곡선 DataFrame, 선택된 반복 횟수)
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    yv = validate_binary_target(y).reset_index(drop=True)
    cut = int(len(X) * (1 - valid_frac))
    if cut < 50 or len(X) - cut < 50:
        raise ValueError("조기 종료용 분할에 쓸 표본이 너무 적습니다.")
    probe = fit_gbm(X.iloc[:cut], yv.iloc[:cut], cat_mask, monotonic, max_iter=max_iter, **params)
    curve = staged_log_loss(probe, X.iloc[:cut], yv.iloc[:cut], X.iloc[cut:], yv.iloc[cut:])
    best = int(curve.loc[curve["valid_log_loss"].idxmin(), "iteration"])
    final = fit_gbm(X, yv, cat_mask, monotonic, max_iter=best, **params)
    return final, curve, best


def tune_gbm_timecv(X, y, grid: dict, cat_mask=None, monotonic=None, n_splits=4, **fixed) -> pd.DataFrame:
    """격자 탐색 + 시간순 교차검증(평균 AUROC). 한 행 = 한 설정. 점수 높은 순.

    CV 최고 점수는 '여러 설정 중 최댓값'이라 낙관적이다. 선택 후에는 봉인된 테스트를 한 번만 본다.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    yv = validate_binary_target(y).reset_index(drop=True)
    keys = list(grid)
    rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        aucs = []
        for tr, va in TimeSeriesSplit(n_splits=n_splits).split(X):
            if yv.iloc[tr].nunique() < 2 or yv.iloc[va].nunique() < 2:
                continue
            m = fit_gbm(X.iloc[tr], yv.iloc[tr], cat_mask, monotonic, **{**fixed, **params})
            aucs.append(roc_auc_score(yv.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
        rows.append({**params, "mean_auc": float(np.mean(aucs)) if aucs else np.nan,
                     "std_auc": float(np.std(aucs)) if aucs else np.nan, "n_folds": len(aucs)})
    return pd.DataFrame(rows).sort_values("mean_auc", ascending=False).reset_index(drop=True)
