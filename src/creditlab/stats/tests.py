"""가설검정: 순열검정, DeLong(상관된 두 AUROC 비교)."""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import rankdata

from ..data.target import validate_binary_target


def permutation_test(a, b, n_perm: int = 5000, seed: int = 0) -> dict:
    """평균 차이(b − a)의 양측 순열검정. p = (극단 횟수 + 1)/(n_perm + 1) (0 이 나오지 않게 하는 표준 보정)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("각 집단에 최소 2개의 관측이 필요합니다.")
    rng = np.random.default_rng(seed)
    obs = b.mean() - a.mean()
    pooled = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(pooled[na:].mean() - pooled[:na].mean()) >= abs(obs) - 1e-15:
            count += 1
    return {"observed_diff": float(obs), "p_value": float((count + 1) / (n_perm + 1)), "n_perm": n_perm}


def _components(y: np.ndarray, s: np.ndarray):
    """DeLong 구조 성분(V10: 부도별, V01: 정상별)과 AUC. 중앙순위(midrank)로 O(N log N)."""
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    r_all = rankdata(s)
    r_pos, r_neg = rankdata(pos), rankdata(neg)
    v10 = (r_all[y == 1] - r_pos) / n
    v01 = 1.0 - (r_all[y == 0] - r_neg) / m
    return v10, v01, float(v10.mean())


def delong_test(y, scores_1, scores_2, confidence: float = 0.95) -> dict:
    """같은 표본에서 두 모형 AUROC 차이(1−2)의 DeLong 검정. 고객이 여러 번 나오면(군집) 가정이 깨진다.

    반환: auc1, auc2, diff, se, z, p_value, ci_lo, ci_hi. 두 점수가 완전히 같아 분산이 0 이면 z=0, p=1.
    """
    yv = validate_binary_target(y).to_numpy()
    s1, s2 = np.asarray(scores_1, float), np.asarray(scores_2, float)
    if not (len(yv) == len(s1) == len(s2)):
        raise ValueError("y, scores_1, scores_2 의 길이가 같아야 합니다.")
    m, n = int((yv == 1).sum()), int((yv == 0).sum())
    if m < 2 or n < 2:
        raise ValueError("부도와 정상이 각각 2건 이상 있어야 DeLong 검정을 할 수 있습니다.")
    v10_1, v01_1, a1 = _components(yv, s1)
    v10_2, v01_2, a2 = _components(yv, s2)
    S10 = np.cov(np.vstack([v10_1, v10_2]))
    S01 = np.cov(np.vstack([v01_1, v01_2]))
    S = S10 / m + S01 / n
    var = float(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    diff = a1 - a2
    if var <= 1e-18:
        z, p, se = (0.0, 1.0, 0.0) if abs(diff) < 1e-15 else (float("inf") * np.sign(diff), 0.0, 0.0)
    else:
        se = float(np.sqrt(var))
        z = diff / se
        p = float(2 * stats.norm.sf(abs(z)))
    q = stats.norm.ppf(0.5 + confidence / 2)
    return {"auc1": a1, "auc2": a2, "diff": float(diff), "se": se, "z": float(z), "p_value": p,
            "ci_lo": float(diff - q * se), "ci_hi": float(diff + q * se), "n_bad": m, "n_good": n}
