"""선택 편향(승인자만 결과를 관찰) 시뮬레이션. 가정에 의존한 *교육용 실험*이며 실제 편향 크기를 알려주지 않는다."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..evaluation import auroc


def selection_bias_experiment(df: pd.DataFrame, features, target: str = "default", accept_share: float = 0.6,
                              officer_weight: float = 1.5, officer_noise: float = 1.0, train_frac: float = 0.7, seed: int = 0) -> dict:
    """과거 승인 규칙(연체·사용률·조회) + 심사역 인상(결과와 상관, 데이터에 없음)으로 accept_share 만 승인했다고 가정.

    officer_weight=0 이면 승인이 *데이터에 있는 변수*만으로 결정되어 편향이 작고, 크면 *데이터에 없는 정보*에 의존해 커진다.
    """
    if not 0 < accept_share < 1:
        raise ValueError("accept_share 는 0~1 사이여야 합니다.")
    need = {"num_delinq_12m", "card_utilization", "inquiries_6m", target, *features}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"필요한 열이 없습니다: {sorted(miss)}")
    d = df.dropna(subset=list(need)).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rule = d["num_delinq_12m"] * 1.5 + d["card_utilization"] * 3 + d["inquiries_6m"] * 0.3
    officer = d[target] * 1.0 + rng.normal(0, officer_noise, len(d))
    z = lambda s: (s - s.mean()) / s.std()
    policy = z(rule) + officer_weight * z(officer)
    accepted = (policy <= policy.quantile(accept_share)).to_numpy()
    tr = np.arange(len(d)) < int(len(d) * train_frac)
    dev, hold = d[tr], d[~tr]
    acc_dev, acc_hold = dev[accepted[tr]], hold[accepted[~tr]]
    make = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    m_acc = make().fit(acc_dev[features], acc_dev[target])
    p_hold = m_acc.predict_proba(hold[features])[:, 1]
    return {
        "bad_rate_accepted_dev": float(acc_dev[target].mean()), "bad_rate_all_dev": float(dev[target].mean()),
        "auroc_on_accepted": auroc(acc_hold[target], m_acc.predict_proba(acc_hold[features])[:, 1]),
        "auroc_on_all": auroc(hold[target], p_hold),
        "mean_pd_all_applicants": float(p_hold.mean()), "bad_rate_all_applicants": float(hold[target].mean()),
        "pd_underestimate_ratio": float(p_hold.mean() / hold[target].mean()),
        "n_dev_accepted": int(len(acc_dev)), "n_dev_all": int(len(dev)),
    }
