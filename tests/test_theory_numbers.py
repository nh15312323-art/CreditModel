"""이론 페이지의 손계산 예제 숫자를 고정한다. 페이지 숫자를 바꾸면 여기도 함께 바꿔야 한다."""
import math

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

from helpers import close


def test_stage9_metrics():
    s = np.array([0.9, 0.7, 0.4, 0.8, 0.5, 0.3, 0.2, 0.1])
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    auc = roc_auc_score(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    assert close(auc, 0.8) and close(2 * auc - 1, 0.6)
    assert close((tpr - fpr).max(), 0.6)
    assert close(brier_score_loss(y, s), 0.18625)
    assert abs(log_loss(y, s) - 0.5458) < 1e-4


def test_stage10_psi():
    e = np.array([0.10, 0.20, 0.40, 0.20, 0.10])
    a = np.array([0.08, 0.17, 0.38, 0.24, 0.13])
    assert abs(((a - e) * np.log(a / e)).sum() - 0.0255) < 5e-5


def test_stage11_gini_gain():
    g = lambda b, n: 2 * (b / (b + n)) * (1 - b / (b + n))
    assert close(g(4, 6) - (0.5 * g(1, 4) + 0.5 * g(3, 2)), 0.08)


def test_stage12_cutoff():
    assert close(1.0 / (1.0 + 8.0), 1 / 9)


def test_stage14_multiple_testing():
    assert abs((1 - 0.95**20) - 0.6415) < 1e-4


def test_score_scaling():
    factor = 20 / math.log(2)
    offset = 600 - factor * math.log(50)
    assert abs(factor - 28.8539) < 1e-4 and abs(offset - 487.123) < 1e-3


def test_stage15_ecl():
    assert close(0.05 * 0.45 * 1000, 22.5)


def test_stage5_hand_example_matches_library():
    from creditlab.features import woe_iv_from_counts

    t = woe_iv_from_counts([8, 2], [2, 8], smoothing=0)
    assert close(t["woe"][0], math.log(4)) and close(t["iv"].sum(), 1.2 * math.log(4))
