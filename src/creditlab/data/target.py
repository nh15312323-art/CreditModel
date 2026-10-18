"""타깃(부도 여부) 정의 검증과 기본 부도율 계산.

약속(convention): 이 프로그램에서 타깃은 항상 1 = 부도(bad), 0 = 정상(good) 이다.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

LABELS = {0: "정상(0)", 1: "부도(1)"}


def validate_binary_target(y, name: str = "target") -> pd.Series:
    """타깃이 결측 없는 0/1 값인지 검사하고 정수형 Series 로 돌려준다.

    불리언(True/False)은 1/0 으로 바꾼다. 그 밖의 값(문자열 'Y'/'N', 2, 결측 등)이
    있으면 무엇이 문제인지 알려주며 ValueError 를 낸다. 조용히 고쳐 쓰지 않는다.
    """
    s = pd.Series(y).copy()
    if len(s) == 0:
        raise ValueError(f"'{name}' 가 비어 있습니다. 데이터를 먼저 확인하세요.")
    if s.isna().any():
        raise ValueError(
            f"'{name}' 에 결측이 {int(s.isna().sum())}건 있습니다. "
            "결측 타깃을 정상으로 간주하면 부도율이 왜곡되므로, 먼저 원인을 확인하고 "
            "제외/재정의 규칙을 문서화하세요."
        )
    if pd.api.types.is_bool_dtype(s):
        s = s.astype(int)
    unique_values = s.drop_duplicates().tolist()
    if not set(unique_values) <= {0, 1}:
        raise ValueError(
            f"'{name}' 는 0/1 이어야 하는데 {unique_values[:10]} 값이 있습니다. "
            "(1=부도, 0=정상) 규칙으로 인코딩을 먼저 정리하세요."
        )
    return s.astype(int)


def default_rate(y) -> float:
    """부도율 = 부도 건수 / 전체 건수."""
    s = validate_binary_target(y)
    return float(s.mean())


def target_distribution(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """타깃 값별 건수와 비율 표. 컬럼: value, label, count, ratio."""
    if target not in df.columns:
        raise KeyError(f"타깃 컬럼 '{target}' 이(가) 없습니다. 컬럼: {list(df.columns)[:15]} ...")
    y = validate_binary_target(df[target], target)
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    total = int(counts.sum())
    return pd.DataFrame(
        {
            "value": counts.index,
            "label": [LABELS[v] for v in counts.index],
            "count": counts.values.astype(int),
            "ratio": counts.values / total,
        }
    )


def default_rate_ci(y, confidence: float = 0.95) -> tuple[float, float]:
    """부도율의 Wilson 점수 신뢰구간.

    p̂ = x/n,  z = 정규분포 분위수 일 때
        center = (p̂ + z²/2n) / (1 + z²/n)
        half   = z·sqrt(p̂(1-p̂)/n + z²/4n²) / (1 + z²/n)
    이벤트가 적어도 0~1 범위를 벗어나지 않아 단순 정규근사보다 안정적이다.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence 는 0과 1 사이여야 합니다.")
    s = validate_binary_target(y)
    n = len(s)
    p_hat = float(s.mean())
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def sample_sufficiency(y, min_events: int = 100, min_non_events: int = 100) -> dict:
    """표본 수와 이벤트(부도) 수가 학습에 충분한지 '경고'를 만든다.

    min_events 기본값 100 은 학습용 경험칙일 뿐 통계적 법칙이 아니다.
    변수 개수, 모형 복잡도, 목적에 따라 필요한 이벤트 수가 달라진다.
    """
    s = validate_binary_target(y)
    events = int(s.sum())
    non_events = int(len(s) - events)
    warnings_ = []
    if events < min_events:
        warnings_.append(f"부도(이벤트) 건수 {events}건 < 기준 {min_events}건: 추정이 불안정할 수 있습니다.")
    if non_events < min_non_events:
        warnings_.append(f"정상 건수 {non_events}건 < 기준 {min_non_events}건: 추정이 불안정할 수 있습니다.")
    return {
        "n": int(len(s)),
        "events": events,
        "non_events": non_events,
        "ok": not warnings_,
        "warnings": warnings_,
    }
