"""교육용 합성 소액대출 데이터 생성기.

주의: 알려진 확률식으로 만든 가짜 데이터다. 이 데이터에서 얻은 성능이나 부도율은
실제 모집단에 대한 근거가 될 수 없다. 오프라인 실습과 테스트를 위한 용도다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "default"
ID_COL = "application_id"

DESCRIPTIONS = {
    "application_id": "대출 신청 번호(가상)",
    "application_date": "신청일",
    "age": "신청 시점 나이(세)",
    "income_10k": "연소득(만원)",
    "loan_amount_10k": "신청 금액(만원)",
    "employment_years": "현 직장 근속연수(년)",
    "card_utilization": "카드 한도 대비 사용률(0~1)",
    "num_delinq_12m": "최근 12개월 연체 건수",
    "inquiries_6m": "최근 6개월 신용조회 횟수",
    "housing_type": "주거 형태(own/rent/family/other)",
    "region": "지역 코드(A~D, 가상)",
    "default": "타깃: 신청 후 12개월 내 90일 이상 연체 여부(가상), 1=부도, 0=정상",
}

# 업무 상식에 근거한 값 범위 규칙 (품질 점검 실습에서 사용)
RANGE_RULES = {
    "age": (19, 100),
    "card_utilization": (0.0, 1.0),
    "income_10k": (0, None),
    "loan_amount_10k": (0, None),
}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_synthetic_credit_data(
    n: int = 5000,
    seed: int = 42,
    inject_quality_issues: bool = False,
) -> pd.DataFrame:
    """합성 대출 신청 데이터를 만든다.

    같은 (n, seed, inject_quality_issues) 이면 항상 같은 결과가 나온다.

    inject_quality_issues=True 이면 품질 점검 연습용으로 다음을 일부러 넣는다.
      * income_10k 결측(근속연수가 짧을수록 더 자주 빠짐 → 결측이 무작위가 아님)
      * employment_years, housing_type 소량 결측
      * 완전 중복 행 약 1%
      * 범위 오류(나이 -1/150, 사용률 > 1)
      * inquiries_6m 이 문자열로 저장되고 "N/A" 가 섞임
    """
    if n < 50:
        raise ValueError("n 은 50 이상이어야 합니다.")
    rng = np.random.default_rng(seed)

    application_id = np.arange(100001, 100001 + n)
    start = np.datetime64("2022-01-01")
    application_date = start + rng.integers(0, 730, size=n).astype("timedelta64[D]")
    age = np.clip(np.round(rng.normal(40, 11, n)), 20, 75).astype(int)
    income = np.round(np.exp(rng.normal(8.4, 0.45, n)))  # 만원, 중앙값 약 4,400
    loan_amount = np.round(income * rng.uniform(0.1, 1.2, n))
    employment_years = np.round(np.clip(rng.gamma(2.0, 3.0, n), 0, 40), 1)
    utilization = np.round(rng.beta(2, 3, n), 3)
    num_delinq = rng.poisson(0.2 + 0.9 * utilization)
    inquiries = rng.poisson(1.0 + 0.8 * utilization)
    housing = rng.choice(["own", "rent", "family", "other"], size=n, p=[0.35, 0.40, 0.20, 0.05])
    region = rng.choice(["A", "B", "C", "D"], size=n, p=[0.4, 0.3, 0.2, 0.1])

    # 진짜 부도 확률식(학습자는 이 식을 모른다고 가정하고 모형으로 복원을 시도한다)
    lti = loan_amount / income
    z = (
        -3.5
        + 2.0 * (utilization - 0.4)
        + 0.55 * num_delinq
        + 1.1 * lti
        - 0.025 * (age - 40)
        - 0.05 * employment_years
        + 0.30 * (housing == "rent")
        + 0.15 * inquiries
    )
    p_default = _sigmoid(z)
    default = (rng.random(n) < p_default).astype(int)

    df = pd.DataFrame(
        {
            "application_id": application_id,
            "application_date": pd.to_datetime(application_date),
            "age": age,
            "income_10k": income,
            "loan_amount_10k": loan_amount,
            "employment_years": employment_years,
            "card_utilization": utilization,
            "num_delinq_12m": num_delinq.astype(int),
            "inquiries_6m": inquiries.astype(int),
            "housing_type": housing,
            "region": region,
            "default": default,
        }
    )

    if inject_quality_issues:
        df = _inject_quality_issues(df, rng)

    df.attrs["source"] = "synthetic"
    df.attrs["seed"] = seed
    return df


def _inject_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    # 1) 소득 결측: 근속연수 1년 미만이면 결측 확률이 훨씬 높다(MAR 성격)
    p_missing_income = np.where(df["employment_years"] < 1.0, 0.25, 0.03)
    df.loc[rng.random(n) < p_missing_income, "income_10k"] = np.nan

    # 2) 소량 무작위 결측
    df.loc[rng.random(n) < 0.02, "employment_years"] = np.nan
    df["housing_type"] = df["housing_type"].astype(object)
    df.loc[rng.random(n) < 0.01, "housing_type"] = np.nan

    # 3) 범위 오류
    bad_age_idx = rng.choice(n, size=4, replace=False)
    df.loc[bad_age_idx[:2], "age"] = -1
    df.loc[bad_age_idx[2:], "age"] = 150
    bad_util_idx = rng.choice(n, size=5, replace=False)
    df.loc[bad_util_idx, "card_utilization"] = 1.5

    # 4) 문자형 숫자 + "N/A"
    inquiries_text = df["inquiries_6m"].astype(str).astype(object)
    inquiries_text[rng.random(n) < 0.015] = "N/A"
    df["inquiries_6m"] = inquiries_text

    # 5) 완전 중복 행 약 1% 추가 후 순서 섞기
    dup = df.sample(n=max(1, n // 100), random_state=int(rng.integers(0, 2**31 - 1)))
    df = pd.concat([df, dup], ignore_index=True)
    df = df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
    return df
