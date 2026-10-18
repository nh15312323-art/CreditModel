"""실습 과제 자동 점검기. 정답을 알려주기보다 '빠진 것'을 짚어 준다."""

from __future__ import annotations

import pandas as pd

from ..data.validation import range_violations

TARGET_DEFINITION_FIELDS = {
    "observation_unit": "관측단위 (행 1개는 무엇인가? 예: 대출 신청 1건, 고객 1명)",
    "observation_point": "기준시점 (설명변수를 언제 시점까지 쓰는가?)",
    "performance_window": "성과관찰기간 (기준시점 이후 몇 개월 동안 부도를 보는가?)",
    "default_definition": "부도 정의 (어떤 상태를 부도(1)로 보는가?)",
    "population": "개발 모집단 (어떤 신청/고객이 포함되는가?)",
    "exclusions": "제외 기준 (무엇을 왜 제외하는가?)",
    "data_source": "데이터 출처와 라이선스",
}

_PLACEHOLDERS = {"", "todo", "tbd", "?", "모름", "미정", "none"}


def check_target_definition(definition: dict | None) -> dict:
    """타깃 정의서 딕셔너리의 빈 항목을 찾아 알려준다. 내용의 옳고 그름은 판단하지 않는다."""
    definition = definition or {}
    missing = []
    for key in TARGET_DEFINITION_FIELDS:
        value = definition.get(key)
        if value is None or str(value).strip().lower() in _PLACEHOLDERS:
            missing.append(key)
    result = {"ok": not missing, "missing": missing}
    if missing:
        print("아직 채우지 않은 항목:")
        for key in missing:
            print(f"  - {key}: {TARGET_DEFINITION_FIELDS[key]}")
    else:
        print("모든 항목이 채워졌습니다. 이제 각 항목이 '측정 가능하게' 쓰였는지 스스로 읽어 보세요.")
    return result


def check_cleaning_result(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame | None,
    id_col: str,
    text_numeric_cols: list[str],
    range_rules: dict,
) -> dict:
    """정제 결과가 기본 요구사항을 지켰는지 점검한다.

    요구사항: (1) 원본 비수정  (2) 완전 중복 제거  (3) 문자형 숫자 변환
              (4) 범위 오류는 결측으로 바꾸거나 제거(값을 그대로 두지 않음)
    """
    if cleaned is None:
        print("cleaned 가 비어 있습니다. 위 실습 셀에서 cleaned 를 만들어 주세요.")
        return {"ok": False, "problems": ["cleaned 없음"]}
    problems = []
    if cleaned is raw:
        problems.append("cleaned 가 raw 와 같은 객체입니다. raw.copy() 로 복사본에서 작업하세요.")
    if cleaned.duplicated().sum() > 0:
        problems.append(f"완전 중복 행이 {int(cleaned.duplicated().sum())}개 남아 있습니다.")
    if cleaned[id_col].duplicated().sum() > 0:
        problems.append(f"{id_col} 가 아직 중복입니다.")
    for col in text_numeric_cols:
        if not pd.api.types.is_numeric_dtype(cleaned[col]):
            problems.append(f"'{col}' 이(가) 아직 숫자형이 아닙니다. pd.to_numeric(..., errors='coerce') 를 써 보세요.")
    remaining = range_violations(cleaned, range_rules)["n_violations"].sum()
    if remaining > 0:
        problems.append(f"범위를 벗어난 값이 {int(remaining)}건 남아 있습니다 (결측 처리 또는 제거 + 이유 기록).")
    if problems:
        print("확인할 점:")
        for p in problems:
            print("  -", p)
    else:
        print("기본 요구사항을 모두 만족합니다. 각 처리의 '이유'를 cleaning_log 에 적었는지도 확인하세요.")
    return {"ok": not problems, "problems": problems}
