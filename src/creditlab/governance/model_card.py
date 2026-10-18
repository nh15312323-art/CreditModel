"""모형 카드(Mitchell et al., 2019) 생성과 완성도 점검. 템플릿: docs/model_card_template.md"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..env import environment_info
from ..reporting import dataframe_fingerprint

REQUIRED_FIELDS = {
    "overview": ["model_name", "intended_use", "out_of_scope"],
    "data": ["source", "period", "default_definition", "quality_issues", "population_gap"],
    "method": ["split", "variable_selection", "model_type", "score_formula"],
    "performance": ["metrics_table", "calibration", "grade_table", "challenger"],
    "monitoring": ["psi_plan", "alert_thresholds", "owner"],
    "limits": ["known_limitations", "assumptions", "fairness_check"],
    "reproducibility": ["code_version", "seed", "data_hash", "library_versions"],
}
PLACEHOLDER = "(미기입)"


def reproducibility_info(df: pd.DataFrame, seed: int, code_version: str = "unknown") -> dict:
    env = environment_info(seed)
    return {"code_version": code_version, "seed": seed, "data_hash": dataframe_fingerprint(df),
            "library_versions": ", ".join(f"{k} {v}" for k, v in env["packages"].items()) + f", python {env['python']}"}


def _is_filled(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return len(v.strip()) >= 3 and v.strip() != PLACEHOLDER
    if isinstance(v, (list, tuple, dict)):
        return len(v) > 0
    if isinstance(v, pd.DataFrame):
        return len(v) > 0
    return True


def check_model_card_completeness(info: dict, min_limitations: int = 3) -> dict:
    """빠진 항목과 '한계·가정' 최소 개수를 점검한다. 항목이 채워졌다는 것이 내용이 맞다는 뜻은 아니다."""
    missing = []
    for section, fields in REQUIRED_FIELDS.items():
        sec = info.get(section, {})
        for f in fields:
            if not _is_filled(sec.get(f)):
                missing.append(f"{section}.{f}")
    lim = info.get("limits", {}).get("known_limitations", [])
    short = isinstance(lim, (list, tuple)) and len([x for x in lim if isinstance(x, str) and len(x.strip()) >= 8]) < min_limitations
    return {"missing": missing, "limitations_enough": not short, "complete": not missing and not short}


def _fmt(v) -> str:
    if isinstance(v, pd.DataFrame):
        from ..reporting import md_table
        return md_table(v)
    if isinstance(v, (list, tuple)):
        return "\n".join(f"  - {x}" for x in v)
    if isinstance(v, dict):
        return "\n".join(f"  - {k}: {x}" for k, x in v.items())
    return str(v) if _is_filled(v) else PLACEHOLDER


def build_model_card(info: dict) -> str:
    """info(섹션 → 필드 → 값)로 마크다운 모형 카드를 만든다. 비어 있는 항목은 '(미기입)' 으로 드러낸다."""
    labels = {"overview": "1. 개요", "data": "2. 데이터", "method": "3. 방법", "performance": "4. 성능",
              "monitoring": "5. 모니터링", "limits": "6. 한계·가정·윤리", "reproducibility": "7. 재현 정보"}
    names = {"model_name": "모형 이름/버전", "intended_use": "의도된 용도", "out_of_scope": "적용하면 안 되는 범위",
             "source": "출처", "period": "기간", "default_definition": "부도 정의", "quality_issues": "품질 이슈와 처리",
             "population_gap": "학습 모집단과 적용 모집단의 차이", "split": "분할", "variable_selection": "변수 선택·비닝 근거",
             "model_type": "모형 종류", "score_formula": "점수 변환 수식", "metrics_table": "성능 지표", "calibration": "보정 결과",
             "grade_table": "등급별 부도율", "challenger": "챌린저 비교", "psi_plan": "PSI 계획", "alert_thresholds": "경보 기준",
             "owner": "조치 담당", "known_limitations": "알려진 한계", "assumptions": "가정", "fairness_check": "공정성 점검",
             "code_version": "코드 버전", "seed": "시드", "data_hash": "데이터 해시", "library_versions": "라이브러리 버전"}
    out = ["# 모형 카드", ""]
    for sec, fields in REQUIRED_FIELDS.items():
        out.append(f"## {labels[sec]}")
        for f in fields:
            val = info.get(sec, {}).get(f)
            body = _fmt(val)
            if "\n" in body:
                out.append(f"- **{names[f]}**:\n{body}")
            else:
                out.append(f"- **{names[f]}**: {body}")
        out.append("")
    return "\n".join(out)
