"""데이터 사전과 품질 진단 (결측·중복·범위·문자형 숫자·상수열).

모든 함수는 DataFrame 을 읽기만 하고 원본을 바꾸지 않는다.
"""

from __future__ import annotations

import pandas as pd
from pandas.api import types as ptypes


def infer_kind(series: pd.Series) -> str:
    """변수 종류를 boolean / datetime / numeric / categorical 중 하나로 분류."""
    if ptypes.is_bool_dtype(series):
        return "boolean"
    if ptypes.is_datetime64_any_dtype(series):
        return "datetime"
    if ptypes.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def data_dictionary(df: pd.DataFrame, descriptions: dict | None = None, example_n: int = 3) -> pd.DataFrame:
    """컬럼별 자료형·종류·결측·고유값·범위·예시를 담은 데이터 사전 표.

    descriptions 에 {컬럼명: 설명} 을 주면 'description' 컬럼에 채운다.
    설명이 없는 컬럼은 빈 문자열이므로, 학습자가 직접 채우는 것이 과제다.
    """
    descriptions = descriptions or {}
    n_rows = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        kind = infer_kind(s)
        non_null = s.dropna()
        n_missing = int(s.isna().sum())
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "kind": kind,
            "n_unique": int(non_null.nunique()),
            "n_missing": n_missing,
            "missing_pct": (n_missing / n_rows * 100) if n_rows else 0.0,
            "min": None,
            "median": None,
            "max": None,
            "examples": ", ".join(str(v) for v in non_null.drop_duplicates().head(example_n)),
            "is_unique_key": bool(n_rows > 0 and n_missing == 0 and non_null.nunique() == n_rows),
            "description": descriptions.get(col, ""),
        }
        if kind == "numeric" and len(non_null):
            row["min"], row["median"], row["max"] = (
                float(non_null.min()),
                float(non_null.median()),
                float(non_null.max()),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼별 결측 건수와 결측률(%). 결측률이 높은 순으로 정렬."""
    n_rows = len(df)
    out = pd.DataFrame(
        {
            "column": df.columns,
            "n_missing": [int(df[c].isna().sum()) for c in df.columns],
        }
    )
    out["missing_pct"] = out["n_missing"] / n_rows * 100 if n_rows else 0.0
    return out.sort_values(["missing_pct", "column"], ascending=[False, True], kind="stable").reset_index(drop=True)


def n_rows_with_missing(df: pd.DataFrame) -> int:
    """하나라도 결측이 있는 행의 수."""
    return int(df.isna().any(axis=1).sum())


def duplicate_report(df: pd.DataFrame, id_col: str | None = None) -> dict:
    """중복 진단.

    - n_full_duplicate_rows: 모든 컬럼 값이 같은 행 중 '첫 번째를 제외한' 중복 행 수
    - n_repeated_id_rows:   id_col 값이 이미 앞에 나온 행 수 (id_col 지정 시)
    - n_conflicting_id_rows: 완전 중복을 제거한 뒤에도 id 가 겹치는 행 수.
      같은 고객인데 값이 다르다는 뜻이므로 어느 행이 맞는지 업무 확인이 필요하다.
    """
    report = {
        "n_rows": int(len(df)),
        "n_full_duplicate_rows": int(df.duplicated(keep="first").sum()),
    }
    if id_col is not None:
        if id_col not in df.columns:
            raise KeyError(f"id 컬럼 '{id_col}' 이(가) 없습니다.")
        deduped = df.drop_duplicates()
        report["n_repeated_id_rows"] = int(df.duplicated(subset=[id_col], keep="first").sum())
        report["n_conflicting_id_rows"] = int(deduped.duplicated(subset=[id_col], keep="first").sum())
    return report


def range_violations(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """업무 규칙 범위 위반 건수.

    rules = {"age": (18, 100), "card_utilization": (0, 1)} 처럼 (하한, 상한) 포함 범위.
    하한/상한에 None 을 주면 그쪽은 검사하지 않는다. 결측은 위반으로 세지 않는다
    (결측은 missing_report 에서 본다).
    """
    rows = []
    for col, (lower, upper) in rules.items():
        if col not in df.columns:
            raise KeyError(f"범위 규칙의 컬럼 '{col}' 이(가) 데이터에 없습니다.")
        s = pd.to_numeric(df[col], errors="coerce")
        n_below = int((s < lower).sum()) if lower is not None else 0
        n_above = int((s > upper).sum()) if upper is not None else 0
        rows.append(
            {
                "column": col,
                "lower": lower,
                "upper": upper,
                "n_below": n_below,
                "n_above": n_above,
                "n_violations": n_below + n_above,
            }
        )
    return pd.DataFrame(rows, columns=["column", "lower", "upper", "n_below", "n_above", "n_violations"])


def find_numeric_like_text(df: pd.DataFrame, min_ratio: float = 0.9) -> pd.DataFrame:
    """문자형인데 사실상 숫자인 컬럼을 찾는다 (예: "3", "0", "N/A" 가 섞인 컬럼).

    비결측 값 중 숫자로 변환되는 비율이 min_ratio 이상이고 100% 는 아닌 컬럼도 포함한다.
    변환 불가 값의 예시를 같이 보여주므로, 어떤 문자열이 섞였는지 확인할 수 있다.
    """
    rows = []
    for col in df.columns:
        s = df[col]
        if infer_kind(s) != "categorical":
            continue
        non_null = s.dropna().astype(str).str.strip()
        if non_null.empty:
            continue
        converted = pd.to_numeric(non_null, errors="coerce")
        ratio = float(converted.notna().mean())
        if ratio >= min_ratio:
            bad_values = non_null[converted.isna()].drop_duplicates().head(5).tolist()
            rows.append(
                {
                    "column": col,
                    "numeric_ratio": ratio,
                    "n_non_numeric": int(converted.isna().sum()),
                    "non_numeric_examples": bad_values,
                }
            )
    return pd.DataFrame(rows, columns=["column", "numeric_ratio", "n_non_numeric", "non_numeric_examples"])


def constant_columns(df: pd.DataFrame) -> list[str]:
    """비결측 값이 하나뿐(또는 전부 결측)인 컬럼. 모형에 정보를 주지 못한다."""
    return [c for c in df.columns if df[c].dropna().nunique() <= 1]


def quality_summary(df: pd.DataFrame, id_col: str | None = None) -> dict:
    """데이터 품질 핵심 수치를 한 번에 모은 요약."""
    n_rows, n_cols = df.shape
    n_cells = n_rows * n_cols
    n_missing_cells = int(df.isna().sum().sum())
    kinds = [infer_kind(df[c]) for c in df.columns]
    summary = {
        "n_rows": int(n_rows),
        "n_columns": int(n_cols),
        "n_missing_cells": n_missing_cells,
        "missing_cell_pct": (n_missing_cells / n_cells * 100) if n_cells else 0.0,
        "n_rows_with_missing": n_rows_with_missing(df),
        "n_numeric_columns": kinds.count("numeric"),
        "n_categorical_columns": kinds.count("categorical"),
        "n_datetime_columns": kinds.count("datetime"),
        "n_boolean_columns": kinds.count("boolean"),
        "constant_columns": constant_columns(df),
    }
    summary.update({f"dup_{k}": v for k, v in duplicate_report(df, id_col).items() if k != "n_rows"})
    return summary
