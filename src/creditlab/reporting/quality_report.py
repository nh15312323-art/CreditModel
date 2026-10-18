"""데이터 품질 보고서(Markdown) 생성.

명세서 11장 보고서 표준 중 1~5번(목적·출처·타깃·모집단·품질)과 15번(재현 정보)을 다룬다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from ..data.target import default_rate, default_rate_ci, sample_sufficiency, target_distribution
from ..data.validation import (
    data_dictionary,
    duplicate_report,
    find_numeric_like_text,
    missing_report,
    quality_summary,
    range_violations,
)
from ..env import environment_info


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """DataFrame 내용의 SHA-256 지문(행 순서·컬럼명 포함). 같은 데이터인지 확인하는 용도."""
    h = hashlib.sha256()
    h.update("|".join(map(str, df.columns)).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:,.4g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_table(df: pd.DataFrame) -> str:
    """DataFrame 을 Markdown 표 문자열로 바꾼다(tabulate 불필요)."""
    if df.empty:
        return "_(해당 없음)_"
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def build_quality_report(
    df: pd.DataFrame,
    target: str,
    meta: dict | None = None,
    id_col: str | None = None,
    descriptions: dict | None = None,
    range_rules: dict | None = None,
    seed: int | None = None,
) -> str:
    """데이터 품질 보고서를 Markdown 문자열로 만든다. 계산은 모두 이 함수가 실제로 수행한다."""
    meta = meta or {}
    id_col = id_col or meta.get("id_col")
    env = environment_info(seed)
    dist = target_distribution(df, target)
    rate = default_rate(df[target])
    ci_low, ci_high = default_rate_ci(df[target])
    suff = sample_sufficiency(df[target])
    summary = quality_summary(df, id_col)
    missing = missing_report(df)
    missing = missing[missing["n_missing"] > 0]

    lines = [
        "# 데이터 품질 보고서",
        "",
        f"- 생성 시각(UTC): {env['generated_at_utc']}",
        "- 성격: **교육용** 보고서입니다. 실제 은행 모형개발 보고서나 규제 승인 자료가 아닙니다.",
        "",
        "## 1. 목적 및 사용 범위",
        "신용평가모형 개발 전 단계에서 데이터의 크기·타깃·결측·중복·범위 오류를 확인한다. "
        "이 보고서만으로 모형 사용 가능 여부를 판단하지 않는다.",
        "",
        "## 2. 데이터 출처",
        f"- 데이터셋: {meta.get('dataset_id', '(미기록)')}",
        f"- 구분: {meta.get('source', '(미기록)')}",
        f"- 라이선스: {meta.get('license', '(미기록)')}",
        f"- 파일 해시(SHA-256): {meta.get('sha256') or '(파일 없음/합성)'}",
        f"- 메모리상 데이터 지문(SHA-256): {dataframe_fingerprint(df)}",
        f"- 비고: {meta.get('note', '')}",
    ]
    if meta.get("fallback_reason"):
        lines.append(f"- ⚠️ 대체 사유: {meta['fallback_reason']}")
    lines += [
        "",
        "## 3. 타깃 정의와 부도율",
        f"- 타깃 컬럼: `{target}` (1=부도, 0=정상)",
        f"- 전체 {len(df):,}건, 부도율 {rate:.2%} (95% Wilson 신뢰구간 {ci_low:.2%} ~ {ci_high:.2%})",
        "",
        md_table(dist.assign(ratio=(dist["ratio"] * 100).round(2)).rename(columns={"ratio": "ratio(%)"})),
        "",
    ]
    if suff["warnings"]:
        lines += ["**표본·이벤트 수 경고(경험칙 기준)**", *[f"- {w}" for w in suff["warnings"]], ""]
    else:
        lines += ["표본·이벤트 수 경고 없음(기준은 학습용 경험칙이며 충분성을 보장하지 않음).", ""]

    lines += [
        "## 4. 데이터 품질",
        f"- 행 {summary['n_rows']:,} × 열 {summary['n_columns']}",
        f"- 결측 셀 {summary['n_missing_cells']:,}개 ({summary['missing_cell_pct']:.2f}%), "
        f"결측이 있는 행 {summary['n_rows_with_missing']:,}개",
        f"- 완전 중복 행(첫 행 제외) {summary['dup_n_full_duplicate_rows']:,}개",
    ]
    if id_col:
        dup = duplicate_report(df, id_col)
        lines += [
            f"- `{id_col}` 값이 반복된 행 {dup['n_repeated_id_rows']:,}개, "
            f"그중 내용까지 충돌하는 행 {dup['n_conflicting_id_rows']:,}개",
        ]
    lines += [f"- 상수 컬럼: {summary['constant_columns'] or '없음'}", "", "### 컬럼별 결측", md_table(missing), ""]

    if range_rules:
        lines += ["### 값 범위 점검", md_table(range_violations(df, range_rules)), ""]
    text_like = find_numeric_like_text(df)
    lines += ["### 문자형 숫자 후보", md_table(text_like), ""]

    lines += ["## 5. 데이터 사전", md_table(data_dictionary(df, descriptions)), ""]

    lines += [
        "## 6. 재현 정보",
        f"- creditlab {env['creditlab']}, Python {env['python']}",
        f"- 패키지: {', '.join(f'{k} {v}' for k, v in env['packages'].items())}",
        f"- 난수 시드: {seed if seed is not None else '(미지정)'}",
        "",
        "## 7. 한계",
        "- 이 보고서는 자동 점검 결과이며, 결측·이상치의 원인 판단과 제외 결정은 사람이 업무 지식으로 해야 한다.",
        "- 타깃 누수 점검은 Phase 2 에서 다룬다. 이 보고서에는 포함되지 않았다.",
        "",
    ]
    return "\n".join(lines)


def save_report(text: str, path) -> Path:
    """보고서를 UTF-8 파일로 저장한다(상위 폴더는 자동 생성)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p
