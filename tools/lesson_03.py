from nbkit import SETUP_CELL, code, md

TITLE = "레슨 1-3. 결측치와 데이터 품질"


def cells():
    return [
        md(
            """
# 레슨 1-3. 결측치와 데이터 품질

**예상 시간** 60분 · **선행** 레슨 1-2 · **이론 페이지** `docs/theory/stage_02.html`

## 학습 목표
1. 결측·중복·범위 오류·문자형 숫자·상수 열을 *코드로* 찾는다.
2. 결측이 **무작위인지** 확인하는 방법(그룹별 결측률 비교)을 익힌다.
3. 정제는 **복사본에서**, **이유와 건수를 기록하며** 한다.
4. 품질 보고서를 자동 생성하고 내용을 해석한다.

> 이 레슨은 연습용으로 **오류를 일부러 심은 합성 데이터**를 씁니다. 공식적으로 결측이 없다고 설명된 UCI 데이터는 과제에서 다룹니다.
"""
        ),
        SETUP_CELL,
        md(
            """
## 1. 개념 — 품질 점검 6가지

| 점검 | 질문 | 도구 |
|---|---|---|
| 결측 | 어떤 열이, 얼마나, 왜 비어 있나? | `missing_report` |
| 중복 | 같은 행/같은 ID 가 반복되나? 값이 충돌하나? | `duplicate_report` |
| 범위 | 업무상 불가능한 값(나이 150)이 있나? | `range_violations` |
| 자료형 | 숫자인데 문자로 저장됐나?("N/A" 혼입) | `find_numeric_like_text` |
| 상수 열 | 값이 하나뿐이라 정보가 없나? | `constant_columns` |
| 타깃 | 결측·범위 밖 값이 있나? | `validate_binary_target` |

### 결측의 세 가지 직관 (Rubin, 1976)
- **MCAR**: 비는 것이 완전히 무작위. 삭제해도 편향이 적다(드문 경우).
- **MAR**: 비는 확률이 *관측된 다른 변수*와 관련. (예: 근속연수가 짧을수록 소득을 안 적음)
- **MNAR**: 비는 확률이 *비어 있는 값 자체*와 관련. (예: 소득이 낮은 사람이 소득을 숨김) — 데이터만으로는 구분하기 어렵다.

> **원칙**: 삭제·대체는 *마지막*에 합니다. 먼저 **왜 비었는지** 확인합니다. 그리고 대체값(평균 등)은 **학습 데이터로만 계산**해야 누수가 없습니다(2·8단계에서 구현).
"""
        ),
        md("## 2. 오류가 심어진 데이터 불러오기"),
        code(
            '''
from creditlab.data import DESCRIPTIONS, RANGE_RULES, load_learning_data, quality_summary

df_raw, meta = load_learning_data("synthetic", seed=SEED, inject_quality_issues=True)
TARGET, ID_COL = meta["target"], meta["id_col"]
print(f"행 {len(df_raw):,} × 열 {df_raw.shape[1]} | 합성 데이터(연습용, 오류 주입됨)")
pd.Series(quality_summary(df_raw, ID_COL))
'''
        ),
        md("### ✋ 먼저 예측\n`income_10k`(연소득) 열의 결측률은 몇 %일까요? 여러 열 중 결측률이 가장 높은 열은 어디일까요?"),
        code("pred_missing_pct = None  # ← 예: 5\npred_worst_column = None  # ← 예: \"age\""),
        code(
            '''
from creditlab.data import missing_report

mr = missing_report(df_raw)
top = mr.iloc[0]
print(f"가장 결측률이 높은 열: {top['column']} ({top['missing_pct']:.2f}%)")
if pred_missing_pct is not None:
    actual = mr.loc[mr["column"] == "income_10k", "missing_pct"].iloc[0]
    print(f"income_10k 결측률 예측 {pred_missing_pct}% / 실제 {actual:.2f}%")
if pred_worst_column is not None:
    print("열 예측:", "맞음" if pred_worst_column == top["column"] else f"틀림(정답 {top['column']})")

fig, ax = plt.subplots(figsize=(6, 3))
shown = mr[mr["n_missing"] > 0]
ax.barh(shown["column"][::-1], shown["missing_pct"][::-1], color="#c8553d")
ax.set_xlabel("missing (%)")
ax.set_title("Missing rate by column")
plt.show()
mr
'''
        ),
        md(
            """
## 3. 결측은 무작위일까? — 그룹별 결측률 비교
소득이 비는 비율이 **근속연수 1년 미만** 그룹과 **1년 이상** 그룹에서 같다면 무작위(MCAR)에 가깝습니다. 다르다면 결측이 다른 변수와 관련(MAR 가능성)이 있다는 신호입니다.
"""
        ),
        code("guess = None  # ← 근속 1년 미만의 소득 결측률은 1년 이상보다: \"높다\" / \"비슷하다\" / \"낮다\""),
        code(
            '''
short = df_raw["employment_years"] < 1.0
rate_short = df_raw.loc[short, "income_10k"].isna().mean()
rate_long = df_raw.loc[~short, "income_10k"].isna().mean()
print(f"근속 1년 미만: 소득 결측률 {rate_short:.1%} (n={short.sum():,})")
print(f"근속 1년 이상: 소득 결측률 {rate_long:.1%} (n={(~short).sum():,})")
print(f"비율 차이: {rate_short / rate_long:.1f}배")
if guess:
    print("내 추측:", guess)
print("\\n→ 결측이 무작위가 아니라는 신호입니다. 소득이 빈 행을 그냥 삭제하면 '근속이 짧은 사람'이 데이터에서 사라집니다.")

# 결측 여부와 부도율의 관계 (결측 자체가 정보일 수 있는지 탐색)
from creditlab.data import default_rate_ci

print("\\n[소득 결측 여부별 부도율]")
for is_missing, g in df_raw.groupby(df_raw["income_10k"].isna()):
    lo, hi = default_rate_ci(g[TARGET])
    print(f"소득 결측={is_missing!s:5}: n={len(g):,}, 부도 {int(g[TARGET].sum())}건, 부도율 {g[TARGET].mean():.1%} (95% 구간 {lo:.1%} ~ {hi:.1%})")
'''
        ),
        md(
            """
**읽는 법**: 결측 그룹은 부도 건수가 적어(약 10건대) 구간이 넓습니다. 두 그룹의 부도율이 달라 보여도 **표본이 작으면 우연일 수 있습니다.** 이 합성 데이터에서 결측은 근속연수에 의존해 만들었을 뿐 부도 여부와 직접 연결하지 않았습니다. 그래서 여기서 보이는 차이를 "결측이면 안전하다" 같은 규칙으로 읽으면 안 됩니다.

실제 업무에서도 마찬가지입니다. 결측 여부를 변수로 쓰려면 ① 충분한 부도 건수, ② 안정적인 차이, ③ **운영 시에도 같은 이유로 비는지**(입력 방식·심사 프로세스 변경 여부)를 모두 확인해야 합니다.
"""
        ),
        md("## 4. 중복·범위·문자형 숫자"),
        code(
            '''
from creditlab.data import duplicate_report, find_numeric_like_text, range_violations

print("중복:", duplicate_report(df_raw, ID_COL))
print("\\n범위 규칙:", RANGE_RULES)
display_df = range_violations(df_raw, RANGE_RULES)
print(display_df.to_string(index=False))
print("\\n범위를 벗어난 행 예시:")
bad_rows = df_raw[(df_raw["age"] < 19) | (df_raw["age"] > 100) | (df_raw["card_utilization"] > 1)]
print(bad_rows[[ID_COL, "age", "card_utilization"]].head(8).to_string(index=False))
'''
        ),
        code(
            '''
fl = find_numeric_like_text(df_raw)
print(fl.to_string(index=False))
print("\\ninquiries_6m 값 분포(상위):")
df_raw["inquiries_6m"].value_counts().head(8)
'''
        ),
        md(
            """
## 5. ✏️ 실습 — 정제하기 (복사본에서!)
요구사항
1. 원본 `df_raw` 는 **그대로 둡니다** (`df_raw.copy()` 로 시작).
2. **완전 중복 행**을 제거합니다.
3. `inquiries_6m` 을 숫자로 바꿉니다(바꿀 수 없는 값은 결측으로).
4. 범위를 벗어난 `age`, `card_utilization` 값은 **결측으로 바꾸거나 행을 제거**합니다. 어느 쪽이든 *건수와 이유*를 `cleaning_log` 에 적습니다.
5. **결측 대체(평균 등)는 하지 않습니다.** 대체는 데이터를 분할한 뒤 학습 데이터로만 계산해야 하므로 2단계·8단계에서 합니다.

힌트 단계: 힌트1 개념 → 힌트2 절차 → 힌트3 코드 일부 → 아래 접힌 정답 예시.
"""
        ),
        code(
            '''
cleaning_log = []   # 예: {"step": "중복 제거", "rows_affected": 50, "reason": "완전히 같은 행"}
cleaned = None      # ← 여기에 정제 결과 DataFrame 을 만드세요

# 힌트1: 원본을 바꾸지 않으려면? → df_raw.copy()
# 힌트2: 순서는 '중복 제거 → 자료형 변환 → 범위 오류 처리' 가 자연스럽습니다.
# 힌트3(코드 일부):
#   cleaned = df_raw.drop_duplicates().copy()
#   cleaned["inquiries_6m"] = pd.to_numeric(cleaned["inquiries_6m"], errors="coerce")
#   cleaned.loc[<조건>, "age"] = np.nan
'''
        ),
        code(
            '''
from creditlab.learning import check_cleaning_result

check_cleaning_result(df_raw, cleaned, ID_COL, ["inquiries_6m"], RANGE_RULES);
'''
        ),
        md(
            """
<details>
<summary>정답 예시 (먼저 직접 해 본 뒤 펼치세요)</summary>

```python
cleaned = df_raw.copy().drop_duplicates()
cleaning_log.append({"step": "완전 중복 제거", "rows_affected": len(df_raw) - len(cleaned),
                     "reason": "모든 열이 같은 행은 같은 신청이 중복 적재된 것으로 판단"})

before_na = cleaned["inquiries_6m"].isna().sum()
cleaned["inquiries_6m"] = pd.to_numeric(cleaned["inquiries_6m"], errors="coerce")
cleaning_log.append({"step": "inquiries_6m 숫자 변환", "rows_affected": int(cleaned["inquiries_6m"].isna().sum() - before_na),
                     "reason": "'N/A' 문자열을 결측으로 변환(대체는 하지 않음)"})

bad_age = (cleaned["age"] < 19) | (cleaned["age"] > 100)
cleaned.loc[bad_age, "age"] = np.nan
cleaning_log.append({"step": "나이 범위 오류 → 결측", "rows_affected": int(bad_age.sum()),
                     "reason": "업무 규칙(19~100세) 위반. 입력 오류로 추정하되 원인은 담당자 확인 필요"})

bad_util = cleaned["card_utilization"] > 1
cleaned.loc[bad_util, "card_utilization"] = np.nan
cleaning_log.append({"step": "사용률 범위 오류 → 결측", "rows_affected": int(bad_util.sum()),
                     "reason": "사용률은 0~1 이어야 함"})
```
</details>
"""
        ),
        code(
            '''
if cleaned is not None:
    print(pd.DataFrame(cleaning_log).to_string(index=False) if cleaning_log else "cleaning_log 가 비어 있습니다. 각 처리의 건수와 이유를 기록하세요.")
    print(f"\\n원본 {len(df_raw):,}행 → 정제 후 {len(cleaned):,}행 (원본은 그대로: {len(df_raw):,}행)")
else:
    print("cleaned 를 먼저 만들어 보세요.")
'''
        ),
        md("## 6. 품질 보고서 자동 생성\n지금까지의 점검을 **한 문서로** 남깁니다. 숫자는 모두 실제로 계산된 값이며, 실행 환경·시드·데이터 지문이 같이 기록됩니다."),
        code(
            '''
from creditlab.reporting import build_quality_report, save_report

report = build_quality_report(
    df_raw, TARGET, meta, id_col=ID_COL, descriptions=DESCRIPTIONS, range_rules=RANGE_RULES, seed=SEED
)
path = save_report(report, "artifacts/quality_report_lesson_1_3.md")
print("저장 위치:", path)
print("\\n".join(report.splitlines()[:28]))
# Colab 에서 내려받기: from google.colab import files; files.download(str(path))
'''
        ),
        md("## 7. 퀴즈"),
        code(
            '''
from creditlab.learning import Question, check, grade, hint, show

QUIZ = [
    Question("Q1", "결측 메커니즘", "근속연수가 짧은 사람일수록 소득이 자주 비어 있습니다. 가장 적절한 해석은?",
             {"A": "완전 무작위(MCAR)이므로 삭제해도 안전하다",
              "B": "다른 관측 변수와 관련된 결측(MAR 가능성). 삭제하면 특정 집단이 사라져 편향될 수 있다",
              "C": "의미 없는 현상이다", "D": "항상 평균으로 대체하면 된다"},
             "B", "그룹별 결측률 차이는 결측이 무작위가 아니라는 신호입니다. 삭제·대체 전에 원인을 확인합니다.",
             ("그룹별 결측률이 같았다면 어떤 결론이었을까요?",)),
    Question("Q2", "누수", "결측을 평균으로 대체하려 합니다. 평균은 어떤 데이터로 계산해야 할까요?",
             {"A": "전체 데이터", "B": "학습 데이터만 (검증·테스트에는 그 평균을 적용만 한다)", "C": "테스트 데이터", "D": "아무거나 같다"},
             "B", "검증·테스트 정보가 학습에 스며들면 성능이 부풀려집니다. 변환은 학습 데이터에만 fit 합니다.",
             ("미래(처음 보는) 데이터의 평균을 지금 알 수 있을까요?",)),
    Question("Q3", "범위 오류", "나이 150 이 발견되었습니다. 올바른 처리는?",
             {"A": "그대로 둔다", "B": "입력 오류로 보고 결측 처리/제거하되, 건수와 이유를 기록하고 원인을 확인한다",
              "C": "모두 100으로 바꾼다", "D": "나이 열 전체를 삭제한다"},
             "B", "근거 없이 값을 만들어 넣지 않고, 처리 기록을 남깁니다.", ("업무 규칙과 기록, 두 가지를 떠올리세요.",)),
    Question("Q4", "중복", "같은 ID 인데 소득 값이 다른 두 행이 있습니다. 올바른 접근은?",
             {"A": "자동으로 첫 행만 남긴다", "B": "어느 행이 맞는지 업무(원천 시스템) 확인이 필요하다", "C": "두 값의 평균을 쓴다", "D": "둘 다 삭제한다"},
             "B", "값이 충돌하는 중복은 시스템 오류·재신청·정정 등 원인이 다양합니다. 기계적 규칙으로 정하지 않습니다.",
             ("duplicate_report 의 n_conflicting_id_rows 가 무엇이었는지 보세요.",)),
    Question("Q5", "결측의 정보", "'결측 자체가 정보일 수 있다'는 말의 올바른 이해는?",
             {"A": "결측이면 항상 부도다", "B": "결측 여부와 부도율의 관계를 확인해 지표 변수 후보로 검토하되, 운영 시에도 같은 이유로 비는지 확인한다",
              "C": "결측은 정보가 전혀 없다", "D": "결측률이 높으면 무조건 변수를 삭제한다"},
             "B", "결측 사유가 운영 환경에서 바뀌면 지표 변수의 의미도 바뀝니다. 탐색 → 검증 → 모니터링이 필요합니다.",
             ("개발 때의 결측 이유가 운영에서도 같을까요?",)),
]

answers = {"Q1": None, "Q2": None, "Q3": None, "Q4": None, "Q5": None}  # ← "A"~"D"

for q in QUIZ:
    show(q)
    print()
'''
        ),
        code(
            '''
for q in QUIZ:
    check(q, answers[q.qid])
if all(v is not None for v in answers.values()):
    r = grade(QUIZ, answers)
    print("\\n점수:", r["score"], "/", r["total"], "| 통과(75% 기준):", r["passed"], "| 다시 볼 개념:", r["concepts_to_review"])
'''
        ),
        md(
            """
## 8. 과제와 정리
**과제**
1. 레슨 1-2 에서 쓴 **UCI 데이터**(`load_learning_data("uci")`)에 오늘의 점검 도구를 모두 적용하세요. 공식 설명은 "결측 없음"이지만, *결측 0건인데도 확인해야 할 것*을 3가지 이상 찾아 적으세요. (힌트: `PAY_0`·`EDUCATION`·`MARRIAGE` 의 값 분포, 설명서와의 비교)
2. 이번 레슨의 `cleaning_log` 를 표로 정리하고, 가장 *확신이 낮은* 처리 한 가지와 그 이유를 쓰세요.
3. 결측 대체를 *분할 이후*에 해야 하는 이유를 3문장으로 설명하세요.

**체크포인트**: 퀴즈 4/5 이상 · `check_cleaning_result` 통과 · 품질 보고서 생성

**Phase 1 완료**: 다음은 Phase 2(통계·탐색 분석, 누수 점검)입니다.

**참고자료**
- Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3).
- Little, R. J. A. & Rubin, D. B., *Statistical Analysis with Missing Data* (Wiley).
- Kaufman, S. et al. (2012). Leakage in data mining. *ACM TKDD*, 6(4).
- 이론: `docs/theory/stage_02.html`
"""
        ),
    ]
