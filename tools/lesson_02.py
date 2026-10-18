from nbkit import SETUP_CELL, code, md

TITLE = "레슨 1-2. 행·열·타깃 이해하기"


def cells():
    return [
        md(
            """
# 레슨 1-2. 행·열·타깃 이해하기

**예상 시간** 50분 · **선행** 레슨 1-1 · **이론 페이지** `docs/theory/stage_01.html`, `docs/theory/stage_02.html`

## 학습 목표
1. 데이터의 **행(관측단위)**, **열(변수)**, **타깃**, **ID** 를 구분한다.
2. 타깃 규칙(**1 = 부도, 0 = 정상**)을 지키고, 규칙에 어긋난 값은 조용히 고치지 않고 *멈춰서 확인*한다.
3. 부도율과 그 신뢰구간, 이벤트(부도) 건수가 충분한지 확인한다.
4. 데이터 사전을 만들고, **설명서와 실제 값이 다를 수 있음**을 확인한다.
5. **타깃 정의서** 초안을 작성한다.
"""
        ),
        SETUP_CELL,
        md(
            """
## 1. 개념

| 용어 | 뜻 | 질문해 볼 것 |
|---|---|---|
| 행 = 관측단위 | 표의 한 줄이 가리키는 대상 | 고객 1명? 계좌 1개? 신청 1건? 계좌-월 1건? |
| 열 = 변수 | 행에 대한 정보 | 신청 시점에 알 수 있는 정보인가? |
| 타깃 | 맞히려는 결과 | 부도의 정의·관측기간은? |
| ID | 행을 식별하는 값 | 중복되는가? 같은 고객이 여러 행인가? |

**이 프로그램의 약속**: 타깃은 항상 `1 = 부도(bad)`, `0 = 정상(good)`.
"""
        ),
        md(
            """
## 2. 데이터 불러오기
`load_learning_data("uci")` 는 ① 로컬 `data/raw/` 의 UCI 파일 → ② 공식 주소 다운로드 순으로 시도합니다.
둘 다 안 되면 **합성 데이터로 대체하고 그 사실을 크게 표시**합니다. 아래 안내 문구를 꼭 읽으세요.
"""
        ),
        code(
            '''
from creditlab.data import load_learning_data

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    df, meta = load_learning_data("uci", raw_dir="data/raw", seed=SEED)

TARGET, ID_COL = meta["target"], meta["id_col"]
print("데이터 구분:", meta["source"], "|", meta["dataset_id"])
print("라이선스/비고:", meta["license"], "/", meta["note"])
if meta["source"] == "synthetic" and meta.get("fallback_reason"):
    print("\\n⚠️  UCI 데이터를 못 불러와 합성 데이터로 대체했습니다.")
    print("    이유:", meta["fallback_reason"])
    print("    → 실데이터로 하려면: data/raw/ 에 UCI zip/xls 를 넣고(또는 Colab 에서 인터넷 연결 확인) 다시 실행하세요.")
print(f"\\n행 {df.shape[0]:,}개 × 열 {df.shape[1]}개 | 타깃 컬럼: {TARGET} | ID 컬럼: {ID_COL}")
df.head()
'''
        ),
        md("### 자료형 확인\n숫자여야 할 열이 문자로 읽히지는 않았나요? (레슨 1-3 에서 자세히 봅니다)"),
        code("df.dtypes"),
        md(
            """
## 3. 데이터 사전 만들기
`data_dictionary` 는 열마다 자료형·종류·결측·고유값·범위·예시를 정리합니다. **설명(description)** 은 사람이 채워야 하는 부분입니다.
"""
        ),
        code(
            '''
from creditlab.data import DESCRIPTIONS, data_dictionary

# UCI 공식 설명 중 확인된 항목만 적었습니다. 나머지 열의 설명은 직접 채우는 것이 과제입니다.
UCI_DESCRIPTIONS = {
    "ID": "고객 번호",
    "LIMIT_BAL": "신용한도(대만 달러)",
    "SEX": "성별 (1=남, 2=여)",
    "EDUCATION": "학력 (공식 설명: 1=대학원, 2=대학, 3=고교, 4=기타)",
    "MARRIAGE": "혼인 (공식 설명: 1=기혼, 2=미혼, 3=기타)",
    "AGE": "나이(세)",
    "PAY_0": "2005-09 상환 상태 (-1=정상 납부, 1~8=연체 개월 수, 9=9개월 이상 연체)",
    "default": "타깃: 다음 달 결제 불이행 (1=불이행, 0=정상)",
}
descriptions = UCI_DESCRIPTIONS if meta["source"] == "uci" else DESCRIPTIONS

dd = data_dictionary(df, descriptions)
dd[["column", "dtype", "kind", "n_unique", "n_missing", "min", "max", "is_unique_key", "description"]]
'''
        ),
        md(
            """
### ✏️ 실습 A — 설명이 빈 열 채우기
`my_descriptions` 에 설명이 비어 있는 열 2~3개의 설명을 직접 적어 보세요. 모르는 열은 *모른다고* 적는 것이 정직한 기록입니다.
"""
        ),
        code(
            '''
my_descriptions = {
    # "열이름": "내가 이해한 설명",
}
dd2 = data_dictionary(df, {**descriptions, **my_descriptions})
empty = dd2.loc[dd2["description"] == "", "column"].tolist()
print(f"아직 설명이 없는 열 {len(empty)}개:", empty[:12], "..." if len(empty) > 12 else "")
'''
        ),
        md(
            """
### 설명서와 실제 값이 같은가?
UCI 데이터는 `PAY_0` 이 -1 과 1~9 라고 설명하지만, 실제 파일에는 설명에 없는 값(예: 0, -2)이 들어 있다고 알려져 있습니다.
*알려져 있다*는 말을 그대로 믿지 말고 직접 세어 봅니다. (합성 데이터를 쓰는 중이라면 이 셀은 건너뜁니다.)
"""
        ),
        code(
            '''
if "PAY_0" in df.columns:
    print(df["PAY_0"].value_counts().sort_index())
    print("\\n설명서(-1, 1~9)에 없는 값이 있다면 → 기록하고, 의미를 모르면 가정을 명시하세요. 삭제하거나 임의로 바꾸지 마세요.")
else:
    print("PAY_0 컬럼이 없는 데이터(합성)입니다. 건너뜁니다.")
'''
        ),
        md(
            """
## 4. 타깃과 부도율

### ✋ 먼저 예측
이 데이터의 부도율은 몇 %일까요? (연체가 드문 일반 대출은 한 자릿수, 카드 현금서비스·소액신용은 더 높습니다.)
"""
        ),
        code(
            '''
prediction_pct = None  # ← 예측(%)을 적어 보세요. 예: 15
'''
        ),
        code(
            '''
from creditlab.data import default_rate, default_rate_ci, sample_sufficiency, target_distribution

dist = target_distribution(df, TARGET)
rate = default_rate(df[TARGET])
low, high = default_rate_ci(df[TARGET])
print(f"부도율 {rate:.2%}  (95% 신뢰구간 {low:.2%} ~ {high:.2%}, n={len(df):,})")
if prediction_pct is not None:
    print(f"내 예측 {prediction_pct}% → 차이 {abs(prediction_pct - rate * 100):.1f}%p")
else:
    print("예측을 적지 않았습니다. 다음에는 먼저 적어 보세요.")

fig, ax = plt.subplots(figsize=(4.5, 3))
ax.bar(["good (0)", "bad (1)"], dist["count"], color=["#6b8fb5", "#c8553d"])
ax.set_title(f"Target distribution (default rate {rate:.1%})")
ax.set_ylabel("count")
plt.show()
dist
'''
        ),
        md(
            """
### 이벤트(부도) 건수는 충분한가?
부도율이 낮으면 *전체 건수는 많아도 부도 건수는 적을* 수 있습니다. 모형은 부도 건수로 배웁니다.
(`min_events=100` 은 학습용 경험칙일 뿐 통계적 법칙이 아닙니다.)
"""
        ),
        code("sample_sufficiency(df[TARGET], min_events=100, min_non_events=100)"),
        md(
            """
## 5. 타깃 규칙이 깨지면? — 조용히 고치지 않기
아래는 일부러 잘못된 타깃을 넣어 본 것입니다. 함수가 *멈추고 이유를 알려 주는지* 확인하세요.
"""
        ),
        code(
            '''
from creditlab.data import validate_binary_target

bad_examples = {
    "문자 인코딩": ["Y", "N", "Y"],
    "결측 포함": [0, 1, None, 0],
    "범위 밖 값": [0, 1, 2],
}
for name, values in bad_examples.items():
    try:
        validate_binary_target(values)
    except ValueError as exc:
        print(f"[{name}] ValueError → {exc}\\n")
'''
        ),
        md(
            """
**왜 멈출까?** 결측 타깃을 0(정상)으로 바꾸면 부도율이 *아무도 모르게* 낮아집니다. 어떤 규칙으로 바꿀지는 업무 판단이고, 그 규칙은 문서에 남겨야 합니다.
"""
        ),
        md("## 6. ID 점검 — 행이 정말 1행 = 1대상인가?"),
        code(
            '''
from creditlab.data import duplicate_report

print("ID 고유 여부:", df[ID_COL].is_unique)
duplicate_report(df, ID_COL)
'''
        ),
        md(
            """
## 7. 실습 B — 타깃 정의서 초안
데이터를 모델링하기 전에 *문서로* 정의를 고정합니다. 아래 딕셔너리의 `"TODO"` 를 자기 말로 바꿔 쓰세요.
(합성 데이터 예: 관측단위=대출 신청 1건 / 성과관찰기간=12개월 / 부도=90일 이상 연체)
점검기는 *빈 칸만* 알려 줍니다. 내용이 타당한지는 스스로(또는 동료가) 판단해야 합니다.
"""
        ),
        code(
            '''
from creditlab.learning import check_target_definition

my_definition = {
    "observation_unit": "TODO",
    "observation_point": "TODO",
    "performance_window": "TODO",
    "default_definition": "TODO",
    "population": "TODO",
    "exclusions": "TODO",
    "data_source": "TODO",
}
check_target_definition(my_definition);
'''
        ),
        md("## 8. 퀴즈"),
        code(
            '''
from creditlab.learning import Question, check, grade, hint, show

QUIZ = [
    Question("Q1", "관측단위", "데이터의 한 행이 '고객 1명'이 아니라 '계좌-월 1건'일 수도 있습니다. 가장 먼저 확인할 것은?",
             {"A": "열 개수", "B": "관측단위(행의 의미)와 ID 중복 구조", "C": "파일 용량", "D": "컬럼 순서"},
             "B", "같은 고객이 여러 행이면 분할·집계·검증 단위가 모두 달라집니다(고객 단위 분할은 8단계).",
             ("행이 무엇을 가리키는지부터 정해야 합니다.",)),
    Question("Q2", "타깃 인코딩", "타깃에 'Y'/'N' 문자와 결측이 섞여 있습니다. 올바른 처리는?",
             {"A": "결측은 0(정상)으로 바꾼다", "B": "'Y'→1 규칙을 문서화하고, 결측 원인을 확인한 뒤 제외/재정의를 결정한다",
              "C": "결측은 평균으로 채운다", "D": "그냥 무시하고 진행한다"},
             "B", "결측 타깃을 임의로 0으로 바꾸면 부도율이 왜곡됩니다. 규칙은 업무 판단이며 기록해야 합니다.",
             ("validate_binary_target 이 왜 멈췄는지 다시 보세요.",)),
    Question("Q3", "설명서 vs 실제", "설명서에는 PAY_0 이 -1, 1~9 라고 되어 있는데 실제 데이터에는 0, -2 도 있습니다. 어떻게 할까요?",
             {"A": "설명서가 맞으니 그 값은 삭제한다", "B": "불일치를 기록하고 의미를 출처·업무 담당자에게 확인하며, 모르면 가정을 명시한다",
              "C": "무시한다", "D": "0은 모두 결측으로 본다"},
             "B", "값을 마음대로 삭제·재해석하면 근거 없는 모형이 됩니다. 불일치 자체가 데이터 품질 발견 사항입니다.",
             ("설명서와 현실이 다르면 누가 답을 알고 있을까요?",)),
    Question("Q4", "신뢰구간", "표본 크기가 같을 때, 부도율 5%인 데이터와 50%인 데이터 중 Wilson 95% 구간 폭이 더 좁은 쪽은?",
             {"A": "5% 쪽", "B": "50% 쪽", "C": "같다", "D": "알 수 없다"},
             "A", "구간 폭은 대략 sqrt(p(1-p)/n) 에 비례합니다. p(1-p) 가 p=0.05 에서는 0.0475, p=0.5 에서는 0.25 라서 5% 쪽이 더 좁습니다.",
             ("p(1-p) 를 두 경우에 계산해 보세요.",)),
    Question("Q5", "ID 고유성", "ID 컬럼의 고유값 수가 행 수보다 적습니다. 의미는?",
             {"A": "정상이다", "B": "같은 ID 가 반복된다. 완전 중복인지 값이 충돌하는지 확인해야 한다",
              "C": "다른 컬럼과 합쳐야 한다", "D": "ID 컬럼을 삭제한다"},
             "B", "완전 중복은 제거 후보, 값이 다른 중복(충돌)은 어느 행이 맞는지 업무 확인이 필요합니다.",
             ("duplicate_report 의 세 숫자가 각각 무엇을 세는지 떠올려 보세요.",)),
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
## 9. 과제와 정리
**과제**
1. 실습 B 의 타깃 정의서를 모두 채우고, 가장 *모호하다고 느낀 항목*과 이유를 한 줄씩 적으세요.
2. 부도율의 신뢰구간이 의미하는 바를 비전공자 동료에게 설명하듯 3문장으로 쓰세요.
3. `df.shape`, 부도율, 이벤트 수, ID 중복 결과를 한 표로 정리하세요.

**체크포인트**: 퀴즈 4/5 이상 · 타깃 정의서 7개 항목 작성

**참고자료**
- UCI Machine Learning Repository, *Default of Credit Card Clients* (CC BY 4.0, 확인일 2026-10-18): https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Yeh, I-C. & Lien, C-h. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2).
- 이론: `docs/theory/stage_01.html`, `docs/theory/stage_02.html`

**다음 레슨**: 1-3. 결측치와 데이터 품질
"""
        ),
    ]
