from nbkit import SETUP_CELL, code, md

TITLE = "레슨 2-1. 단변량·다변량 분석과 변수 진단"


def cells():
    return [
        md(
            """
# 레슨 2-1. 단변량·다변량 분석과 변수 진단

**예상 시간** 70분 · **선행** 레슨 1-3 · **이론 페이지** `docs/theory/stage_03.html`, `docs/theory/stage_04.html`

## 학습 목표
1. 변수를 **하나씩**(단변량) 그리고 **함께**(다변량) 살펴보고, 둘의 차이를 설명한다.
2. 부도/정상 평균 비교에서 **p-value 와 효과크기(Cohen's d)** 를 구분해 읽는다.
3. 상관계수·**VIF**·조건수로 변수들이 같은 정보를 담고 있는지 진단한다.
4. 상관계수가 놓치는 **비선형 관계**를 확인한다.
5. **누수 후보**를 규칙으로 찾고, 최종 판단은 사람이 한다는 것을 이해한다.
6. 변수 **유지/제외 결정을 이유와 함께 기록**한다.

> 모든 진단은 **개발(학습) 구간**에서만 합니다. 나중에 쓸 테스트 구간을 보고 변수를 고르면 누수입니다.
"""
        ),
        SETUP_CELL,
        md(
            """
## 1. 개념

| | 단변량 분석 | 다변량 분석 |
|---|---|---|
| 보는 것 | 변수 **하나**의 분포·결측·타깃과의 관계 | 변수 **여럿**의 관계(상관, 공선성, 교란) |
| 질문 | 이 변수에 정보가 있나? 데이터는 깨끗한가? | 이 변수들이 서로 같은 말을 하나? 함께 쓰면 안정적인가? |
| 도구 | 요약표, 부도율, 평균 비교 | 상관행렬, VIF, 조건수 |

**통계적 유의성만으로 변수를 고르면 안 됩니다.** 변수 선택은 *업무 논리 · 신청 시점의 가용성 · 안정성 · 데이터 품질 · 법적·공정성 검토 · 통계적 증거*를 함께 보고, **이유를 기록**하는 일입니다.
"""
        ),
        md("## 2. 데이터와 개발 구간\n신청일 순으로 앞 70%를 **개발 구간**, 뒤 30%를 **테스트 구간**으로 나누고, 이번 레슨에서는 개발 구간만 봅니다."),
        code(
            '''
from creditlab.data import load_learning_data

df_all, meta = load_learning_data("synthetic", seed=SEED)
TARGET = meta["target"]
df_all = df_all.sort_values("application_date").reset_index(drop=True)
cut = int(len(df_all) * 0.7)
dev, test = df_all.iloc[:cut].copy(), df_all.iloc[cut:].copy()   # test 는 이 레슨에서 열어 보지 않습니다
print(f"개발 {len(dev):,}행 (부도율 {dev[TARGET].mean():.1%}) | 테스트 {len(test):,}행은 보관")
NUM = ["age", "income_10k", "loan_amount_10k", "employment_years", "card_utilization", "num_delinq_12m", "inquiries_6m"]
CAT = ["housing_type", "region"]
dev.head(3)
'''
        ),
        md("## 3. 단변량 분석"),
        code(
            '''
from creditlab.features import univariate_numeric

uni = univariate_numeric(dev, NUM, TARGET)
uni[["variable", "n", "missing_pct", "mean", "median", "min", "max", "mean_bad", "mean_good"]].round(3)
'''
        ),
        md("### ✋ 먼저 예측\n부도 그룹과 정상 그룹의 **평균 차이가 가장 큰** 변수는 무엇일까요? (단위가 다르니 *표준화한 차이*, 즉 효과크기로 생각해 보세요)"),
        code('pred_top_variable = None  # ← 예: "age"'),
        code(
            '''
from creditlab.features import compare_groups

cmp = pd.DataFrame([compare_groups(dev, c, TARGET) for c in NUM])
cmp["abs_d"] = cmp["cohens_d"].abs()
cmp = cmp.sort_values("abs_d", ascending=False).reset_index(drop=True)
print(cmp[["variable", "mean_bad", "mean_good", "diff", "ci_low", "ci_high", "cohens_d", "welch_p", "mannwhitney_p"]].round(4).to_string(index=False))
top = cmp.loc[0, "variable"]
print(f"\\n효과크기가 가장 큰 변수: {top} (|d|={cmp.loc[0, 'abs_d']:.2f})")
if pred_top_variable:
    print("내 예측:", pred_top_variable, "→", "맞음" if pred_top_variable == top else "틀림")
'''
        ),
        md(
            """
**읽는 법**
- `diff` 는 (부도 평균 − 정상 평균), `cohens_d` 는 그 차이를 표준편차 단위로 바꾼 **효과크기**입니다. 대략 0.2 작음 · 0.5 중간 · 0.8 큼(Cohen 의 관행적 기준, 절대 규칙 아님).
- `welch_p` 는 "두 그룹 평균이 같다고 가정할 때 이만큼 차이날 확률"입니다. **차이가 중요하다는 뜻이 아닙니다.**
- 변수를 7개 검정했으니 **다중검정** 문제가 있습니다. 하나만 p<0.05 가 나와도 우연일 수 있습니다(단계 14).
- 표에서 효과크기가 0 에 가까운 변수가 있나요? 단변량 평균 차이가 작다고 업무적으로 중요한 변수를 바로 버리지는 않습니다. 다른 변수와의 *결합*(비율·상호작용)이나 비선형 관계로 정보를 가질 수 있습니다.
"""
        ),
        md("### p-value 는 표본 크기에 흔들리지만 효과크기는 안정적이다"),
        code(
            '''
var = "age"
rows = []
for n in [300, 1000, len(dev)]:
    for seed in range(3):
        sub = dev.sample(n=n, random_state=seed)
        try:
            r = compare_groups(sub, var, TARGET)
            rows.append({"표본 수": n, "시드": seed, "부도 건수": r["n_bad"], "d": r["cohens_d"], "p-value": r["welch_p"]})
        except ValueError as e:      # 부도가 2건 미만이면 계산하지 않고 알려줌
            rows.append({"표본 수": n, "시드": seed, "부도 건수": "-", "d": np.nan, "p-value": np.nan})
pd.DataFrame(rows).round(4).to_string(index=False)
'''
        ),
        md("표본이 작을수록 같은 변수라도 p-value 가 크게 흔들립니다. 효과크기 `d` 의 *추정 불확실성*도 표본이 작으면 큽니다. **\"유의하지 않다\"가 \"관계가 없다\"는 뜻은 아닙니다.**"),
        md("### 범주형 변수: 범주별 부도율과 신뢰구간"),
        code(
            '''
from creditlab.features import univariate_categorical

for c in CAT:
    print(f"[{c}]")
    print(univariate_categorical(dev, c, TARGET, min_count=300).round(3).to_string(index=False), "\\n")
'''
        ),
        md("`rare=True` 는 건수가 적어 부도율이 불안정한 범주입니다. 구간(`bad_rate_lo`~`hi`)이 겹치면 범주 간 차이를 단정하기 어렵습니다."),
        md("## 4. 다변량 분석 — 상관과 다중공선성"),
        code(
            '''
from creditlab.features import correlation_matrix, high_correlation_pairs

corr = correlation_matrix(dev, NUM, "pearson")
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(NUM)), NUM, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(NUM)), NUM, fontsize=8)
fig.colorbar(im)
ax.set_title("Pearson correlation (dev)")
plt.show()
print("상관 0.5 이상인 쌍:")
high_correlation_pairs(corr, 0.5)
'''
        ),
        md(
            """
위 표에서 상관이 높은 쌍이 있는지(있다면 업무적으로 왜 그런지) 먼저 확인하세요. 이 합성 데이터에는 아주 강한 공선성이 없으므로, **공선성이 있는 상황을 직접 만들어** VIF 가 어떻게 반응하는지 봅니다.

- `age_months` = 나이×12 + 작은 잡음 → 나이와 거의 같은 정보
- `loan_to_income` = 대출금 / 소득 → 두 변수에서 만들어진 파생 변수
"""
        ),
        code("pred_vif = None  # ← 'age' 와 'age_months' 를 함께 넣으면 VIF 는 대략 얼마일까요? (10 미만 / 10~100 / 100 초과 중에서)"),
        code(
            '''
from creditlab.features import condition_number, vif_table

rng = np.random.default_rng(SEED)
work = dev.copy()
work["age_months"] = work["age"] * 12 + rng.normal(0, 3, len(work))
work["loan_to_income"] = work["loan_amount_10k"] / work["income_10k"]

base_cols = ["age", "income_10k", "loan_amount_10k", "employment_years", "card_utilization"]
print("[기본 변수만]")
print(vif_table(work, base_cols).round(2).to_string(index=False))
print("조건수:", round(condition_number(work, base_cols), 2))

more = base_cols + ["age_months", "loan_to_income"]
print("\\n[파생 변수 추가]")
t = vif_table(work, more)
print(t.round(2).to_string(index=False))
print("조건수:", round(condition_number(work, more), 2))
print("\\n(결측이 있는 행은 제외하고 계산:", int(t["n_used"].iloc[0]), "행 사용)")
'''
        ),
        md(
            """
**읽는 법**: VIF 는 "이 변수를 *나머지 변수로* 얼마나 설명할 수 있나"를 뜻합니다. `age` 와 `age_months` 는 서로를 거의 완벽하게 설명하므로 VIF 가 치솟습니다. 5·10 같은 기준은 관행일 뿐이고, **VIF 가 높으면 자동 삭제가 아니라 '어느 변수들이 같은 정보를 담았나'를 업무적으로 확인**해 하나를 고릅니다(예: 둘 중 설명하기 쉽고 안정적인 쪽).

공선성은 *예측 성능*보다 **계수의 해석과 안정성**을 해칩니다. 같은 정보를 담은 변수가 둘이면 계수가 서로 부호를 바꿔 가며 흔들릴 수 있습니다.
"""
        ),
        md("### 상관계수가 놓치는 것 — 비선형 관계"),
        code(
            '''
rng = np.random.default_rng(SEED)
z = rng.normal(size=20000)
p_default = 0.04 + 0.12 * (np.abs(z) > 1.2)          # 값이 아주 낮거나 아주 높으면 위험(U자형)
demo = pd.DataFrame({"z": z, "default": rng.binomial(1, p_default)})

print("Pearson :", round(demo["z"].corr(demo["default"]), 4))
print("Spearman:", round(demo["z"].corr(demo["default"], method="spearman"), 4))
demo["bin"] = pd.qcut(demo["z"], 8)
rates = demo.groupby("bin", observed=True)["default"].mean()
print("\\n구간별 부도율:")
print((rates * 100).round(1).to_string())
'''
        ),
        md("상관계수는 0 에 가깝지만 부도율은 구간에 따라 크게 다릅니다(양 끝이 높음). **상관이 낮다고 변수를 버리면** 이런 변수를 놓칩니다. 그래서 다음 단계(비닝·WOE)에서 구간별 부도율을 직접 봅니다."),
        md("## 5. 누수 후보 점검 — 기계가 찾고, 사람이 판단한다"),
        code(
            '''
from creditlab.features import leakage_suspects

# 일부러 누수 변수 두 개를 심어 봅니다. (실제 데이터에서는 이런 열이 어디 숨어 있을지 모릅니다)
rng = np.random.default_rng(SEED)
leaky = dev.copy()
flip = rng.uniform(size=len(leaky)) < 0.03
leaky["collection_flag"] = np.where(flip, 1 - leaky[TARGET], leaky[TARGET])            # 부도 후 추심 여부(결과 이후 정보)
leaky["score_v2"] = leaky[TARGET] * 2.0 + rng.normal(0, 0.3, len(leaky))               # 이름은 평범하지만 결과와 거의 같음

found = leakage_suspects(leaky, TARGET, columns=NUM + CAT + ["collection_flag", "score_v2"])
found
'''
        ),
        md(
            """
- `이름 규칙`: 결과 이후를 암시하는 단어(예: collection, recovery, status, after…)가 이름에 있는 열
- `단일 변수 AUROC`: 변수 하나만으로 부도를 거의 완벽히 구분하는 열 → 신용 데이터에서는 보통 *너무 좋아서 의심스럽다*

규칙에 걸리지 않았다고 안전한 것은 **아닙니다**. 모든 변수에 대해 *"이 값은 신청(예측) 시점에 이미 알려져 있는가?"* 를 데이터 사전으로 확인해야 합니다. 걸린 변수도 항상 누수는 아닙니다 — 사람이 판단합니다.
"""
        ),
        md(
            """
## 6. ✏️ 실습 — 변수 결정 기록
후보 변수 7개 + 범주형 2개(`NUM + CAT`)에 대해 **keep / drop / review** 를 정하고 **이유를 적으세요.**
- 이유는 통계 결과(효과크기, 결측, VIF)뿐 아니라 **업무 논리 · 신청 시점 가용성 · 법적·공정성 검토 필요 여부**도 고려해 쓰세요.
- 확신이 없으면 `review`(보류)를 쓰고 무엇을 확인해야 하는지 적습니다.
- 이유가 비어 있으면 기록되지 않습니다.
"""
        ),
        code(
            '''
# [exercise]
from creditlab.features import VariableLog

log = VariableLog()
# 예)  log.add("age", "review", "위험과 관련 있으나 연령 사용의 법적·공정성 검토 필요", evidence="d=0.1, 결측 0%")
#      log.add("card_utilization", "keep", "신청 시점에 알 수 있고 업무적으로 타당, 효과크기 큼", evidence="d=0.5")
# 힌트1: 변수 이름은 NUM + CAT 에 있습니다.   힌트2: 통계 결과는 위 표에서 가져오세요.

todo = log.undecided(NUM + CAT)
print(f"아직 결정하지 않은 변수 {len(todo)}개:", todo)
'''
        ),
        md(
            """
<details>
<summary>정답 예시 (먼저 직접 해 본 뒤 펼치세요)</summary>

```python
log.add("age", "review", "위험과 관련 있을 수 있으나 연령 사용의 법적·공정성 검토가 먼저 필요", evidence="결측 0%")
log.add("income_10k", "review", "업무 논리상 상환능력과 직결되지만, 단변량 효과크기가 작고 대출금액과 상관이 있어 대출/소득 같은 비율 변수로 재구성할지 검토", evidence="효과크기·상관표 참고")
log.add("loan_amount_10k", "keep", "대출 규모는 위험과 직결, 신청 시점에 알려짐")
log.add("employment_years", "keep", "소득 안정성의 대리 변수, 신청 시점에 확인 가능")
log.add("card_utilization", "keep", "효과크기가 큰 대표적 위험 변수, 신청 시점에 알려짐")
log.add("num_delinq_12m", "keep", "과거 연체 이력은 핵심 위험 정보, 신청 시점 기준 12개월")
log.add("inquiries_6m", "keep", "최근 조회 건수. 값이 작은 정수라 비닝 시 구간 수 주의")
log.add("housing_type", "review", "범주 간 부도율 구간이 겹침. 'other' 범주 건수 부족 → 묶기 여부 검토")
log.add("region", "review", "지역은 보호 속성의 대리 변수가 될 수 있어 공정성 검토 필요")
```
</details>
"""
        ),
        code(
            '''
decided = log.to_frame()
if len(decided):
    print(decided.to_string(index=False))
    print("\\n유지(keep):", log.kept())
    print("미결정:", log.undecided(NUM + CAT))
else:
    print("아직 기록이 없습니다. 위 실습 셀에서 log.add(...) 를 채워 보세요.")
'''
        ),
        md("## 7. 퀴즈"),
        code(
            '''
from creditlab.learning import Question, check, grade, show

QUIZ = [
    Question("Q1", "p-value", "표본이 100만 건일 때 두 그룹의 평균 차이가 아주 작은데 p-value 가 0.0001 입니다. 가장 적절한 해석은?",
             {"A": "이 변수는 매우 중요하다", "B": "통계적으로는 유의하지만 효과크기를 함께 봐야 실무적 중요성을 판단할 수 있다",
              "C": "p-value 가 작으니 차이가 크다", "D": "귀무가설이 참일 확률이 0.01% 이다"},
             "B", "표본이 크면 사소한 차이도 유의해집니다. 효과크기·신뢰구간·업무 의미를 함께 봅니다.",
             ("p-value 는 무엇에 대한 확률일까요?",)),
    Question("Q2", "상관 vs 비선형", "Pearson 상관이 0.01 입니다. 이 변수에 대한 올바른 판단은?",
             {"A": "정보가 없으니 삭제한다", "B": "선형 관계가 약할 뿐이다. 구간별 부도율을 확인해 비선형 관계를 점검한다",
              "C": "항상 중요하다", "D": "상관이 0 이면 독립이다"},
             "B", "U자형처럼 상관은 0 인데 관계가 뚜렷한 경우가 있습니다.", ("우리가 만든 U자형 예제를 떠올리세요.",)),
    Question("Q3", "VIF", "두 변수의 VIF 가 모두 200 입니다. 올바른 대응은?",
             {"A": "VIF 가 큰 변수를 자동으로 모두 삭제한다", "B": "어떤 변수들이 같은 정보를 담았는지 확인하고, 업무적으로 설명하기 쉬운 쪽을 고른다",
              "C": "무시한다", "D": "변수를 하나 더 추가한다"},
             "B", "VIF 는 경보입니다. 원인을 확인하고 업무 논리로 선택합니다.", ("age 와 age_months 를 떠올려 보세요.",)),
    Question("Q4", "누수", "누수 점검 규칙에 걸리지 않은 변수는?",
             {"A": "안전하다", "B": "안전하다고 단정할 수 없다. 각 변수가 예측 시점에 알려져 있는지 데이터 사전으로 확인해야 한다",
              "C": "규칙에 안 걸리면 모형에 바로 쓴다", "D": "AUROC 가 낮으니 안전하다"},
             "B", "규칙은 후보를 찾는 도구일 뿐, 최종 판정은 사람이 합니다.", ("이름이 평범한 score_v2 는 어땠나요?",)),
    Question("Q5", "변수 선택", "다음 중 변수 선택 근거로 가장 적절한 것은?",
             {"A": "p-value 가 0.05 미만이다", "B": "업무 논리·신청 시점 가용성·안정성·데이터 품질·법적/공정성 검토와 통계적 증거를 종합하고 이유를 기록한다",
              "C": "IV 가 가장 높다", "D": "상관이 가장 낮다"},
             "B", "통계 하나로 결정하지 않고, 결정과 이유를 기록해 검증 가능하게 합니다.", ("VariableLog 에 무엇을 적었나요?",)),
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
1. UCI 데이터(`load_learning_data("uci")`)의 `LIMIT_BAL`, `AGE`, `BILL_AMT1~6`, `PAY_AMT1~6` 에 오늘의 도구를 적용하세요. `BILL_AMT1~6` 의 VIF 는 어떤가요? 그 이유를 한 문장으로 쓰세요.
2. 효과크기가 가장 큰 변수 3개와 그 효과크기를 적고, 그 변수들이 **신청 시점에 알려진 정보인지** 데이터 설명서로 확인하세요.
3. 자신의 `VariableLog` 를 표로 저장하고, 확신이 가장 낮은 결정 한 가지와 이유를 쓰세요.

**체크포인트**: 퀴즈 4/5 이상 · 변수 결정 기록(이유 포함) 작성

**다음**: 레슨 2-2 — 비닝·WOE·IV. 상관이 놓친 비선형 관계를 구간별로 직접 봅니다.

**참고자료**
- Cohen, J., *Statistical Power Analysis for the Behavioral Sciences* (효과크기 관행적 기준).
- Kutner, M. et al., *Applied Linear Statistical Models* (VIF 와 다중공선성).
- Kaufman, S. et al. (2012). Leakage in data mining. *ACM TKDD*, 6(4).
- 이론: `docs/theory/stage_03.html`, `docs/theory/stage_04.html`
"""
        ),
    ]
