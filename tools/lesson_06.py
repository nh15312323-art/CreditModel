from nbkit import SETUP_CELL, code, md

TITLE = "레슨 3-1. 로지스틱 회귀의 수학과 해석"


def cells():
    return [
        md(
            """
# 레슨 3-1. 로지스틱 회귀의 수학과 해석

**예상 시간** 80분 · **선행** 레슨 2-2 · **이론 페이지** `docs/theory/stage_06.html`

## 학습 목표
1. **확률 ↔ 오즈 ↔ 로그오즈**를 오가며 계산한다.
2. 선형회귀를 확률 예측에 그대로 쓰기 어려운 이유를 확인한다.
3. 로지스틱 계수가 **로그 오즈비**임을 2×2 표로 직접 검산한다.
4. 계수·표준오차·p-value·**오즈비와 신뢰구간**을 읽고, 단위가 해석을 바꾼다는 것을 안다.
5. **완전 분리·이벤트 수 부족·공선성** 경고를 이해한다.
6. **규제(L2/L1)** 가 계수를 어떻게 바꾸는지 보고, 규제 강도 C 는 *학습 구간 안에서만* 고른다.

> 모든 적합은 **개발 구간**에서 합니다. 테스트 구간은 마지막에 한 번만 사용합니다.
"""
        ),
        SETUP_CELL,
        md("## 1. 확률·오즈·로그오즈"),
        code("pred_odds_half = None  # ← 부도확률이 0.5 일 때 오즈(부도:정상)는?  그때 로그오즈는?"),
        code(
            '''
from creditlab.models import logit, odds, sigmoid

p = np.array([0.01, 0.05, 0.10, 0.50, 0.90])
tbl = pd.DataFrame({"확률 p": p, "오즈 p/(1-p)": odds(p), "로그오즈 ln(오즈)": logit(p)})
print(tbl.round(4).to_string(index=False))
print("\\n로그오즈 → 확률 되돌리기(sigmoid):", np.round(sigmoid(tbl["로그오즈 ln(오즈)"]), 4).tolist())
if pred_odds_half is not None:
    print("내 예측:", pred_odds_half)
'''
        ),
        md(
            """
- 확률은 0~1 이지만 **로그오즈는 −∞~+∞** 입니다. 로지스틱 회귀는 이 *로그오즈를 변수들의 선형식*으로 모형화합니다.
  `ln(PD/(1−PD)) = β0 + β1·x1 + β2·x2 + …`
- 확률이 작을 때(예: 1%, 5%)는 오즈 ≈ 확률입니다. 확률이 커지면 둘은 크게 달라집니다.
- 계수의 의미: `x` 가 1 단위 늘면 **로그오즈가 β 만큼, 오즈가 exp(β)배**가 됩니다. (확률이 exp(β)배가 되는 것이 *아닙니다.*)
"""
        ),
        md("## 2. 데이터와 개발 구간"),
        code(
            '''
from creditlab.data import load_learning_data

df_all, meta = load_learning_data("synthetic", seed=SEED)
TARGET = meta["target"]
df_all = df_all.sort_values("application_date").reset_index(drop=True)
cut = int(len(df_all) * 0.7)
dev, test = df_all.iloc[:cut].copy(), df_all.iloc[cut:].copy()
FEATS = ["age", "card_utilization", "num_delinq_12m", "inquiries_6m", "employment_years"]
print(f"개발 {len(dev):,}행(부도 {int(dev[TARGET].sum())}건) | 테스트 {len(test):,}행은 마지막까지 보관")
'''
        ),
        md("## 3. 왜 선형회귀로는 부족한가"),
        code(
            '''
Xd = np.column_stack([np.ones(len(dev)), dev[FEATS].to_numpy(dtype=float)])
beta_lin, *_ = np.linalg.lstsq(Xd, dev[TARGET].to_numpy(dtype=float), rcond=None)
lin_pred = Xd @ beta_lin
print(f"선형확률모형 예측의 범위: {lin_pred.min():.3f} ~ {lin_pred.max():.3f}")
print(f"확률로서 불가능한 값(<0 또는 >1)의 개수: {int(((lin_pred < 0) | (lin_pred > 1)).sum())}")

# 선형모형은 극단적인 입력에서 쉽게 범위를 벗어납니다 (관측 범위 밖 시나리오)
scenario = pd.DataFrame({"age": [60], "card_utilization": [-0.2], "num_delinq_12m": [0], "inquiries_6m": [0], "employment_years": [30]})
s_lin = float(np.r_[1, scenario.iloc[0].to_numpy(dtype=float)] @ beta_lin)
print(f"\\n극단 시나리오(사용률이 매우 낮고 연체 없음)의 선형 예측: {s_lin:.3f}  (음수면 확률이 될 수 없음)")
'''
        ),
        md(
            """
선형확률모형은 ① 예측이 0~1 을 벗어날 수 있고 ② 잔차의 분산이 일정하지 않으며(`p(1−p)` 에 의존) ③ 계수를 오즈로 해석할 수 없습니다. 로지스틱 함수 `1/(1+e^−z)` 는 어떤 z 에서도 0~1 안에 머물게 만듭니다.
"""
        ),
        md(
            """
## 4. 계수는 로그 오즈비다 — 2×2 표로 검산
주거 형태가 `rent` 인 신청자와 `own` 인 신청자의 부도 건수로 오즈비를 **손으로** 계산하고, 로지스틱 회귀 결과와 비교합니다.
"""
        ),
        code("pred_or_rent = None  # ← rent 의 오즈가 own 의 오즈보다 몇 배일까요? (1보다 큰가 작은가부터)"),
        code(
            '''
from creditlab.models import fit_logistic

sub = dev[dev["housing_type"].isin(["rent", "own"])].copy()
sub["is_rent"] = (sub["housing_type"] == "rent").astype(float)
ct = pd.crosstab(sub["housing_type"], sub[TARGET])
print(ct.to_string())
bad_r, good_r = ct.loc["rent", 1], ct.loc["rent", 0]
bad_o, good_o = ct.loc["own", 1], ct.loc["own", 0]
or_hand = (bad_r / good_r) / (bad_o / good_o)
se_hand = np.sqrt(1 / bad_r + 1 / good_r + 1 / bad_o + 1 / good_o)      # Woolf 표준오차
print(f"\\n손계산 오즈비 = ({bad_r}/{good_r}) / ({bad_o}/{good_o}) = {or_hand:.4f}, ln = {np.log(or_hand):.4f}, 표준오차 = {se_hand:.4f}")

f2 = fit_logistic(sub[["is_rent"]], sub[TARGET])
row = f2.summary().set_index("variable").loc["is_rent"]
print(f"로지스틱 회귀   계수 = {row['coef']:.4f}, exp(계수) = {row['odds_ratio']:.4f}, 표준오차 = {row['std_err']:.4f}")
print("두 값이 같은가:", np.isclose(np.log(or_hand), row["coef"], atol=1e-6), "/", np.isclose(se_hand, row["std_err"], atol=1e-6))
if pred_or_rent is not None:
    print("내 예측:", pred_or_rent)
'''
        ),
        md("**이것이 계수의 정체입니다**: 변수가 0→1 일 때의 **로그 오즈비**. 표준오차도 정보행렬로 계산한 값과 손계산(Woolf)이 일치합니다."),
        md("## 5. 여러 변수 — 계수·오즈비·신뢰구간·p-value"),
        code("pred_signs = None  # ← 5개 변수의 계수 부호를 적어 보세요. 예: {\"age\": \"-\", \"card_utilization\": \"+\", ...}"),
        code(
            '''
fit = fit_logistic(dev[FEATS], dev[TARGET])
summ = fit.summary()
print(summ[["variable", "coef", "std_err", "z", "p_value", "odds_ratio", "or_ci_low", "or_ci_high"]].round(4).to_string(index=False))
print(f"\\n수렴: {fit.converged} ({fit.n_iter}회) | 표본 {fit.n:,} | 부도 {fit.n_events} | 의사 R²(McFadden): {fit.pseudo_r2:.3f}")
print("경고:", fit.warnings or "없음")
if pred_signs:
    print("내 예측:", pred_signs)
'''
        ),
        md(
            """
**읽는 법**
- `odds_ratio` = exp(계수): 그 변수가 **1 단위** 늘 때 *다른 변수를 고정하면* 오즈가 몇 배가 되는가.
- 신뢰구간이 **1 을 포함**하면(`or_ci_low < 1 < or_ci_high`) 효과의 방향도 불확실합니다.
- **단위가 해석을 바꿉니다.** `card_utilization` 은 0~1 이라 "1 단위"가 사실상 *전체 범위*입니다. 0.1 단위로 보려면 계수에 0.1 을 곱한 뒤 exp 를 취합니다.
- p-value 는 유의성이지 중요도가 아닙니다. 의사 R²(McFadden)는 선형회귀 R² 와 같은 의미가 아니며 값이 낮게 나오는 것이 흔합니다(위 결과에서도 낮습니다). 모형의 쓸모는 변별력·보정 지표로 따로 평가합니다(Phase 4).
"""
        ),
        md("## 6. ✏️ 실습 — 오즈비로 말하기"),
        code(
            '''
# [exercise]
or_delinq = None       # ← 연체 횟수(num_delinq_12m)가 1건 늘면 오즈는 몇 배? (소수 둘째 자리)
or_util_01 = None      # ← 카드 사용률이 0.1(=10%p) 오르면 오즈는 몇 배?  (힌트: exp(계수 × 0.1))
# 힌트1: 위 표의 coef 또는 fit.slopes 를 쓰세요.   힌트2: 사용률 0.1 증가 = 계수에 0.1 을 곱하고 exp.

if or_delinq is not None and or_util_01 is not None:
    true_delinq = float(np.exp(fit.slopes["num_delinq_12m"]))
    true_util = float(np.exp(fit.slopes["card_utilization"] * 0.1))
    print(f"연체 +1건 → 오즈 {or_delinq:.2f}배 (정답 {true_delinq:.2f}) :", "맞음" if abs(or_delinq - true_delinq) < 0.02 else "다시 계산해 보세요")
    print(f"사용률 +0.1 → 오즈 {or_util_01:.2f}배 (정답 {true_util:.2f}) :", "맞음" if abs(or_util_01 - true_util) < 0.02 else "다시 계산해 보세요")
else:
    print("값을 채우고 다시 실행하세요.")
'''
        ),
        md(
            """
<details>
<summary>정답 예시 (먼저 직접 해 본 뒤 펼치세요)</summary>

```python
or_delinq = round(float(np.exp(fit.slopes["num_delinq_12m"])), 2)
or_util_01 = round(float(np.exp(fit.slopes["card_utilization"] * 0.1)), 2)
```
</details>
"""
        ),
        md("## 7. 완전 분리와 경고들"),
        code(
            '''
# (1) 결과를 거의 그대로 담은 누수 변수가 있으면: 완전/준완전 분리
rng = np.random.default_rng(SEED)
leak = dev[FEATS].copy()
leak["post_collection"] = dev[TARGET].to_numpy() * 3.0 + rng.uniform(0, 1, len(dev))      # 부도면 항상 3 이상, 아니면 3 미만
fl = fit_logistic(leak, dev[TARGET])
print("[누수 변수 포함] 경고:")
for w in fl.warnings:
    print("  -", w)
print("  post_collection 계수:", round(float(fl.slopes["post_collection"]), 1), "← 비현실적으로 큼")

# (2) 표본이 작아 이벤트 수가 부족하면
small = dev.sample(400, random_state=SEED)
fs = fit_logistic(small[FEATS], small[TARGET])
print(f"\\n[400건만 사용] 부도 {fs.n_events}건, 변수 {len(FEATS)}개")
for w in fs.warnings:
    print("  -", w)
print("  계수 표준오차(작은 표본):", fs.summary()["std_err"].iloc[1:].round(3).tolist())
print("  계수 표준오차(전체 개발):", fit.summary()["std_err"].iloc[1:].round(3).tolist())
'''
        ),
        md("완전 분리에서는 **최대우도 추정치가 존재하지 않아** 계수가 한없이 커집니다. 분리가 보이면 변수가 *결과 이후 정보*는 아닌지 먼저 의심하세요(레슨 2-1). 이벤트가 적으면 표준오차가 커져 계수를 믿기 어렵습니다."),
        md("## 8. 규제 — L2, L1 과 규제 강도 C"),
        code(
            '''
from sklearn.preprocessing import StandardScaler
from creditlab.models import regularization_path, select_C_by_cv

scaler = StandardScaler().fit(dev[FEATS])                       # 개발 구간으로만 fit
Zd = pd.DataFrame(scaler.transform(dev[FEATS]), columns=FEATS, index=dev.index)
Zt = pd.DataFrame(scaler.transform(test[FEATS]), columns=FEATS, index=test.index)

Cs = [100, 1, 0.1, 0.01, 0.001]
print("[L2] C 가 작을수록(강한 규제) 계수가 0 쪽으로 줄어듭니다")
print(regularization_path(Zd, dev[TARGET], Cs, "l2").round(3).to_string(index=False))
print("\\n[L1] 일부 계수가 정확히 0 이 됩니다(변수 선택 효과)")
print(regularization_path(Zd, dev[TARGET], Cs, "l1").round(3).to_string(index=False))
'''
        ),
        md(
            """
- 규제는 **계수 크기에 벌점**을 줘서 과적합을 줄입니다. 입력 단위가 다르면 규제가 불공정하므로 **표준화**를 먼저 합니다(여기서는 개발 구간 평균·표준편차로만).
- L1 은 일부 계수를 0 으로 만들어 **변수 선택** 효과가 있지만, 상관된 변수 중 **무엇이 살아남는지가 불안정**할 수 있습니다.
- 규제 모형의 p-value·신뢰구간은 표준 공식이 맞지 않습니다. *해석 중심 추론은 규제 없는 모형으로*, *예측 안정성은 규제 모형으로* 나눠 씁니다.
"""
        ),
        md("### 규제 강도 C 는 학습 구간 안에서 시간 순서 교차검증으로"),
        code(
            '''
cv = select_C_by_cv(Zd, dev[TARGET], [10, 1, 0.1, 0.01, 0.001], penalty="l2", n_splits=4)   # dev 는 시간순 정렬됨
print(cv.round(5).to_string(index=False))
best_C = float(cv.loc[cv["best"], "C"].iloc[0])
print("\\n선택된 C:", best_C, "(테스트 구간은 쓰지 않았습니다)")

from sklearn.metrics import log_loss
from creditlab.models import fit_regularized

m_none = fit_regularized(Zd, dev[TARGET], "none")
m_best = fit_regularized(Zd, dev[TARGET], "l2", C=best_C)
print("\\n[마지막에 테스트에서 단 한 번 확인] log loss (작을수록 좋음)")
print("  규제 없음:", round(log_loss(test[TARGET], m_none.predict_proba(Zt)[:, 1]), 4))
print(f"  L2(C={best_C}):", round(log_loss(test[TARGET], m_best.predict_proba(Zt)[:, 1]), 4))
'''
        ),
        md("데이터가 충분하고 변수가 적으면 규제의 효과는 작을 수 있습니다. 규제가 항상 이기는 것은 아닙니다 — 변수가 많거나 이벤트가 적거나 공선성이 있을 때 도움이 됩니다."),
        md("## 9. 퀴즈"),
        code(
            '''
from creditlab.learning import Question, check, grade, show

QUIZ = [
    Question("Q1", "계수 해석", "num_delinq_12m 의 계수가 0.5 입니다. 다른 변수를 고정하고 연체가 1건 늘면?",
             {"A": "부도확률이 0.5 만큼 늘어난다", "B": "로그오즈가 0.5 늘고, 오즈는 exp(0.5)≈1.65 배가 된다",
              "C": "부도확률이 1.65 배가 된다", "D": "부도율이 50% 가 된다"},
             "B", "계수는 로그오즈의 변화량입니다. 확률의 변화는 기준 확률에 따라 달라집니다.", ("선형식은 어떤 양에 대한 것이었나요?",)),
    Question("Q2", "선형회귀", "선형회귀로 부도확률을 예측하면 생기는 문제는?",
             {"A": "문제 없다", "B": "예측이 0~1 을 벗어날 수 있고 오차의 분산이 일정하지 않다",
              "C": "항상 더 정확하다", "D": "계수를 계산할 수 없다"},
             "B", "로지스틱 함수는 출력을 0~1 로 제한합니다.", ("극단 시나리오 예측이 어땠나요?",)),
    Question("Q3", "분리", "어떤 변수 하나만으로 부도/정상이 완전히 갈립니다. 가장 먼저 의심할 것은?",
             {"A": "최고의 변수를 찾았다", "B": "결과 이후 정보(누수)이거나 정의/데이터 오류이다. 최대우도 추정치가 존재하지 않아 계수가 비정상적으로 커진다",
              "C": "표본이 너무 크다", "D": "규제를 끄면 해결된다"},
             "B", "신용 데이터에서 완벽한 분리는 대개 누수의 신호입니다.", ("post_collection 변수를 떠올리세요.",)),
    Question("Q4", "규제", "L1 규제의 특징으로 옳은 것은?",
             {"A": "계수를 절대 0 으로 만들지 않는다", "B": "일부 계수를 정확히 0 으로 만들어 변수 선택 효과가 있으나, 상관된 변수 사이 선택은 불안정할 수 있다",
              "C": "항상 L2 보다 성능이 좋다", "D": "표준화가 필요 없다"},
             "B", "L1 은 희소한 해를 주지만 상관된 변수에서는 선택이 흔들립니다.", ("C 를 줄였을 때 n_nonzero 열을 보세요.",)),
    Question("Q5", "튜닝과 누수", "규제 강도 C 를 고르는 올바른 방법은?",
             {"A": "테스트 AUROC 가 가장 높은 C 를 고른다", "B": "개발(학습) 구간 안에서 시간 순서 교차검증으로 고르고, 테스트는 마지막에 한 번만 본다",
              "C": "C 는 항상 1 이다", "D": "전체 데이터로 교차검증한다"},
             "B", "테스트를 보고 설정을 고르면 테스트가 더 이상 '미래'가 아닙니다.", ("select_C_by_cv 에 어떤 데이터를 넣었나요?",)),
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
## 10. 과제와 정리
**과제**
1. UCI 데이터의 `LIMIT_BAL`, `AGE`, `PAY_0`, `BILL_AMT1` 로 로지스틱 모형을 적합하고, `PAY_0` 이 1 단계 늘 때의 오즈비와 신뢰구간을 한 문장으로 해석하세요. (표본이 크면 p-value 가 거의 항상 작다는 점도 언급)
2. 변수 하나의 단위를 바꿔(예: 0.1 단위) 계수·오즈비가 어떻게 변하는지, 모형의 예측이 바뀌는지 확인하고 설명하세요.
3. 규제 없는 모형과 L2 모형의 계수를 표로 비교하고, 어느 쪽을 해석용/예측용으로 쓸지 이유를 쓰세요.

**체크포인트**: 퀴즈 4/5 이상 · 오즈비 실습 통과 · 2×2 표 검산

**다음**: 레슨 3-2 — WOE 변수로 모형을 만들고 **점수카드**로 바꿉니다.

**참고자료**
- Hosmer, D., Lemeshow, S., Sturdivant, R., *Applied Logistic Regression* (Wiley).
- Agresti, A., *Categorical Data Analysis* (Wiley).
- scikit-learn: Logistic regression 사용자 가이드 (규제·solver).
- 이론: `docs/theory/stage_06.html`
"""
        ),
    ]
