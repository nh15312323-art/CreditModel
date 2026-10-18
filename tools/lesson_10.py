from nbkit import SETUP_CELL, code, md

TITLE = "레슨 5-1. 트리·부스팅·설명 (챌린저 모형)"


def cells():
    return [
        md(
            """
# 레슨 5-1. 트리·부스팅·설명 (챌린저 모형)

**예상 시간** 100분 · **선행** 레슨 4-2 · **이론 페이지** `docs/theory/stage_11.html`

## 학습 목표
1. 지니 불순도와 **분할 이득**을 손으로 계산한다.
2. 깊은 단일 트리의 **과적합**을 학습/테스트 AUROC 격차로 확인한다.
3. 부스팅의 **반복 횟수**를 *시간 순서 검증*으로 정한다(조기 종료).
4. 변수에 **단조 제약**을 걸어 예측이 업무 논리를 벗어나지 않게 한다.
5. **순열 중요도**와 부분의존으로 모형을 설명하고, 그 한계를 안다.
6. 스코어카드(챔피언)와 부스팅(챌린저)을 **구간과 함께** 비교한다.

> 이 레슨은 scikit-learn 만 사용합니다. XGBoost/CatBoost/SHAP 은 설치되지 않아 **실행하지 못했습니다.**
> 합성데이터 결과는 실제 성능이 아닙니다. 챌린저가 챔피언을 이겨도 **즉시 교체 근거가 되지 않습니다**(설명·안정성·규제 검토가 필요).
"""
        ),
        SETUP_CELL,
        md("## 1. 분할 이득 손계산"),
        code("pred_best = None  # ← 부모 노드 (부도 40, 정상 160). 분할 A: 왼쪽(30,70) 오른쪽(10,90) / 분할 B: 왼쪽(5,95) 오른쪽(35,65). 어느 쪽 이득이 클까요?"),
        code(
            '''
from creditlab.challengers import gini_impurity, split_gain

print(f"이론 예제: 부모(4,6)={gini_impurity(4, 6):.2f}, 왼쪽(1,4)={gini_impurity(1, 4):.2f}, 오른쪽(3,2)={gini_impurity(3, 2):.2f}, 이득={split_gain((1, 4), (3, 2)):.2f}")
print()
rows = []
for name, left, right in [("A", (30, 70), (10, 90)), ("B", (5, 95), (35, 65))]:
    rows.append({"분할": name, "왼쪽(bad,good)": left, "오른쪽(bad,good)": right, "이득": split_gain(left, right)})
print(pd.DataFrame(rows).round(4).to_string(index=False))
if pred_best is not None:
    print("내 예측:", pred_best)
'''
        ),
        md("지니 불순도는 `2p(1−p)` 입니다. **분할 이득 = 부모 불순도 − 자식 불순도의 가중 평균**. 트리는 모든 변수·분할점 후보에서 이득이 가장 큰 곳을 고르는 일을 반복합니다(탐욕적)."),
        md("## 2. 데이터와 챔피언(스코어카드)"),
        code(
            '''
from creditlab.data import load_learning_data
from creditlab.evaluation import auroc, compare_models, paired_bootstrap_diff
from creditlab.features import fit_binners, transform_woe_frame
from creditlab.models import fit_logistic
from creditlab.scorecard import ScoreScaling, Scorecard
from creditlab.challengers import MLFeatureEncoder

df_all, meta = load_learning_data("synthetic", seed=SEED)
TARGET = meta["target"]
df_all = df_all.sort_values("application_date").reset_index(drop=True)
cut = int(len(df_all) * 0.7)
dev, test = df_all.iloc[:cut].copy(), df_all.iloc[cut:].copy()

# 챔피언: 레슨 3-2 의 WOE 스코어카드
VARS = ["num_delinq_12m", "card_utilization", "age", "inquiries_6m", "employment_years", "loan_amount_10k"]
binners = fit_binners(dev, VARS, TARGET, method="quantile", max_bins=5, min_events=10)
fit = fit_logistic(transform_woe_frame(dev, binners), dev[TARGET])
card = Scorecard.from_fit(binners, fit, scaling=ScoreScaling(600, 50, 20))
pd_champ = card.predict_pd(test)

# 챌린저용 입력: 개발 구간에서만 범주를 배우는 인코더 (결측은 NaN 그대로)
NUM = ["age", "income_10k", "loan_amount_10k", "employment_years", "card_utilization", "num_delinq_12m", "inquiries_6m"]
CAT = ["housing_type", "region"]
enc = MLFeatureEncoder(NUM, CAT).fit(dev)
Xd, Xt = enc.transform(dev), enc.transform(test)
yd, yt = dev[TARGET].reset_index(drop=True), test[TARGET].reset_index(drop=True)
print(f"개발 {len(dev):,}행(부도 {int(yd.sum())}) | 테스트 {len(test):,}행(부도 {int(yt.sum())}) | 챔피언 테스트 AUROC = {auroc(yt, pd_champ):.4f}")
'''
        ),
        md("## 3. 단일 트리의 과적합"),
        code("pred_tree = None  # ← 깊이 제한이 없는 트리의 학습 AUROC 는 얼마나 높을까요(0.5~1)? 테스트 AUROC 는 어떻게 될까요?"),
        code(
            '''
from creditlab.challengers import fit_tree

rows = []
for depth, leaf in [(3, 50), (None, 1)]:
    t = fit_tree(Xd.fillna(-1), yd, max_depth=depth, min_samples_leaf=leaf)
    rows.append({"max_depth": depth, "min_samples_leaf": leaf, "학습 AUROC": auroc(yd, t.predict_proba(Xd.fillna(-1))[:, 1]),
                 "테스트 AUROC": auroc(yt, t.predict_proba(Xt.fillna(-1))[:, 1]), "잎 개수": int(t.get_n_leaves())})
print(pd.DataFrame(rows).round(3).to_string(index=False))
if pred_tree is not None:
    print("내 예측:", pred_tree)
'''
        ),
        md("깊이·잎 크기 제한이 없으면 학습 데이터를 거의 암기해 학습 AUROC 가 1 에 가까워지지만 **테스트에서는 크게 떨어집니다.** 학습 성능은 성능이 아닙니다."),
        md("## 4. 배깅(포레스트)과 부스팅, 그리고 조기 종료"),
        code(
            '''
from creditlab.challengers import fit_forest, fit_gbm_early_stop, fit_gbm

rf = fit_forest(Xd.fillna(-1), yd, n_estimators=300, min_samples_leaf=30)
gbm, curve, best_it = fit_gbm_early_stop(Xd, yd, enc.cat_mask, learning_rate=0.05, max_depth=3, min_samples_leaf=50, max_iter=600)
overfit = fit_gbm(Xd, yd, enc.cat_mask, learning_rate=0.05, max_depth=3, min_samples_leaf=50, max_iter=600)   # 조기 종료 없이 600회

tab = pd.DataFrame({
    "모형": ["스코어카드(챔피언)", "랜덤포레스트", f"부스팅(조기 종료, {best_it}회)", "부스팅(600회, 조기 종료 없음)"],
    "테스트 AUROC": [auroc(yt, pd_champ), auroc(yt, rf.predict_proba(Xt.fillna(-1))[:, 1]),
                    auroc(yt, gbm.predict_proba(Xt)[:, 1]), auroc(yt, overfit.predict_proba(Xt)[:, 1])]})
print(tab.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.plot(curve["iteration"], curve["train_log_loss"], label="train", color="#3d6fc8")
ax.plot(curve["iteration"], curve["valid_log_loss"], label="valid (later period)", color="#c8553d")
ax.axvline(best_it, ls=":", color="k"); ax.set_xlabel("boosting iterations"); ax.set_ylabel("log loss"); ax.legend()
plt.tight_layout(); plt.show()
print(f"선택된 반복 횟수 = {best_it} (검증 구간 log loss 최저점)")
'''
        ),
        md(
            """
- 학습 곡선은 계속 내려가지만 **미래 구간에서의 검증 곡선은 어느 지점부터 올라갑니다.** 그 최저점이 적정 반복 횟수입니다.
- 조기 종료의 검증 구간은 **시간상 뒤쪽**을 씁니다. 무작위 분할을 쓰면 미래 정보가 섞여 낙관적이 됩니다.
- 테스트 구간은 *반복 횟수를 정하는 데 쓰지 않았습니다.* 표의 숫자는 한 번만 본 값입니다.
"""
        ),
        md("## 5. 단조 제약"),
        code(
            '''
from creditlab.challengers import partial_dependence, monotone_violation

MONO = {"num_delinq_12m": 1, "card_utilization": 1}        # 연체·사용률이 클수록 부도확률은 비감소
free = fit_gbm(Xd, yd, enc.cat_mask, max_iter=best_it, learning_rate=0.05, max_depth=3, min_samples_leaf=50)
mono = fit_gbm(Xd, yd, enc.cat_mask, monotonic=MONO, max_iter=best_it, learning_rate=0.05, max_depth=3, min_samples_leaf=50)

rows = []
for var in MONO:
    vals = np.unique(np.quantile(Xt[var].dropna(), np.linspace(0.05, 0.95, 15)))
    for name, m in [("제약 없음", free), ("단조 제약", mono)]:
        pdp = partial_dependence(m, Xt, var, values=vals)
        rows.append({"변수": var, "모형": name, "부분의존 최대 위반폭": monotone_violation(pdp["mean_pd"], 1)})
print(pd.DataFrame(rows).round(5).to_string(index=False))
print(f"\\n테스트 AUROC: 제약 없음 {auroc(yt, free.predict_proba(Xt)[:, 1]):.4f} | 단조 제약 {auroc(yt, mono.predict_proba(Xt)[:, 1]):.4f}")
'''
        ),
        md("단조 제약이 있으면 **부분의존 곡선이 한 방향으로만** 움직입니다(위반폭 0). 제약이 없는 모형은 표본의 우연한 굴곡을 학습해 업무 논리에 어긋나는 구간이 생길 수 있습니다. 제약의 성능 비용은 위 AUROC 로 확인합니다. *이 합성데이터의 실제 관계는 단조라서* 비용이 작지만, 실제 데이터에서는 다를 수 있습니다."),
        md("## 6. 설명 — 순열 중요도와 부분의존"),
        code(
            '''
from creditlab.challengers import permutation_importance

imp = permutation_importance(mono, Xt, yt, n_repeats=8, seed=SEED)
print(f"테스트 기준 AUROC = {imp.attrs['baseline_metric']:.4f}  (값을 섞으면 이만큼 떨어지는 변수가 중요)")
print(imp.round(4).to_string(index=False))

pdp = partial_dependence(mono, Xt, "card_utilization", n_points=8)
print("\\n카드 사용률 부분의존:")
print(pdp.round(4).to_string(index=False))
'''
        ),
        md(
            """
- 중요도가 **0 근처이거나 음수**인 변수는 이 모형·이 데이터에서 예측에 거의 기여하지 않습니다.
- 순열 중요도·부분의존은 *모형이 어떻게 예측하는가*의 설명이며 **현실의 인과 효과가 아닙니다.** 상관된 변수끼리는 중요도가 나뉘고, 부분의존은 비현실적인 조합(사용률은 낮은데 연체는 많은 사람)도 평균에 섞입니다.
- 불이익 사유 통지 같은 규제 설명 용도로는 승인된 방법론 문서가 필요합니다.
"""
        ),
        code(
            '''
# [SHAP] 이 환경에는 shap 패키지가 없다 — 있으면 가산성(기준값 + 기여도 합 = 모형 출력)을 확인한다.
try:
    import shap
    ex = shap.TreeExplainer(overfit)
    print("shap 사용 가능:", shap.__version__)
except ImportError:
    print("shap 미설치: SHAP 코드는 이 환경에서 실행하지 못했습니다. (Colab 에서는 `pip install shap` 후 실행)")
'''
        ),
        md("## 7. 챔피언 vs 챌린저 — 차이가 우연인가?"),
        code(
            '''
p_gbm = mono.predict_proba(Xt)[:, 1]
cmp_tab = compare_models(yt, {"스코어카드": pd_champ, "부스팅(단조)": p_gbm})
print(cmp_tab[["auroc", "ks", "brier", "log_loss", "ece", "cal_slope", "mean_pd"]].round(4).to_string())

d = paired_bootstrap_diff(auroc, yt, p_gbm, pd_champ, n_boot=500, seed=SEED)
print(f"\\nAUROC 차이(부스팅 − 스코어카드) = {d['diff']:+.4f}, 95% 쌍체 부트스트랩 구간 [{d['lo']:+.4f}, {d['hi']:+.4f}], 0 포함: {d['includes_zero']}")
'''
        ),
        md(
            """
차이의 **구간이 0 을 포함하면** 같은 표본에서 본 우위가 우연일 수 있습니다(공식 검정은 단계 14 의 DeLong 등). 구간이 0 을 포함하지 않아도 교체의 근거는 **성능 하나가 아닙니다.**

| | 스코어카드 | 부스팅 |
|---|---|---|
| 설명 | 점수표로 직접 | 사후 설명(중요도·SHAP) |
| 안정성 | 구간 점수라 안정적 | 데이터 변화에 민감할 수 있음 |
| 규제·검증 | 확립된 관행 | 별도 방법론 문서·검증 필요 |
"""
        ),
        md("## 8. ✏️ 실습 — 분할 이득으로 분할 고르기"),
        code(
            '''
# [exercise]
parent = (40, 160)               # (부도, 정상)
split_a = ((30, 70), (10, 90))
split_b = ((5, 95), (35, 65))

gain_a = None      # ← 분할 A 의 이득 (지니 불순도 2p(1−p), 자식은 건수 가중 평균)
gain_b = None      # ← 분할 B 의 이득
choice = None      # ← 이득이 더 큰 분할 "A" 또는 "B"

if None not in (gain_a, gain_b, choice):
    ta, tb = split_gain(*split_a), split_gain(*split_b)
    print(f"A {gain_a:.4f} (정답 {ta:.4f}) :", "맞음" if abs(gain_a - ta) < 1e-3 else "다시 계산")
    print(f"B {gain_b:.4f} (정답 {tb:.4f}) :", "맞음" if abs(gain_b - tb) < 1e-3 else "다시 계산")
    print("선택", choice, ":", "맞음" if choice == ("A" if ta > tb else "B") else "다시 확인")
else:
    print("gain_a, gain_b, choice 를 채우고 다시 실행하세요.")
'''
        ),
        md(
            """
<details>
<summary>정답 예시 (먼저 직접 해 본 뒤 펼치세요)</summary>

```python
def g(b, n_good):
    p = b / (b + n_good)
    return 2 * p * (1 - p)

def gain(parent, left, right):
    nl, nr = sum(left), sum(right)
    return g(*parent) - (nl * g(*left) + nr * g(*right)) / (nl + nr)

gain_a = gain(parent, *split_a)
gain_b = gain(parent, *split_b)
choice = "A" if gain_a > gain_b else "B"
```
</details>
"""
        ),
        md("## 9. 퀴즈"),
        code(
            '''
from creditlab.learning import Question, check, grade, show

QUIZ = [
    Question("Q1", "분할 이득", "분할 이득이란?",
             {"A": "자식 노드의 불순도 합", "B": "부모 노드 불순도 − 자식 노드 불순도의 (건수 가중) 평균", "C": "분할한 변수의 IV", "D": "자식 노드의 건수 차이"},
             "B", "이득이 클수록 분할이 두 집단을 더 잘 구분합니다.", ("분할 전후 불순도를 비교해 보세요.",)),
    Question("Q2", "과적합", "깊이 제한이 없는 트리의 학습 AUROC 0.99, 테스트 AUROC 0.62 입니다. 의미는?",
             {"A": "모형이 매우 우수하다", "B": "학습 데이터를 암기해 과적합했다. 성능은 테스트 값으로 말한다",
              "C": "테스트 데이터가 틀렸다", "D": "AUROC 를 평균내서 0.8 로 보고한다"},
             "B", "학습 성능은 성능이 아닙니다.", ("두 값의 격차를 보세요.",)),
    Question("Q3", "조기 종료", "부스팅 반복 횟수를 정할 때 검증 구간은?",
             {"A": "무작위로 뽑은 20%", "B": "학습 구간보다 시간상 뒤쪽 구간", "C": "테스트 구간", "D": "학습 구간 전체"},
             "B", "시간 순서를 지켜야 미래 정보가 섞이지 않습니다. 테스트를 쓰면 오염입니다.", ("모형은 과거로 배우고 미래에 쓰입니다.",)),
    Question("Q4", "설명의 한계", "순열 중요도가 높은 변수를 '이 변수를 바꾸면 부도가 줄어든다'고 말해도 될까요?",
             {"A": "된다. 중요도는 인과 효과다", "B": "안 된다. 모형의 예측 의존도일 뿐이며 상관된 변수가 있으면 나뉘어 보일 수 있다",
              "C": "중요도가 0.5 이상일 때만 된다", "D": "SHAP 이면 된다"},
             "B", "설명 방법은 모형이 예측하는 방식을 설명하며 현실의 원인을 증명하지 않습니다.", ("인과와 예측은 다릅니다.",)),
    Question("Q5", "챌린저 평가", "부스팅이 스코어카드보다 AUROC 가 0.01 높습니다. 쌍체 부트스트랩 구간이 [−0.01, +0.03] 입니다. 올바른 결론은?",
             {"A": "부스팅이 확실히 우수하다", "B": "차이가 우연일 수 있다. 설명·안정성·규제 요건과 함께 판단한다", "C": "스코어카드가 틀렸다", "D": "표본을 줄이면 해결된다"},
             "B", "구간이 0 을 포함하면 우위를 확신할 수 없습니다.", ("구간에 0 이 들어 있나요?",)),
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
1. 부스팅의 `max_depth` 를 2·3·5 로 바꿔 **시간순 검증 곡선의 최저점(반복 횟수)** 이 어떻게 달라지는지 표로 만드세요. 테스트는 보지 마세요.
2. 단조 제약 변수를 `num_delinq_12m` 한 개로 줄였을 때와 비교해 AUROC 와 부분의존이 어떻게 변하는지 쓰세요.
3. 순열 중요도를 **개발 구간**에서 계산해 테스트에서 계산한 값과 비교하고, 어느 쪽을 보고서에 써야 하는지 이유를 쓰세요.

**체크포인트**: 퀴즈 4/5 이상 · 분할 이득 실습 통과 · 단조 제약 위반폭 0 을 확인

**다음 레슨**: 레슨 5-2 — 하이퍼파라미터 튜닝의 선택 편향, 앙상블·스태킹, 확률 보정(Platt/Isotonic), 컷오프 정책.

**참고자료**
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (Springer, 2009).
- Friedman, J., "Greedy function approximation: a gradient boosting machine", *Annals of Statistics* (2001).
- Lundberg & Lee, "A unified approach to interpreting model predictions", NeurIPS (2017).
- 이론: `docs/theory/stage_11.html`
"""
        ),
    ]
