# 단계 11. 트리·부스팅·SHAP

> 트리는 "조건 분기"로 데이터를 쪼개고, 부스팅은 그런 작은 트리를 *앞선 모형의 실수를 고치는 방향*으로 계속 더한다. 성능은 올라가지만 설명 가능성과 안정성의 비용을 치른다.

**연결 실습**: 노트북 `10_ml_challengers.ipynb` · 코드 `creditlab.challengers`(`fit_gbm`, `fit_gbm_early_stop`, `permutation_importance`, `partial_dependence`). 이 페이지는 sklearn 만으로 실행된다. XGBoost/CatBoost 는 같은 아이디어의 구현체다.

## 학습 목표

- 불순도(지니)와 분할 이득을 손으로 계산한다.
- 단일 트리의 과적합과 배깅(랜덤포레스트)·부스팅의 차이를 설명한다.
- 그래디언트 부스팅의 "잔차(그래디언트)를 적합하는 반복"을 구현으로 확인한다.
- 학습률·트리 수·깊이의 상호작용과 조기 종료를 이해한다.
- 단조성 제약이 왜 신용 모형에서 중요한지 말한다.
- SHAP 의 의미와 *인과가 아님*을 구분한다.

## 1. 왜 생겼는가

로지스틱 회귀는 변수의 효과가 *가산적·선형적*이라고 가정한다. 현실에는 상호작용(소득이 낮고 *동시에* 사용률이 높을 때만 위험)과 비선형이 있다. 트리는 이를 자동으로 잡는다. 단일 트리는 불안정해서 배깅(Breiman, 2001)과 부스팅(Friedman, 2001)이 발전했고, 이후 XGBoost(2016), CatBoost(2018) 등이 정형 데이터의 표준 도구가 되었다.

## 2. 왜 필요한가

- 비선형·상호작용을 *수작업 변수 설계 없이* 포착한다.
- 결측과 범주형 처리가 상대적으로 쉽다.
- 로지스틱 모형 대비 변별력 개선 여지가 있다. 단, **개선폭은 데이터마다 다르며** 작을 수도 있다.

## 3. 무엇을 하는가

| 모형 | 방식 | 성격 |
|---|---|---|
| 의사결정나무 | 조건 분기로 구역 분할 | 해석 쉬움, 불안정 |
| 랜덤포레스트 | 부트스트랩+변수 무작위 트리를 *독립적으로* 평균 | 분산 감소 |
| 그래디언트 부스팅 | 트리를 *순차적으로* 더해 손실을 줄임 | 편향 감소, 튜닝 민감 |

## 4. 어떻게 작동하는가

### 4-1. 지니 불순도와 분할 이득 (손계산)

노드의 부도율 $p$ 에 대해 지니 $G=2p(1-p)$ (이진 분류). 10건(부도 4, 정상 6)을 변수 $x$ 로 분할한다고 하자.

- 부모: $p=0.4$, $G=2(0.4)(0.6)=0.48$
- 왼쪽 노드 5건(부도 1, 정상 4): $p=0.2$, $G=0.32$
- 오른쪽 노드 5건(부도 3, 정상 2): $p=0.6$, $G=0.48$
- 자식 가중평균: $0.5(0.32)+0.5(0.48)=0.40$
- **분할 이득 = 0.48 − 0.40 = 0.08**

```python
def gini(bad, good):
    n = bad + good
    p = bad / n
    return 2 * p * (1 - p)

parent = gini(4, 6)
left, right = gini(1, 4), gini(3, 2)
child = (5 / 10) * left + (5 / 10) * right
gain = parent - child
print(round(parent, 2), round(left, 2), round(right, 2), round(child, 2), round(gain, 2))
assert abs(parent - 0.48) < 1e-12 and abs(left - 0.32) < 1e-12 and abs(right - 0.48) < 1e-12
assert abs(gain - 0.08) < 1e-12
```

트리는 모든 변수·모든 분할점 후보 중 이득이 가장 큰 곳을 고르는 일을 반복한다(탐욕적). 정지 조건(최대 깊이, 잎 최소 표본)이 없으면 학습 데이터를 완벽 암기한다.

### 4-2. 단일 트리의 과적합

```python
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from creditlab.data import make_synthetic_credit_data

df = make_synthetic_credit_data(n=8000, seed=31).sort_values("application_date")
df["housing_type"] = df["housing_type"].astype("category").cat.codes
df["region"] = df["region"].astype("category").cat.codes
feats = ["age", "income_10k", "loan_amount_10k", "employment_years", "card_utilization",
         "num_delinq_12m", "inquiries_6m", "housing_type", "region"]
tr, te = df.iloc[:5500], df.iloc[5500:]
Xtr, ytr = tr[feats].fillna(-1), tr["default"]
Xte, yte = te[feats].fillna(-1), te["default"]

res = {}
for depth in [3, None]:
    m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=1 if depth is None else 50, random_state=0).fit(Xtr, ytr)
    res[depth] = (roc_auc_score(ytr, m.predict_proba(Xtr)[:, 1]), roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
print({k: tuple(round(x, 3) for x in v) for k, v in res.items()})
assert res[None][0] > 0.99            # 깊이 제한이 없으면 학습 데이터를 거의 암기
assert res[None][1] < res[None][0] - 0.2
assert res[3][1] > res[None][1]       # 얕은 트리가 오히려 테스트에서 낫다
```

학습 AUROC 와 테스트 AUROC 의 격차가 과적합의 신호다. **학습 성능은 성능이 아니다.**

### 4-3. 배깅과 부스팅

- **랜덤포레스트**: 서로 다른 부트스트랩 표본·변수 부분집합으로 *깊은* 트리 여러 개를 독립적으로 만들고 평균 → 분산 감소.
- **그래디언트 부스팅**: 현재 예측의 손실 *그래디언트의 반대 방향*을 얕은 트리로 적합해 학습률 $\eta$ 만큼 더한다.

$$ F_m(x)=F_{m-1}(x)+\eta\,h_m(x),\qquad h_m \approx -\frac{\partial L}{\partial F}\Big|_{F_{m-1}} $$

로그 손실에서 $-\partial L/\partial F = y-p$ (잔차)이다. 아래는 이를 *직접 구현*해 sklearn 과 비슷한 성능이 나오는지 확인한다.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

X, y = Xtr.to_numpy(), ytr.to_numpy().astype(float)
F = np.full(len(y), np.log(y.mean() / (1 - y.mean())))        # 초기값: 로그 오즈
trees, eta = [], 0.1
for _ in range(100):
    resid = y - sigmoid(F)                                      # 음의 그래디언트 = 잔차
    t = DecisionTreeRegressor(max_depth=3, min_samples_leaf=50, random_state=0).fit(X, resid)
    F += eta * t.predict(X)        # (단순화: 잎 값 뉴턴 보정 생략)
    trees.append(t)

def predict(Xn):
    f = np.full(len(Xn), np.log(y.mean() / (1 - y.mean())))
    for t in trees:
        f += eta * t.predict(Xn)
    return sigmoid(f)

mine = roc_auc_score(yte, predict(Xte.to_numpy()))
ref = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_leaf=50, random_state=0).fit(Xtr, ytr)
ref_auc = roc_auc_score(yte, ref.predict_proba(Xte)[:, 1])
print(round(mine, 3), round(ref_auc, 3))
assert mine > 0.6 and abs(mine - ref_auc) < 0.03
```

### 4-4. 핵심 하이퍼파라미터

| 파라미터 | 올리면 | 위험 |
|---|---|---|
| 트리 수 | 적합 증가 | 과적합 |
| 학습률 | 빨리 학습 | 불안정, 과적합 |
| 최대 깊이 | 상호작용 포착 | 과적합 |
| 잎 최소 표본 | 안정 | 과소적합 |
| 서브샘플/열 샘플 | 분산 감소 | 약간의 편향 |
| 정규화(L1/L2) | 과적합 억제 | 과소적합 |

**학습률은 낮추고 트리 수는 늘리는 것**이 보통 안정적이다. 트리 수는 *검증 세트의 손실이 개선되지 않는 시점*(조기 종료)에서 정하되, **조기 종료에 쓴 검증 세트는 최종 평가에 쓰지 않는다**(단계 12).

### 4-5. 단조성 제약

연체 횟수가 늘수록 위험이 *줄어드는* 구간이 모형에 생기면 심사자와 규제 당국은 수용하기 어렵다. 부스팅 라이브러리는 변수별 **단조 제약**(예: 연체횟수↑ → 위험 비감소)을 지원한다. 제약은 약간의 성능을 포기하는 대신 *설명 가능성·정책 일관성*을 얻는다. sklearn 의 `HistGradientBoostingClassifier(monotonic_cst=...)` 로 확인한다.

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

cst = [0] * len(feats)
cst[feats.index("num_delinq_12m")] = 1          # 연체횟수↑ → 부도확률 비감소
hgb = HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.1, monotonic_cst=cst, random_state=0).fit(Xtr, ytr)

grid = Xte.copy().iloc[:300]
probs = []
for v in range(0, 6):
    g = grid.copy()
    g["num_delinq_12m"] = v
    probs.append(hgb.predict_proba(g)[:, 1].mean())
print(np.round(probs, 4))
assert all(b >= a - 1e-12 for a, b in zip(probs, probs[1:]))
```

### 4-6. SHAP

SHAP 은 게임이론의 Shapley 값을 이용해 각 변수가 *개별 예측을 기준값에서 얼마나 밀어 올리거나 내렸는지* 가산적으로 분해한다.

$$ f(x)=\phi_0+\sum_{j}\phi_j(x) $$

트리 모형에는 정확·빠른 알고리즘(TreeSHAP)이 있다. **그러나 SHAP 은 모형이 *어떻게 예측하는지*의 설명이지 *현실에서 무엇이 원인인지*가 아니다.** 상관된 변수가 있으면 기여도가 나뉘고, 배경 데이터 선택에 따라 값이 달라진다. 규제 설명(불이익 사유 통지) 용도로는 승인된 방법론 문서가 필요하다.

```python
# [requires: shap]
import numpy as np
import shap

explainer = shap.TreeExplainer(ref)
sv = explainer.shap_values(Xte.iloc[:200])
base = explainer.expected_value
raw = ref.decision_function(Xte.iloc[:200])
assert np.allclose(base + sv.sum(axis=1), raw, atol=1e-6)    # 가산성: 기준값 + 기여도 합 = 모형 출력(로그오즈)
```

## 5. 가정

- 학습 데이터의 구조가 미래에도 유지된다(단계 10).
- 트리 분할은 *관측된 범위 안*에서만 의미가 있다. 범위 밖에서는 마지막 구간의 값을 그대로 쓴다(외삽 불가).
- 부스팅은 레이블 노이즈에 민감하다. 부도 정의가 틀리면 노이즈를 학습한다.

## 6. 한계와 오용

!!! warning "자주 하는 오용"
    - 학습 성능이 높다고 좋은 모형이라 판단한다.
    - 테스트 세트로 파라미터를 반복 조정한다(테스트 오염).
    - 변수 중요도(gain)를 *인과 효과*로 읽는다. 상관된 변수끼리 중요도가 쪼개진다.
    - SHAP 을 "이 변수를 바꾸면 이만큼 좋아진다"로 해석한다.
    - 시간 순서 없이 무작위 분할로 평가한다(단계 8).
    - 단조성이 필요한 변수에 제약 없이 배포한다.

## 7. 대안 및 비교

| | 규제 로지스틱 CSS | 랜덤포레스트 | 그래디언트 부스팅 |
|---|---|---|---|
| 해석 | 계수·점수표로 직접 | 중요도·부분의존 | SHAP 등 사후 설명 |
| 비선형·상호작용 | 수작업 | 자동 | 자동 |
| 안정성 | 높음 | 중간 | 튜닝 민감 |
| 규제 수용 | 쉬움 | 어려움 | 문서화 필요 |
| 변별력 | 기준선 | 대체로 높음 | 대체로 가장 높음(데이터 의존) |

## 8. 실무 적용

- AI 모형은 **로지스틱 CSS 의 도전자(challenger)** 로 두고 같은 분할·같은 지표로 비교한다. 개선폭이 신뢰구간 안이면 복잡도를 정당화하기 어렵다.
- 단조 제약, 변수 수 제한, 변수 사용 승인 목록으로 설명 가능성을 확보한다.
- 모형 카드에 하이퍼파라미터, 조기 종료 방식, 시드, 라이브러리 버전을 기록한다.

## 9. 지식 확인

**Q1.** 부모 노드 부도 5·정상 5 를 (부도 4·정상 1) / (부도 1·정상 4) 로 나눴다. 지니 이득은?

<details markdown="1"><summary>정답 보기</summary>

부모 $G=2(0.5)(0.5)=0.5$. 자식 각 $G=2(0.8)(0.2)=0.32$, 가중평균 0.32. 이득 $=0.5-0.32=0.18$.

</details>

**Q2.** 랜덤포레스트와 부스팅의 가장 큰 구조 차이는?

<details markdown="1"><summary>정답 보기</summary>

랜덤포레스트는 깊은 트리를 *독립적으로* 만들어 평균(분산 감소), 부스팅은 얕은 트리를 *순차적으로* 앞선 실수를 고치며 더한다(편향 감소).

</details>

**Q3.** 학습 AUROC 0.99, 테스트 0.62. 무엇을 의심하나?

<details markdown="1"><summary>정답 보기</summary>

과적합. 깊이·잎 최소 표본·트리 수·정규화를 조정하고, 데이터 누수가 없는지도 확인한다.

</details>

**Q4.** SHAP 에서 '소득' 기여도가 +0.3 이다. 소득을 올리면 부도확률이 내려가는가?

<details markdown="1"><summary>정답 보기</summary>

아니다. SHAP 은 *모형 예측의 분해*이며 인과 효과가 아니다. 상관 변수와 배경 데이터에 따라 달라진다.

</details>

**실습 과제**: 합성 데이터로 로지스틱과 부스팅의 테스트 AUROC 와 부트스트랩 신뢰구간을 비교하고, 부스팅에 `num_delinq_12m` 단조 제약을 걸었을 때 성능 변화를 보고하라. 개선이 구간 안인지 판단하라.

## 10. 참고자료

- Breiman, L. (2001). Random forests. *Machine Learning*, 45.
- Friedman, J. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5).
- Chen, T. & Guestrin, C. (2016). XGBoost: a scalable tree boosting system. *KDD*.
- Prokhorenkova, L. et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS*.
- Lundberg, S. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Lundberg, S. et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2.
- scikit-learn 앙상블: <https://scikit-learn.org/stable/modules/ensemble.html>
- 확인일 2026-10-18
