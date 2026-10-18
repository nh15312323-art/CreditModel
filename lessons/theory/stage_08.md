# 단계 8. Train/Validation/Test, 교차검증과 누수 방지

> 모형의 성적은 "처음 보는 데이터"에서만 의미가 있다. 데이터를 어떻게 나누고, 전처리를 어디에서 하며, 테스트를 몇 번 보느냐가 결과의 신뢰성을 정한다.

**연결 실습**: Phase 4 노트북(예정) · 이 페이지의 코드로 *누수 있는/없는* 파이프라인을 직접 비교해 보자.

## 학습 목표

- 학습/검증/테스트 집합의 역할을 구분한다.
- 무작위 분할, **시간 기준 분할**, 그룹 분할, 층화 분할을 상황에 맞게 고른다.
- 교차검증과 nested CV 가 해결하는 문제를 설명한다.
- 전처리·비닝·변수 선택·스케일링을 **파이프라인 안**에서 `fit` 해야 하는 이유를 보인다.
- 데이터 누수의 전형적 패턴을 열거하고 점검표를 만든다.

## 1. 왜 생겼는가

모형을 만든 데이터로 성능을 재면 항상 좋아 보인다. 모형은 *외운* 것일 수도 있기 때문이다(과적합). 그래서 **보지 않은 데이터**로 평가하는 분할이 필요해졌다. 그런데 분할만 하면 끝이 아니다. 전처리·변수 선택·하이퍼파라미터 조정이 *평가 데이터의 정보를 몰래 쓰면* 분할의 의미가 사라진다. 이것이 누수(leakage)다. 교차검증은 데이터가 적을 때 분할의 불안정성을 줄이려는 방법이다.

## 2. 왜 필요한가

- 운영에서 기대할 성능을 *추정*하려면 평가가 운영과 닮아야 한다.
- 규제·검증 환경에서는 **독립 테스트(표본 외, 시점 외)** 결과를 요구한다.
- 하이퍼파라미터를 조정하는 순간, 같은 평가 데이터를 다시 쓰면 *조정에 쓰인 데이터*가 된다.

## 3. 무엇을 하는가

| 집합 | 용도 | 몇 번 보는가 |
|---|---|---|
| 학습(train) | 모형 파라미터·전처리·비닝 학습 | 계속 |
| 검증(validation) | 모형 선택, 하이퍼파라미터, 조기 종료 | 여러 번 (그래서 점점 *오염*됨) |
| 테스트(test) | 최종 성능 보고 | **원칙적으로 한 번** |

## 4. 어떻게 작동하는가

### 4-1. 분할 방식

| 방식 | 언제 | 주의 |
|---|---|---|
| 무작위 | 시간 의존이 없고 행이 독립 | 신용 데이터는 시간·고객 의존이 있어 *낙관적* |
| **층화**(stratified) | 부도가 드물 때 | 분할마다 부도율이 비슷해짐 |
| **시간 기준**(out-of-time) | 미래에 적용할 모형 | 가장 현실적. 경기·정책 변화 포착 |
| **그룹**(고객·계좌·가구) | 같은 고객이 여러 행 | 같은 고객이 train 과 test 에 동시에 있으면 누수 |

**핵심 원칙**: 미래에 배포할 모형의 성능은 **시간 기준 분할**로 추정한다. 무작위 분할 성능만으로 실제 미래 성능을 보장할 수 없다.

**이벤트 수 점검**: 10,000건, 부도율 3% → 부도 300건. 5-fold CV 에서 검증 fold 당 부도 60건. 부도율 1% 면 총 100건, fold 당 20건이라 지표가 크게 흔들린다. **분할 전에 분할별 이벤트 수를 계산**한다.

### 4-2. 교차검증(CV)

데이터를 $K$ 조각으로 나눠 *돌아가며* 한 조각을 검증, 나머지를 학습으로 쓰고 $K$ 개 성능의 평균과 *산포*를 본다. 장점: 적은 데이터로도 안정적 추정. 주의: 시간·그룹 구조를 무시하면 누수(→ `StratifiedKFold`, `GroupKFold`, `TimeSeriesSplit`).

**Nested CV**: 하이퍼파라미터 선택(안쪽 CV)과 성능 추정(바깥쪽 CV)을 분리한다. 안쪽 CV 점수를 최종 성능이라고 보고하면, *선택에 쓴* 데이터로 평가한 낙관적 값이다(Cawley & Talbot, 2010).

### 4-3. 전처리는 학습 데이터에서만 `fit`

스케일링·결측 대체·비닝·변수 선택은 모두 *데이터에서 무언가를 배우는* 단계다. 이 단계들이 검증/테스트 데이터를 보면 안 된다. scikit-learn `Pipeline` 은 **교차검증의 각 fold 마다 학습 구간에서만 `fit`** 하게 해 준다.

**고전적 실험(ESL §7.10.2).** 타깃과 *아무 관계 없는* 무작위 변수 1,000개로 만든 데이터 100건이 있다. 전체 데이터로 "타깃과 가장 관련 있어 보이는" 변수 20개를 *먼저* 고른 뒤 교차검증하면 성능이 좋아 보인다. 그러나 선택을 파이프라인 안에서 하면 성능은 우연 수준으로 돌아온다.

```python
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 1000))          # 전부 무작위 잡음 변수
y = rng.integers(0, 2, 100)               # 타깃도 무작위: 진짜 예측력은 0(AUROC≈0.5)
cv = StratifiedKFold(5, shuffle=True, random_state=0)

X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)      # ✗ 전체 데이터(타깃 포함)로 변수 선택
leaky = cross_val_score(LogisticRegression(max_iter=1000), X_sel, y, cv=cv, scoring="roc_auc").mean()

pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
correct = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()   # ✓ 각 fold 의 학습 구간에서만 선택
print(round(leaky, 3), round(correct, 3))
assert leaky > 0.8 and correct < 0.7 and leaky - correct > 0.2
```

의미 없는 데이터에서 AUROC ≈ 0.94 가 나온다. 이것이 누수의 위력이다. 현실의 누수는 더 미묘해서(비닝·IV 선택·결측 대체) 같은 방식으로 *과대평가*가 슬며시 들어온다.

### 4-4. 시간 기준 분할 구현

```python
import pandas as pd
from creditlab.data import make_synthetic_credit_data

df = make_synthetic_credit_data(n=6000, seed=3).sort_values("application_date")
cut_dev, cut_val = df["application_date"].quantile([0.6, 0.8])
train = df[df["application_date"] <= cut_dev]                       # 과거 60%
valid = df[(df["application_date"] > cut_dev) & (df["application_date"] <= cut_val)]   # 다음 20%
test = df[df["application_date"] > cut_val]                          # 가장 최근 20% (마지막에 한 번)
print(len(train), len(valid), len(test))
assert train["application_date"].max() <= valid["application_date"].min()
assert valid["application_date"].max() <= test["application_date"].min()
assert len(train) + len(valid) + len(test) == len(df)
assert set(train["application_id"]).isdisjoint(test["application_id"])
```

### 4-5. 누수 점검표

| 패턴 | 예 | 방지 |
|---|---|---|
| 전체 데이터 통계 사용 | 평균 대체·스케일링·IV 선택을 분할 전에 | Pipeline, 분할 *후* fit |
| 타깃 기반 변환 | 비닝·타깃 인코딩을 전체로 | 학습 구간에서만 |
| 중복/그룹 | 같은 고객이 양쪽에 | 그룹 분할, 중복 제거 |
| 시간 | 미래 데이터로 학습해 과거 예측 | 시간 기준 분할 |
| 변수 자체 | 결과 이후 정보 | 단계 2 점검표 |
| 테스트 반복 사용 | 테스트 성능을 보고 모형 수정 | 검증 데이터로만 조정, 테스트는 1회 |
| 불균형 처리 순서 | 오버샘플링을 분할 *전에* 하면 복제본이 양쪽에 | 분할 *후* 학습 구간에만 |

## 5. 가정

- 테스트 데이터가 **운영에서 만날 데이터**를 대표한다.
- 분할된 집합들이 서로 **독립**이다(고객·시간 누수 없음).
- 검증 성능이 *선택 때문에 낙관적*일 수 있음을 인정하고, 최종 테스트로 확인한다.
- 이벤트 수가 지표 추정에 충분하다.

## 6. 한계와 오용

!!! warning "자주 하는 오용"
    - 테스트 성능이 마음에 안 들어 *모형을 고치고 다시 테스트*한다 → 테스트가 검증이 되어 낙관적.
    - 무작위 분할 성능을 운영 성능처럼 보고한다.
    - CV 점수의 *평균만* 보고하고 산포(fold 간 차이)를 보지 않는다.
    - 시계열 구조가 있는데 `KFold(shuffle=True)` 를 쓴다.
    - 시간 분할인데 *가장 최근 기간*이 다른 경기 국면이라 성능이 다르게 나온 것을 모형 문제로만 해석한다(환경 변화일 수 있음).
    - 이벤트가 적을 때 검증 fold 를 너무 잘게 나눈다.

## 7. 대안 및 비교

| 방법 | 장점 | 한계 |
|---|---|---|
| 단순 holdout | 간단, 빠름 | 데이터가 적으면 불안정 |
| K-fold CV | 안정적 | 계산량, 시간·그룹 구조 주의 |
| 시계열 분할(rolling/expanding) | 시간 변화 평가 | 초기 구간 학습 데이터가 적음 |
| 부트스트랩 | 신뢰구간 | 의존 구조 주의 |
| 외부 검증 데이터 | 가장 신뢰 | 확보 어려움 |

## 8. 실무 적용

- **규제 로지스틱 CSS**: 비닝·WOE·변수 선택까지 *학습 구간에서만* 수행하고, 같은 경계를 검증·테스트(**표본 외, 시점 외**)에 적용한다. 개발 문서에 분할 기간과 건수·이벤트 수를 표로 기재한다.
- **AI 모형**: 하이퍼파라미터 튜닝은 검증(또는 시간 기준 CV)에서만, 최종 성능은 *한 번의* 테스트에서 보고한다. 두 모형은 **동일한 분할**을 쓴다.
- 분할 규칙·시드·결과를 설정 파일로 저장해 검증팀이 *같은 분할을 재현*하게 한다.
- 모니터링 단계(단계 15)의 "개발 vs 현재" 비교도 같은 정의를 쓴다.

## 9. 지식 확인

**Q1.** 결측을 전체 평균으로 대체한 뒤 train/test 로 나눴다. 문제는?

<details markdown="1"><summary>정답 보기</summary>

test 데이터의 값이 평균에 섞여 학습 단계에 새어 들어간다. 분할 *후* train 평균으로 대체하고 test 에는 적용만 한다.

</details>

**Q2.** 한 고객이 여러 계좌를 가졌다. 계좌 단위로 무작위 분할하면?

<details markdown="1"><summary>정답 보기</summary>

같은 고객의 계좌가 train 과 test 에 모두 들어가 *고객 특성을 외운* 모형이 높은 성능을 보인다(누수). **고객 단위 그룹 분할**을 쓴다.

</details>

**Q3.** 5-fold CV 에서 fold 별 AUROC 가 0.62~0.78 로 크게 흔들린다. 무엇을 의심하는가?

<details markdown="1"><summary>정답 보기</summary>

fold 당 이벤트 수가 적어 지표가 불안정하거나, fold 간 분포 차이(시간·그룹)가 있다. 이벤트 수를 확인하고 층화/그룹/시간 구조를 점검한다.

</details>

**Q4.** 테스트에서 AUROC 가 0.01 낮아 하이퍼파라미터를 바꿨다. 올바른 행동은?

<details markdown="1"><summary>정답 보기</summary>

테스트는 이미 *보았으므로* 조정에 쓰면 안 된다. 조정은 검증 데이터로 하고, 테스트는 새로운(더 최근) 데이터가 생기면 다시 평가한다. 불가능하면 한계로 기록한다.

</details>

**실습 과제**: 합성 데이터에서 (a) 전체 데이터로 IV 상위 3개 변수를 선택한 뒤 CV, (b) 학습 구간에서만 선택하는 파이프라인 CV 를 비교하고 차이를 보고하라. 시간 기준 분할 3구간의 건수와 부도 건수를 표로 쓰라.

## 10. 참고자료

- Hastie, T., Tibshirani, R., Friedman, J. *The Elements of Statistical Learning* — 7장 모형 평가·선택, §7.10.2 (잘못된 교차검증 사례).
- Cawley, G. & Talbot, N. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *JMLR*, 11.
- Kaufman, S. et al. (2012). Leakage in data mining. *ACM TKDD*, 6(4).
- scikit-learn 모델 평가·교차검증 가이드: <https://scikit-learn.org/stable/modules/cross_validation.html>
- 확인일 2026-10-18
