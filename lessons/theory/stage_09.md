# 단계 9. 평가 지표와 검증

> 모형이 "좋다"는 말은 지표마다 다른 뜻이다. 순위를 잘 맞히는가(변별력), 확률 값이 맞는가(보정), 특정 컷오프에서 어떤 오류가 나는가(분류 성능)를 *따로* 본다.

**연결 실습**: 노트북 `08_evaluation_metrics.ipynb` · 코드 `creditlab.evaluation`(`auroc`, `gini`, `ks_statistic`, `confusion_at`, `calibration_table`, `bootstrap_ci`) · 이 페이지의 8건 예제를 손으로 먼저 풀어 보자.

## 학습 목표

- 혼동행렬과 정확도·정밀도·재현율·특이도·F1 을 계산하고 **임계값에 의존**함을 설명한다.
- ROC·AUROC, **Gini = 2×AUROC − 1**, KS, PR-AUC 의 정의와 해석을 말한다.
- log loss, Brier score, calibration curve 로 *확률의 질*을 평가한다.
- AR(Accuracy Ratio)의 정의를 **명시적으로 고정**해야 하는 이유를 설명한다.
- 순위 변별력과 확률 정확성의 차이를 구분한다.
- 부트스트랩 신뢰구간과 기간별 성능 변화를 본다.

## 1. 왜 생겼는가

초기에는 정확도(맞힌 비율)로 모형을 평가했다. 그러나 부도가 1% 인 데이터에서 "모두 정상"이라고 예측하면 정확도 99% 다. 불균형 데이터에서는 정확도가 쓸모 없고, **임계값과 무관하게** 순위를 평가하는 ROC/AUROC 와 KS, 확률을 직접 평가하는 log loss·Brier·보정이 필요해졌다. 신용 분야에서는 CAP 곡선과 Accuracy Ratio(AR)도 쓰인다.

## 2. 왜 필요한가

- 모형 선택·비교·승인의 근거가 지표다.
- 지표마다 **약점**이 있어 하나로 결론내리면 틀린다.
- 변별력이 좋아도 확률이 틀리면 PD 로 쓸 수 없고(보정), 확률이 맞아도 순위가 약하면 심사에 도움이 안 된다.

## 3. 무엇을 하는가

| 질문 | 지표 |
|---|---|
| 누가 더 위험한지 순위를 잘 매기나? | AUROC, Gini, AR, KS |
| 드문 부도 중에서 얼마나 잘 찾나? | PR-AUC(평균 정밀도), 재현율@k |
| 예측 확률이 실제 부도율과 맞나? | calibration curve, Brier, log loss |
| 특정 컷오프에서 오류는? | 혼동행렬, 정밀도·재현율 |
| 성능이 시간에 안정적인가? | 기간별 AUROC/KS, 신뢰구간 |

## 4. 어떻게 작동하는가

### 4-1. 8건 손계산 예제

부도(1) 3건의 위험점수: 0.9, 0.7, 0.4 / 정상(0) 5건: 0.8, 0.5, 0.3, 0.2, 0.1. 점수가 높을수록 위험하다.

#### 혼동행렬 (임계값 0.45 이상이면 "부도 예측")

|  | 예측 부도 | 예측 정상 |
|---|---|---|
| 실제 부도 | TP=2 (0.9, 0.7) | FN=1 (0.4) |
| 실제 정상 | FP=2 (0.8, 0.5) | TN=3 (0.3, 0.2, 0.1) |

- 정확도 $=(2+3)/8=0.625$
- 정밀도 $=TP/(TP+FP)=2/4=0.5$
- 재현율(민감도) $=TP/(TP+FN)=2/3=0.667$
- 특이도 $=TN/(TN+FP)=3/5=0.6$
- F1 $=2TP/(2TP+FP+FN)=4/7=0.571$

**임계값을 바꾸면 모든 숫자가 바뀐다.** 임계값은 *사업 정책*(단계 12)이므로 혼동행렬 지표는 정책과 함께 해석한다.

#### AUROC

> AUROC = 무작위로 뽑은 (부도, 정상) 한 쌍에서 부도의 점수가 더 높을 확률(동점은 0.5).

부도 3 × 정상 5 = 15 쌍. 부도 0.9 는 5쌍 모두 이김(5), 0.7 은 0.8 에게 지고 나머지 4쌍을 이김(4), 0.4 는 0.3, 0.2, 0.1 을 이김(3) → $12/15=0.8$.

#### Gini와 AR

$$ \mathrm{Gini}=2\times\mathrm{AUROC}-1=0.6 $$

완벽한 모형은 1, 무작위는 0 이다. **AR(Accuracy Ratio)** 는 CAP 곡선으로 정의된다.

$$ \mathrm{AR}=\frac{\text{CAP 곡선과 무작위선 사이 면적}}{\text{완벽 모형 CAP 곡선과 무작위선 사이 면적}} $$

표준 CAP 정의에서는 $\mathrm{AR}=2\times\mathrm{AUROC}-1$ 이 수학적으로 성립한다(Engelmann, Hayden & Tasche, 2003). 그러나 **조직·문헌별로 AR 계산이 다를 수 있다**(표본 구성, 동점 처리, 점수 대신 *등급 단위*로 계산하는 경우의 계단형 곡선, 부호 방향 등). 따라서 이 프로그램은 AR 을 자동으로 Gini 로 환산하지 않고, **프로젝트 설정에 산식·방향·분모를 명시**하도록 한다.

#### KS 통계량

부도 누적분포와 정상 누적분포의 **최대 차이**: $\mathrm{KS}=\max_s|F_{bad}(s)-F_{good}(s)|$.

| 임계값(이상) | 부도 중 비율(TPR) | 정상 중 비율(FPR) | 차이 |
|---|---|---|---|
| 0.9 | 0.333 | 0.0 | 0.333 |
| 0.8 | 0.333 | 0.2 | 0.133 |
| 0.7 | 0.667 | 0.2 | 0.467 |
| 0.5 | 0.667 | 0.4 | 0.267 |
| **0.4** | **1.0** | **0.4** | **0.6** |
| 0.3 | 1.0 | 0.6 | 0.4 |
| 0.2 | 1.0 | 0.8 | 0.2 |

KS = 0.6 (임계값 0.4 에서). KS 는 *한 점*의 차이만 보고, 표본 구성·부도율에 영향받을 수 있다.

#### Brier score와 log loss

점수를 확률로 보면, 

$$ \mathrm{Brier}=\frac1n\sum(p_i-y_i)^2,\qquad \mathrm{LogLoss}=-\frac1n\sum\big[y_i\ln p_i+(1-y_i)\ln(1-p_i)\big] $$

예제: Brier $=1.49/8=0.18625$, log loss $\approx0.5458$. 둘 다 *작을수록 좋다*. log loss 는 틀린 확신(예: 정상에 0.99 부여)을 크게 벌한다.

```python
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve

s = np.array([0.9, 0.7, 0.4, 0.8, 0.5, 0.3, 0.2, 0.1])
y = np.array([1, 1, 1, 0, 0, 0, 0, 0])

auc = roc_auc_score(y, s)
fpr, tpr, _ = roc_curve(y, s)
ks = (tpr - fpr).max()
print(round(auc, 3), round(2 * auc - 1, 3), round(ks, 3), round(brier_score_loss(y, s), 5), round(log_loss(y, s), 4))
assert abs(auc - 0.8) < 1e-12 and abs(2 * auc - 1 - 0.6) < 1e-12 and abs(ks - 0.6) < 1e-12
assert abs(brier_score_loss(y, s) - 0.18625) < 1e-12 and abs(log_loss(y, s) - 0.5458) < 1e-4

# 쌍 비교로 직접 계산한 AUROC (정의 확인)
bad, good = s[y == 1], s[y == 0]
pairs = [(b > g) + 0.5 * (b == g) for b in bad for g in good]
assert abs(np.mean(pairs) - auc) < 1e-12

# 임계값 0.45 혼동행렬
pred = (s >= 0.45).astype(int)
tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
assert (tp, fp, fn, tn) == (2, 2, 1, 3)
assert abs(2 * tp / (2 * tp + fp + fn) - 4 / 7) < 1e-12
```

### 4-2. 불균형과 정확도, PR-AUC

부도율 1% 에서 "모두 정상" 예측은 정확도 99%, 재현율 0%. **PR 곡선**은 정밀도–재현율 관계를 보여 주며 *무작위 모형의 정밀도가 부도율*(기저율)과 같다. 부도가 드물수록 ROC 는 낙관적으로 보일 수 있어, 상위 위험군 선별이 중요한 문제에서는 PR-AUC(평균 정밀도)를 함께 본다(Saito & Rehmsmeier, 2015). 단, PR-AUC 는 부도율에 따라 값 자체가 달라지므로 *다른 부도율의 데이터끼리* 직접 비교하지 않는다.

### 4-3. 변별력과 보정은 다르다

- **변별력(discrimination)**: 위험 순서를 맞히는 능력. AUROC/KS/Gini.
- **보정(calibration)**: "예측 5%"인 고객들의 실제 부도율이 5% 인가. Brier, log loss, 보정곡선.

모든 예측확률에 2를 곱해도 순서는 같아 AUROC 는 변하지 않지만 확률은 틀어진다. PD 모형은 *둘 다* 필요하다.

**보정곡선(reliability diagram)**: 예측확률을 구간으로 나누어 각 구간의 *평균 예측*과 *실제 부도율*을 비교한다. 대각선이면 잘 보정된 것이다. 구간 요약 지표로 ECE(기대 보정 오차) 등이 쓰이지만 구간 정의에 민감하다.

```python
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from creditlab.data import make_synthetic_credit_data

df = make_synthetic_credit_data(n=6000, seed=11).sort_values("application_date")
feats = ["age", "card_utilization", "num_delinq_12m", "employment_years"]
train, test = df.iloc[:4000], df.iloc[4000:]
model = LogisticRegression(C=np.inf, max_iter=2000).fit(train[feats], train["default"])
p = model.predict_proba(test[feats])[:, 1]
frac_pos, mean_pred = calibration_curve(test["default"], p, n_bins=5, strategy="quantile")
print(np.round(mean_pred, 3), np.round(frac_pos, 3))
assert len(frac_pos) == len(mean_pred) <= 5
assert abs(p.mean() - test["default"].mean()) < 0.03   # 전체 평균 수준은 크게 어긋나지 않음
```

### 4-4. 신뢰구간과 기간별 변화

지표는 *표본에서 계산된 추정치*다. 같은 모형도 표본이 달라지면 AUROC 가 달라진다. **부트스트랩**(복원추출 반복)으로 구간을 만들고, **시점별**로 성능이 어떻게 변하는지 함께 본다. 이벤트가 적으면 구간이 넓다.

```python
import numpy as np
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
y_t, p_t = test["default"].to_numpy(), p
auc_hat = roc_auc_score(y_t, p_t)
boot = []
for _ in range(300):
    idx = rng.integers(0, len(y_t), len(y_t))
    if y_t[idx].min() == y_t[idx].max():        # 한 클래스만 있으면 AUROC 정의 불가 → 건너뜀
        continue
    boot.append(roc_auc_score(y_t[idx], p_t[idx]))
lo, hi = np.percentile(boot, [2.5, 97.5])
print(round(auc_hat, 3), (round(lo, 3), round(hi, 3)))
assert lo < auc_hat < hi
```

### 4-5. 정의되지 않는 경우

테스트 구간에 부도가 0건이거나 정상이 0건이면 AUROC 는 *정의되지 않는다*. 숫자를 억지로 만들지 말고 "계산 불가(한 클래스만 존재)" 로 표시한다.

## 5. 가정

- 평가 데이터가 **운영 데이터를 대표**한다(단계 8).
- 레이블(부도)이 **정의대로 정확히** 붙어 있다.
- AUROC/KS 는 표본의 순위 정보만 쓰므로 *보정 오류를 못 본다*.
- 신뢰구간(부트스트랩)은 관측이 독립이라고 가정한다(고객 군집이면 군집 부트스트랩).

## 6. 한계와 오용

!!! warning "자주 하는 오용"
    - 정확도·정밀도 등 혼동행렬 지표를 *임계값을 말하지 않고* 비교한다.
    - AUROC 하나로 "좋은 모형"을 선언한다. 보정이 틀리거나 상위 위험군에서 약할 수 있다(Hand, 2009: AUROC 의 비용 가정 문제).
    - 부도율이 다른 두 표본의 KS·PR-AUC 를 직접 비교한다.
    - AR 을 정의 확인 없이 Gini 와 같다고 보고한다.
    - 개발 표본(학습) 지표를 *성능*으로 보고한다.
    - 지표 차이가 0.002 인데 "개선"이라고 한다(신뢰구간과 단계 14 참조).

## 7. 대안 및 비교

| 지표 | 보는 것 | 장점 | 약점 |
|---|---|---|---|
| AUROC / Gini | 전체 순위 | 임계값 불필요, 널리 쓰임 | 보정·상위 구간 무시, 불균형에서 낙관적 |
| KS | 최대 분리 | 직관적, 신용 실무 관행 | 한 점만 봄, 표본 구성 영향 |
| PR-AUC | 드문 부도 탐지 | 불균형에 민감 | 기저율에 따라 값이 달라짐 |
| Brier | 확률 오차 | 변별+보정 통합 | 불균형에서 해석 어려움 |
| log loss | 확률의 질 | 확신 오류 벌점 | 극단값 민감 |
| 보정곡선 | 확률 보정 | 시각적 | 구간 정의에 민감 |
| H-measure 등 | 비용 가정 보완 | 이론적 보완 | 실무 채택 적음 |

## 8. 실무 적용

- **규제 로지스틱 CSS 와 AI 모형 비교**: 같은 테스트 데이터에서 AUROC/Gini/KS/PR-AUC, Brier/log loss, 보정곡선, **등급별 부도율**을 모두 표로 비교한다.
- **PD 모형**은 변별력과 *보정*이 모두 필수다. 변별력이 같다면 보정을 사후에 개선할 수 있다(단계 12).
- 보고서에 **AR 산식과 부호 방향**, KS 계산 표본, 기간별 값과 신뢰구간을 명시한다.
- 개발 시점 성능은 모니터링의 *기준선*이 된다(단계 10, 15).

## 9. 지식 확인

**Q1.** AUROC 0.8 이면 Gini 는? 무작위 모형의 AUROC 와 Gini 는?

<details markdown="1"><summary>정답 보기</summary>

Gini $=2\times0.8-1=0.6$. 무작위는 AUROC 0.5, Gini 0.

</details>

**Q2.** 모든 예측확률에 2를 곱했다. AUROC, Brier 는 어떻게 되는가?

<details markdown="1"><summary>정답 보기</summary>

AUROC 는 **변하지 않고**(순서 동일), Brier 와 log loss 는 일반적으로 **나빠진다**(확률이 틀어짐). 변별력과 보정은 다르다.

</details>

**Q3.** 부도율 0.5% 인 데이터에서 정확도 99.5% 모형을 얻었다. 의미 있는 성과인가?

<details markdown="1"><summary>정답 보기</summary>

"모두 정상" 예측만으로도 99.5% 가 나온다. 정확도는 의미가 없다. 재현율·정밀도·PR-AUC·AUROC 와 보정을 본다.

</details>

**Q4.** 테스트 기간에 부도가 한 건도 없다. AUROC 를 어떻게 보고하는가?

<details markdown="1"><summary>정답 보기</summary>

"계산 불가(정의되지 않음: 한 클래스만 존재)"로 보고한다. 기간을 합치거나 더 긴 관찰을 기다린다.

</details>

**Q5 (계산).** 부도 점수 {0.8, 0.3}, 정상 점수 {0.6, 0.2, 0.1}. AUROC 를 쌍 비교로 계산하라.

<details markdown="1"><summary>정답 보기</summary>

6쌍. 0.8 은 3쌍 모두 이김(3), 0.3 은 0.6 에게 지고 0.2, 0.1 을 이김(2) → $5/6\approx0.833$.

</details>

**실습 과제**: 합성 데이터에서 시간 기준 분할 후 로지스틱 모형을 적합하고, 테스트 구간 AUROC/Gini/KS/Brier/log loss 와 보정곡선을 구하라. 분기별 AUROC 와 부트스트랩 구간을 표로 쓰고, 5등급의 등급별 부도율을 함께 제시하라. AR 의 정의를 문장으로 명시하라.

## 10. 참고자료

- Engelmann, B., Hayden, E., Tasche, D. (2003). Testing rating accuracy. *Risk*, 16(1).
- Hand, D. (2009). Measuring classifier performance: a coherent alternative to the area under the ROC curve. *Machine Learning*, 77.
- Saito, T. & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3).
- Niculescu-Mizil, A. & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML*.
- Basel Committee on Banking Supervision, *Studies on the Validation of Internal Rating Systems*, Working Paper No. 14 (2005).
- scikit-learn 모델 평가: <https://scikit-learn.org/stable/modules/model_evaluation.html>
- 확인일 2026-10-18
