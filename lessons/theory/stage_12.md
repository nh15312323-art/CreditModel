# 단계 12. 튜닝·앙상블·보정

> 튜닝은 "검증 데이터에 맞춰 설정을 고르는 일"이므로, 그 데이터는 이미 *오염*되었다. 최종 성능은 **한 번도 쓰지 않은** 데이터로만 말한다. 그리고 순위가 좋은 모형도 확률이 틀리면 PD 가 아니다.

**연결 실습**: 노트북 `11_tuning_calibration.ipynb` · 코드 `creditlab.calibration`(`PlattCalibrator`, `IsotonicCalibrator`, `StackedModel`, `profit_by_cutoff`), `creditlab.challengers.tune_gbm_timecv`.

## 학습 목표

- 학습·검증·테스트 세트의 역할을 구분하고 **테스트 오염**을 설명한다.
- 시간 순서를 지키는 교차검증(Time-series split)과 그리드/랜덤 탐색을 구현한다.
- 앙상블(평균·스태킹)의 이점과 위험을 말한다.
- Platt scaling·Isotonic 보정을 적용하고 보정 전후를 비교한다.
- 임계값(컷오프)이 *모형이 아니라 정책*임을 설명한다.

## 1. 왜 생겼는가

모형 설정(하이퍼파라미터)을 바꿀 때마다 성능을 확인하면, 우리는 *확인에 쓴 데이터에 유리한 설정*을 고르게 된다. 이것이 선택 편향이다. 이를 막기 위해 데이터를 역할별로 나누고(학습/검증/테스트), 교차검증으로 안정적 추정을 얻는다. 부스팅·신경망은 확률을 *과신*하거나 *과소신*하는 경향이 있어 사후 보정 기법(Platt 2000, Zadrozny & Elkan 2002)이 쓰인다.

## 2. 왜 필요한가

- 튜닝 없이는 부스팅이 과적합하고, 튜닝만 하면 검증 데이터에 과적합한다.
- PD 는 *확률 값 자체*가 자본·충당금·가격에 쓰인다 → 보정이 필수다.
- 컷오프는 승인율·손실률의 *경영 의사결정*이다.

## 3. 무엇을 하는가

| 단계 | 사용 데이터 | 목적 |
|---|---|---|
| 학습 | train | 모수 추정 |
| 튜닝/조기종료/보정 | validation(또는 train 내 CV) | 설정 선택 |
| 최종 평가 | **test (1회만)** | 성능 보고 |

## 4. 어떻게 작동하는가

### 4-1. 시간 순서 교차검증

시계열 신용 데이터에서 미래로 과거를 예측해서는 안 된다. `TimeSeriesSplit` 은 *항상 과거로 학습, 직후 구간으로 검증*한다.

```python
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from creditlab.data import make_synthetic_credit_data

df = make_synthetic_credit_data(n=9000, seed=41).sort_values("application_date").reset_index(drop=True)
df["housing_type"] = df["housing_type"].astype("category").cat.codes
df["region"] = df["region"].astype("category").cat.codes
feats = ["age", "income_10k", "loan_amount_10k", "employment_years", "card_utilization",
         "num_delinq_12m", "inquiries_6m", "housing_type", "region"]
dev, test = df.iloc[:7000], df.iloc[7000:]          # 마지막 구간은 최종 테스트로 봉인
Xdev, ydev = dev[feats].fillna(-1), dev["default"]
Xte, yte = test[feats].fillna(-1), test["default"]

tscv = TimeSeriesSplit(n_splits=4)
for tr_idx, va_idx in tscv.split(Xdev):
    assert tr_idx.max() < va_idx.min()          # 검증은 항상 학습보다 미래
print([(len(a), len(b)) for a, b in tscv.split(Xdev)])
```

### 4-2. 탐색과 선택 편향

```python
import itertools
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

grid = list(itertools.product([2, 3, 5], [0.03, 0.1], [100, 300]))
rows = []
for depth, lr, n_iter in grid:
    scores = []
    for tr_idx, va_idx in tscv.split(Xdev):
        m = HistGradientBoostingClassifier(max_depth=depth, learning_rate=lr, max_iter=n_iter,
                                           min_samples_leaf=50, random_state=0).fit(Xdev.iloc[tr_idx], ydev.iloc[tr_idx])
        scores.append(roc_auc_score(ydev.iloc[va_idx], m.predict_proba(Xdev.iloc[va_idx])[:, 1]))
    rows.append(((depth, lr, n_iter), float(np.mean(scores))))
rows.sort(key=lambda r: -r[1])
best_params, best_cv = rows[0]
print(best_params, round(best_cv, 4))

final = HistGradientBoostingClassifier(max_depth=best_params[0], learning_rate=best_params[1], max_iter=best_params[2],
                                       min_samples_leaf=50, random_state=0).fit(Xdev, ydev)
test_auc = roc_auc_score(yte, final.predict_proba(Xte)[:, 1])
print("CV 최고:", round(best_cv, 4), "테스트(1회):", round(test_auc, 4))
assert 0.5 < test_auc < 1
```

CV 최고 점수는 "여러 설정 중 최대값"이므로 **낙관적**이다. 테스트 점수가 보통 더 낮거나 비슷하다. 테스트를 본 뒤 설정을 다시 바꾸면 그 테스트는 더 이상 테스트가 아니다.

### 4-3. 앙상블

- **평균**: 서로 다른 모형의 예측을 평균. 오류가 *서로 덜 상관*될수록 이득이 크다.
- **스태킹**: 기본 모형 예측을 입력으로 메타 모형(대개 로지스틱)을 학습. *메타 모형은 기본 모형이 보지 않은 데이터*(OOF 예측)로 학습해야 누수를 막는다.

앙상블은 변별력을 소폭 올릴 수 있으나 설명·문서화·유지 비용이 커진다.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

lr_model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(Xdev, ydev)
p_lr = lr_model.predict_proba(Xte)[:, 1]
p_gb = final.predict_proba(Xte)[:, 1]
from scipy.stats import rankdata
rank_avg = (rankdata(p_lr) + rankdata(p_gb)) / 2
print({"lr": round(roc_auc_score(yte, p_lr), 4), "gb": round(roc_auc_score(yte, p_gb), 4),
       "avg": round(roc_auc_score(yte, rank_avg), 4)})
assert roc_auc_score(yte, rank_avg) > min(roc_auc_score(yte, p_lr), roc_auc_score(yte, p_gb)) - 0.02
```

### 4-4. 보정: Platt 과 Isotonic

- **Platt scaling**: 모형 점수 $s$ 에 대해 $p=\sigma(a s+b)$ 를 적합. 단순·안정, 왜곡이 *시그모이드형*일 때 적합.
- **Isotonic**: 단조 증가 계단함수. 유연하지만 *표본이 적으면 과적합*.
- 보정은 **보정 전용 데이터**(학습에 쓰지 않은 구간)에서 적합한다.

부도율이 전체적으로 이동했을 때(prior shift)의 *단순 절편 보정*도 있다: 로그 오즈에 상수를 더해 평균 예측을 목표 부도율에 맞춘다.

```python
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

# 보정 전용 구간: 개발 데이터의 마지막 1500건을 따로 떼고, 모형은 그 앞에서만 학습
fit_part, cal_part = dev.iloc[:5500], dev.iloc[5500:]
base = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.1, max_iter=400, min_samples_leaf=20, random_state=0)
base.fit(fit_part[feats].fillna(-1), fit_part["default"])           # 과적합하기 쉬운 설정(과신 유도)
s_cal = base.predict_proba(cal_part[feats].fillna(-1))[:, 1]
s_te = base.predict_proba(Xte)[:, 1]

logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
platt = LogisticRegression(C=np.inf, max_iter=1000).fit(logit(s_cal).reshape(-1, 1), cal_part["default"])
p_platt = platt.predict_proba(logit(s_te).reshape(-1, 1))[:, 1]
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s_cal, cal_part["default"])
p_iso = iso.predict(s_te)

for name, p in [("보정 전", s_te), ("Platt", p_platt), ("Isotonic", p_iso)]:
    print(name, "Brier", round(brier_score_loss(yte, p), 5), "AUROC", round(roc_auc_score(yte, p), 4))
assert brier_score_loss(yte, p_platt) <= brier_score_loss(yte, s_te) + 1e-9    # 과신 모형은 보정 후 Brier 가 낮아진다
assert abs(roc_auc_score(yte, p_platt) - roc_auc_score(yte, s_te)) < 1e-9      # 단조 변환: 순위 불변
```

Platt 은 단조 변환이므로 **AUROC 가 변하지 않는다**. 보정은 확률의 *값*만 고친다. Isotonic 은 동점(계단)을 만들어 AUROC 가 약간 변할 수 있다.

### 4-5. 임계값은 정책이다

승인 컷오프 $t$ 에서 기대 손익은 대략

$$ \text{기대이익}(t)=\sum_{i:\,p_i<t}\big[(1-p_i)\,m-p_i\,L\big] $$

($m$ = 정상 고객의 건당 마진, $L$ = 부도 시 손실). *마진과 손실은 사업이 정한다.* 같은 모형도 마진·손실 가정이 바뀌면 최적 컷오프가 바뀐다.

```python
import numpy as np

m_gain, loss = 1.0, 8.0            # 가정: 정상 1단위 이익, 부도 8단위 손실
p = p_platt
order = np.sort(p)
profits = []
cands = np.quantile(p, np.linspace(0.3, 1.0, 36))
for t in cands:
    approve = p < t
    profits.append(((1 - p[approve]) * m_gain - p[approve] * loss).sum())
best_t = cands[int(np.argmax(profits))]
print("최적 컷오프(확률)", round(float(best_t), 4), "승인율", round(float((p < best_t).mean()), 3))
# 기대이익이 양수인 고객(1-p)m - pL > 0  ⇔  p < m/(m+L) 만 승인하는 것이 이론적 최적
theory = m_gain / (m_gain + loss)
print("이론 컷오프 m/(m+L) =", round(theory, 4))
assert abs(theory - 1 / 9) < 1e-12
```

보정된 확률에서 이론 최적 컷오프는 $p<m/(m+L)$ 이다. 이 공식은 **확률이 잘 보정되어 있을 때만** 성립하므로 보정이 정책의 전제다.

## 5. 가정

- 검증·테스트 구간이 운영 데이터를 대표한다.
- 보정 데이터와 운영 데이터의 부도율 관계가 같다.
- 마진·손실 가정이 고정이다(실제로는 경기·상품에 따라 바뀐다).

## 6. 한계와 오용

!!! warning "자주 하는 오용"
    - 테스트 세트를 보고 설정을 바꾼 뒤 같은 테스트로 다시 보고한다.
    - 보정 데이터를 학습에도 쓴다.
    - 시간 순서 없는 CV 로 튜닝한다.
    - 앙상블 스태킹에서 기본 모형의 *학습 예측*으로 메타 모형을 학습한다(누수).
    - 보정 후 AUROC 가 *오른다*고 기대한다(Platt 은 불변).
    - 부도가 적은 구간에 Isotonic 을 적용한다(과적합).
    - 컷오프를 모형 성능의 일부로 최적화한다.

## 7. 대안 및 비교

| 방법 | 장점 | 약점 |
|---|---|---|
| 그리드 탐색 | 단순·재현 | 차원 폭발 |
| 랜덤 탐색 | 효율적(Bergstra & Bengio, 2012) | 재현엔 시드 필요 |
| 베이지안 최적화 | 호출 수 절감 | 복잡 |
| Platt | 안정 | 모양 가정 |
| Isotonic | 유연 | 표본 많이 필요 |
| 등급별 재보정 | 규제 관행(등급 PD) | 계단형 |

## 8. 실무 적용

- 개발 시 *train / validation(튜닝·보정) / OOT(최종 시점 외 테스트)* 로 구분하고, 최종 테스트는 모형 확정 *후* 한 번만 사용한다.
- 로지스틱 CSS 는 등급별 PD 를 *보정 구간 부도율*로 재산출하고, AI 모형은 확률 보정 후 같은 등급 구조로 비교한다.
- 컷오프·승인율은 리스크·영업 정책 승인 절차와 함께 문서화한다.

## 9. 지식 확인

**Q1.** 튜닝에 쓴 검증 세트의 AUROC 를 최종 성능으로 보고해도 되는가?

<details markdown="1"><summary>정답 보기</summary>

안 된다. 여러 설정 중 *최대*를 골랐으므로 낙관적이다. 봉인된 테스트로 한 번만 평가한다.

</details>

**Q2.** Platt scaling 후 AUROC 는 오르는가?

<details markdown="1"><summary>정답 보기</summary>

아니다. 점수의 *단조 변환*이라 순위가 같아 AUROC 는 불변이다. Brier/log loss/보정곡선이 개선된다.

</details>

**Q3.** $m=1$, $L=9$ 일 때 이론 컷오프는?

<details markdown="1"><summary>정답 보기</summary>

$m/(m+L)=1/10=0.1$. PD 0.1 미만만 승인. 단, 확률이 잘 보정되어 있어야 한다.

</details>

**Q4.** 스태킹 메타 모형을 기본 모형의 *학습 데이터 예측*으로 학습하면?

<details markdown="1"><summary>정답 보기</summary>

기본 모형이 이미 본 데이터이므로 예측이 과도하게 좋아 메타 모형이 과신한다(누수). OOF(폴드 밖) 예측을 쓴다.

</details>

**실습 과제**: 합성 데이터에서 시간 순서 CV 로 부스팅을 튜닝하고, 보정 전용 구간으로 Platt 과 Isotonic 을 비교하라. 보정곡선과 Brier 를 표로 제시하고, 마진·손실 가정 2가지에서 최적 컷오프가 어떻게 달라지는지 보고하라.

## 10. 참고자료

- Platt, J. (2000). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods.
- Zadrozny, B. & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *KDD*.
- Niculescu-Mizil, A. & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML*.
- Guo, C. et al. (2017). On calibration of modern neural networks. *ICML*.
- Bergstra, J. & Bengio, Y. (2012). Random search for hyper-parameter optimization. *JMLR*, 13.
- Elkan, C. (2001). The foundations of cost-sensitive learning. *IJCAI*.
- scikit-learn 보정: <https://scikit-learn.org/stable/modules/calibration.html>
- 확인일 2026-10-18
