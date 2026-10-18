# 단계 14. 가설검정·p-value·DeLong·다중검정

> "AUROC 가 0.003 올랐다"는 것만으로는 개선이 아니다. 그 차이가 *표본이 달라져도 반복될 변동*보다 큰지를 따져야 한다. 검정은 그 질문에 답하는 도구이고, p-value 는 흔히 오해된다.

**연결 실습**: Phase 6 노트북(예정). 이 페이지의 코드만으로 DeLong 검정을 직접 구현한다.

## 학습 목표

- 귀무가설·유의수준·p-value 의 정의와 *p-value 가 아닌 것*을 말한다.
- 순열검정으로 p-value 를 직접 계산하고 t-검정과 비교한다.
- 같은 데이터로 평가한 두 모형의 AUROC 를 DeLong 검정으로 비교한다.
- 다중검정 문제와 Bonferroni·BH(FDR) 보정을 계산한다.
- 통계적 유의성과 *실무적 유의성*을 구분한다.

## 1. 왜 생겼는가

표본은 우연히 흔들린다. 그 흔들림 안의 차이를 "개선"이라 부르면 우연을 성과로 착각한다. Fisher 와 Neyman–Pearson 이 가설검정 틀을 만들었고, 신용 분야에서는 *같은 고객으로 평가한 두 모형의 AUROC 비교*가 상관되어 있다는 점을 반영한 DeLong 검정(DeLong et al., 1988)이 표준 도구가 되었다.

## 2. 왜 필요한가

- 모형 교체는 비용이 크다. 우연한 개선에 근거해 교체하면 안 된다.
- 변수·모형·하이퍼파라미터를 많이 시도할수록 *우연히 좋아 보이는 것*이 반드시 나온다.
- 결과를 신뢰구간·효과 크기와 함께 보고해야 의사결정이 가능하다.

## 3. 무엇을 하는가

| 질문 | 도구 |
|---|---|
| 두 집단 평균이 다른가? | t-검정, 순열검정 |
| 두 모형 AUROC 가 다른가(같은 표본)? | DeLong |
| 여러 개를 동시에 검정했을 때 | Bonferroni, Holm, BH |
| 차이가 얼마나 큰가? | 신뢰구간, 효과 크기 |

## 4. 어떻게 작동하는가

### 4-1. 가설검정의 구조

1. 귀무가설 $H_0$: "차이가 없다"를 가정.
2. 검정통계량 계산.
3. **p-value** = $H_0$ 가 참이라고 *가정할 때* 관측값만큼 또는 더 극단적인 결과가 나올 확률.
4. $p<\alpha$(관행 0.05)이면 $H_0$ 를 기각.

**p-value 는 "귀무가설이 참일 확률"이 아니다.** 효과의 크기도, 효과가 *중요한지*도 알려 주지 않는다. 표본이 매우 크면 사소한 차이도 유의해진다. p>0.05 는 "차이가 없다"의 증명이 아니라 "차이를 *못 찾았다*"이다.

### 4-2. 순열검정으로 p-value 직접 만들기

$H_0$ 가 참이면 집단 라벨은 교환 가능하다. 라벨을 무작위로 섞어 통계량의 분포를 만들고, 관측값이 그 분포에서 얼마나 극단인지 본다.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
a = rng.normal(0.0, 1.0, 60)
b = rng.normal(0.5, 1.0, 60)            # 진짜 평균 차이 0.5

obs = b.mean() - a.mean()
pooled = np.concatenate([a, b])
perm = np.empty(5000)
for i in range(5000):
    rng.shuffle(pooled)
    perm[i] = pooled[60:].mean() - pooled[:60].mean()
p_perm = (np.abs(perm) >= abs(obs)).mean()
p_t = stats.ttest_ind(b, a, equal_var=False).pvalue
print(round(obs, 3), "순열 p =", round(p_perm, 4), "Welch t p =", round(p_t, 4))
assert abs(p_perm - p_t) < 0.05
```

### 4-3. DeLong 검정 (직접 구현)

AUROC 는 U-통계량이다. 부도 $i$ 에 대한 *구조 성분* $V_{10,i}=\frac1n\sum_j\psi(s_i,s_j)$ ($j$ 는 정상), 정상 $j$ 에 대한 $V_{01,j}=\frac1m\sum_i\psi(s_i,s_j)$ 로 분산과 공분산을 추정한다. 여기서 $\psi(x,y)=1$ (x>y), $0.5$ (x=y), $0$ (x<y). 두 모형이 **같은 고객**을 평가하므로 공분산을 반영한다.

$$ Z=\frac{\widehat{AUC}_1-\widehat{AUC}_2}{\sqrt{\operatorname{Var}_1+\operatorname{Var}_2-2\operatorname{Cov}}} $$

```python
import numpy as np
from scipy import stats

def psi_matrix(pos, neg):
    d = pos[:, None] - neg[None, :]
    return (d > 0) + 0.5 * (d == 0)

def delong_components(y, scores):
    pos, neg = scores[y == 1], scores[y == 0]
    k = psi_matrix(pos, neg)
    return k.mean(axis=1), k.mean(axis=0), k.mean()     # V10 (부도별), V01 (정상별), AUC

def delong_test(y, s1, s2):
    m, n = int((y == 1).sum()), int((y == 0).sum())
    v10_1, v01_1, a1 = delong_components(y, s1)
    v10_2, v01_2, a2 = delong_components(y, s2)
    S10 = np.cov(np.vstack([v10_1, v10_2]))
    S01 = np.cov(np.vstack([v01_1, v01_2]))
    S = S10 / m + S01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(var)
    return a1, a2, z, 2 * stats.norm.sf(abs(z))

# 단계 9 의 8건 예제: AUROC 0.8 확인 (구조 성분 평균이 AUC)
s = np.array([0.9, 0.7, 0.4, 0.8, 0.5, 0.3, 0.2, 0.1]); y8 = np.array([1, 1, 1, 0, 0, 0, 0, 0])
v10, v01, auc = delong_components(y8, s)
print(np.round(v10, 3), np.round(v01, 3), auc)
assert abs(auc - 0.8) < 1e-12 and abs(v10.mean() - 0.8) < 1e-12 and abs(v01.mean() - 0.8) < 1e-12
assert np.allclose(v10, [1.0, 0.8, 0.6])
```

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from creditlab.data import make_synthetic_credit_data

df = make_synthetic_credit_data(n=12000, seed=61).sort_values("application_date").reset_index(drop=True)
df["housing_type"] = df["housing_type"].astype("category").cat.codes
df["region"] = df["region"].astype("category").cat.codes
feats = ["age", "income_10k", "loan_amount_10k", "employment_years", "card_utilization",
         "num_delinq_12m", "inquiries_6m", "housing_type", "region"]
tr, te = df.iloc[:8000], df.iloc[8000:]
Xtr, ytr, Xte, yte = tr[feats].fillna(-1), tr["default"], te[feats].fillna(-1), te["default"].to_numpy()

s_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
s_gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=50,
                                      random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
a1, a2, z, p = delong_test(yte, s_gb, s_lr)
print("AUROC gb", round(a1, 4), "lr", round(a2, 4), "z", round(z, 2), "p", round(p, 4))
assert abs(a1 - roc_auc_score(yte, s_gb)) < 1e-12 and abs(a2 - roc_auc_score(yte, s_lr)) < 1e-12

# 모형 순서를 바꾸면 z 의 부호만 뒤집히고 p-value 는 같다 (대칭성)
b1, b2, z_rev, p_rev = delong_test(yte, s_lr, s_gb)
assert abs(z + z_rev) < 1e-9 and abs(p - p_rev) < 1e-12

# 같은 점수를 단조 변환(순위 불변)하면 AUROC 도, 검정 결과도 그대로다
_, _, z_mono, p_mono = delong_test(yte, s_gb, np.log(s_lr / (1 - s_lr)))
assert abs(z - z_mono) < 1e-9
```

주의: DeLong 은 *같은 표본*으로 두 모형을 평가한 경우에 쓰며, 이벤트 수가 적으면 근사가 불안정하다. 고객이 여러 번 나오면(군집) 독립 가정이 깨져 군집 부트스트랩이 필요하다. 결과는 **p-value 와 함께 AUROC 차이의 신뢰구간**으로 보고한다.

### 4-4. 다중검정

20개 변수를 각각 5% 수준으로 검정하면, 모두 귀무가설이 참이어도 최소 하나가 유의할 확률이

$$ 1-(1-0.05)^{20}\approx 0.6415 $$

이다. 해결책:

- **Bonferroni**: $\alpha/m$ 로 기각(보수적, FWER 통제).
- **Holm**: Bonferroni 의 단계적 개선.
- **Benjamini–Hochberg(BH)**: 거짓 발견 비율(FDR)을 통제. p 를 오름차순 정렬해 $p_{(k)}\le \frac{k}{m}q$ 를 만족하는 *가장 큰 k* 까지 기각.

```python
import numpy as np

assert abs((1 - 0.95**20) - 0.6415) < 1e-4

def bh_reject(pvals, q=0.05):
    p = np.asarray(pvals); m = len(p)
    order = np.argsort(p)
    passed = p[order] <= (np.arange(1, m + 1) / m) * q
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    out = np.zeros(m, bool); out[order[:k]] = True
    return out

pv = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
bonf = np.array(pv) < 0.05 / len(pv)
bh = bh_reject(pv, 0.05)
print("개별 5%:", int((np.array(pv) < 0.05).sum()), "Bonferroni:", int(bonf.sum()), "BH:", int(bh.sum()))
assert (np.array(pv) < 0.05).sum() == 5 and bonf.sum() == 1 and bh.sum() == 2
```

```python
import numpy as np

rng = np.random.default_rng(7)
false_pos = []
for _ in range(2000):
    p = rng.uniform(size=20)                      # 20개 모두 귀무가설이 참
    false_pos.append((p < 0.05).any())
print("한 번이라도 '유의'가 나올 비율", round(float(np.mean(false_pos)), 3))
assert abs(np.mean(false_pos) - 0.6415) < 0.04
```

변수 선택, 구간 분할, 모형 비교, 하이퍼파라미터 탐색 모두 *암묵적 다중검정*이다. 시도한 횟수를 기록하고, 최종 판단은 **봉인된 테스트**에서 한다(단계 12).

### 4-5. 통계적 유의성 vs 실무적 유의성

AUROC 차이 0.002 가 표본 100만 건에서는 '유의'할 수 있지만 승인율·손실에 미치는 영향은 무시할 수준일 수 있다. 반대로 소표본에서 0.03 차이가 '유의하지 않음'일 수 있다. 어느 쪽이든 **효과 크기와 신뢰구간, 사업 영향(예: 같은 승인율에서 부도율 변화)** 으로 판단한다.

## 5. 가정

- 검정에 쓰는 표본이 독립(또는 군집 구조를 반영)이다.
- 검정할 가설을 **데이터를 보기 전에** 정했다(사후에 고르면 p-value 가 무의미).
- DeLong: 표본이 충분하고 두 모형이 같은 표본으로 평가되었다.

## 6. 한계와 오용

!!! warning "자주 하는 오용"
    - p-value 를 "가설이 맞을 확률"로 읽는다.
    - p=0.049 와 0.051 에 질적 차이를 둔다. 유의수준은 관행이다.
    - 여러 모형을 시도해 가장 좋은 것만 검정하고 보정하지 않는다(p-hacking).
    - 독립 표본용 t-검정을 *같은 고객으로 평가한 두 AUROC*에 적용한다.
    - 고객당 여러 행인데 행을 독립으로 취급한다.
    - 유의하지 않다고 "동등하다"고 결론짓는다(동등성 검정은 별도).

## 7. 대안 및 비교

| 도구 | 용도 | 비고 |
|---|---|---|
| 순열검정 | 분포 가정 최소 | 계산량 |
| 부트스트랩 구간 | 지표 변동 추정 | 군집이면 군집 부트스트랩 |
| DeLong | AUROC 비교(상관 반영) | 소표본 주의 |
| McNemar | 같은 임계값의 오분류 비교 | 임계값 정책 의존 |
| 베이지안 비교 | 확률적 진술 | 사전분포 필요 |

## 8. 실무 적용

- 챌린저 모형 승인 시: 테스트 AUROC 차이와 **DeLong p-value, 95% 신뢰구간, 같은 승인율에서의 부도율/손실 차이**를 함께 제출한다.
- 변수 후보를 많이 시험했다면 시도 횟수와 보정(BH 등) 방법을 모형 문서에 쓴다.
- 임계값(0.05)·FDR 수준은 *사전에* 정하고 문서화한다.

## 9. 지식 확인

**Q1.** p=0.03 의 올바른 해석은?

<details markdown="1"><summary>정답 보기</summary>

"차이가 없다는 가정 하에서 이만큼 극단적인 결과가 나올 확률이 3%". 귀무가설이 참일 확률이 3% 라는 뜻이 아니다.

</details>

**Q2.** 50개 변수를 개별 5% 수준으로 검정해 3개가 유의했다. 안전한가?

<details markdown="1"><summary>정답 보기</summary>

모두 귀무가설이 참이어도 기대 2.5개가 유의하다. 3개는 우연과 구별되지 않는다. BH/Bonferroni 같은 보정이 필요하다.

</details>

**Q3.** 같은 테스트 고객으로 평가한 두 모형의 AUROC 를 독립 t-검정으로 비교하면?

<details markdown="1"><summary>정답 보기</summary>

두 AUROC 가 같은 고객에 기반해 *양의 상관*이므로 공분산을 무시하면 분산이 과대평가되어 검정력이 떨어진다. DeLong 같은 상관 반영 방법을 쓴다.

</details>

**Q4.** 표본 100만 건에서 AUROC 차이 0.001, p<0.001. 모형을 교체해야 하나?

<details markdown="1"><summary>정답 보기</summary>

통계적으로 유의해도 실무적으로 미미할 수 있다. 같은 승인율에서의 손실 변화, 설명가능성·운영 비용을 종합해 판단한다.

</details>

**Q5 (계산).** p 값 {0.001, 0.02, 0.03, 0.2} 에 BH(q=0.05) 를 적용하면 몇 개 기각?

<details markdown="1"><summary>정답 보기</summary>

기준 $k/4\times0.05$: 0.0125, 0.025, 0.0375, 0.05. 정렬 p: 0.001≤0.0125 ✓, 0.02≤0.025 ✓, 0.03≤0.0375 ✓, 0.2≤0.05 ✗ → 가장 큰 k=3, **3개 기각**.

</details>

**실습 과제**: 합성 데이터에서 로지스틱과 부스팅을 시간 분할 테스트로 평가하고, DeLong 검정과 부트스트랩 신뢰구간으로 AUROC 차이를 보고하라. 변수 20개 각각의 부도 연관성을 검정하고 개별 5%, Bonferroni, BH 결과를 비교하라.

## 10. 참고자료

- DeLong, E., DeLong, D., Clarke-Pearson, D. (1988). Comparing the areas under two or more correlated ROC curves. *Biometrics*, 44(3).
- Sun, X. & Xu, W. (2014). Fast implementation of DeLong's algorithm. *IEEE Signal Processing Letters*, 21(11).
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS B*, 57(1).
- Wasserstein, R. & Lazar, N. (2016). The ASA statement on p-values. *The American Statistician*, 70(2).
- Efron, B. & Tibshirani, R. *An Introduction to the Bootstrap* (1993).
- 확인일 2026-10-18 (서지 정보는 학습 시점에 원문으로 재확인할 것)
