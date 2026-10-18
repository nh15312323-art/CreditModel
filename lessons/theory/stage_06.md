# 단계 6. 로지스틱 회귀의 수학과 해석

> 확률을 직접 직선으로 맞추지 않고 *로그오즈*를 직선으로 맞춘다. 계수 하나가 "오즈를 몇 배로 바꾸는가"로 읽힌다는 점이 신용모형의 해석력의 근간이다.

**연결 실습**: Phase 3 노트북(예정) · 이 페이지의 코드(NumPy IRLS ↔ scikit-learn 비교)를 직접 실행해 보자.

## 학습 목표

- 확률·오즈·로그오즈(logit)를 서로 변환한다.
- 로지스틱 함수, 최대우도추정, 손실함수, 경사하강법/뉴턴 방법의 직관을 설명한다.
- 계수·오즈비·절편·기준범주·스케일링·부호를 해석한다.
- L1/L2 규제, 완전분리, 희소 이벤트 문제를 설명한다.
- 계수의 신뢰구간·p-value 해석과 한계를 말한다.

## 1. 왜 생겼는가

타깃이 0/1 일 때 선형회귀 $p=\beta_0+\beta_1x$ 를 쓰면 예측이 **0 미만이나 1 초과**가 되고, 오차의 분산이 $p(1-p)$ 로 달라져 가정이 깨진다. 확률은 0~1 로 묶여 있지만 *로그오즈*는 $-\infty\sim+\infty$ 의 값을 가진다. 그래서 *로그오즈를 선형으로* 모형화하는 로지스틱 회귀가 널리 쓰이게 되었다. 계수가 *오즈의 곱셈 효과*로 해석되고, 최대우도로 안정적으로 추정되며, 모형이 투명해 규제 환경에 잘 맞는다.

## 2. 왜 필요한가

- **규제 로지스틱 CSS** 의 중심 모형이다. 점수카드는 이 모형의 계수를 점수로 바꾼 것이다(단계 7).
- 계수의 부호와 크기가 *업무 논리 검토*의 대상이다.
- AI 모형과의 비교에서 **기준선(baseline)** 역할을 한다.

## 3. 무엇을 하는가

입력 $x_1,\dots,x_k$ → 부도확률 $p$.

$$ \ln\frac{p}{1-p}=\beta_0+\beta_1x_1+\cdots+\beta_kx_k\quad\Longleftrightarrow\quad p=\frac{1}{1+e^{-(\beta_0+\sum\beta_jx_j)}} $$

## 4. 어떻게 작동하는가

### 4-1. 확률 ↔ 오즈 ↔ 로그오즈

오즈 $=p/(1-p)$, 로그오즈 $=\ln(\text{오즈})$.

| $p$ | 오즈 | 로그오즈 |
|---|---|---|
| 0.01 | 0.0101 | -4.5951 |
| 0.05 | 0.0526 | -2.9444 |
| 0.10 | 0.1111 | -2.1972 |
| 0.20 | 0.25 | -1.3863 |
| 0.50 | 1 | 0 |
| 0.90 | 9 | 2.1972 |

$p=0.5$ 에서 로그오즈는 0 이고, $p$ 와 $1-p$ 는 로그오즈의 부호만 바꾼다(대칭).

### 4-2. 계수의 해석: 오즈비

$x_j$ 가 1 단위 늘 때 로그오즈가 $\beta_j$ 만큼 변하므로 **오즈는 $e^{\beta_j}$ 배**가 된다(다른 변수는 고정).

$\beta=0.7$ → $e^{0.7}=2.0138$. 즉 오즈가 약 2배. 그러나 **확률이 2배가 되는 것은 아니다.**

| 시작 확률 | 시작 오즈 | 2.0138배 후 오즈 | 새 확률 |
|---|---|---|---|
| 5% | 0.0526 | 0.1060 | 9.6% |
| 50% | 1.0 | 2.0138 | 66.8% |

같은 계수라도 시작 확률이 다르면 확률의 변화량이 다르다. 그래서 *계수는 오즈비로*, 확률 변화는 *구체적 시점에서* 설명한다.

```python
import numpy as np

p = np.array([0.01, 0.05, 0.10, 0.20, 0.50, 0.90])
odds = p / (1 - p)
logit = np.log(odds)
assert np.allclose(np.round(logit, 4), [-4.5951, -2.9444, -2.1972, -1.3863, 0.0, 2.1972])

orr = np.exp(0.7)
def after(p0, ratio):
    o = p0 / (1 - p0) * ratio
    return o / (1 + o)
print(round(orr, 4), round(after(0.05, orr), 4), round(after(0.5, orr), 4))
assert abs(orr - 2.0138) < 1e-4
assert abs(after(0.05, orr) - 0.0958) < 1e-4 and abs(after(0.5, orr) - 0.6682) < 1e-4
```

### 4-3. 절편·기준범주·스케일링·부호

- **절편** $\beta_0$: 모든 변수가 0 일 때의 로그오즈. 변수를 *평균으로 중심화*하면 "평균 고객"의 로그오즈가 된다.
- **범주형 변수**는 더미(0/1)로 바꾸고 **기준범주**를 하나 정한다. 계수는 *기준범주 대비* 효과다. 기준을 바꾸면 계수가 바뀐다(해석이 달라질 뿐 모형은 같다).
- **스케일링**: 나이(수십 단위)와 소득(수만 단위)은 계수 크기를 직접 비교할 수 없다. 표준화하거나 *단위를 명시*한다.
- **부호**: 업무 논리와 반대면 공선성·표본·정의를 의심한다. WOE 입력(본 프로그램 규칙: 위험할수록 +)이면 계수가 **양수·1 근처**여야 자연스럽다.

### 4-4. 최대우도추정(MLE)과 손실함수

각 고객이 부도 $y_i\in\{0,1\}$, 모형 확률이 $p_i$ 일 때 우도는 $\prod p_i^{y_i}(1-p_i)^{1-y_i}$. 로그를 취한 **로그우도**를 최대화한다.

$$ \ell(\beta)=\sum_{i}\big[y_i\ln p_i+(1-y_i)\ln(1-p_i)\big],\qquad \frac{\partial\ell}{\partial\beta}=X^\top(y-p) $$

음의 평균 로그우도가 곧 **로그 손실(log loss)** 이다. 기울기는 *(실제−예측)* 를 변수에 곱해 더한 형태라 직관적이다. 예측이 틀릴수록 큰 수정이 일어난다.

- **경사하강법**: 기울기 방향으로 조금씩 이동.
- **뉴턴/IRLS**(반복 재가중 최소제곱): 2차 정보(헤시안 $X^\top WX$, $W=\mathrm{diag}\,p_i(1-p_i)$)를 써 빠르게 수렴. statsmodels 가 이 방식이다.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from creditlab.data import make_synthetic_credit_data

df = make_synthetic_credit_data(n=4000, seed=42)
X = df[["age", "card_utilization", "num_delinq_12m"]].to_numpy(float)
y = df["default"].to_numpy()
Xs = (X - X.mean(0)) / X.std(0)                       # 표준화: 계수 크기 비교 가능

def fit_irls(X, y, n_iter=25):
    Xb = np.c_[np.ones(len(X)), X]
    beta = np.zeros(Xb.shape[1])
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-Xb @ beta))
        grad = Xb.T @ (y - p)
        hess = Xb.T @ (Xb * (p * (1 - p))[:, None])
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta

beta = fit_irls(Xs, y)
# scikit-learn 은 기본이 L2 규제(C=1.0)이므로, 규제 없는 추정과 비교하려면 C=np.inf (1.8 이상; 이전 버전은 penalty=None)
sk = LogisticRegression(C=np.inf, max_iter=2000).fit(Xs, y)
print(np.round(beta, 3), np.round(np.r_[sk.intercept_, sk.coef_[0]], 3))
assert np.abs(beta - np.r_[sk.intercept_, sk.coef_[0]]).max() < 5e-3
assert beta[1] < 0 < beta[2]                           # 나이↑ → 위험↓, 사용률↑ → 위험↑ (생성식과 일치)
```

!!! note "라이브러리 API 는 바뀐다"
    scikit-learn 1.8 에서 `penalty` 인자가 deprecated 되어 `C`, `l1_ratio` 로 규제를 지정한다(`C=np.inf` = 규제 없음, 1.9.1 문서 확인). **설치된 버전의 공식 문서**를 확인하고, 보고서에 버전을 적는다.

### 4-5. 규제: L2, L1

과적합과 불안정을 줄이려 계수 크기에 벌점을 준다.

$$ \min_\beta\;-\ell(\beta)+\lambda\Big(\alpha\|\beta\|_1+\tfrac{1-\alpha}{2}\|\beta\|_2^2\Big) $$

- **L2(릿지)**: 계수를 0 쪽으로 *부드럽게* 줄임. 공선성에 안정적. 계수는 0 이 되지 않음.
- **L1(라쏘)**: 일부 계수를 *정확히 0* 으로 만들어 변수 선택 효과. 상관 변수 중 하나만 고르는 경향.
- scikit-learn 의 `C` 는 규제 세기의 **역수**(작을수록 강함)다. 규제를 쓰면 계수는 *편향*되고, **p-value 해석이 달라진다**.

### 4-6. 완전분리와 희소 이벤트

어떤 변수 값을 경계로 부도/정상이 *완벽히* 갈리면(**완전분리**) 우도는 계수를 키울수록 계속 커져 최대값이 없다. 계수가 매우 크게 나오거나 수렴 경고가 뜬다. 구간 내 부도 0건인 WOE 구간, 소표본 더미에서 흔하다. 대응: 구간 병합, 규제, 변수 제외, Firth 보정 같은 편향 감소 방법.

### 4-7. 신뢰구간과 p-value

Wald 검정: $z=\hat\beta/\mathrm{SE}(\hat\beta)$. 95% 구간은 $\hat\beta\pm1.96\,\mathrm{SE}$ 이고, 오즈비 구간은 양끝에 $\exp$ 를 씌운다. 표준오차는 헤시안의 역행렬에서 얻는다. p-value 는 *계수가 0 인지*에 대한 증거일 뿐 변수의 중요도나 예측력이 아니다(단계 3, 4).

statsmodels 로 추론용 결과를, scikit-learn 으로 예측 파이프라인을 만드는 것이 일반적인 역할 분담이다.

```python
# [requires: statsmodels]
import numpy as np
import statsmodels.api as sm

res = sm.Logit(y, sm.add_constant(Xs)).fit(disp=0)
print(res.summary2().tables[1].round(3))
assert np.allclose(res.params, beta, atol=1e-5)       # 위의 NumPy IRLS 와 같은 해
```

## 5. 가정

- **선형성**: 로그오즈가 설명변수의 선형 결합이다(비닝·WOE·스플라인으로 완화).
- **독립성**: 관측이 독립이다(같은 고객 반복 행이면 위반).
- **공선성이 심하지 않다**: 계수 해석을 위해.
- **충분한 이벤트 수**와 완전분리가 없음.
- 기준시점에 이용 가능한 변수만 사용(단계 1).

## 6. 한계와 오용

!!! warning "자주 하는 오용"
    - 계수를 "확률이 몇 %p 오른다"로 읽는다. → 오즈의 곱셈 효과다.
    - 서로 다른 스케일의 변수의 계수 크기를 비교해 "더 중요한 변수"라고 말한다.
    - 규제 모형의 p-value 를 규제 없는 모형의 p-value 처럼 해석한다.
    - 기준범주를 바꿨더니 계수가 달라졌다고 모형이 불안정하다고 말한다(해석이 달라질 뿐).
    - 수렴 경고·거대 계수를 무시한다.
    - 학습 데이터의 부도율로 *진짜 PD 수준*이 보장된다고 가정한다(표본추출·기간·정책에 따라 보정 필요, 단계 12·15).

## 7. 대안 및 비교

| 모형 | 장점 | 한계 |
|---|---|---|
| 로지스틱 | 투명·안정·계수 해석 | 비선형·상호작용은 사람이 설계 |
| 프로빗 | 정규분포 기반 | 해석이 오즈보다 덜 직관적, 신용에선 로지스틱이 관행 |
| 선형확률모형 | 단순 | 확률 범위 위반 |
| 트리 부스팅 | 비선형·상호작용 자동 | 설명·안정성 근거 추가 필요(단계 11) |

## 8. 실무 적용

- **규제 로지스틱 CSS**: WOE 변수 + 로지스틱 + (필요 시) 규제 → 계수 부호·크기 점검 → 점수 변환. 계수가 음수/비정상이면 *변수 제외·구간 병합*을 검토한다.
- **AI 모형과 공정 비교**: 같은 분할·같은 성능 지표·같은 확률 보정 기준으로 비교한다. 로지스틱은 *최소한 이만큼은 해야 한다*는 기준선이다.
- 결과는 계수표(계수, SE, 오즈비, 신뢰구간), 수렴 여부, 공선성 진단과 함께 문서화한다.

## 9. 지식 확인

**Q1.** $p=0.2$ 의 오즈와 로그오즈는?

<details markdown="1"><summary>정답 보기</summary>

오즈 $0.2/0.8=0.25$, 로그오즈 $\ln0.25\approx-1.3863$.

</details>

**Q2.** $\beta=0.7$ 이면 변수가 1 증가할 때 부도확률이 2배가 되는가?

<details markdown="1"><summary>정답 보기</summary>

아니다. **오즈**가 약 2.01배가 된다. 확률 변화는 시작 확률에 따라 다르다(5%→9.6%, 50%→66.8%).

</details>

**Q3.** 모든 부도가 한 구간에 몰리고 다른 구간은 부도가 0건이다. 계수 추정에서 일어날 일은?

<details markdown="1"><summary>정답 보기</summary>

(준)완전분리로 계수가 매우 커지거나 수렴하지 않는다. 구간 병합, 규제, 최소 이벤트 수 기준을 검토한다.

</details>

**Q4.** scikit-learn 의 기본 `LogisticRegression()` 결과를 statsmodels `Logit` 결과와 비교했더니 계수가 약간 다르다. 가장 먼저 의심할 것은?

<details markdown="1"><summary>정답 보기</summary>

scikit-learn 의 **기본 L2 규제**(C=1.0). 규제 없는 추정과 비교하려면 `C=np.inf`(또는 이전 버전의 `penalty=None`)로 맞춘다.

</details>

**실습 과제**: 합성 데이터에서 3~5개 변수로 로지스틱 모형을 적합하고, 각 계수의 오즈비와 직관적 설명을 *한 문장씩* 작성하라. 변수 하나를 평균으로 중심화했을 때 절편의 해석이 어떻게 달라지는지 쓰라.

## 10. 참고자료

- Hosmer, D., Lemeshow, S., Sturdivant, R. *Applied Logistic Regression*, 3rd ed. (Wiley, 2013).
- Hastie, T., Tibshirani, R., Friedman, J. *The Elements of Statistical Learning* (Springer) — 4장 분류, 규제.
- Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, 80(1).
- scikit-learn `LogisticRegression`: <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>
- statsmodels `Logit`: <https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html>
- 확인일 2026-10-18 (scikit-learn 1.9.1 문서의 `C`/`l1_ratio`/`penalty` 설명 확인)
