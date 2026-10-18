# 단계 10. PSI·안정성·드리프트

> 모형은 "개발 때 본 세상"에서 만들어졌다. 세상이 변하면 입력 분포도, 변수와 부도의 관계도 변한다. PSI 는 *분포가 얼마나 달라졌는가*를 재는 첫 번째 경보기일 뿐, 모형이 틀렸다는 증명이 아니다.

**연결 실습**: 노트북 `09_psi_stability.ipynb` · 코드 `creditlab.evaluation`(`psi_numeric`, `psi_categorical`, `csi_table`, `expected_psi_noise`) · 이 페이지의 PSI 코드와 시뮬레이션을 먼저 실행해 보자.

## 학습 목표

- PSI 를 정의하고 구간별로 손계산한다.
- 기준기간·비교기간·**고정된 bin 경계**의 중요성을 설명한다.
- 0 빈도 처리와 표본 수가 PSI 에 미치는 영향을 안다.
- **데이터 분포 변화(covariate shift)** 와 **관계 변화(concept drift)** 를 구분한다.
- 변수별 PSI, 점수 PSI, 등급 구성비, 부도율 모니터링을 설계한다.
- PSI 임계값이 관행적 경보 기준임을 이해한다.

## 1. 왜 생겼는가

대출 모형은 몇 년씩 쓰인다. 그 사이 고객 구성이 바뀌고(신규 채널, 상품 변화), 경기가 변하고, 데이터 입력 방식이 바뀐다. 성능을 직접 보려면 부도 결과가 나올 때까지 *수개월~1년* 기다려야 한다. 그래서 **결과를 기다리지 않고** 입력·점수 분포의 변화를 먼저 감지하는 지표로 PSI(Population Stability Index)가 쓰이게 되었다.

## 2. 왜 필요한가

- 입력 분포가 크게 변하면 모형이 *본 적 없는 영역*에서 쓰이는 것이므로 성능이 떨어질 수 있다.
- 성능 지표(AUROC, 부도율)는 **지연**된다. PSI 는 *즉시* 계산된다.
- 시스템 오류(항목 누락, 단위 변경, 결측 급증)를 조기에 발견한다.

## 3. 무엇을 하는가

기준(expected, 보통 개발 표본) 분포와 비교(actual, 현재 표본) 분포를 **같은 구간(bin)** 으로 나누어 비교한다.

## 4. 어떻게 작동하는가

### 4-1. 정의

구간 $i$ 의 기준 비율 $e_i$, 비교 비율 $a_i$ 일 때

$$ \mathrm{PSI}=\sum_{i=1}^{k}(a_i-e_i)\,\ln\frac{a_i}{e_i} $$

각 항은 항상 0 이상이고, 분포가 같으면 0 이다. PSI 는 두 방향 KL 발산의 합이다: $\mathrm{PSI}=D_{KL}(a\|e)+D_{KL}(e\|a)$ (대칭).

**손계산 예제.** 5개 구간.

| 구간 | 기준 $e_i$ | 현재 $a_i$ | $(a_i-e_i)\ln(a_i/e_i)$ |
|---|---|---|---|
| 1 | 0.10 | 0.08 | 0.0045 |
| 2 | 0.20 | 0.17 | 0.0049 |
| 3 | 0.40 | 0.38 | 0.0010 |
| 4 | 0.20 | 0.24 | 0.0073 |
| 5 | 0.10 | 0.13 | 0.0079 |
| 합 | | | **0.0255** (반올림 합 0.0256) |

```python
import numpy as np

e = np.array([0.10, 0.20, 0.40, 0.20, 0.10])
a = np.array([0.08, 0.17, 0.38, 0.24, 0.13])
terms = (a - e) * np.log(a / e)
psi = terms.sum()
kl = lambda p, q: (p * np.log(p / q)).sum()
print(np.round(terms, 4), round(psi, 4))
assert np.allclose(np.round(terms, 4), [0.0045, 0.0049, 0.0010, 0.0073, 0.0079])
assert abs(psi - 0.0255) < 5e-5
assert abs(psi - (kl(a, e) + kl(e, a))) < 1e-12        # 대칭 KL 의 합
```

### 4-2. 해석 기준은 관행이다

실무에서는 PSI < 0.10 이면 안정, 0.10~0.25 면 주의(조사), 0.25 초과면 큰 변화로 보는 *경보 기준*이 널리 쓰인다. 이는 통계 법칙이 아니라 **업계 관행**이고, 구간 수·표본 크기·변수 성격에 따라 적절한 값이 달라진다. 기준은 조직이 정하고 *문서화*한다.

### 4-3. bin 경계는 기준기간에서 한 번 정해 고정

- 구간은 **기준기간(개발 표본)** 에서 분위수(예: 10분위)나 비닝 구간으로 만들고, **현재 표본에도 같은 경계**를 적용한다. 현재 표본으로 경계를 다시 만들면 *항상 비슷한 분포*로 보여 변화가 사라진다.
- 경계 양끝은 열어 둔다(−∞, +∞) 그래야 새로운 극단값도 들어간다.
- 범주형 변수는 범주별 비율을 쓰고, 새 범주는 별도 구간으로 둔다. 결측도 별도 구간.

```python
import numpy as np
import pandas as pd
from creditlab.data import make_synthetic_credit_data

def psi_from_samples(expected, actual, bins=10, eps=1e-4):
    """기준(expected) 분위수 경계를 고정해 actual 에 적용한 PSI. 0 비율은 eps 로 바닥을 둔다."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = pd.cut(pd.Series(expected), edges).value_counts(sort=False).to_numpy() / len(expected)
    a = pd.cut(pd.Series(actual), edges).value_counts(sort=False).to_numpy() / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(((a - e) * np.log(a / e)).sum())

df = make_synthetic_credit_data(n=8000, seed=21).sort_values("application_date")
dev, now = df.iloc[:4000], df.iloc[4000:]
print("같은 분포의 두 시점:", round(psi_from_samples(dev["card_utilization"], now["card_utilization"]), 4))

shifted = now["card_utilization"] + 0.15                      # 사용률이 전반적으로 0.15 올랐다고 가정(분포 이동 주입)
psi_shift = psi_from_samples(dev["card_utilization"], shifted)
print("분포 이동 주입 후:", round(psi_shift, 4))
assert psi_from_samples(dev["card_utilization"], now["card_utilization"]) < 0.05
assert psi_shift > 0.25
```

### 4-4. 0 빈도와 표본 수

- 비교기간에 어떤 구간이 비면 $\ln(0)$ 때문에 PSI 가 무한대가 된다 → **작은 값(eps)으로 바닥**을 두거나 인접 구간과 합친다. eps 선택에 따라 값이 크게 달라질 수 있으므로 *방법을 문서화*한다.
- **표본이 작으면 PSI 는 우연만으로도 커진다.** 분포 변화가 전혀 없어도 기대 PSI 는 대략 $(k-1)\,(1/n_e+1/n_a)$ 이다. $k=10$, $n_e=n_a=1{,}000$ 이면 약 0.018. 표본이 작은 월별 모니터링에서는 *임계값을 표본 수에 맞게 해석*한다.

```python
import numpy as np

rng = np.random.default_rng(5)
k, n = 10, 1000
vals = []
for _ in range(400):
    exp = rng.normal(size=n)
    act = rng.normal(size=n)                     # 같은 분포에서 뽑은 두 표본: 진짜 변화는 0
    vals.append(psi_from_samples(exp, act, bins=k))
print(round(float(np.mean(vals)), 4), "이론 근사:", (k - 1) * (2 / n))
assert 0.010 < np.mean(vals) < 0.030
```

### 4-5. 분포 변화 vs 관계 변화

| 변화 | 의미 | PSI 로 보이나 | 예 |
|---|---|---|---|
| **공변량 이동**(covariate shift) | 입력 $X$ 의 분포가 변함, $P(Y\mid X)$ 는 그대로 | 보임 | 신규 채널로 젊은 고객 증가 |
| **개념 변화**(concept drift) | $X$ 와 $Y$ 의 *관계*가 변함 | **안 보일 수 있음** | 경기 침체로 같은 소득도 위험해짐 |
| **사전확률 변화**(prior shift) | 부도율 자체가 변함 | 입력엔 안 보임 | 전반적 부도율 상승 |

PSI 가 작아도 관계가 변했을 수 있고, PSI 가 커도 모형이 여전히 잘 작동할 수 있다. 그래서 *성능 지표(AUROC/KS)·보정·등급별 부도율*을 같이 본다. **PSI 하나만으로 모형 안정성을 판정할 수 없다.**

### 4-6. 모니터링 설계

| 대상 | 지표 | 주기 | 경보 후 조치 |
|---|---|---|---|
| 변수 분포 | 변수별 PSI(CSI 라고도 함) | 월/분기 | 원인 조사(데이터 오류? 모집단 변화?) |
| 점수 분포 | 점수 PSI, 등급 구성비 | 월 | 정책·한도 영향 검토 |
| 변별력 | AUROC/KS(성과 관찰 후) | 분기/반기 | 재개발 검토 |
| 보정 | 등급별 예측 PD vs 실제 부도율 | 분기/반기 | 재보정 검토(단계 12, 15) |
| 데이터 품질 | 결측률, 범위 오류 | 매 배치 | 시스템 점검 |

*경보 → 조사 → 조치 → 재검토*의 책임과 기한을 사전에 정해 둔다.

## 5. 가정

- 기준기간이 **정상 상태**를 대표한다(기준 자체가 이상하면 비교가 무의미).
- 구간 경계와 정의가 시간에 걸쳐 **동일**하다.
- 변화의 방향·크기는 *표본 변동*보다 충분히 크다.

## 6. 한계와 오용

!!! warning "자주 하는 오용"
    - 현재 표본으로 구간을 다시 만들어 PSI 를 계산한다.
    - 0.1/0.25 를 *법칙*으로 여기고 표본 수·구간 수를 무시한다.
    - PSI 가 작으니 "모형 안정"이라고 결론낸다(관계 변화 미탐지).
    - PSI 가 크니 즉시 재개발한다(원인이 데이터 오류일 수 있음).
    - 월별 소표본 PSI 의 변동을 추세로 해석한다.
    - 구간 수가 다른 PSI 를 직접 비교한다.

## 7. 대안 및 비교

| 지표 | 특징 | 비고 |
|---|---|---|
| PSI | 구간 기반, 해석 관행 확립 | 구간 선택에 민감 |
| KS 두 표본 검정 | 연속형 분포 차이 검정 | 큰 표본에서 사소한 차이도 유의 |
| Wasserstein 거리 | 이동량 해석 | 신용 실무에선 덜 쓰임 |
| 평균·분산 이동 | 단순 | 분포 모양 변화 놓침 |
| 모형 기반 탐지(분류기로 두 기간 구분) | 다변량 변화 탐지 | 해석 복잡 |

## 8. 실무 적용

- **규제 로지스틱 CSS**: 개발 시 *점수 분포·등급 구성·변수 구간 분포*를 기준으로 저장하고, 운영에서 같은 구간으로 PSI 를 계산한다. 보고서에 기준기간·구간·임계값·경보 절차를 넣는다.
- **AI 모형**: 입력 변수가 많아 변수별 PSI 와 *점수 PSI* 를 중심으로 보고, 중요 변수(예: SHAP 상위)를 우선 모니터링한다. 변수 사이의 관계 변화는 성능·보정 지표로 확인한다.
- 모니터링 결과는 *경보 이력*으로 남기고, 임계값 초과 시 조치(조사→재보정/재개발)와 승인 절차를 정의한다.

## 9. 지식 확인

**Q1.** 기준 분포와 현재 분포가 완전히 같으면 PSI 는?

<details markdown="1"><summary>정답 보기</summary>

0. 모든 $a_i=e_i$ 이면 각 항이 0.

</details>

**Q2.** 현재 표본으로 분위수 구간을 새로 만들어 PSI 를 계산했더니 항상 거의 0 이다. 왜?

<details markdown="1"><summary>정답 보기</summary>

구간을 현재 분포에 맞춰 다시 만들면 각 구간의 비율이 항상 비슷해져 변화가 지워진다. **기준기간 경계를 고정**해야 한다.

</details>

**Q3.** 변수별 PSI 는 모두 0.03 인데 부도율이 급등했다. 가능한 설명은?

<details markdown="1"><summary>정답 보기</summary>

입력 분포는 그대로지만 **관계가 변했거나(concept drift)** 부도율 자체가 변했을 수 있다(경기 변화 등). PSI 만으론 알 수 없으므로 성능·보정·등급별 부도율을 본다.

</details>

**Q4.** 월 표본이 300건이다. PSI 0.08 은 변화인가?

<details markdown="1"><summary>정답 보기</summary>

소표본에서는 우연만으로 PSI 가 커질 수 있다($k=10$, $n=300$ 이면 기대값 약 0.06). 0.08 만으로 결론내리지 않고 추세·다른 지표와 함께 본다.

</details>

**실습 과제**: 합성 데이터에서 신청일 기준 앞 절반을 기준기간, 뒤 절반을 현재로 하여 5개 변수의 PSI 를 구하고, 한 변수에 *인위적 분포 이동*을 주입해 PSI 와 AUROC 의 변화를 비교하라. PSI 가 큰데 AUROC 가 거의 같을 수 있는 이유와 그 반대의 경우를 각각 설명하라.

## 10. 참고자료

- Siddiqi, N. *Credit Risk Scorecards* (Wiley, 2006) — PSI/CSI 모니터링 관행.
- Yurdakul, B. (2018). *Statistical Properties of Population Stability Index* (Western Michigan Univ. 박사논문) — PSI 의 통계적 성질.
- Quiñonero-Candela, J. et al. *Dataset Shift in Machine Learning* (MIT Press, 2009) — 공변량 이동·사전확률 이동.
- Basel Committee on Banking Supervision, *Studies on the Validation of Internal Rating Systems*, Working Paper No. 14 (2005).
- 확인일 2026-10-18 (서지 정보는 학습 시점에 원문으로 재확인할 것)
