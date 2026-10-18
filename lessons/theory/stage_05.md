# 단계 5. 비닝·WOE·IV

> 연속형 변수를 구간으로 나누고, 구간마다 "얼마나 위험한가"를 로그 비율(WOE)로 바꾼 뒤, 변수 전체의 정보량(IV)을 요약한다. 스코어카드의 재료가 되는 과정이다.

**연결 실습**: 노트북 `05_binning_woe_iv` · 코드 `creditlab.features.WoeBinner`, `woe_iv_from_counts`, `iv_summary` · 이 페이지의 손계산을 먼저 해 보자.

## 학습 목표

- 비닝이 필요한 이유와 **정보 손실의 대가**를 설명한다.
- 등폭·등빈도·업무 기준·트리 기반 비닝을 구분한다.
- WOE·IV 를 손으로 계산하고 방향 규칙을 고정한다.
- 단조성·최소 bin 크기·이벤트 수·안정성을 점검한다.
- **개발 표본에서만** 경계를 만들고 다른 표본에는 *같은 경계*를 적용한다.

## 1. 왜 생겼는가

로지스틱 회귀는 변수와 로그오즈의 **직선 관계**를 가정한다. 그러나 나이, 소득, 사용률은 대개 직선이 아니다(예: 사용률이 매우 낮거나 매우 높을 때 모두 위험). 변수를 구간으로 나누면 비선형을 *구간별 값*으로 흡수할 수 있고, 극단값·결측·이상치도 한 구간으로 다룰 수 있다. 여기에 구간별 위험을 *같은 척도(로그 비율)* 로 바꾸는 **WOE** 가 더해져, 서로 다른 변수들을 하나의 로지스틱 모형에 넣기 쉬워졌다. 이 방식은 신용 스코어카드에서 오래 쓰여 왔다.

## 2. 왜 필요한가

- 설명 가능하다: "이 구간이면 이 점수"라고 고객·감독당국에게 보여 줄 수 있다.
- 결측·극단값 처리 규칙이 점수표에 *명시*된다.
- WOE 로 바꾼 변수는 단위·분포가 달라도 비교하기 쉽고, **단조 방향**을 사전에 검토할 수 있다.
- IV 로 변수의 *전체 정보량*을 한 숫자로 요약해 후보를 거를 수 있다.

## 3. 무엇을 하는가

입력: 연속형/범주형 변수 + 이진 타깃. 출력: ① 구간 경계, ② 구간별 건수·부도율·WOE, ③ 변수 IV.

## 4. 어떻게 작동하는가

### 4-1. 비닝 방법

| 방법 | 설명 | 장점 | 주의 |
|---|---|---|---|
| 등폭 | 값 범위를 같은 폭으로 | 단순 | 치우친 분포에서 구간이 비어 있음 |
| 등빈도 | 건수가 같도록 분위수로 | 구간 크기 안정 | 같은 값이 많으면 경계 중복 |
| 업무 기준 | 정책상 의미 있는 경계(예: 연체 0/1/2+) | 설명 쉬움 | 데이터와 어긋날 수 있음 |
| 트리 기반 | 타깃을 보고 최적 분할 | 정보 보존 | **과적합** 위험, 단조 제약 필요 |

**원칙**: 경계는 **개발(학습) 표본에서만** 만들고, 검증·테스트·운영에는 *그 경계를 그대로* 쓴다. 테스트 표본을 보고 경계를 고치면 누수다.

### 4-2. WOE 정의 — 이 프로그램의 약속

이 프로그램에서는 타깃이 1=부도(bad), 0=정상(good) 이다. 구간 $i$ 에서

$$ \mathrm{WOE}_i=\ln\frac{\text{부도 분포}_i}{\text{정상 분포}_i}=\ln\frac{B_i/B}{G_i/G} $$

여기서 $B_i, G_i$ 는 구간 $i$ 의 부도·정상 건수, $B, G$ 는 전체 부도·정상 건수다. WOE > 0 이면 **전체 평균보다 위험**한 구간이다. 이렇게 정의하면 WOE 변수로 *부도 로그오즈*를 모형화할 때 계수가 **양수**(이상적으로 1 근처)가 된다.

!!! warning "방향 규칙 주의"
    교재(예: Siddiqi)에서는 $\ln(\text{good 분포}/\text{bad 분포})$ 를 쓰는 경우가 많다. 부호만 반대다($\mathrm{WOE}_\text{교재}=-\mathrm{WOE}_\text{본 프로그램}$). **IV 는 부호와 무관하게 같다.** 코드·문서·보고서에서 방향을 하나로 고정하고 적어 둔다.

### 4-3. IV (Information Value)

$$ \mathrm{IV}=\sum_i\left(\frac{B_i}{B}-\frac{G_i}{G}\right)\mathrm{WOE}_i $$

각 항은 $(\text{부도 분포}-\text{정상 분포})\times\ln(\text{비})$ 로 *항상 0 이상*이다. 구간 간 부도·정상 분포가 다를수록 IV 가 커진다.

**손 계산 예제.** 정상 9,000건, 부도 1,000건. 변수 "최근 12개월 연체 건수"를 3구간으로 나눈다.

| 구간 | 정상 $G_i$ | 부도 $B_i$ | 부도율 | 정상 분포 | 부도 분포 | WOE | IV 기여 |
|---|---|---|---|---|---|---|---|
| 0건 | 6,100 | 520 | 7.9% | 0.6778 | 0.5200 | -0.2650 | 0.0418 |
| 1건 | 1,900 | 260 | 12.0% | 0.2111 | 0.2600 | 0.2083 | 0.0102 |
| 2건 이상 | 1,000 | 220 | 18.0% | 0.1111 | 0.2200 | 0.6831 | 0.0744 |
| 합계 | 9,000 | 1,000 | 10.0% | 1 | 1 | | **0.1264** |

예를 들어 1건 구간의 WOE 는 $\ln(0.26/0.2111)=\ln1.2316=0.2083$ 이다. 부도율이 높은 구간일수록 WOE 가 커져 **단조 증가**한다.

```python
import numpy as np

G = np.array([6100, 1900, 1000])
B = np.array([520, 260, 220])
dist_g, dist_b = G / G.sum(), B / B.sum()
woe = np.log(dist_b / dist_g)                    # 본 프로그램 규칙: ln(부도 분포 / 정상 분포)
iv_terms = (dist_b - dist_g) * woe
print(np.round(woe, 4), np.round(iv_terms, 4), round(iv_terms.sum(), 4))
assert np.allclose(np.round(woe, 4), [-0.2650, 0.2083, 0.6831])
assert np.allclose(np.round(iv_terms, 4), [0.0418, 0.0102, 0.0744])
assert abs(iv_terms.sum() - 0.1264) < 1e-4
assert np.all(np.diff(B / (G + B)) > 0)          # 부도율이 구간 순서대로 증가(단조)
```

### 4-4. IV 해석: 관행이지 법칙이 아니다

흔히 IV < 0.02 는 거의 무정보, 0.02~0.1 약함, 0.1~0.3 중간, 0.3~0.5 강함, 0.5 초과는 *의심*(누수 가능)으로 읽는다. 이는 업계 관행이다.

- IV 는 **구간 수가 많을수록 커지는 경향**이 있다(과적합).
- IV 는 인과 효과나 *미래* 예측력을 증명하지 않는다. 기간별 안정성을 따로 본다.
- 표본이 작으면 IV 도 흔들린다. 부도 건수가 적은 구간은 신뢰할 수 없다.

### 4-5. 0 빈도와 스무딩

어떤 구간에 부도가 0건이면 $\ln(0)$ 이 되어 WOE 가 $-\infty$ 다. 해결은 **스무딩**이다. 예를 들어 모든 구간의 정상·부도 건수에 0.5 를 더해 분포를 계산한다.

```python
import numpy as np

G = np.array([100, 500, 400])
B = np.array([0, 30, 70])
with np.errstate(divide="ignore"):
    raw = np.log((B / B.sum()) / (G / G.sum()))
smooth = np.log(((B + 0.5) / (B + 0.5).sum()) / ((G + 0.5) / (G + 0.5).sum()))
print(raw, np.round(smooth, 3))
assert np.isneginf(raw[0]) and np.all(np.isfinite(smooth))

# 이 프로그램의 구현(creditlab.features.woe_iv_from_counts)과 같은 정의인지 확인
from creditlab.features import woe_iv_from_counts
assert np.allclose(woe_iv_from_counts(B, G, smoothing=0.5)["woe"], smooth)
```

스무딩 상수(0.5 등)는 *선택*이며, 어떤 값을 썼는지 문서화한다. 더 나은 방법은 **최소 구간 크기와 최소 이벤트 수**를 두어 0 빈도가 생기지 않도록 구간을 합치는 것이다.

### 4-6. 단조성·최소 bin·안정성

- **단조성**: 업무 논리상 한 방향이어야 하는 변수(예: 연체 많을수록 위험)는 구간이 순서대로 부도율이 *증가*해야 설명이 쉽다. 단조가 깨지면 이유를 설명하거나 구간을 합친다. U자형 변수(사용률 등)는 단조를 *강요하지 않는다*.
- **최소 bin 크기**: 예를 들어 전체의 5% 이상, 이벤트 N건 이상 같은 *내부 기준*을 둔다(수치는 정책).
- **안정성**: 개발 기간을 나눠(예: 연도별) 구간별 부도율·WOE 순서가 유지되는지 본다.
- **범주형**: 희소 범주는 합치고, **미지 범주**(운영에서 처음 보는 값)에 대한 규칙(별도 구간 또는 WOE 0)을 미리 정한다.
- **결측**: 별도 구간으로 두고 그 구간의 WOE 를 계산한다(결측 의미가 운영에서도 같은지 확인).

### 4-7. 개발 표본에서 경계를 만들고 그대로 적용하기

```python
import numpy as np
import pandas as pd
from creditlab.data import make_synthetic_credit_data

df = make_synthetic_credit_data(n=6000, seed=1)
df = df.sort_values("application_date")
train, test = df.iloc[:4500], df.iloc[4500:]           # 시간 순 분할(단계 8)

# 1) 경계는 train 에서만: 사용률을 5개 등빈도 구간으로
_, edges = pd.qcut(train["card_utilization"], q=5, retbins=True, duplicates="drop")
edges[0], edges[-1] = -np.inf, np.inf                    # 새 데이터의 극단값도 받도록 양끝을 열어 둔다

# 2) 같은 경계를 train/test 모두에 적용 (test 로 경계를 다시 계산하지 않는다)
train_bin = pd.cut(train["card_utilization"], bins=edges)
test_bin = pd.cut(test["card_utilization"], bins=edges)

tab = train.groupby(train_bin, observed=True)["default"].agg(n="size", bad="sum", rate="mean")
print(tab)
assert test_bin.notna().all()                            # 경계 밖 값이 없다
assert len(edges) == 6 and (np.diff(edges[1:-1]) > 0).all()
```

## 5. 가정

- 구간 내부에서 위험이 대체로 **같다**(구간 내 균질성).
- 개발 표본의 구간별 위험 관계가 **미래에도 유지**된다(안정성).
- 구간별 표본·이벤트 수가 WOE 를 믿을 만큼 충분하다.
- WOE 변환 후 변수와 로그오즈의 관계가 **근사적으로 선형**이다(단계 6의 계수 ≈ 1 점검).

## 6. 한계와 오용

!!! warning "자주 하는 오용"
    - **타깃 기반 최적 비닝을 전체 데이터에** 적용한 뒤 분할한다 → 누수. 비닝은 *학습 데이터*에서만.
    - 구간을 너무 잘게 나눈다 → IV 가 부풀고, 새 데이터에서 순서가 뒤바뀐다.
    - IV 가 높다고 **인과**나 **미래 성능**이 보장된다고 말한다.
    - 구간 수를 늘리면 *정보 손실이 없다*고 가정한다. 비닝은 구간 내부의 정보를 버리는 **정보 손실**이다.
    - WOE 방향·스무딩·결측 처리 규칙을 문서화하지 않는다 → 재현 불가.

## 7. 대안 및 비교

| 접근 | 장점 | 한계 |
|---|---|---|
| 비닝 + WOE + 로지스틱 | 설명 쉬움, 비선형·결측 처리, 규제 친화 | 정보 손실, 상호작용 포착 못함 |
| 원변수 + 스플라인/다항 | 정보 보존 | 점수표 형태 변환이 복잡 |
| 트리 부스팅 | 비선형·상호작용 자동 | 설명·안정성 근거 별도 필요 |
| 원-핫(구간별 더미) | WOE 없이 같은 비선형 | 계수가 많아져 변수당 자유도 증가 |

## 8. 실무 적용

- **규제 로지스틱 CSS**: 최종 *구간표*가 점수카드의 뼈대다. 구간 규칙·WOE 방향·스무딩·결측/미지 범주 규칙·개발 기간을 문서에 남긴다.
- **AI 모형**: 비닝 없이 원변수를 쓰는 것이 보통이지만, **공정 비교**를 위해 같은 정제·분할을 쓴다. WOE 변수를 트리 모형에 넣으면 안 되는 것은 아니나 이득이 제한적이다.
- 비닝 결과(경계, WOE)는 **파일로 저장**하고 운영 시스템이 같은 경계로 변환하는지 *일치 테스트*를 한다.

## 9. 지식 확인

**Q1.** 구간 $i$ 의 정상 분포가 0.30, 부도 분포가 0.45 이면 WOE(본 프로그램 규칙)와 그 구간의 위험은?

<details markdown="1"><summary>정답 보기</summary>

$\ln(0.45/0.30)=\ln1.5\approx0.405>0$ → 평균보다 **위험한** 구간.

</details>

**Q2.** 교재 규칙($\ln$ good/bad)으로 계산한 WOE 가 -0.405 이다. 본 프로그램의 값은?

<details markdown="1"><summary>정답 보기</summary>

부호만 반대이므로 **+0.405**. IV 는 같다.

</details>

**Q3.** 비닝을 전체 데이터로 한 뒤 train/test 로 나눴더니 test AUROC 가 높게 나왔다. 무엇이 문제인가?

<details markdown="1"><summary>정답 보기</summary>

타깃을 이용한 경계가 test 정보까지 반영해 만들어졌으므로 **누수**다. 분할 후 train 에서만 비닝하고, test 에는 그 경계를 적용만 해야 한다.

</details>

**Q4.** 변수 A 의 IV 가 0.9 다. 첫 행동은?

<details markdown="1"><summary>정답 보기</summary>

누수·정의 오류·소표본 구간 의심을 먼저 한다. 변수의 생성 시점, 결과 이후 값 여부, 구간별 이벤트 수를 확인한다.

</details>

**실습 과제**: 합성 데이터의 `num_delinq_12m`, `card_utilization`, `age` 에 대해 train 에서 비닝·WOE·IV 를 계산하고, test 에 같은 경계를 적용했을 때 구간별 부도율 순서가 유지되는지 비교하라. WOE 방향과 스무딩 값을 표 위에 명시하라.

## 10. 참고자료

- Siddiqi, N. *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring* (Wiley, 2006).
- Thomas, L. *Consumer Credit Models* (Oxford University Press, 2009).
- scikit-learn `KBinsDiscretizer`: <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html>
- pandas `qcut`/`cut`: <https://pandas.pydata.org/docs/reference/api/pandas.qcut.html>
- 확인일 2026-10-18
