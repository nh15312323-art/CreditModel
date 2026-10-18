# 단계 7. 규제형 로지스틱 CSS와 스코어카드

> 로지스틱 모형이 만든 로그오즈를 "점수"로 바꾸고, 점수를 "등급"으로 묶고, 등급별 부도율을 검토한다. 모형(예측)과 정책(컷오프)은 분리한다.

**연결 실습**: 노트북 `07_scorecard.ipynb` · 코드 `creditlab.scorecard`(`ScoreScaling`, `Scorecard`, `make_grade_edges`, `grade_table`, `cutoff_analysis`) · 이 페이지의 점수 변환 코드를 먼저 실행해 보자.

## 학습 목표

- WOE → 로지스틱 → 점수카드로 이어지는 구조를 설명한다.
- 기준점·기준 오즈·**PDO** 로 점수를 변환하고 *점수 ↔ 오즈 ↔ 확률*을 검산한다.
- 점수 방향(높을수록 안전)과 오즈의 정의를 일관되게 유지한다.
- 점수 구간 등급화 방법(동일 건수·동일 부도율·업무 기준)의 차이를 안다.
- 등급별 표본 수·부도율·신뢰구간·단조성을 검토한다.
- **컷오프(정책)** 와 예측모형을 구분하고 override·수동심사의 의미를 이해한다.

## 1. 왜 생겼는가

로지스틱 회귀의 출력(확률, 로그오즈)은 심사 현장에서 쓰기 불편하다. 심사자·영업점·고객에게는 "720점", "3등급" 같은 **정수 점수·등급**이 익숙하다. 또 "점수가 20점 오를 때마다 위험이 절반이 된다" 같은 *일정한 척도*가 있으면 점수 의미가 시간이 지나도 일관된다. 이렇게 모형 출력을 *운영 가능한 척도*로 바꾸려고 스코어카드가 발전했다.

## 2. 왜 필요한가

- **설명 가능성**: 변수·구간마다 몇 점인지 표로 보여 줄 수 있다(규제 환경의 요구와 맞음).
- **운영 일관성**: 점수의 의미가 고정되면 정책(컷오프·한도·금리)을 점수로 설계할 수 있다.
- **모니터링**: 점수 분포(PSI)와 등급별 부도율을 시간에 따라 추적한다(단계 10).

## 3. 무엇을 하는가

`변수 구간 → WOE → 로지스틱 계수 → 점수(구간별 점수의 합) → 등급 → 등급별 PD`

## 4. 어떻게 작동하는가

### 4-1. 구조

단계 5 의 WOE 변수 $W_j$ 를 입력으로 부도 로그오즈 모형을 적합한다.

$$ z=\ln\frac{p}{1-p}=\beta_0+\sum_{j=1}^{m}\beta_j W_j $$

WOE 를 쓰는 이유는 각 변수 구간이 *독립적으로 점수에 더해지는 표*가 되기 때문이다. 장점은 투명성이고, 단점은 **비닝으로 인한 정보 손실**과 *변수 간 상호작용을 직접 표현하지 못한다*는 점이다.

### 4-2. 점수 변환: 기준점·기준 오즈·PDO

점수를 *높을수록 안전*하게 만들기 위해 **좋은:나쁜 오즈**(good odds $=(1-p)/p$)를 쓴다.

$$ \mathrm{Score}=\mathrm{Offset}+\mathrm{Factor}\times\ln(\text{odds}_{good}),\qquad \mathrm{Factor}=\frac{\mathrm{PDO}}{\ln 2},\quad \mathrm{Offset}=\mathrm{Score}_0-\mathrm{Factor}\times\ln(\text{odds}_0) $$

- $\mathrm{PDO}$(points to double the odds): 오즈를 **2배**로 만드는 점수 증가량.
- $\text{odds}_0$, $\mathrm{Score}_0$: 기준 오즈와 그에 대응하는 기준점.

**숫자 예제.** 기준점 600 에서 good:bad 오즈 50:1, PDO=20 이라 하자.

- Factor $=20/\ln2=28.8539$
- Offset $=600-28.8539\times\ln50=600-112.877=487.123$

| good:bad 오즈 | 점수 | 부도확률 $1/(1+\text{odds})$ |
|---|---|---|
| 25 : 1 | 580 | 3.85% |
| 50 : 1 | 600 | 1.96% |
| 100 : 1 | 620 | 0.99% |
| 200 : 1 | 640 | 0.50% |

점수가 20점 오를 때마다 오즈가 2배, 부도확률은 *대략* 절반(정확히는 $1/(1+2\,\text{odds})$)이 된다.

```python
import math

pdo, base_score, base_odds = 20, 600, 50
factor = pdo / math.log(2)
offset = base_score - factor * math.log(base_odds)
score = lambda odds_good: offset + factor * math.log(odds_good)
pd_from_odds = lambda odds_good: 1 / (1 + odds_good)

print(round(factor, 4), round(offset, 3))
assert abs(factor - 28.8539) < 1e-4 and abs(offset - 487.123) < 1e-3
for odds, s in [(25, 580), (50, 600), (100, 620), (200, 640)]:
    assert abs(score(odds) - s) < 1e-9                 # 20점마다 오즈 2배
assert abs(pd_from_odds(50) - 0.0196) < 1e-4 and abs(pd_from_odds(100) - 0.0099) < 1e-4
```

### 4-3. 구간별 점수 배분

모형이 부도 로그오즈 $z$ 를 주므로 $\ln(\text{odds}_{good})=-z$ 이고

$$ \mathrm{Score}=\mathrm{Offset}-\mathrm{Factor}\times\Big(\beta_0+\sum_j\beta_jW_j\Big)=\sum_{j}\Big[-\mathrm{Factor}\,\beta_jW_{j}-\tfrac{\mathrm{Factor}\,\beta_0-\mathrm{Offset}}{m}\Big] $$

변수 $j$ 의 구간 $i$ 점수는 $-\mathrm{Factor}\,\beta_jW_{ij}$ 에 절편·기준점 몫을 *변수 수 $m$ 으로 나눠* 더한 값이다. 점수표는 보통 반올림한 **정수 점수**를 쓰므로, 반올림 오차가 합산되어 원래 점수와 몇 점 달라질 수 있다 → 검산 필수.

**단일 변수 예제.** 단계 5 의 연체 건수 3구간(정상 6,100/1,900/1,000, 부도 520/260/220)에서 $\beta=1$ 이면 모형 오즈가 *구간의 실제 오즈*와 같아진다(단일 변수·포화 구간). 그 구간의 good 오즈를 위 점수식에 넣으면

| 구간 | good:bad 오즈 | 점수 |
|---|---|---|
| 0건 | 11.73 | 558.2 |
| 1건 | 7.31 | 544.5 |
| 2건 이상 | 4.55 | 530.8 |

변수 하나라 점수 폭이 좁다. 실제 점수카드는 여러 변수의 점수가 합산되어 넓게 퍼진다.

```python
import numpy as np

G, B = np.array([6100, 1900, 1000]), np.array([520, 260, 220])
odds_good = G / B
scores = offset + factor * np.log(odds_good)       # 앞 코드의 factor/offset
print(np.round(odds_good, 2), np.round(scores, 1))
assert np.allclose(np.round(scores, 1), [558.2, 544.5, 530.8])
assert np.all(np.diff(scores) < 0)                 # 위험한 구간일수록 점수가 낮다
```

### 4-4. 점수 구간과 등급화

| 방식 | 설명 | 장점 | 주의 |
|---|---|---|---|
| 동일 건수 | 같은 인원씩 | 등급별 표본 안정 | 부도율 간격이 불규칙 |
| 동일 부도율 간격 | 인접 등급 부도율이 비슷하게 벌어짐 | 해석 직관적 | 희소 등급 발생 가능 |
| 업무 기준 | 정책·상품 요건 반영 | 운영 편리 | 통계적 근거 약할 수 있음 |
| 최적화(규칙 기반) | 변별력·단조·최소 건수 제약 | 객관적 | 개발 표본 과적합 가능 |

**등급 검토 항목**: ① 등급별 건수·부도 건수, ② 등급별 부도율과 **신뢰구간**(Wilson), ③ **단조성**(등급이 나빠질수록 부도율 증가), ④ 인접 등급 구분 가능성(구간이 겹치면 합치는 것을 검토), ⑤ 등급별 PD 추정치(장기 평균·보수성 반영은 단계 15).

```python
import numpy as np
import pandas as pd
from creditlab.data import default_rate_ci, make_synthetic_credit_data

df = make_synthetic_credit_data(n=8000, seed=7)
# 설명용 단순 점수: 위험할수록 낮은 점수. (실제 점수는 로지스틱 계수로 만든다)
risk = 2.0 * df["card_utilization"] + 0.6 * df["num_delinq_12m"] - 0.03 * df["age"]
df["score"] = 600 - 40 * (risk - risk.mean()) / risk.std()
df["grade"] = pd.qcut(df["score"], q=5, labels=["E", "D", "C", "B", "A"])   # 동일 건수 5등급 (A=가장 안전)
rows = []
for g, sub in df.groupby("grade", observed=True):
    lo, hi = default_rate_ci(sub["default"])
    rows.append({"grade": g, "n": len(sub), "bad": int(sub["default"].sum()), "rate": sub["default"].mean(), "ci_low": lo, "ci_high": hi})
table = pd.DataFrame(rows).sort_values("grade")   # A → E (가장 안전 → 가장 위험)
print(table.round(4).to_string(index=False))
assert table["n"].sum() == len(df) and (table["ci_low"] <= table["rate"]).all()
assert table["rate"].is_monotonic_increasing                         # A→E 로 갈수록 부도율 증가
```

### 4-5. 모형 vs 정책: 컷오프, override

- **모형**은 점수를 낸다. **컷오프**는 "몇 점 미만은 거절/심사 강화" 같은 *정책*이다. 컷오프는 승인율, 기대손실, 수익, 위험선호, 규제·고객 보호 정책 등을 반영해 *사업이 정한다*(단계 12 임계값 선택).
- **Override**: 심사자가 점수와 다르게 결정하는 것. 사유를 기록하고 비율을 모니터링한다. 모형이 못 보는 정보가 있으면 정당할 수 있으나 남용되면 모형의 효과가 사라진다.
- **Reject/수동심사**: 점수가 경계 구간이면 사람 심사로 보내는 설계가 흔하다.
- 모형의 성능과 **정책 성과**(손익, 승인율)는 따로 평가한다(단계 15).

## 5. 가정

- 로지스틱 모형이 로그오즈를 합리적으로 설명한다(WOE 변환 후 선형 근사).
- **점수–오즈 관계**가 시간이 지나도 유지된다(점수의 의미가 안정적).
- 개발 모집단과 적용 모집단이 충분히 비슷하다(단계 1).
- 등급별 부도율 순서가 미래에도 유지된다.

## 6. 한계와 오용

!!! warning "자주 하는 오용"
    - 점수 방향(높을수록 안전/위험)과 오즈 정의(good:bad vs bad:good)를 섞어 쓴다.
    - 정수 반올림 후 *검산을 안 해서* 점수와 확률이 어긋난다.
    - 등급을 "부도율이 좋아 보이게" 개발 표본에 맞춰 과도하게 최적화한다.
    - 컷오프를 모형 성능의 일부로 취급한다(승인율이 바뀌면 모형이 나빠진 것처럼 오해).
    - 등급 PD 를 개발 기간의 *관측 부도율*로만 두고 장기·하방 경기 반영 필요를 검토하지 않는다(단계 15).
    - 한 변수 구간의 점수만 보고 "이 변수가 부도를 *일으킨다*"고 해석한다.

## 7. 대안 및 비교

| 접근 | 장점 | 한계 |
|---|---|---|
| WOE 로지스틱 스코어카드 | 투명, 점수표 직접 제공 | 정보 손실, 상호작용 약함 |
| 더미 변수 로지스틱 | 정보 손실 적고 구현 간단 | 계수 수 증가, 점수표 변환이 덜 직관 |
| GAM(일반화가법모형) | 비선형을 부드럽게 | 설명이 표보다 복잡 |
| 트리 부스팅 점수화 | 높은 변별력 | 점수의 *구간별 설명* 어려움, 설명 도구 필요 |

## 8. 실무 적용

- **규제 로지스틱 CSS**: 계수 → 점수표 → 점수 분포 → 등급표 → 등급별 PD. 점수표와 시스템 구현 일치 테스트(같은 입력에 같은 점수)를 한다.
- **AI 모형**: 확률을 같은 PDO 규칙으로 **점수로 변환**할 수 있다. 이렇게 하면 동일한 등급 체계에서 로지스틱과 비교할 수 있다. 다만 *구간별 점수표*처럼 변수 기여를 보여 줄 수는 없어 설명 도구(SHAP 등)가 별도로 필요하다.
- 모델 카드에 점수 방향·PDO·기준점·기준 오즈, 등급 경계, 컷오프 *정책 문서의 위치*를 명시한다.

## 9. 지식 확인

**Q1.** PDO=20 에서 점수가 40점 올라가면 good:bad 오즈는 몇 배가 되는가?

<details markdown="1"><summary>정답 보기</summary>

20점마다 2배이므로 40점이면 $2^2=4$배.

</details>

**Q2.** 기준 600점 = 오즈 50:1, PDO 20. 오즈 100:1 의 점수와 부도확률은?

<details markdown="1"><summary>정답 보기</summary>

620점. 부도확률 $1/(1+100)\approx0.99\%$.

</details>

**Q3.** 등급 B 의 부도율이 등급 C 보다 높게 나왔다(단조성 위반). 가능한 원인과 대응은?

<details markdown="1"><summary>정답 보기</summary>

표본 부족(신뢰구간이 겹침), 등급 경계 설정, 개발·검증 기간의 변화 때문일 수 있다. 신뢰구간 확인 → 건수 부족이면 등급 병합 → 지속되면 모형·변수 재검토.

</details>

**Q4.** 컷오프를 600→620 으로 올렸더니 승인율이 하락했다. 모형이 나빠졌는가?

<details markdown="1"><summary>정답 보기</summary>

아니다. 정책이 바뀐 것이다. 모형 성능(변별력·보정)은 점수 자체를 평가하고, 정책 성과(승인율·손익)는 분리해 평가한다.

</details>

**실습 과제**: 합성 데이터로 3~4개 변수의 WOE 로지스틱 모형을 만들고(단계 5·6), PDO=20, 기준점 600, 기준 오즈 50:1 로 점수표를 만든 뒤, 점수→오즈→확률을 5개 사례에서 검산하라. 5등급 표에 건수·부도율·신뢰구간을 쓰고 단조성을 평가하라.

## 10. 참고자료

- Siddiqi, N. *Credit Risk Scorecards* (Wiley, 2006) — 점수 스케일링, 등급화.
- Thomas, L., Edelman, D., Crook, J. *Credit Scoring and Its Applications* (SIAM, 2002).
- Basel Committee on Banking Supervision, *Studies on the Validation of Internal Rating Systems*, Working Paper No. 14 (2005) — 등급 체계 검증. <https://www.bis.org/bcbs/>
- 실제 적용: 해당 시점의 감독규정·내부 방법론 원문 확인.
- 확인일 2026-10-18
