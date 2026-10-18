# 단계 15. 고급 주제와 운영: 거절 추론·충당금·설명·공정성·모형위험관리

> 모형은 개발이 끝나면 시작이다. 운영 환경에서는 "누구의 데이터로 배웠는가(선택 편향)", "확률이 돈이 되는 방식(충당금)", "고객에게 어떻게 설명하는가", "누가 불리해지는가", "누가 책임지는가"가 문제가 된다.

**연결 실습**: Phase 6 노트북(예정). 이 페이지는 각 주제의 *핵심 계산*을 작게 재현한다.

## 학습 목표

- 거절 추론이 필요한 이유(선택 편향)를 재현해 설명한다.
- PD·LGD·EAD 로 기대신용손실(ECL)을 계산한다.
- 로지스틱 모형에서 개별 고객의 *불이익 사유* 후보를 기여도로 산출한다.
- 공정성 지표(승인율 비율)를 계산하고, 그 한계와 프록시 변수 문제를 설명한다.
- 모형위험관리(MRM)의 구성요소와 모형 카드에 담을 내용을 말한다.

## 1. 왜 생겼는가

- **선택 편향**: 과거에 승인된 사람의 결과만 관찰된다. 거절된 사람이 부도 났을지는 *모른다*.
- **회계·자본 규제**: 부도 확률은 충당금(IFRS 9/CECL)과 자본(바젤)에 쓰여 *보정된 확률*이 필요하다.
- **소비자 보호**: 거절 시 사유 설명 요구, 차별 금지 요구가 있다.
- **모형 사고**: 잘못된 모형이 대규모 손실을 낸 사례들 후에 모형위험관리 원칙이 정립되었다(예: 미국 SR 11-7, 2011).

## 2. 왜 필요한가

- 승인 모집단에서만 학습한 모형을 *전체 신청자*에 쓰면 성능과 확률이 틀어진다.
- 충당금·가격·한도는 확률 값에 직접 의존한다.
- 법·규제·평판 위험이 모형 품질만큼 중요하다.

## 3. 무엇을 하는가

| 주제 | 질문 | 도구 |
|---|---|---|
| 거절 추론 | 거절자의 결과를 어떻게 다루나 | 확대(augmentation), 파셀링, 일부 무작위 승인, 외부 데이터 |
| 충당금 | 확률을 금액으로 | ECL = PD × LGD × EAD |
| 설명 | 이 고객은 왜 거절되었나 | 기여도 기반 사유 코드 |
| 공정성 | 집단 간 격차가 있나 | 승인율/오류율 비교, 프록시 점검 |
| MRM | 누가 어떻게 검증·승인하나 | 모형 인벤토리, 독립 검증, 모니터링 |

## 4. 어떻게 작동하는가

### 4-1. 선택 편향 재현

과거에 위험 점수가 낮고 심사역 인상이 좋은 60%만 승인했다고 하자. 승인자만으로 학습·평가하면 *위험 범위가 잘려* 변별력이 낮아 보이고, 부도율과 예측 PD 가 전체 신청자보다 크게 낮다.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from creditlab.data import make_synthetic_credit_data

df = make_synthetic_credit_data(n=20000, seed=71).dropna().reset_index(drop=True)
feats = ["age", "income_10k", "loan_amount_10k", "employment_years", "card_utilization", "num_delinq_12m", "inquiries_6m"]
rule = df["num_delinq_12m"] * 1.5 + df["card_utilization"] * 3 + df["inquiries_6m"] * 0.3   # 과거 심사 규칙(가정)
rng = np.random.default_rng(0)
officer = df["default"] * 1.0 + rng.normal(0, 1, len(df))     # 심사역의 비정형 인상: 결과와 상관, 하지만 데이터 변수에는 없음
z = lambda s: (s - s.mean()) / s.std()
policy = z(rule) + 1.5 * z(officer)
accepted = policy <= policy.quantile(0.60)

train_idx = np.arange(len(df)) < 14000
dev, hold = df[train_idx], df[~train_idx]
acc_dev = dev[accepted[train_idx]]
acc_hold = hold[accepted[~train_idx]]

make = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
m_acc = make().fit(acc_dev[feats], acc_dev["default"])        # 승인자만으로 학습
m_all = make().fit(dev[feats], dev["default"])                # (참고) 전체로 학습: 현실에선 거절자 결과를 모름

print("부도율 승인자", round(acc_dev["default"].mean(), 3), "전체", round(dev["default"].mean(), 3))
auc_on_acc = roc_auc_score(acc_hold["default"], m_acc.predict_proba(acc_hold[feats])[:, 1])
auc_on_all = roc_auc_score(hold["default"], m_acc.predict_proba(hold[feats])[:, 1])
print("승인자 학습 모형 AUROC  승인자 구간:", round(auc_on_acc, 3), " 전체 신청자:", round(auc_on_all, 3))
pd_pred = m_acc.predict_proba(hold[feats])[:, 1].mean()
print("전체 신청자 예측 평균 PD", round(pd_pred, 3), "실제", round(hold["default"].mean(), 3))
assert acc_dev["default"].mean() < dev["default"].mean()       # 승인자의 부도율이 더 낮다
assert auc_on_acc < auc_on_all                                 # 범위 제한(range restriction)으로 승인자 안에서는 변별력이 낮아 보인다
assert pd_pred < hold["default"].mean()                        # 승인자만 본 모형은 전체 모집단의 위험을 과소추정
```

**왜 이렇게 크게 틀어지는가**: 과거 승인이 *데이터에 있는 변수*(연체·사용률)만으로 결정되었다면, 올바르게 설정된 모형은 승인자만으로 학습해도 전체를 꽤 잘 예측한다(관측 변수에 의한 선택). 편향이 커지는 것은 승인이 **데이터에 없는 정보**(심사역의 인상, 서류, 영업 판단)에 의존해 *결과와 상관*될 때다. 위 코드는 그 상황을 `officer` 로 주입했다. 실제 데이터에서는 `officer` 가 얼마나 큰지 알 수 없다는 것이 문제다.

거절 추론 기법은 *가정에 의존*한다. 어떤 가정도 거절자의 진짜 결과를 대체하지 못한다. 가장 신뢰할 만한 방법은 **소액·통제된 무작위 승인(실험)** 이나 외부 신용정보로 *거절자의 이후 성과*를 관찰하는 것이다. 이 프로그램은 거절 추론 기법을 *결과를 개선하는 마법*이 아니라 *편향의 크기를 가늠하는 도구*로 소개한다.

### 4-2. 기대신용손실(ECL)

$$ \mathrm{ECL}=\mathrm{PD}\times\mathrm{LGD}\times\mathrm{EAD} $$

- PD: 부도확률(12개월 또는 전체 만기), LGD: 부도 시 손실률, EAD: 부도 시 노출액.
- IFRS 9 은 신용위험이 *유의하게 증가했는지*에 따라 12개월 ECL(1단계) 또는 전체 만기 ECL(2·3단계)을 쓴다. 보정된 PD 가 핵심이다.

```python
pd_, lgd, ead = 0.05, 0.45, 1000.0
ecl = pd_ * lgd * ead
print(ecl)
assert abs(ecl - 22.5) < 1e-12

# 포트폴리오: 등급별 합산
grades = [  # (고객 수, PD, LGD, EAD)
    (1000, 0.01, 0.45, 1000.0),
    (500, 0.05, 0.45, 1000.0),
    (100, 0.20, 0.45, 1000.0),
]
total = sum(n * p * l * e for n, p, l, e in grades)
print(total)
assert abs(total - (1000 * 0.01 * 450 + 500 * 0.05 * 450 + 100 * 0.20 * 450)) < 1e-9   # 4500+11250+9000
assert abs(total - 24750.0) < 1e-9
```

보정이 틀려 PD 가 절반으로 과소추정되면 충당금도 절반으로 과소계상된다. 단계 12 의 보정이 *회계 숫자*로 이어진다.

### 4-3. 설명: 불이익 사유 후보

로지스틱 모형에서 변수 $j$ 의 기여도(평균 고객 대비)는 $c_j=\beta_j(x_j-\bar x_j)$ 이며, $\text{logit}(x)-\text{logit}(\bar x)=\sum_j c_j$ 이다. 기여도가 큰 양수(위험 증가) 변수가 거절 사유 후보다. 사용자에게 보이는 문구, 사유 개수, 변수-문구 매핑은 법무·소비자 보호 부서의 기준을 따른다.

```python
import numpy as np

lr = m_all.named_steps["logisticregression"]
sc = m_all.named_steps["standardscaler"]
Xh = hold[feats].to_numpy()
i = 0
z = sc.transform(Xh[[i]])[0]                     # 표준화 점수(= (x-평균)/표준편차)
contrib = lr.coef_[0] * z                       # 평균 고객 대비 로그오즈 기여도
logit_i = lr.intercept_[0] + (lr.coef_[0] * z).sum()
logit_mean = lr.intercept_[0]                   # 표준화 평균 고객의 로그오즈
assert abs((logit_i - logit_mean) - contrib.sum()) < 1e-12        # 기여도의 합 = 평균 고객과의 로그오즈 차이
top = [feats[j] for j in np.argsort(-contrib)[:3]]
print("위험을 가장 많이 올린 변수 후보:", top, np.round(np.sort(contrib)[::-1][:3], 3))
```

*상관된 변수*가 있으면 기여도가 나뉘어 사유가 불안정해진다. 사유 산출 방법은 고정하고 검증한다.

### 4-4. 공정성: 계산은 쉽고 해석은 어렵다

가장 단순한 지표는 집단별 승인율의 비율이다.

$$ \text{승인율 비율}=\frac{\text{승인율}_{\text{집단 A}}}{\text{승인율}_{\text{집단 B}}} $$

(미국 일부 지침의 "4/5 규칙"은 0.8 을 경고 신호로 쓰지만 *법적 판단 기준이 아니고 국가·분야마다 다르다*.) 지표는 여러 가지(승인율, 오류율, 보정)이며 **동시에 모두 만족시킬 수 없는 경우가 있다**(Kleinberg et al., 2016; Chouldechova, 2017).

```python
import numpy as np

def approval_ratio(approved, group):
    approved, group = np.asarray(approved), np.asarray(group)
    rates = {g: approved[group == g].mean() for g in np.unique(group)}
    lo, hi = min(rates.values()), max(rates.values())
    return rates, lo / hi

rates, ratio = approval_ratio([1] * 6 + [0] * 4 + [1] * 8 + [0] * 2, ["A"] * 10 + ["B"] * 10)
print(rates, round(ratio, 3))
assert abs(ratio - 0.75) < 1e-12

# 우리 합성 데이터의 '지역'별 승인율 (지역이 보호 속성의 프록시일 수 있음에 유의)
cut = np.quantile(m_all.predict_proba(dev[feats])[:, 1], 0.7)
approved = (m_all.predict_proba(hold[feats])[:, 1] < cut).astype(int)
rates_r, ratio_r = approval_ratio(approved, hold["region"].to_numpy())
print({k: round(float(v), 3) for k, v in rates_r.items()}, round(float(ratio_r), 3))
assert 0 < ratio_r <= 1
```

**프록시 변수**: 보호 속성(성별, 인종 등)을 쓰지 않아도, 지역·직업·소비 패턴 같은 변수가 그 대리 역할을 할 수 있다. 변수 후보를 심사할 때 *비즈니스 타당성*과 *차별적 영향*을 함께 본다. 어떤 속성을 사용·수집할 수 있는지는 법·규제가 정하며 이 문서는 법률 자문이 아니다.

### 4-5. 모형위험관리와 모형 카드

| 구성요소 | 내용 |
|---|---|
| 모형 인벤토리 | 사용 중 모형 목록, 목적, 책임자, 위험 등급 |
| 개발 문서 | 데이터, 정의, 방법, 한계, 대안 비교 |
| **독립 검증** | 개발자와 분리된 팀이 재현·도전(챌린저)·결과 확인 |
| 승인·변경관리 | 버전·시드·데이터 해시·변경 이력 |
| 모니터링 | 단계 10 의 PSI, 성능, 보정, 경보·조치 |
| 사용 한도 | 모형이 쓰이면 안 되는 상황(적용 범위 밖) 명시 |

**모형 카드**(Mitchell et al., 2019)에는 의도된 용도, 학습 데이터, 평가 지표(집단별 포함), 한계, 윤리적 고려를 쓴다. 이 프로그램의 `docs/model_card_template.md` 를 사용한다.

## 5. 가정

- 거절 추론은 *대체 가정*(거절자의 위험이 승인자와 비슷하게 이어진다 등)에 의존한다. 가정을 문서화한다.
- ECL 은 PD·LGD·EAD 각각의 정확성에 의존한다. 이 프로그램은 PD 중심이다.
- 사유 설명·공정성은 *해당 관할의 규정*을 따른다. 이 문서의 수치 기준은 예시다.

## 6. 한계와 오용

!!! warning "자주 하는 오용"
    - 승인자 데이터만으로 학습하고 *전체 신청자 성능*이라고 보고한다.
    - 거절 추론으로 *성능이 올랐다*고 보고한다(검증할 진짜 결과가 없다).
    - 보정 없는 점수를 PD 로 충당금에 쓴다.
    - SHAP/기여도 사유를 *인과적 조언*으로 고객에게 전달한다.
    - 4/5 규칙을 법적 safe harbor 로 오해한다.
    - 보호 속성을 제거했으니 공정하다고 결론짓는다(프록시).
    - 개발자가 자기 모형을 검증한다(독립성 부재).

## 7. 대안 및 비교

| 문제 | 접근 | 장점 | 약점 |
|---|---|---|---|
| 선택 편향 | 무작위 승인 실험 | 가장 신뢰 | 비용·위험 |
| | 파셀링/확대 | 저비용 | 가정 의존 |
| | 외부 성과 데이터 | 실제 결과 | 접근·동일 정의 문제 |
| 설명 | 선형 기여도 | 단순·일관 | 선형 모형 한정 |
| | SHAP | 일반적 | 상관·배경 의존 |
| 공정성 | 승인율 비교 | 단순 | 오류율/보정과 상충 가능 |

## 8. 실무 적용

- 모형 문서에 **승인/거절 정책과 학습 모집단**을 쓰고, 적용 모집단과의 차이(선택 편향)를 명시한다.
- 충당금 용도라면 *보정 검증 결과*와 *PD 기간(12개월/전체 만기)*을 명시한다.
- 불이익 사유 산출 로직은 고정·테스트하고 변경 시 승인 절차를 둔다.
- 개발자와 독립된 검증 담당자가 *재현*(같은 데이터·시드로 같은 결과)부터 확인한다.

## 9. 지식 확인

**Q1.** 승인자 데이터로만 학습한 모형의 평균 PD 가 전체 신청자에서 낮게 나오는 이유는?

<details markdown="1"><summary>정답 보기</summary>

과거 정책이 위험한 신청자를 거절해 관찰 표본이 *낮은 위험 구간으로 잘렸기* 때문이다(선택 편향). 특히 승인이 데이터에 없는 정보에 좌우되었다면 변수만으로는 보정되지 않는다.

</details>

**Q2.** PD 2%, LGD 40%, EAD 5,000 의 ECL 은?

<details markdown="1"><summary>정답 보기</summary>

$0.02\times0.4\times5000=40$.

</details>

**Q3.** 보호 속성을 모형에서 제외하면 공정성이 보장되는가?

<details markdown="1"><summary>정답 보기</summary>

아니다. 다른 변수가 프록시 역할을 해 집단 간 격차가 생길 수 있다. 결과(승인율·오류율)를 집단별로 점검한다.

</details>

**Q4.** 모형 개발자와 검증자를 분리하는 이유는?

<details markdown="1"><summary>정답 보기</summary>

개발자는 자기 가정을 의심하기 어렵다. 독립 검증이 오류·편향·과적합을 *도전적*으로 확인하고 책임을 분리한다.

</details>

**실습 과제**: 합성 데이터에서 과거 정책 승인자만으로 모형을 만들고 전체 신청자에서 성능·평균 PD 를 비교하라. 그 뒤 모형 카드 템플릿에 *의도된 용도, 학습 모집단, 한계(선택 편향), 모니터링 계획*을 작성하라.

## 10. 참고자료

- Federal Reserve / OCC. SR 11-7: *Guidance on Model Risk Management* (2011).
- Basel Committee. *Principles for the sound management of credit risk* (2000, 개정 2023 관련 문서).
- IASB. IFRS 9 *Financial Instruments* — 손상(ECL) 모형 관련 조항.
- Crook, J. & Banasik, J. (2004). Does reject inference really improve the performance of application scoring models? *Journal of Banking & Finance*, 28(4).
- Kleinberg, J., Mullainathan, S., Raghavan, M. (2016). Inherent trade-offs in the fair determination of risk scores.
- Chouldechova, A. (2017). Fair prediction with disparate impact. *Big Data*, 5(2).
- Mitchell, M. et al. (2019). Model cards for model reporting. *FAT\**.
- 금융위원회·금융감독원 등 국내 모형 검증·AI 가이드라인은 *최신 원문*을 확인할 것.
- 확인일 2026-10-18 (규제 문서는 개정될 수 있으므로 학습 시점에 원문 재확인)
