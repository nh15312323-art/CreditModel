# 단계 16. 종합 프로젝트: 처음부터 끝까지

> 앞 단계의 모든 선택(목표 정의, 정제, 분할, 비닝, 모형, 평가, 보정, 모니터링, 문서화)을 *하나의 재현 가능한 흐름*으로 묶는다. 합격 기준은 높은 AUROC 가 아니라 **각 결정을 근거와 함께 설명할 수 있는가**이다.

**연결 실습**: 노트북 `15_capstone` (`creditlab.capstone.run_capstone`). 이 페이지는 *전체 흐름의 축소판*과 *제출 기준*을 제공한다.

## 학습 목표

- 신용평가모형 개발의 전 과정을 순서대로 수행하고 각 단계의 산출물을 남긴다.
- 로지스틱 CSS 를 WOE → 로지스틱 → 점수 변환 → 등급화까지 직접 구현한다.
- 챌린저(부스팅)와 같은 조건에서 비교하고 유의성을 판단한다.
- 모형 문서(모형 카드)와 모니터링 계획을 작성한다.
- 결과의 한계를 *숨기지 않고* 보고한다.

## 1. 왜 생겼는가

개별 기법을 아는 것과 *전 과정을 일관되게 수행*하는 것은 다르다. 실무 사고의 대부분은 기법이 아니라 연결 지점에서 난다(정의와 데이터의 불일치, 분할 전 전처리, 평가 데이터 재사용, 문서와 코드의 불일치). 캡스톤은 그 연결을 직접 해 보는 단계다.

## 2. 왜 필요한가

- 모든 산출물이 서로 *추적 가능*해야 검증·감사가 가능하다.
- 같은 데이터·시드로 *다시 만들 수 있어야* 한다(재현성).
- 실무에서는 코드보다 문서와 의사결정 근거가 오래 남는다.

## 3. 무엇을 하는가: 산출물 지도

| 순서 | 단계 | 산출물 |
|---|---|---|
| 1 | 목표 정의 | 대상·관찰기간·부도 정의서, 데이터 사전 |
| 2 | 데이터 품질 | 품질 보고서, cleaning_log |
| 3 | 분할 | 시간 기준 분할 근거(개발/검증/OOT) |
| 4 | 변수 처리 | 비닝표(WOE/IV), 변수 선택 근거 |
| 5 | 모형 | 계수·부호 검토, 점수표 |
| 6 | 평가 | AUROC/Gini/KS/Brier/보정, 신뢰구간 |
| 7 | 챌린저 | 부스팅 비교, DeLong 결과 |
| 8 | 등급·보정 | 등급별 PD·부도율 |
| 9 | 운영 | PSI 기준, 경보·조치 계획 |
| 10 | 문서 | 모형 카드, 한계 목록 |

## 4. 어떻게 작동하는가

### 4-1. 축소판 전체 흐름 (재현 가능)

아래는 *교육용 최소 구현*이다. 실제 프로젝트에서는 각 단계를 별도 모듈·테스트로 분리한다. 모든 적합(비닝 경계, 대체값, 모형)은 **개발 구간에서만** 수행한다.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from creditlab.data import make_synthetic_credit_data
from creditlab import set_seed

set_seed(42)
df = make_synthetic_credit_data(n=15000, seed=81).sort_values("application_date").reset_index(drop=True)
TARGET = "default"
num_cols = ["age", "income_10k", "loan_amount_10k", "employment_years", "card_utilization", "num_delinq_12m", "inquiries_6m"]

# 1) 시간 기준 분할: 개발 60% / 검증 20% / OOT 20%
n = len(df)
dev, val, oot = df.iloc[: int(n * .6)], df.iloc[int(n * .6): int(n * .8)], df.iloc[int(n * .8):]
assert dev["application_date"].max() <= val["application_date"].min() <= oot["application_date"].min()

# 2) 비닝 경계와 WOE 는 개발 구간에서만 학습 (WOE = ln(%bad/%good), 높을수록 위험)
def fit_bins(s, y, q=5):
    edges = np.unique(np.quantile(s.dropna(), np.linspace(0, 1, q + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    b = pd.cut(s, edges)
    g = pd.DataFrame({"bin": b.astype(object).where(b.notna(), "MISSING"), "y": y})
    t = g.groupby("bin", observed=True)["y"].agg(bad="sum", n="count")
    t["good"] = t["n"] - t["bad"]
    adj = 0.5                                                  # 0 빈도 보호(스무딩)
    t["woe"] = np.log(((t["bad"] + adj) / (t["bad"].sum() + adj * len(t))) / ((t["good"] + adj) / (t["good"].sum() + adj * len(t))))
    return edges, t["woe"].to_dict()

def apply_bins(s, edges, woe):
    b = pd.cut(s, edges)
    lab = b.astype(object).where(b.notna(), "MISSING")
    return lab.map(woe).fillna(0.0).astype(float)       # 개발에 없던 구간은 중립(0)

bins = {c: fit_bins(dev[c], dev[TARGET]) for c in num_cols}
W = lambda d: pd.DataFrame({c: apply_bins(d[c], *bins[c]) for c in num_cols})
Wd, Wv, Wo = W(dev), W(val), W(oot)

# IV 로 변수 정렬 (개발 구간)
def iv(c):
    edges, woe = bins[c]
    b = pd.cut(dev[c], edges); lab = b.astype(object).where(b.notna(), "MISSING")
    tab = pd.crosstab(lab, dev[TARGET]); pb, pg = tab[1] / tab[1].sum(), tab[0] / tab[0].sum()
    return float(((pb - pg) * np.log((pb + 1e-9) / (pg + 1e-9))).sum())
ivs = pd.Series({c: iv(c) for c in num_cols}).sort_values(ascending=False)
print(ivs.round(3).to_dict())

# 3) 로지스틱: WOE 입력, 계수 부호 검토 (WOE 정의상 +, 높을수록 위험)
lr = LogisticRegression(C=np.inf, max_iter=2000).fit(Wd, dev[TARGET])
coefs = dict(zip(num_cols, np.round(lr.coef_[0], 3)))
print("계수", coefs)
assert (lr.coef_[0] > -0.2).all()           # 거의 모두 양(+) 이어야 정상 (심한 음수는 다중공선성 의심)

# 4) 점수 변환: Score = Offset + Factor * ln(odds_good), 600점@50:1, PDO 20
PDO, BASE, ODDS = 20, 600, 50
factor = PDO / np.log(2)
offset = BASE - factor * np.log(ODDS)
assert abs(factor - 28.8539) < 1e-4 and abs(offset - 487.123) < 1e-3
logit_pd = lambda W_: lr.intercept_[0] + W_.to_numpy() @ lr.coef_[0]
score = lambda W_: offset + factor * (-logit_pd(W_))               # ln(odds_good) = -logit(PD)
s_val, s_oot = score(Wv), score(Wo)
assert abs(np.corrcoef(s_val, -logit_pd(Wv))[0, 1] - 1) < 1e-12     # 점수와 로그오즈는 정확히 선형

# 5) 평가: 검증 / OOT, PD 보정 확인
for name, Wx, d in [("검증", Wv, val), ("OOT", Wo, oot)]:
    p = 1 / (1 + np.exp(-logit_pd(Wx)))
    a = roc_auc_score(d[TARGET], p)
    print(name, "AUROC", round(a, 3), "Gini", round(2 * a - 1, 3), "Brier", round(brier_score_loss(d[TARGET], p), 4),
          "평균PD", round(p.mean(), 3), "실제", round(d[TARGET].mean(), 3))
auc_oot = roc_auc_score(oot[TARGET], 1 / (1 + np.exp(-logit_pd(Wo))))
assert auc_oot > 0.65
assert abs((1 / (1 + np.exp(-logit_pd(Wo)))).mean() - oot[TARGET].mean()) < 0.03
```

### 4-2. 등급화와 단조성 확인

```python
import numpy as np
import pandas as pd

p_dev = 1 / (1 + np.exp(-logit_pd(Wd)))
cuts = np.quantile(p_dev, [0.2, 0.4, 0.6, 0.8])          # 개발 구간에서 5등급 경계 고정
def grade(p):
    return pd.cut(p, [-np.inf, *cuts, np.inf], labels=list("ABCDE"))     # A 가 가장 안전
tab = []
for name, Wx, d in [("개발", Wd, dev), ("검증", Wv, val), ("OOT", Wo, oot)]:
    g = grade(1 / (1 + np.exp(-logit_pd(Wx))))
    r = d.groupby(g.to_numpy(), observed=True)[TARGET].mean().reindex(list("ABCDE"))
    tab.append(r.rename(name))
tab = pd.concat(tab, axis=1)
print(tab.round(3))
assert tab["개발"].is_monotonic_increasing                     # 개발 구간에서 단조 증가
assert tab["OOT"].iloc[0] < tab["OOT"].iloc[-1]                # OOT 에서도 A < E
```

### 4-3. PSI 와 챌린저

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

def psi(e, a, bins=10, eps=1e-4):
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1))); edges[0], edges[-1] = -np.inf, np.inf
    pe = np.clip(np.histogram(e, edges)[0] / len(e), eps, None)
    pa = np.clip(np.histogram(a, edges)[0] / len(a), eps, None)
    return float(((pa - pe) * np.log(pa / pe)).sum())

s_dev = score(Wd)
print("점수 PSI 검증", round(psi(s_dev, s_val), 4), "OOT", round(psi(s_dev, s_oot), 4))
assert psi(s_dev, s_val) < 0.1

gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=50, random_state=0)
gb.fit(dev[num_cols].fillna(-1), dev[TARGET])
a_gb = roc_auc_score(oot[TARGET], gb.predict_proba(oot[num_cols].fillna(-1))[:, 1])
print("OOT AUROC  CSS", round(auc_oot, 4), " 챌린저", round(a_gb, 4))
assert 0.5 < a_gb < 1
```

여기서 챌린저가 크게 앞서지 않으면 *설명 가능한 로지스틱 CSS 를 채택하는 것이 합리적*일 수 있다. 앞선다면 단계 14 의 DeLong 검정과 신뢰구간, 단계 11 의 설명·단조성, 단계 15 의 운영 비용을 함께 따져 결정한다.

## 5. 제출 기준(루브릭)

| 영역 | 확인 항목 | 배점 |
|---|---|---|
| 정의·데이터 | 목표·관찰기간 명시, 품질 보고서, cleaning_log | 15 |
| 누수 방지 | 시간 분할, 개발 구간에서만 적합, 사후 정보 변수 점검 | 20 |
| 모형 | WOE/IV 근거, 계수 부호 검토, 점수 변환 수식 검산 | 20 |
| 평가 | 지표 정의(AR 포함), 신뢰구간, 보정, 등급별 부도율 | 20 |
| 운영·문서 | PSI 계획, 모형 카드, 한계·가정 명시 | 15 |
| 재현성 | 시드·버전·데이터 해시, 처음부터 다시 실행 가능 | 10 |

감점 사유: OOT/테스트를 보고 설정을 바꿈, 학습 성능만 보고, 한계 미기재, 합성데이터 결과를 실제 성능으로 주장.

## 6. 한계와 오용

!!! warning "캡스톤에서 자주 나오는 문제"
    - 비닝·결측 대체를 전체 데이터로 한 뒤 분할한다.
    - OOT 결과가 나쁘다고 설정을 바꾸고 같은 OOT 로 다시 보고한다.
    - 교육용 합성데이터의 AUROC 를 *현실 성능*처럼 말한다.
    - 점수 변환 수식의 방향(높을수록 안전)을 문서와 코드가 다르게 쓴다.
    - 계수 부호가 이상한데 이유를 확인하지 않는다.

## 7. 대안 및 비교

| 선택지 | 언제 |
|---|---|
| 규제 로지스틱 CSS | 설명·승인·감사가 중요할 때 (기본값) |
| 단조 제약 부스팅 | 변별력 개선이 유의하고 설명 체계가 있을 때 |
| 두 모형 병행(챌린저) | 의사결정 보조·모니터링 비교 |

## 8. 실무 적용: 모형 카드 체크리스트

- 목적, 적용 범위(상품·모집단), *적용하면 안 되는 범위*
- 데이터: 출처, 기간, 부도 정의, 품질 이슈와 처리
- 방법: 분할, 비닝, 변수 선택, 모형, 점수 변환 수식
- 성능: 개발/검증/OOT, 신뢰구간, 보정, 등급별 부도율
- 모니터링: PSI·성능 지표·경보 기준·조치 담당
- 한계와 가정, 알려진 편향(선택 편향 등)
- 재현 정보: 코드 버전, 시드, 데이터 해시, 라이브러리 버전

## 9. 지식 확인

**Q1.** 비닝 경계를 전체 데이터로 만든 뒤 시간 분할했다. 무엇이 문제인가?

<details markdown="1"><summary>정답 보기</summary>

검증/OOT 구간의 분포가 개발 단계에 새어 들어간 누수다. 경계·WOE·대체값은 *개발 구간*에서만 학습한다.

</details>

**Q2.** 점수 600 에서 PDO 20 이면 점수 620 의 odds(good:bad)는?

<details markdown="1"><summary>정답 보기</summary>

PDO 는 odds 가 *2배*가 되는 점수 증가량이다. 600점이 50:1 이면 620점은 100:1.

</details>

**Q3.** OOT AUROC 가 개발 구간보다 크게 낮다. 가능한 원인 세 가지는?

<details markdown="1"><summary>정답 보기</summary>

과적합, 모집단·분포 변화(PSI 확인), 거시환경 변화(관계 변화), 레이블 성숙도 부족, 누수 때문에 개발 성능이 낙관적이었을 가능성 등.

</details>

**Q4.** 챌린저가 OOT AUROC 를 0.004 높였다. 교체하는가?

<details markdown="1"><summary>정답 보기</summary>

DeLong 신뢰구간과 같은 승인율에서의 손실 변화, 설명·운영 비용을 본 뒤 결정한다. 차이가 구간 안이면 교체 근거가 약하다.

</details>

**최종 과제**: UCI 신용카드 데이터(가능하면) 또는 합성데이터로 위 10개 산출물을 모두 만들어 제출한다. 결과가 좋지 않아도 한계와 원인 분석을 정직하게 쓰면 높은 점수를 받을 수 있다.

## 10. 참고자료

- Siddiqi, N. *Credit Risk Scorecards* (Wiley, 2006).
- Thomas, L. *Consumer Credit Models* (Oxford, 2009).
- Baesens, B. et al. (2003). Benchmarking state-of-the-art classification algorithms for credit scoring. *JORS*, 54.
- Lessmann, S. et al. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: an update. *EJOR*, 247.
- Yeh, I.-C. & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2).
- UCI 데이터셋 페이지(라이선스 CC BY 4.0, 확인일 2026-10-18).
- 확인일 2026-10-18
