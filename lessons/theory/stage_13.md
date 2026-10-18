# 단계 13. GAN과 합성데이터

> 합성데이터는 "진짜 데이터의 대용품"이 아니라 "진짜 데이터의 *통계적 그림자*"다. 그림자가 얼마나 닮았는지(충실도), 그림자로 학습해도 현실에서 통하는지(유용성), 원본 개인을 드러내지 않는지(프라이버시)를 *각각* 확인해야 한다.

**연결 실습**: 노트북 `12_synthetic_data` (`creditlab.synthesis`: 가우시안 코퓰라 합성기, 충실도·TSTR·DCR). 이 프로그램의 `creditlab.data.synthetic` 은 *규칙으로 만든 교육용 가상 데이터*이고, 이 단계의 합성데이터는 *주어진 데이터를 학습해서 만든* 데이터를 다룬다. 둘은 다르다. GAN 계열(CTGAN 등)은 설치되지 않아 실행하지 못했고, 코퓰라로 같은 품질 점검 절차를 연습한다.

## 학습 목표

- 합성데이터를 쓰는 이유(프라이버시, 희소 부도, 데이터 공유)와 한계를 설명한다.
- 가우시안 코퓰러 방식의 합성기를 직접 구현한다.
- GAN(CTGAN 계열)의 구조와 표 형태 데이터에서의 어려움을 이해한다.
- 충실도·유용성(TSTR)·프라이버시 세 축으로 합성데이터를 평가한다.
- **소수 클래스 오버샘플링의 데이터 누수**를 재현해 피한다.

## 1. 왜 생겼는가

신용 데이터는 개인정보라서 공유·외부 위탁·교육 사용이 어렵다. 부도는 드물어 학습 표본이 부족하다. 그래서 *원본의 통계적 성질을 유지하는 인공 데이터*를 만들려는 시도가 생겼다. 초기에는 SMOTE(Chawla et al., 2002) 같은 보간 기반 오버샘플링, 통계적 모형(코퓰러), 이후 GAN(Goodfellow et al., 2014)과 표 형태 전용 CTGAN(Xu et al., 2019)이 나왔다.

## 2. 왜 필요한가

- 외부 연구자·교육·테스트 환경에 *실데이터 없이* 줄 수 있는 후보다.
- 희소 사건의 학습 보조(단, 효과는 데이터마다 다르고 **보장되지 않음**).
- 시나리오·스트레스 데이터를 만든다.

## 3. 무엇을 하는가

| 방식 | 핵심 | 비고 |
|---|---|---|
| 단순 오버샘플링 | 소수 클래스 행 복제 | 정보 증가 없음, 누수 위험 |
| SMOTE | 이웃 사이 보간 | 표형 혼합 변수에 약함 |
| 가우시안 코퓰러 | 주변분포 + 상관구조 | 단순·설명 쉬움, 비선형 의존 약함 |
| CTGAN/TVAE | 신경망으로 결합분포 학습 | 튜닝·평가 어려움, 프라이버시 별도 검증 |

## 4. 어떻게 작동하는가

### 4-1. 가우시안 코퓰러 합성기 (직접 구현)

1. 각 수치 열을 순위 → 표준정규 점수로 바꾼다.
2. 변환된 열들의 상관행렬을 추정한다.
3. 다변량 정규분포에서 표본을 뽑는다.
4. 각 열을 원래 분포의 경험적 분위수로 되돌린다.

주변분포(각 열의 모양)와 *선형적* 상관을 보존한다.

```python
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from creditlab.data import make_synthetic_credit_data

def fit_copula(df):
    """수치 열 DataFrame 에서 (상관행렬, 정렬된 원자료) 를 추정."""
    z = np.column_stack([norm.ppf(rankdata(df[c]) / (len(df) + 1)) for c in df.columns])
    return np.corrcoef(z, rowvar=False), {c: np.sort(df[c].to_numpy()) for c in df.columns}

def sample_copula(corr, sorted_cols, n, seed):
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(np.zeros(corr.shape[0]), corr, size=n)
    u = norm.cdf(z)
    out = {}
    for j, (c, srt) in enumerate(sorted_cols.items()):
        out[c] = np.quantile(srt, u[:, j])
    return pd.DataFrame(out)

real = make_synthetic_credit_data(n=8000, seed=51).dropna()
cols = ["age", "income_10k", "loan_amount_10k", "employment_years", "card_utilization", "num_delinq_12m", "inquiries_6m", "default"]
real = real[cols].reset_index(drop=True)
r_train, r_hold = real.iloc[:5000], real.iloc[5000:]
corr, srt = fit_copula(r_train)
synth = sample_copula(corr, srt, 5000, seed=0)
synth["default"] = (synth["default"] > 0.5).astype(int)      # 이진 열은 반올림해 복원(단순화)

print(round(r_train["default"].mean(), 3), round(synth["default"].mean(), 3))
print(r_train[["age", "income_10k", "card_utilization"]].mean().round(2).to_dict())
print(synth[["age", "income_10k", "card_utilization"]].mean().round(2).to_dict())
assert abs(r_train["age"].mean() - synth["age"].mean()) < 1.0               # 주변분포(평균) 보존
diff = np.abs(np.corrcoef(r_train.drop(columns="default"), rowvar=False) - np.corrcoef(synth.drop(columns="default"), rowvar=False)).max()
print("상관행렬 최대 차이", round(float(diff), 3))
assert diff < 0.1
```

### 4-2. 충실도: 닮았는가

- 열별: 평균·분위수·KS 통계량, 범주 비율
- 열 사이: 상관행렬 차이, 목표변수와의 관계(예: 변수 구간별 부도율)
- **목표변수와의 관계가 약해지기 쉽다**: 코퓰러가 비선형·상호작용을 놓치면 $P(Y\mid X)$ 가 흐려진다.

### 4-3. 유용성: TSTR

**TSTR(Train on Synthetic, Test on Real)**: 합성데이터로 모형을 학습하고 *진짜 보류 데이터*로 평가해 TRTR(진짜로 학습·진짜로 평가)과 비교한다.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

feat = [c for c in cols if c != "default"]
def fit_eval(train_df):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(train_df[feat], train_df["default"])
    return roc_auc_score(r_hold["default"], m.predict_proba(r_hold[feat])[:, 1])

trtr, tstr = fit_eval(r_train), fit_eval(synth)
print("TRTR", round(trtr, 3), "TSTR", round(tstr, 3))
assert tstr > 0.55                     # 합성데이터로도 순위 정보는 일부 전달됨
assert tstr < trtr + 0.03              # 합성이 실데이터보다 눈에 띄게 좋을 수는 없다
```

TSTR 이 TRTR 에 크게 못 미치면 합성기가 $P(Y\mid X)$ 를 못 잡은 것이다.

### 4-4. 프라이버시: 닮음과 복제는 다르다

합성기가 학습 행을 **외워서 내보내면** 개인정보가 노출된다. 가장 가까운 원본과의 거리(DCR, Distance to Closest Record)로 확인한다. 합성 행이 원본과 *정확히 같으면* 복제다.

```python
import numpy as np
from scipy.spatial import cKDTree

feat_cols = feat
scale = r_train[feat_cols].std().replace(0, 1)
tree = cKDTree((r_train[feat_cols] / scale).to_numpy())

def dcr(df):
    d, _ = tree.query((df[feat_cols] / scale).to_numpy(), k=1)
    return d

copula_dcr = dcr(synth)
naive = r_train.sample(1000, random_state=0)          # '합성'이랍시고 원본 행을 그대로 재표집
naive_dcr = dcr(naive)
holdout_dcr = dcr(r_hold)
print("복제기 최소거리", float(naive_dcr.min()), "| 코퓰러 최소거리", round(float(copula_dcr.min()), 4),
      "| 보류데이터 중앙값", round(float(np.median(holdout_dcr)), 3))
assert naive_dcr.max() == 0                     # 복제는 거리 0 → 즉시 탐지
assert copula_dcr.min() > 0
```

DCR>0 이라고 안전한 것은 아니다. 희귀한 극단 행(소득이 매우 높은 사람 등)은 가까운 합성 행으로 *재식별될 수 있고*, 합성 모형이 특정 개인의 존재 여부를 드러내는지(멤버십 추론) 평가해야 한다. 형식적 보장은 **차등 프라이버시**(DP) 학습 같은 별도 기법이 필요하다. **합성데이터 ≠ 자동 익명화.**

### 4-5. GAN 은 어떻게 학습하나

```python
# [설명용] 개념 의사코드 — 실제 학습은 CTGAN 등 라이브러리 사용
# G: 잡음 z -> 합성 행,  D: 행 -> '진짜일 확률'
for step in range(n_steps):
    real_batch = sample(real_data)
    fake_batch = G(z=random_noise())
    # 판별자: 진짜는 1, 가짜는 0 으로 분류하도록 학습
    loss_D = bce(D(real_batch), 1) + bce(D(fake_batch), 0)
    # 생성자: 판별자를 속이도록 학습
    loss_G = bce(D(fake_batch), 1)
```

표 형태 데이터에서의 어려움: 연속·범주 혼합, 극단 비대칭 분포(소득), 불균형 범주(부도). CTGAN 은 *모드 특이 정규화*와 *조건부 샘플링*으로 이를 다룬다. 학습이 **불안정**하고 모드 붕괴(일부 패턴만 생성)가 일어날 수 있어 평가가 필수다.

### 4-6. 오버샘플링의 누수 (가장 흔한 사고)

분할 **전에** 소수 클래스를 복제·보간하면 같은 행의 사본이 학습과 검증에 *동시에* 들어가 성능이 거짓으로 높아진다.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

X, y = real[feat], real["default"]
# 올바른 방법: 먼저 분할, 학습 구간에서만 증식 (여기선 증식 없이 기준선)
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
honest = roc_auc_score(yb, RandomForestClassifier(200, min_samples_leaf=1, random_state=0).fit(Xa, ya).predict_proba(Xb)[:, 1])

# 잘못된 방법: 부도 행을 5배로 복제한 뒤 무작위 분할
pos = real[real["default"] == 1]
dup = pd.concat([real] + [pos] * 4, ignore_index=True)
Xc, Xd, yc, yd = train_test_split(dup[feat], dup["default"], test_size=0.3, random_state=0, stratify=dup["default"])
leaky = roc_auc_score(yd, RandomForestClassifier(200, min_samples_leaf=1, random_state=0).fit(Xc, yc).predict_proba(Xd)[:, 1])
print("정직한 AUROC", round(honest, 3), "누수된 AUROC", round(leaky, 3))
assert leaky > honest + 0.1
```

## 5. 가정

- 합성기가 학습한 분포가 *미래*에도 유효하다. 합성데이터는 *원본에 있는 정보만* 담으며 **새 정보를 만들지 못한다**.
- 평가용 진짜 보류 데이터가 있다.
- 합성 과정에서 학습 데이터의 개인이 드러나지 않는다는 *별도 증거*가 있다.

## 6. 한계와 오용

!!! warning "자주 하는 오용"
    - 합성데이터를 "익명 데이터"로 취급한다.
    - 분할 전에 오버샘플링하거나 합성 데이터를 검증/테스트에 섞는다.
    - 합성데이터로 *성능을 개선했다*고 보고하면서 평가는 합성 포함 데이터로 한다.
    - 합성데이터에서 발견한 변수 효과를 현실의 *인과*로 해석한다.
    - 부도 레이블이 합성기에 의해 약화되었는데도 TSTR 을 확인하지 않는다.
    - 교육용 규칙 기반 가상 데이터(이 프로그램)와 학습형 합성데이터를 혼동한다.

## 7. 대안 및 비교

| 목적 | 합성데이터 | 대안 |
|---|---|---|
| 프라이버시 | 후보(별도 검증 필요) | 가명처리, 접근통제, 연합학습, DP |
| 희소 부도 | 보조 가능 | 가중치 학습, 비용 민감 학습, 더 긴 관찰기간 |
| 교육/테스트 | 매우 적합 | 마스킹된 샘플 |
| 시나리오 | 적합 | 규칙 기반 시뮬레이션 |

## 8. 실무 적용

- 합성데이터는 *개발 보조*나 *교육·공유*에 한정하고, **최종 모형 검증은 항상 실데이터**로 한다.
- 데이터 레지스트리에 합성 여부·생성 방법·시드·평가(충실도/TSTR/프라이버시)를 기록한다.
- 합성기 학습에 쓴 원본 데이터의 사용 권한과 목적 제한을 확인한다(개인정보 보호 관련 법규·내부 지침).

## 9. 지식 확인

**Q1.** 합성데이터는 원본의 개인을 보호하는가?

<details markdown="1"><summary>정답 보기</summary>

자동으로 보호하지 않는다. 복제·재식별·멤버십 추론 위험을 DCR 등으로 평가하고, 필요하면 DP 같은 형식적 기법을 쓴다.

</details>

**Q2.** 소수 클래스를 복제한 뒤 무작위 분할했더니 AUROC 가 0.99 다. 무엇이 문제인가?

<details markdown="1"><summary>정답 보기</summary>

같은 행의 사본이 학습·검증에 모두 들어간 누수다. **분할 후, 학습 구간에서만** 증식한다.

</details>

**Q3.** TSTR 이 TRTR 보다 크게 낮다. 의미는?

<details markdown="1"><summary>정답 보기</summary>

합성기가 변수와 부도 사이의 관계 $P(Y\mid X)$ 를 충분히 보존하지 못했다는 신호다.

</details>

**Q4.** 합성데이터가 새로운 정보를 만들어 모형 성능을 올려 줄 수 있는가?

<details markdown="1"><summary>정답 보기</summary>

원칙적으로 원본에 없는 정보는 만들지 못한다. 정규화·증강 효과로 일부 개선될 수는 있으나 보장되지 않으며 반드시 실데이터 보류 세트로 검증한다.

</details>

**실습 과제**: 코퓰러 합성기로 합성데이터를 만들고 충실도(평균·상관), TSTR, DCR 을 표로 보고하라. 그 뒤 일부러 원본 행을 재표집한 '나쁜 합성기'의 DCR 을 비교하고, 데이터 레지스트리 항목(`synthetic: true`, 생성법, 시드, 평가 결과)을 작성하라.

## 10. 참고자료

- Goodfellow, I. et al. (2014). Generative adversarial nets. *NeurIPS*.
- Chawla, N. et al. (2002). SMOTE: synthetic minority over-sampling technique. *JAIR*, 16.
- Xu, L. et al. (2019). Modeling tabular data using conditional GAN. *NeurIPS*.
- Nelsen, R. *An Introduction to Copulas* (Springer, 2006).
- Jordon, J. et al. (2022). Synthetic data — what, why and how? (Royal Society 보고서 관련 논문).
- Dwork, C. & Roth, A. (2014). The algorithmic foundations of differential privacy.
- 확인일 2026-10-18 (서지 정보는 학습 시점에 원문으로 재확인할 것)
