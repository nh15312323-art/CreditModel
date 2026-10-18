from nbkit import SETUP_CELL, code, md

TITLE = "레슨 6-1. 합성데이터: 충실도·유용성·프라이버시"


def cells():
    return [
        md(
            """
# 레슨 6-1. 합성데이터: 충실도·유용성·프라이버시

**예상 시간** 90분 · **선행** 레슨 5-2 · **이론 페이지** `docs/theory/stage_13.html`

## 학습 목표
1. 가우시안 코퓰러로 **실제 표본을 학습한 합성데이터**를 만들고, 무엇이 보존되고 무엇이 약해지는지 본다.
2. 합성데이터를 **충실도(닮았나) · 유용성(TSTR) · 프라이버시(DCR)** 세 축으로 따로 평가한다.
3. **DCR > 0 이 안전을 뜻하지 않는다**는 것을 실험으로 확인한다.
4. **분할 전 오버샘플링의 누수**를 숫자로 재현한다.

> 이 프로그램의 `make_synthetic_credit_data` 는 *규칙으로 만든 가상 데이터*이고, 이 레슨의 합성기는 *주어진 표본을 학습해 흉내 내는* 방법입니다. 둘은 다릅니다.
> 코퓰러는 **교육용**입니다. 합성데이터는 자동 익명화가 아니며, CTGAN 같은 GAN 은 설치되지 않아 **실행하지 못했습니다.**
"""
        ),
        SETUP_CELL,
        md("## 1. 준비: 수치 열만, 결측 제거, 시간순 분할"),
        code(
            '''
from creditlab.data import make_synthetic_credit_data
from creditlab.synthesis import GaussianCopulaSynthesizer

raw = make_synthetic_credit_data(n=8000, seed=51)
COLS = ["age", "income_10k", "loan_amount_10k", "employment_years", "card_utilization", "num_delinq_12m", "inquiries_6m", "default"]
FEATS = COLS[:-1]
real = raw.dropna(subset=COLS).sort_values("application_date")[COLS].reset_index(drop=True)
print(f"원본 {len(raw):,}행 → 결측 행 제거 후 {len(real):,}행 (제거 {len(raw) - len(real):,}행; 제거 방식은 반드시 기록)")
r_train, r_hold = real.iloc[:5000], real.iloc[5000:]          # 합성기는 학습 구간만 본다
print(f"합성기 학습 {len(r_train):,}행 | 평가용 실제 보류 {len(r_hold):,}행")
'''
        ),
        md("## 2. 가우시안 코퓰러 합성기"),
        code("pred_corr = None  # ← 코퓰러가 가장 잘 보존하는 것은? (각 열의 분포 모양 / 변수 쌍의 선형 상관 / 비선형 관계 / 목표변수와의 관계)"),
        code(
            '''
synth = GaussianCopulaSynthesizer(seed=0).fit(r_train).sample(5000)
print("합성 5,000행 샘플:")
print(synth.head(4).round(2).to_string(index=False))
print("\\n이산 열은 관측된 값만 내보냅니다:", sorted(synth["num_delinq_12m"].unique()))
if pred_corr is not None:
    print("내 예측:", pred_corr)
'''
        ),
        md("## 3. 충실도 — 열별, 열 사이, 목표변수와의 관계"),
        code(
            '''
from creditlab.synthesis import correlation_gap, fidelity_table, target_relationship

ft = fidelity_table(r_train, synth)
print(ft[["column", "real_mean", "synth_mean", "real_std", "synth_std", "ks"]].round(3).to_string(index=False))
cg = correlation_gap(r_train[FEATS], synth[FEATS])
print(f"\\n상관행렬 차이: 최대 {cg['max_abs']:.3f}, 평균 {cg['mean_abs']:.3f}")
print("\\n카드 사용률 구간별 부도율 — 실제 vs 합성:")
print(target_relationship(r_train, synth, "card_utilization", "default").round(3).to_string(index=False))
'''
        ),
        md(
            """
- 열별 분포(평균·표준편차·KS)와 변수 쌍의 선형 상관은 잘 보존됩니다.
- **목표변수와의 관계**는 약해지기 쉽습니다: 코퓰러는 비선형·상호작용을 모르므로 위험이 높은 구간의 부도율이 낮아지고(또는 낮은 구간이 높아져) *기울기가 완만해지는* 경향이 있습니다. 아래 TSTR 로 이 손실이 모형 학습에 미치는 영향을 봅니다.
"""
        ),
        md("## 4. 유용성 — TSTR"),
        code(
            '''
from creditlab.synthesis import tstr

res = tstr(r_train, synth, r_hold, FEATS, "default")
print(f"TRTR (실로 학습, 실로 평가) AUROC = {res['trtr']:.4f}")
print(f"TSTR (합성으로 학습, 실로 평가) AUROC = {res['tstr']:.4f}   격차 = {res['gap']:+.4f}")

# 합성 표본 크기를 바꿔 보기
rows = []
for n in [500, 2000, 5000, 20000]:
    s = GaussianCopulaSynthesizer(seed=1).fit(r_train).sample(n)
    rows.append({"합성 행 수": n, "TSTR AUROC": tstr(r_train, s, r_hold, FEATS, "default")["tstr"]})
print()
print(pd.DataFrame(rows).round(4).to_string(index=False))
'''
        ),
        md("TSTR 이 TRTR 에 가까우면 *선형 모형 수준의 순위 정보*가 합성에 남아 있다는 뜻입니다. 합성 표본 크기를 키워도 TSTR 이 **단조롭게 좋아지지는 않습니다**(표집 잡음이 있고, 합성기가 놓친 정보는 크기로 메워지지 않습니다). 합성이 실데이터보다 *눈에 띄게* 좋아 보이면 평가 구성을 의심합니다. 비선형 모형(부스팅)으로는 격차가 더 벌어질 수 있습니다(과제 2)."),
        md("## 5. 프라이버시 — 닮음과 복제는 다르다"),
        code("pred_dcr = None  # ← 원본 행을 그대로 재표집한 '합성'과, 원본에 아주 작은 잡음만 더한 '합성'. DCR 최솟값이 0 인 것은? DCR>0 이면 안전할까요?"),
        code(
            '''
from creditlab.synthesis import dcr_summary

rng = np.random.default_rng(SEED)
naive = r_train.sample(1500, random_state=0)                                       # 복제
scale = r_train[FEATS].std()
jitter = naive.copy()
jitter[FEATS] = jitter[FEATS] + rng.normal(0, 0.01, (len(jitter), len(FEATS))) * scale.to_numpy()   # 복제 + 아주 작은 잡음
cop = synth.iloc[:1500]

rows = {}
for name, d in [("원본 재표집(복제)", naive), ("복제+작은 잡음", jitter), ("코퓰러", cop)]:
    s = dcr_summary(r_train, d, r_hold, FEATS)
    rows[name] = {"DCR 최소": s["synth_min"], "DCR 5%": s["synth_p5"], "완전 복제 건수": s["exact_copies"],
                  "보류 DCR 5%": s["holdout_p5"], "보류보다 가까운 비율": s["share_closer_than_holdout_p5"]}
print(pd.DataFrame(rows).T.round(4).to_string())
if pred_dcr is not None:
    print("내 예측:", pred_dcr)
'''
        ),
        md(
            """
- **복제**는 DCR 이 0 이라 즉시 드러납니다.
- **복제+작은 잡음**은 DCR 최솟값이 0 이 *아니어서* '복제 없음' 검사를 통과하지만, 합성→학습 거리가 **같은 분포의 보류 데이터→학습 거리보다 훨씬 가깝습니다**(비율 열). 즉 "DCR > 0" 만으로는 안전을 말할 수 없고, **보류 데이터와의 상대 비교**가 더 믿을 만합니다.
- 코퓰러는 이 검사에서 보류 데이터와 비슷한 수준입니다. 그래도 *희귀 극단 행의 재식별*, *멤버십 추론*은 따로 평가해야 하고, 형식적 보장은 차등 프라이버시 같은 별도 기법이 필요합니다. **합성데이터 ≠ 자동 익명화.**
"""
        ),
        md("## 6. 오버샘플링의 누수 (가장 흔한 사고)"),
        code(
            '''
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

X, y = real[FEATS], real["default"]
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)        # 올바름: 먼저 분할
honest = roc_auc_score(yb, RandomForestClassifier(200, random_state=0).fit(Xa, ya).predict_proba(Xb)[:, 1])

pos = real[real["default"] == 1]
dup = pd.concat([real] + [pos] * 4, ignore_index=True)                                     # 잘못: 분할 전에 부도 행을 5배로 복제
Xc, Xd, yc, yd = train_test_split(dup[FEATS], dup["default"], test_size=0.3, random_state=0, stratify=dup["default"])
leaky = roc_auc_score(yd, RandomForestClassifier(200, random_state=0).fit(Xc, yc).predict_proba(Xd)[:, 1])
print(f"정직한 AUROC = {honest:.3f} | 누수된 AUROC = {leaky:.3f}  (같은 데이터, 같은 모형)")
share = lambda tr_, va_: va_.merge(tr_.drop_duplicates(), on=FEATS, how="left", indicator=True)["_merge"].eq("both").mean()
print(f"검증 행 중 '똑같은 행이 학습에도 있는' 비율 — 올바른 분할 {share(Xa, Xb):.1%} | 복제 후 분할 {share(Xc, Xd):.1%}  → 후자는 암기한 것을 맞히는 셈")
'''
        ),
        md("분할 **전에** 소수 클래스를 복제·보간하면 같은 행의 사본이 학습과 검증에 *동시에* 들어가 성능이 거짓으로 높아집니다. 증식은 **학습 구간 안에서만**, 평가는 증식하지 않은 구간으로 합니다. 합성데이터로 학습 데이터를 늘릴 때도 마찬가지입니다(TSTR 처럼 *실제 보류*로 평가)."),
        md("## 7. ✏️ 실습 — KS 통계량 직접 계산"),
        code(
            '''
# [exercise]
a = r_train["card_utilization"].to_numpy()
b = synth["card_utilization"].to_numpy()

ks_mine = None        # ← 두 표본의 누적분포(ECDF) 차이의 최댓값
worst_col = None      # ← fidelity_table 에서 KS 가 가장 큰 열 이름(문자열)
# 힌트: 합친 값 grid = np.sort(np.concatenate([a, b])) 에서 np.searchsorted(np.sort(a), grid, side="right") / len(a) 등으로 ECDF

if ks_mine is not None and worst_col is not None:
    from scipy.stats import ks_2samp
    true_ks = float(ks_2samp(a, b).statistic)
    true_worst = ft.loc[ft["ks"].idxmax(), "column"]
    print(f"KS {ks_mine:.4f} (정답 {true_ks:.4f}) :", "맞음" if abs(ks_mine - true_ks) < 1e-9 else "다시 계산")
    print("가장 큰 열:", worst_col, ":", "맞음" if worst_col == true_worst else f"다시 확인 (정답 {true_worst})")
else:
    print("ks_mine, worst_col 을 채우고 다시 실행하세요.")
'''
        ),
        md(
            """
<details>
<summary>정답 예시 (먼저 직접 해 본 뒤 펼치세요)</summary>

```python
grid = np.sort(np.concatenate([a, b]))
Fa = np.searchsorted(np.sort(a), grid, side="right") / len(a)
Fb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
ks_mine = float(np.abs(Fa - Fb).max())
worst_col = ft.loc[ft["ks"].idxmax(), "column"]
```
</details>
"""
        ),
        md("## 8. 퀴즈"),
        code(
            '''
from creditlab.learning import Question, check, grade, show

QUIZ = [
    Question("Q1", "세 축", "합성데이터 평가의 세 축은?",
             {"A": "정확도·재현율·F1", "B": "충실도(닮음)·유용성(TSTR)·프라이버시(복제/재식별)", "C": "AUROC·KS·PSI", "D": "속도·크기·비용"},
             "B", "닮았는지, 학습에 쓸 수 있는지, 원본 개인이 드러나지 않는지를 각각 확인해야 합니다.", ("세 가지는 서로 다른 질문입니다.",)),
    Question("Q2", "TSTR", "TSTR 에서 평가는 어떤 데이터로 하나요?",
             {"A": "합성 데이터", "B": "실제 보류 데이터", "C": "학습에 쓴 합성 데이터의 일부", "D": "아무거나"},
             "B", "합성으로 학습하고 *진짜* 보류로 평가해야 현실에서 통하는지 알 수 있습니다.", ("현실에서 쓸 모형입니다.",)),
    Question("Q3", "코퓰러의 한계", "코퓰러 합성에서 가장 약해지기 쉬운 것은?",
             {"A": "각 열의 평균", "B": "변수 쌍의 선형 상관", "C": "비선형 관계·상호작용과 그에 따른 목표변수와의 관계", "D": "열 이름"},
             "C", "코퓰러는 주변분포와 선형 상관만 보존하므로 P(Y|X) 가 흐려지기 쉽습니다.", ("무엇을 모델링하지 않았나요?",)),
    Question("Q4", "DCR", "합성 행의 DCR 최솟값이 0 보다 큽니다. 올바른 해석은?",
             {"A": "프라이버시가 보장된다", "B": "완전 복제는 없지만, 원본과 매우 가깝거나(보류 대비) 희귀 행이 재식별될 수 있어 추가 평가가 필요하다",
              "C": "합성 품질이 낮다", "D": "DCR 은 무의미하다"},
             "B", "DCR>0 은 안전의 증거가 아닙니다. 보류 데이터와의 상대 거리, 재식별·멤버십 평가가 필요합니다.", ("복제+작은 잡음을 떠올려 보세요.",)),
    Question("Q5", "누수", "부도 행을 5배 복제한 뒤 무작위로 학습/검증을 나눴더니 AUROC 가 크게 올랐습니다. 원인은?",
             {"A": "모형이 좋아졌다", "B": "같은 행의 사본이 학습과 검증에 동시에 들어가 암기한 것을 맞힌다(누수)", "C": "표본이 커져서", "D": "부도율이 낮아져서"},
             "B", "증식은 분할 후 학습 구간에서만 해야 합니다.", ("검증 행의 사본이 학습에도 있나요?",)),
]

answers = {"Q1": None, "Q2": None, "Q3": None, "Q4": None, "Q5": None}  # ← "A"~"D"

for q in QUIZ:
    show(q)
    print()
'''
        ),
        code(
            '''
for q in QUIZ:
    check(q, answers[q.qid])
if all(v is not None for v in answers.values()):
    r = grade(QUIZ, answers)
    print("\\n점수:", r["score"], "/", r["total"], "| 통과(75% 기준):", r["passed"], "| 다시 볼 개념:", r["concepts_to_review"])
'''
        ),
        md(
            """
## 9. 과제와 정리
**과제**
1. 합성 표본 크기를 500·5,000·50,000 으로 바꿔 열별 KS 와 상관 차이가 어떻게 변하는지 표로 만드세요. 크기를 키우면 *합성기의 한계*가 사라지나요?
2. TSTR 을 **부스팅**(레슨 5-1 의 `fit_gbm`)으로 다시 측정해 로지스틱 때의 격차와 비교하고, 코퓰러가 놓친 비선형 정보가 격차로 나타나는지 쓰세요.
3. `jitter` 의 잡음 크기를 0.01 → 0.1 → 0.5 로 키우며 *보류보다 가까운 비율*이 어떻게 변하는지 보고, 프라이버시와 유용성의 맞교환을 설명하세요.

**체크포인트**: 퀴즈 4/5 이상 · KS 실습 통과 · '복제+잡음'이 DCR 최솟값 검사를 통과하지만 상대 비교에서는 걸리는 이유를 설명할 수 있음

**다음 레슨**: 레슨 6-2 — 가설검정: 순열검정, DeLong, 다중검정.

**참고자료**
- Xu, L. 외, "Modeling Tabular Data using Conditional GAN", NeurIPS (2019).
- Goodfellow, I. 외, "Generative Adversarial Nets", NeurIPS (2014).
- Nelsen, R., *An Introduction to Copulas* (Springer, 2006).
- 이론: `docs/theory/stage_13.html`
"""
        ),
    ]
