from nbkit import SETUP_CELL, code, md

TITLE = "레슨 6-2. 가설검정·DeLong·다중검정"


def cells():
    return [
        md(
            """
# 레슨 6-2. 가설검정·DeLong·다중검정

**예상 시간** 90분 · **선행** 레슨 6-1 · **이론 페이지** `docs/theory/stage_14.html`

## 학습 목표
1. **순열검정**으로 p-value 를 직접 만들고 t 검정과 비교한다.
2. 같은 고객으로 평가한 두 모형의 AUROC 차이를 **DeLong 검정**과 신뢰구간으로 보고한다.
3. 검정을 많이 하면 우연한 유의 결과가 늘어남을 보고, **Bonferroni·Holm·BH** 로 보정한다.
4. **통계적 유의성과 실무적 유의성**을 구분한다.

> p-value 는 *귀무가설이 참일 때* 관측만큼 극단적인 결과가 나올 확률입니다. **가설이 참일 확률이 아니며**, 효과의 크기도 아닙니다.
> 합성데이터 결과는 실제 성능이 아닙니다.
"""
        ),
        SETUP_CELL,
        md("## 1. 순열검정으로 p-value 만들기"),
        code("pred_p = None  # ← 진짜 평균 차이가 0.5(표준편차 1), 집단당 60명이면 p-value 는 0.05 보다 작을까요?"),
        code(
            '''
from scipy import stats
from creditlab.stats import permutation_test

rng = np.random.default_rng(SEED)
a = rng.normal(0.0, 1.0, 60)
b = rng.normal(0.5, 1.0, 60)                 # 진짜 평균 차이 0.5
r = permutation_test(a, b, n_perm=5000, seed=SEED)
p_t = stats.ttest_ind(b, a, equal_var=False).pvalue
print(f"관측 평균 차이 {r['observed_diff']:.3f} | 순열 p = {r['p_value']:.4f} | Welch t p = {p_t:.4f}")

# 진짜 차이가 0 일 때 p-value 는 어떤 분포를 갖나? (귀무가설이 참인 200번의 반복)
ps = [stats.ttest_ind(rng.normal(size=60), rng.normal(size=60), equal_var=False).pvalue for _ in range(2000)]
print(f"귀무가설이 참일 때 p<0.05 인 비율 = {np.mean(np.array(ps) < 0.05):.3f} (≈ 0.05 가 되도록 설계됨)")

# 진짜 차이가 0.5 로 '있을' 때, 집단당 60명이면 얼마나 자주 유의하게 나오나? (검정력)
pw = [stats.ttest_ind(rng.normal(0.5, 1, 60), rng.normal(0, 1, 60), equal_var=False).pvalue for _ in range(2000)]
print(f"진짜 차이가 0.5 일 때 p<0.05 인 비율(검정력) = {np.mean(np.array(pw) < 0.05):.3f}")
if pred_p is not None:
    print("내 예측:", pred_p)
'''
        ),
        md("귀무가설이 참이면 p-value 는 **0~1 에서 균등**하게 나옵니다. 그래서 *p<0.05 기준은 5% 의 거짓 양성*을 뜻합니다. 반대로 진짜 차이가 있어도 표본이 작으면 유의하게 나오지 않는 경우가 많습니다(검정력). **p>0.05 는 '차이 없음'의 증거가 아닙니다.** 순열검정은 라벨을 섞어 귀무분포를 직접 만들기 때문에 분포 가정이 적습니다."),
        md("## 2. 모형 데이터 준비"),
        code(
            '''
from creditlab.data import load_learning_data
from creditlab.features import fit_binners, transform_woe_frame
from creditlab.models import fit_logistic
from creditlab.scorecard import ScoreScaling, Scorecard
from creditlab.challengers import MLFeatureEncoder, fit_gbm_early_stop
from creditlab.evaluation import auroc, paired_bootstrap_diff

df_all, meta = load_learning_data("synthetic", seed=SEED)
TARGET = meta["target"]
df_all = df_all.sort_values("application_date").reset_index(drop=True)
cut = int(len(df_all) * 0.7)
dev, test = df_all.iloc[:cut].copy(), df_all.iloc[cut:].copy()
yt = test[TARGET].reset_index(drop=True)

VARS = ["num_delinq_12m", "card_utilization", "age", "inquiries_6m", "employment_years", "loan_amount_10k"]
bn = fit_binners(dev, VARS, TARGET, method="quantile", max_bins=5, min_events=10)
card = Scorecard.from_fit(bn, fit_logistic(transform_woe_frame(dev, bn), dev[TARGET]), scaling=ScoreScaling(600, 50, 20))
p_card = card.predict_pd(test)

# 약한 모형: 변수 2개만
bn2 = fit_binners(dev, ["age", "loan_amount_10k"], TARGET, method="quantile", max_bins=5, min_events=10)
card2 = Scorecard.from_fit(bn2, fit_logistic(transform_woe_frame(dev, bn2), dev[TARGET]), scaling=ScoreScaling(600, 50, 20))
p_weak = card2.predict_pd(test)

# 챌린저: 단조 제약 부스팅
NUM = ["age", "income_10k", "loan_amount_10k", "employment_years", "card_utilization", "num_delinq_12m", "inquiries_6m"]
CAT = ["housing_type", "region"]
enc = MLFeatureEncoder(NUM, CAT).fit(dev)
gbm, _, best_it = fit_gbm_early_stop(enc.transform(dev), dev[TARGET], enc.cat_mask, monotonic={"num_delinq_12m": 1, "card_utilization": 1},
                                     learning_rate=0.05, max_depth=3, min_samples_leaf=50, max_iter=500)
p_gbm = gbm.predict_proba(enc.transform(test))[:, 1]
print(f"테스트 {len(test):,}행(부도 {int(yt.sum())}건) | AUROC — 스코어카드 {auroc(yt, p_card):.4f}, 약한 모형(2변수) {auroc(yt, p_weak):.4f}, 부스팅 {auroc(yt, p_gbm):.4f}")
'''
        ),
        md("## 3. DeLong — 같은 고객으로 평가한 두 AUROC"),
        code("pred_delong = None  # ← 스코어카드(6변수) vs 약한 모형(2변수): 차이가 유의할까요? 스코어카드 vs 부스팅은?"),
        code(
            '''
from creditlab.stats import delong_test

rows = []
for name, pa, pb in [("스코어카드 vs 약한 모형(2변수)", p_card, p_weak), ("부스팅 vs 스코어카드", p_gbm, p_card)]:
    d = delong_test(yt, pa, pb)
    rows.append({"비교(1 vs 2)": name, "AUC1": d["auc1"], "AUC2": d["auc2"], "차이": d["diff"], "SE": d["se"], "z": d["z"],
                 "p-value": d["p_value"], "95% 구간": f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]"})
print(pd.DataFrame(rows).round(4).to_string(index=False))

# 쌍체 부트스트랩(레슨 5-1)과의 교차 확인
pb = paired_bootstrap_diff(auroc, yt, p_card, p_weak, n_boot=500, seed=SEED)
print(f"\\n[교차 확인] 스코어카드 − 약한 모형: 쌍체 부트스트랩 구간 [{pb['lo']:+.4f}, {pb['hi']:+.4f}] (DeLong 과 비슷해야 함)")
if pred_delong is not None:
    print("내 예측:", pred_delong)
'''
        ),
        code(
            '''
# DeLong 의 성질 확인: 순서를 바꾸면 z 의 부호만 바뀌고, 단조 변환에는 불변
d1 = delong_test(yt, p_card, p_weak)
d2 = delong_test(yt, p_weak, p_card)
d3 = delong_test(yt, p_card, np.log(p_weak / (1 - p_weak)))        # 두 번째 점수를 로짓으로 바꿔도 순위는 같다
print(f"순서 바꿈: z {d1['z']:+.4f} → {d2['z']:+.4f}, p {d1['p_value']:.6f} → {d2['p_value']:.6f}")
print(f"단조 변환: z {d1['z']:+.4f} → {d3['z']:+.4f}")
'''
        ),
        md(
            """
- **p-value 와 함께 AUROC 차이와 신뢰구간**을 보고합니다. 구간이 0 을 포함하면 차이를 확신할 수 없습니다.
- 변수 2개 모형은 변수를 4개 더 쓴 모형보다 분명히 나쁘고, 부스팅과 스코어카드의 차이는 (이 표본 크기에서는) 구분되지 않을 수 있습니다 — 위 표의 구간으로 확인하세요.
- DeLong 은 *같은 표본*으로 두 모형을 평가할 때 쓰며, 이벤트(부도)가 적으면 근사가 불안정합니다. 고객이 여러 번 나오면(군집) 독립 가정이 깨집니다.
"""
        ),
        md("## 4. 다중검정 — 많이 검정하면 우연히 유의해진다"),
        code(
            '''
from creditlab.stats import familywise_error_rate, multiple_testing_table, adjust_pvalues

print(f"20번 검정(모두 귀무가설이 참)에서 하나라도 p<0.05 일 확률 = {familywise_error_rate(20):.4f}")
rng = np.random.default_rng(SEED)
any_sig = [(rng.uniform(size=20) < 0.05).any() for _ in range(3000)]
bh_any = [(adjust_pvalues(rng.uniform(size=20), "bh") < 0.05).any() for _ in range(3000)]
print(f"시뮬레이션: 보정 없음 {np.mean(any_sig):.3f} | BH 보정 후 {np.mean(bh_any):.3f}")
'''
        ),
        md("### 변수 9개를 한꺼번에 검정 — 부도/정상의 평균이 다른가?"),
        code(
            '''
CANDS = ["age", "income_10k", "loan_amount_10k", "employment_years", "card_utilization", "num_delinq_12m", "inquiries_6m"]
pv = {}
for c in CANDS:                                    # 개발 구간에서, 결측 제외
    x = dev[[c, TARGET]].dropna()
    pv[c] = stats.ttest_ind(x.loc[x[TARGET] == 1, c], x.loc[x[TARGET] == 0, c], equal_var=False).pvalue
# 범주형 2개는 카이제곱으로
for c in ["housing_type", "region"]:
    ct = pd.crosstab(dev[c].fillna("MISSING"), dev[TARGET])
    pv[c] = stats.chi2_contingency(ct)[1]

mt = multiple_testing_table(list(pv.values()), names=list(pv.keys()), alpha=0.05).sort_values("p")
print(mt[["test", "p", "raw_reject", "bonferroni_reject", "holm_reject", "bh_reject"]].round(4).to_string(index=False))
print("\\n기각 개수 — 보정 없음:", int(mt["raw_reject"].sum()), "| Bonferroni:", int(mt["bonferroni_reject"].sum()),
      "| Holm:", int(mt["holm_reject"].sum()), "| BH:", int(mt["bh_reject"].sum()))
'''
        ),
        md(
            """
- 이 데이터에서는 신호가 강한 변수들이 모두 살아남아 네 방식의 기각 개수가 같습니다. **방식의 차이는 p-value 가 경계 근처일 때** 드러납니다(아래 실습).
- **Bonferroni**(α/m)는 가장 보수적이고, **Holm** 은 같은 보장(FWER)을 유지하며 조금 덜 보수적이며, **BH** 는 *거짓 발견 비율(FDR)* 을 통제해 더 많이 기각합니다. 어느 것을 쓸지는 목적(하나의 오탐도 안 된다 / 일부 오탐은 허용)에 따라 정하고 **사전에 기록**합니다.
- 변수 선택, 구간 분할, 모형 비교, 하이퍼파라미터 탐색도 *암묵적 다중검정*입니다. **시도한 횟수를 기록**하고, 최종 판단은 봉인된 테스트에서 합니다.
- 여기서 유의하게 나온 변수가 모형에 쓰여야 한다는 뜻은 아닙니다. 유의성은 *변수 선별의 참고*일 뿐이고, 업무적 타당성·안정성·공정성 검토가 따로 필요합니다.
"""
        ),
        md("## 5. 통계적 유의성 ≠ 실무적 유의성"),
        code(
            '''
# 시뮬레이션: 표본이 매우 크면 아주 작은 AUROC 차이도 '유의'해진다
rng = np.random.default_rng(SEED)
N = 400_000
y_big = (rng.random(N) < 0.08).astype(int)
base = rng.normal(size=N) + 0.95 * y_big
s_a = base
s_b = base + rng.normal(scale=0.12, size=N)         # 거의 같은 모형 (잡음 추가)
d = delong_test(y_big, s_a, s_b)
print(f"N={N:,}: AUROC 차이 {d['diff']:+.4f}, p = {d['p_value']:.2e}, 95% 구간 [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]")

# 사업 영향: 같은 승인율(80%)에서 승인 건의 부도율
def bad_rate_at_approval(score, y, approve=0.8):
    thr = np.quantile(score, approve)               # 점수(위험)가 낮은 80% 승인
    return y[score <= thr].mean()
ra, rb = bad_rate_at_approval(s_a, y_big), bad_rate_at_approval(s_b, y_big)
print(f"승인율 80% 에서 승인 건 부도율: 모형 A {ra:.4%} | 모형 B {rb:.4%} | 차이 {ra - rb:+.4%}p")
'''
        ),
        md("표본이 크면 통계적으로 '유의'하지만, 같은 승인율에서 부도율에 미치는 영향이 얼마나 되는지는 **별개의 질문**입니다. 반대로 소표본에서는 큰 차이도 '유의하지 않음'이 될 수 있습니다. **효과 크기·신뢰구간·사업 영향**으로 판단합니다."),
        md("## 6. ✏️ 실습 — BH 직접 계산"),
        code(
            '''
# [exercise]
pvals = [0.003, 0.012, 0.019, 0.034, 0.041, 0.20]      # 6개 검정의 p-value (이미 오름차순)
q = 0.05

bonf_count = None    # ← Bonferroni(α/m) 로 기각되는 개수
bh_count = None      # ← BH: p_(k) <= (k/m)·q 를 만족하는 *가장 큰 k* 까지 기각 → 기각 개수
# 힌트: m = 6.  BH 기준선은 (1/6)q, (2/6)q, ... 이고 가장 큰 k 를 찾는다(중간에 못 넘는 k 가 있어도 더 큰 k 가 통과하면 그때까지 모두 기각).

if bonf_count is not None and bh_count is not None:
    t = multiple_testing_table(pvals, alpha=q)
    tb, th = int(t["bonferroni_reject"].sum()), int(t["bh_reject"].sum())
    print(f"Bonferroni {bonf_count} (정답 {tb}) :", "맞음" if bonf_count == tb else "다시 계산")
    print(f"BH {bh_count} (정답 {th}) :", "맞음" if bh_count == th else "다시 계산")
else:
    print("bonf_count, bh_count 를 채우고 다시 실행하세요.")
'''
        ),
        md(
            """
<details>
<summary>정답 예시 (먼저 직접 해 본 뒤 펼치세요)</summary>

```python
m = len(pvals)
bonf_count = sum(p < q / m for p in pvals)
passed = [p <= (k / m) * q for k, p in enumerate(sorted(pvals), start=1)]
bh_count = max([k for k, ok in enumerate(passed, start=1) if ok], default=0)
```
</details>
"""
        ),
        md("## 7. 퀴즈"),
        code(
            '''
from creditlab.learning import Question, check, grade, show

QUIZ = [
    Question("Q1", "p-value", "p-value 0.03 의 올바른 해석은?",
             {"A": "귀무가설이 참일 확률이 3% 이다", "B": "귀무가설이 참이라고 가정할 때 관측만큼 극단적인 결과가 나올 확률이 3% 이다",
              "C": "효과가 3% 이다", "D": "대립가설이 참일 확률이 97% 이다"},
             "B", "p-value 는 귀무가설이 참일 때의 확률입니다. 가설이 참일 확률이 아닙니다.", ("무엇을 가정하고 계산하나요?",)),
    Question("Q2", "DeLong", "두 모형의 AUROC 를 같은 테스트 표본으로 비교할 때 DeLong 검정을 쓰는 이유는?",
             {"A": "표본을 늘리려고", "B": "같은 고객을 평가하므로 두 AUROC 가 상관되어 있어 그 공분산을 반영해야 하기 때문",
              "C": "AUROC 를 계산하려고", "D": "모형을 학습하려고"},
             "B", "독립 가정으로 비교하면 분산을 잘못 계산합니다.", ("두 모형이 같은 사람을 평가했습니다.",)),
    Question("Q3", "다중검정", "20개 변수를 각각 5% 수준으로 검정했더니 1개가 유의했습니다(실제로는 모두 무관). 이상한 일인가요?",
             {"A": "매우 이상하다", "B": "정상이다. 모두 귀무일 때 하나 이상 유의할 확률이 약 64% 이다", "C": "검정이 틀렸다", "D": "표본이 작아서만 생긴다"},
             "B", "1−0.95^20 ≈ 0.64 입니다. 보정이 필요합니다.", ("20번 시도했습니다.",)),
    Question("Q4", "BH vs Bonferroni", "BH 가 Bonferroni 보다 일반적으로 더 많이 기각하는 이유는?",
             {"A": "BH 가 틀려서", "B": "BH 는 거짓 발견 '비율'(FDR)을, Bonferroni 는 하나라도 틀릴 확률(FWER)을 통제해 더 엄격하기 때문",
              "C": "BH 는 p 를 키워서", "D": "둘은 같다"},
             "B", "통제 대상이 달라서 BH 가 덜 보수적입니다.", ("FDR 과 FWER 는 다릅니다.",)),
    Question("Q5", "실무적 유의성", "표본 100만 건에서 AUROC 차이 0.001 이 p<0.001 로 유의했습니다. 올바른 판단은?",
             {"A": "무조건 새 모형으로 교체한다", "B": "통계적으로 유의하나 효과가 작을 수 있으므로 같은 승인율에서의 부도율 변화 등 사업 영향과 비용·설명 가능성을 함께 본다",
              "C": "유의하므로 신뢰구간은 불필요하다", "D": "p-value 가 작으니 효과가 크다"},
             "B", "p-value 는 효과의 크기가 아닙니다. 효과 크기와 사업 영향으로 판단합니다.", ("p-value 는 표본 크기에 민감합니다.",)),
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
## 8. 과제와 정리
**과제**
1. 6장 시뮬레이션의 `scale`(잡음 0.12)을 0.05·0.12·0.5 으로 바꿔 AUROC 차이·p-value·승인율 80% 부도율 차이를 표로 만드세요. 유의하지만 사업 영향이 작은 구간이 있나요?
2. 변수를 9개에서 변수당 3개의 구간 분할 방법까지 늘려 *암묵적 검정 횟수*가 얼마가 되는지 세어 보고, 이를 기록하는 양식(예: 시도 로그)을 한 줄 설계하세요.
3. 레슨 5-2 의 스태킹과 스코어카드를 DeLong 으로 비교하고, 구간과 함께 한 문장으로 결론을 쓰세요.

**체크포인트**: 퀴즈 4/5 이상 · BH 실습 통과 · DeLong 결과를 'p-value + 차이 + 구간'으로 말할 수 있음

**다음 레슨**: 레슨 7-1 — 운영과 거버넌스(선택 편향·ECL·사유 코드·공정성·모형 카드).

**참고자료**
- DeLong, DeLong & Clarke-Pearson, "Comparing the areas under two or more correlated ROC curves", *Biometrics* (1988).
- Benjamini & Hochberg, "Controlling the false discovery rate", *JRSS B* (1995).
- Wasserstein & Lazar, "The ASA statement on p-values", *The American Statistician* (2016).
- 이론: `docs/theory/stage_14.html`
"""
        ),
    ]
