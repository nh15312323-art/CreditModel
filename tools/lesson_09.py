from nbkit import SETUP_CELL, code, md

TITLE = "레슨 4-2. PSI·안정성·드리프트"


def cells():
    return [
        md(
            """
# 레슨 4-2. PSI·안정성·드리프트

**예상 시간** 90분 · **선행** 레슨 4-1 · **이론 페이지** `docs/theory/stage_10.html`

## 학습 목표
1. PSI 를 구간별로 **손으로 계산**하고 라이브러리 값과 맞춘다.
2. 구간 경계를 **기준(개발) 구간에서 한 번 정해 고정**해야 하는 이유를 실험으로 확인한다.
3. **표본이 작으면 변화가 없어도 PSI 가 커지고**, 0 빈도 처리(eps)에 따라 값이 달라짐을 본다.
4. 모형 변수의 CSI, 점수 PSI, 등급 구성비를 한 표로 모니터링한다.
5. **공변량 이동·사전확률 변화·개념 변화**를 시뮬레이션하고, *PSI 가 못 보는 변화*를 구분한다.

> 시뮬레이션은 합성데이터 위에서 일부러 만든 변화입니다. **실제 모니터링 결과가 아니며** 교육 목적입니다.
> PSI 의 경보 기준(0.10/0.25)은 통계 법칙이 아니라 **업계 관행**입니다.
"""
        ),
        SETUP_CELL,
        md(
            """
## 1. PSI 정의와 손계산
`PSI = Σ (a_i − e_i) · ln(a_i / e_i)`  (e: 기준 비율, a: 현재 비율)

각 항은 항상 0 이상이고(a 와 e 가 같으면 0), 구간 비율이 달라질수록 커집니다.
"""
        ),
        code("pred_psi = None  # ← 아래 5구간 예제의 PSI 가 0.01 보다 작을까요, 0.1 보다 클까요? 어느 구간이 가장 크게 기여할까요?"),
        code(
            '''
from creditlab.evaluation import psi_terms, psi_label

e = np.array([0.10, 0.20, 0.40, 0.20, 0.10])
a = np.array([0.08, 0.17, 0.38, 0.24, 0.13])
terms = psi_terms(e, a)
print(pd.DataFrame({"기준 e": e, "현재 a": a, "(a−e)": a - e, "ln(a/e)": np.log(a / e), "항": terms}).round(4).to_string())
print(f"\\nPSI = {terms.sum():.4f}  → {psi_label(terms.sum())}")
print(f"기준·현재를 바꿔도 같다: {psi_terms(a, e).sum():.4f}   (대칭 KL 의 합)")
if pred_psi is not None:
    print("내 예측:", pred_psi)
'''
        ),
        md("**읽는 법**: 비율이 조금씩 움직인 이 예제의 PSI 는 작아서 `안정` 입니다. 기여가 가장 큰 구간은 비율이 가장 많이 변한 쪽이며, 총합만 보지 말고 **어느 구간 때문인지** 항별로 봅니다."),
        md("## 2. 데이터·모형·기준 구간"),
        code(
            '''
from creditlab.data import load_learning_data
from creditlab.features import fit_binners, transform_woe_frame
from creditlab.models import fit_logistic
from creditlab.scorecard import ScoreScaling, Scorecard, make_grade_edges, assign_grades
from creditlab.evaluation import auroc, calibration_slope_intercept

df_all, meta = load_learning_data("synthetic", seed=SEED)
TARGET = meta["target"]
df_all = df_all.sort_values("application_date").reset_index(drop=True)
cut = int(len(df_all) * 0.7)
dev, test = df_all.iloc[:cut].copy(), df_all.iloc[cut:].copy()

VARS = ["num_delinq_12m", "card_utilization", "age", "inquiries_6m", "employment_years", "loan_amount_10k"]
binners = fit_binners(dev, VARS, TARGET, method="quantile", max_bins=5, min_events=10)
fit = fit_logistic(transform_woe_frame(dev, binners), dev[TARGET])
card = Scorecard.from_fit(binners, fit, scaling=ScoreScaling(600, 50, 20))
sc_dev = card.score(dev)
edges_grade = make_grade_edges(sc_dev, 5, "quantile")          # 등급 경계도 개발 구간에서 고정
print(f"개발 {len(dev):,}행 | 현재(=테스트) {len(test):,}행 | 등급 경계 {[round(x) for x in edges_grade]}")
'''
        ),
        md("## 3. 구간 경계는 한 번만 — 다시 만들면 변화가 사라진다"),
        code(
            '''
from creditlab.evaluation import psi_numeric, fixed_edges

x_dev = dev["card_utilization"]
x_now = test["card_utilization"]
x_shift = x_now + 0.15                         # 카드 사용률이 전반적으로 0.15 올랐다고 '가정'(주입)

ok_same = psi_numeric(x_dev, x_now)                     # 정상 방법: 개발 경계 고정
ok_shift = psi_numeric(x_dev, x_shift)
bad_shift = psi_numeric(x_shift, x_shift)               # 잘못된 방법: 현재 표본으로 경계를 다시 만든 것과 같은 효과

print(f"변화 없음(개발 경계)         PSI = {ok_same['psi']:.4f}  {ok_same['label']}")
print(f"+0.15 이동(개발 경계 고정)   PSI = {ok_shift['psi']:.4f}  {ok_shift['label']}")
print(f"+0.15 이동(경계 다시 만듦)    PSI = {bad_shift['psi']:.4f}  ← 이동이 사라진다!")
print()
print(ok_shift["table"].round(3).to_string(index=False))
'''
        ),
        md("개발 구간의 10분위로 만든 구간에서 현재 표본은 **오른쪽 구간에 몰립니다.** 경계를 현재 표본에서 다시 만들면 *언제나 균등하게* 보여 변화가 지워집니다. 경계의 양 끝은 −∞/+∞ 로 열어 새 극단값을 받습니다."),
        md("## 4. 표본 수와 0 빈도"),
        code("pred_noise = None  # ← 변화가 전혀 없어도 표본이 작으면 PSI 는 0 일까요? n=200 과 n=5000 중 어디서 더 클까요?"),
        code(
            '''
from creditlab.evaluation import expected_psi_noise

rng = np.random.default_rng(SEED)
rows = []
for n in [200, 1000, 5000]:
    vals = [psi_numeric(rng.normal(size=n), rng.normal(size=n), bins=10)["psi"] for _ in range(300)]   # 같은 분포 → 진짜 변화 0
    rows.append({"표본 n(각각)": n, "PSI 평균(변화 0)": np.mean(vals), "95% 분위": np.quantile(vals, 0.95), "이론 근사 (k−1)(2/n)": expected_psi_noise(10, n, n)})
print(pd.DataFrame(rows).round(4).to_string(index=False))
if pred_noise is not None:
    print("내 예측:", pred_noise)
'''
        ),
        md("진짜 변화가 0 인데도 **표본이 작을수록 PSI 가 큽니다.** 월별처럼 표본이 작은 모니터링에서는 `0.10` 을 기계적으로 쓰지 말고 *표본 수에 맞게 해석*합니다."),
        code(
            '''
# 0 빈도: 현재 표본에 한 구간이 비었을 때 eps 선택의 영향
e_pct = np.array([0.30, 0.40, 0.30])
a_pct = np.array([0.50, 0.50, 0.00])                      # 세 번째 구간이 비었다
for eps in [1e-2, 1e-3, 1e-4, 1e-6]:
    print(f"eps={eps:g}  PSI = {psi_terms(e_pct, a_pct, eps).sum():.3f}")
'''
        ),
        md("빈 구간은 `ln(0)` 때문에 PSI 를 무한대로 만들 수 있어 작은 값(eps)으로 바닥을 둡니다. **eps 에 따라 결과가 달라지므로 값을 문서에 적고**, 가능하면 인접 구간과 합쳐 빈 구간 자체를 없앱니다."),
        md("## 5. 모니터링 표 — 변수(CSI)·점수·등급"),
        code(
            '''
from creditlab.evaluation import csi_table, psi_from_bins

def monitor(df_now, label):
    csi = csi_table(dev, df_now, binners)
    sc_now = card.score(df_now)
    score_psi = psi_numeric(sc_dev, sc_now)["psi"]
    grade_psi = psi_from_bins(assign_grades(sc_dev, edges_grade), assign_grades(sc_now, edges_grade), order=list("ABCDE"))["psi"]
    return {"scenario": label, "score_psi": score_psi, "grade_psi": grade_psi,
            "max_csi": csi.iloc[0]["psi"], "max_csi_var": csi.iloc[0]["variable"]}

print(pd.DataFrame([monitor(test, "현재(변화 없음)")]).round(4).to_string(index=False))
print()
print(csi_table(dev, test, binners).round(4).to_string(index=False))
'''
        ),
        md("CSI 는 *모형에 쓰인 변수마다* 개발 구간의 비닝 경계를 그대로 적용한 PSI 입니다. 점수 PSI 와 등급 구성비 변화는 **정책·한도에 미치는 영향**을 볼 때 쓰입니다."),
        md("## 6. 세 가지 변화를 일부러 만들어 본다"),
        md(
            """
| 시뮬레이션 | 만드는 방법 | PSI 에 보이는가 |
|---|---|---|
| **공변량 이동** | 현재 표본에서 사용률이 높은 고객을 더 자주 뽑는다. 라벨은 *원래 값 그대로* | 보임 |
| **사전확률 변화** | 같은 기울기로 부도 확률만 일괄 상승시켜 라벨을 다시 뽑는다 | 입력엔 안 보임 |
| **개념 변화** | 입력은 그대로 두고, 위험과 입력의 *관계*를 약하게 바꿔 라벨을 다시 뽑는다 | 안 보일 수 있음 |

라벨을 다시 뽑는 두 시나리오는 **모형 PD 를 기준으로 가정한 시뮬레이션**입니다(실제 세계의 변화가 아닙니다).
"""
        ),
        code(
            '''
from scipy.optimize import brentq
from creditlab.models import logit, sigmoid

rng = np.random.default_rng(SEED)
p0 = card.predict_pd(test)
base_rate = float(test[TARGET].mean())

# 1) 공변량 이동: 사용률이 높은 고객을 더 자주 뽑되 라벨은 그대로 둔다 (P(Y|X) 불변)
w = np.where(test["card_utilization"] > dev["card_utilization"].median(), 3.0, 1.0)
idx = rng.choice(len(test), size=len(test), replace=True, p=w / w.sum())
cov = test.iloc[idx].reset_index(drop=True)

# 2) 사전확률 변화: logit 에 상수를 더해 부도 확률만 일괄 상승
prior = test.copy().reset_index(drop=True)
prior[TARGET] = (rng.random(len(test)) < sigmoid(logit(p0) + 0.7)).astype(int)

# 3) 개념 변화: 관계를 절반 강도로 약화(평균 부도율은 유지)
def weaker(a):
    return sigmoid(a + 0.5 * logit(p0)).mean() - base_rate
a_star = brentq(weaker, -5, 5)
concept = test.copy().reset_index(drop=True)
concept[TARGET] = (rng.random(len(test)) < sigmoid(a_star + 0.5 * logit(p0))).astype(int)

scen = {"기준(변화 없음)": test.reset_index(drop=True), "공변량 이동": cov, "사전확률 변화": prior, "개념 변화": concept}
rows = []
for name, d in scen.items():
    m = monitor(d, name)
    pdn = card.predict_pd(d)
    cs = calibration_slope_intercept(d[TARGET], pdn)
    m.update({"AUROC": auroc(d[TARGET], pdn), "실제 부도율": d[TARGET].mean(), "평균 PD": pdn.mean(),
              "보정 절편": cs["intercept"], "보정 기울기": cs["slope"]})
    rows.append(m)
res = pd.DataFrame(rows).set_index("scenario")
print(res[["score_psi", "max_csi", "max_csi_var", "AUROC", "실제 부도율", "평균 PD", "보정 절편", "보정 기울기"]].round(3).to_string())
'''
        ),
        md(
            """
**읽는 법** (위 표를 보며):
- **공변량 이동**: PSI(점수·CSI)가 커지지만 AUROC·보정은 크게 안 변합니다. *분포가 변했다고 모형이 틀린 것은 아닙니다.*
- **사전확률 변화**: 입력이 그대로라 PSI 는 기준과 비슷한데 **실제 부도율이 평균 PD 보다 높아지고 보정 절편이 움직입니다.**
- **개념 변화**: 입력 분포(PSI)는 그대로인데 **AUROC 가 내려가고 보정 기울기가 1 에서 벗어납니다.**

그래서 **PSI 하나로 모형 안정성을 판정할 수 없습니다.** 분포 지표(PSI/CSI)는 *첫 번째 경보기*이고, 성과가 관찰된 뒤에는 AUROC·KS·보정·등급별 부도율이 함께 필요합니다.
"""
        ),
        code(
            '''
# 시뮬레이션이 의도대로 만들어졌는지 점검 (이 표가 노트북의 근거)
checks = {
    "공변량 이동: 점수 PSI 가 기준보다 큼": res.loc["공변량 이동", "score_psi"] > res.loc["기준(변화 없음)", "score_psi"],
    "사전확률 변화: 실제 부도율 > 평균 PD": res.loc["사전확률 변화", "실제 부도율"] > res.loc["사전확률 변화", "평균 PD"],
    "개념 변화: AUROC 가 기준보다 낮음": res.loc["개념 변화", "AUROC"] < res.loc["기준(변화 없음)", "AUROC"],
    "개념 변화: 점수 PSI 는 기준과 같음(입력 동일)": abs(res.loc["개념 변화", "score_psi"] - res.loc["기준(변화 없음)", "score_psi"]) < 1e-9,
}
for k, v in checks.items():
    print(("OK  " if v else "FAIL"), k)
'''
        ),
        md(
            """
## 7. 모니터링 설계 메모
| 대상 | 지표 | 주기(예) | 경보 후 조치 |
|---|---|---|---|
| 변수 분포 | CSI | 월/분기 | 데이터 오류인가, 모집단 변화인가 조사 |
| 점수·등급 | 점수 PSI, 등급 구성비 | 월 | 정책·한도 영향 검토 |
| 변별력 | AUROC/KS | 성과 관찰 후 분기/반기 | 재개발 검토 |
| 보정 | 등급별 예측 PD vs 실제 | 분기/반기 | 재보정 검토 |

임계값(0.10/0.25), 구간 수, eps, 기준 기간, 책임자와 기한은 **사전에 정해 문서에 적습니다.**
(실제 금융회사의 모니터링 체계·감독 보고 기준은 기관마다 다르며 이 프로그램은 그것을 대신하지 않습니다.)
"""
        ),
        md("## 8. ✏️ 실습 — 건수로 PSI 계산"),
        code(
            '''
# [exercise]
from creditlab.evaluation import psi_from_counts

dev_counts = np.array([200, 300, 300, 200])
now_counts = np.array([100, 250, 350, 300])
psi_mine = None     # ← 건수를 비율로 바꿔 직접 계산한 PSI (소수 넷째 자리까지)
# 힌트: e = dev_counts / dev_counts.sum(),  a = now_counts / now_counts.sum(),  np.sum((a - e) * np.log(a / e))

if psi_mine is not None:
    true = psi_from_counts(dev_counts, now_counts)
    print(f"내 계산 {psi_mine:.4f} | 라이브러리 {true:.4f} :", "맞음" if abs(psi_mine - true) < 1e-3 else "다시 계산", "|", psi_label(true))
else:
    print("psi_mine 을 채우고 다시 실행하세요.")
'''
        ),
        md(
            """
<details>
<summary>정답 예시 (먼저 직접 해 본 뒤 펼치세요)</summary>

```python
e = dev_counts / dev_counts.sum()
a = now_counts / now_counts.sum()
psi_mine = float(np.sum((a - e) * np.log(a / e)))
```
</details>
"""
        ),
        md("## 9. 퀴즈"),
        code(
            '''
from creditlab.learning import Question, check, grade, show

QUIZ = [
    Question("Q1", "경계 고정", "PSI 를 낼 때 구간 경계는 어떻게 정해야 하나요?",
             {"A": "현재 표본의 분위수로 매번 다시 만든다", "B": "기준(개발) 표본에서 한 번 정해 현재 표본에 그대로 적용한다",
              "C": "구간은 상관없다", "D": "양 끝을 닫아 둔다"},
             "B", "현재 표본으로 경계를 다시 만들면 항상 균등하게 보여 변화가 사라집니다.", ("변화를 보려면 무엇이 고정되어야 할까요?",)),
    Question("Q2", "표본 크기", "분포가 전혀 변하지 않았는데 월별 표본(각 300건)의 PSI 가 0.04 였습니다. 올바른 해석은?",
             {"A": "모형이 불안정하다", "B": "표본이 작으면 우연만으로도 PSI 가 0 보다 커진다. 표본 수를 감안해 해석한다",
              "C": "PSI 가 틀렸다", "D": "구간 수를 1 로 줄인다"},
             "B", "기대 PSI 는 대략 (k−1)(1/n_e+1/n_a) 입니다.", ("표본 수가 작아지면 비율의 흔들림은?",)),
    Question("Q3", "개념 변화", "PSI 는 모두 0.05 미만인데 AUROC 가 떨어졌습니다. 가능한 설명은?",
             {"A": "불가능하다", "B": "입력 분포는 같지만 입력과 부도의 관계가 변했을 수 있다(개념 변화)",
              "C": "PSI 계산 오류뿐이다", "D": "AUROC 는 항상 PSI 와 같이 움직인다"},
             "B", "PSI 는 입력·점수 분포만 보므로 관계 변화는 놓칠 수 있습니다.", ("PSI 가 보는 것은 무엇인가요?",)),
    Question("Q4", "경보 기준", "PSI 0.10/0.25 기준은?",
             {"A": "통계적으로 증명된 임계값이다", "B": "업계 관행이며 구간 수·표본 수에 따라 달라질 수 있어 조직이 정해 문서화한다",
              "C": "법적 의무 값이다", "D": "항상 0.5 이다"},
             "B", "관행적 경보 기준이지 법칙이 아닙니다.", ("누가 이 값을 정했나요?",)),
    Question("Q5", "PSI 의 한계", "카드 사용률 PSI 가 0.4 로 크게 나왔습니다. 가장 알맞은 첫 대응은?",
             {"A": "즉시 모형이 틀렸다고 선언하고 폐기한다", "B": "원인(데이터 오류·모집단 변화)을 조사하고, 가능하면 성능·보정 지표와 함께 영향을 본다",
              "C": "무시한다", "D": "기준 구간을 현재 구간으로 바꿔 PSI 를 낮춘다"},
             "B", "분포가 변해도 모형은 여전히 잘 작동할 수 있습니다. 경보 → 조사 → 조치 순서를 지킵니다.", ("경보는 증명이 아닙니다.",)),
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
## 10. 과제와 정리
**과제**
1. 테스트 구간을 분기별로 나눠 **분기별 점수 PSI 와 CSI 최댓값**을 표로 만드세요. 표본 수를 함께 적고, 경보로 볼지 표본 변동으로 볼지 근거를 쓰세요.
2. 6장의 공변량 이동 시뮬레이션에서 가중치(3.0)를 1.5·3.0·6.0 으로 바꾸어 **PSI 와 AUROC 가 어떻게 변하는지** 표로 정리하세요. 왜 AUROC 는 크게 안 변하는지 설명하세요.
3. 구간 수를 5·10·20 으로 바꿔 같은 이동에서의 PSI 를 비교하고, 임계값을 구간 수와 무관하게 쓰기 어려운 이유를 쓰세요.

**체크포인트**: 퀴즈 4/5 이상 · 건수 PSI 실습 통과 · 세 가지 변화가 각각 어떤 지표에 보이는지 말할 수 있음

**다음 Phase**: 5 — 머신러닝 챌린저(트리·부스팅)와 확률 보정을 다룹니다.

**참고자료**
- Siddiqi, N., *Credit Risk Scorecards* (Wiley, 2006) — 모니터링과 PSI.
- Yurdakul, B., *Statistical Properties of Population Stability Index* (Western Michigan Univ. dissertation, 2018).
- Quiñonero-Candela 외, *Dataset Shift in Machine Learning* (MIT Press, 2009).
- 이론: `docs/theory/stage_10.html`
"""
        ),
    ]
