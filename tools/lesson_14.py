from nbkit import SETUP_CELL, code, md

TITLE = "레슨 7-1. 운영과 거버넌스: 선택 편향·충당금·사유·공정성·모형 카드"


def cells():
    return [
        md(
            """
# 레슨 7-1. 운영과 거버넌스: 선택 편향·충당금·사유·공정성·모형 카드

**예상 시간** 100분 · **선행** 레슨 6-2 · **이론 페이지** `docs/theory/stage_15.html`

## 학습 목표
1. **선택 편향**(승인자만 결과를 관찰)이 언제 커지는지 실험으로 본다.
2. **기대신용손실(ECL = PD × LGD × EAD)** 을 계산하고 PD 보정 오류가 충당금으로 번지는 경로를 본다.
3. 스코어카드로 **불이익 사유 후보**를 만들고 가산성을 검산한다.
4. 집단별 승인율·부도율·보정을 점검하되 **해석의 한계**를 안다.
5. **모형 카드**의 빈칸을 드러내는 점검표를 만든다.

> 이 레슨의 LGD·EAD·컷오프는 **교육용 가정**입니다. 공정성 지표는 *법적 판단 기준이 아니며* 국가·분야마다 다릅니다. 법률 자문이 아닙니다.
> 사유 문구, 사용 가능한 변수, 보호 속성의 정의는 법무·소비자 보호·규제 기준을 따릅니다.
"""
        ),
        SETUP_CELL,
        md("## 1. 선택 편향 — 승인자만 결과를 본다면"),
        code("pred_sel = None  # ← 과거 승인이 (a) 데이터에 있는 변수만으로 결정된 경우 / (b) 심사역의 비정형 인상(데이터에 없음)에 크게 의존한 경우, 승인자만으로 학습한 모형의 PD 과소추정은 어느 쪽이 클까요?"),
        code(
            '''
from creditlab.data import make_synthetic_credit_data
from creditlab.governance import selection_bias_experiment

df71 = make_synthetic_credit_data(n=20000, seed=71)
F = ["age", "income_10k", "loan_amount_10k", "employment_years", "card_utilization", "num_delinq_12m", "inquiries_6m"]
rows = []
for w in [0.0, 0.5, 1.0, 1.5, 2.5]:
    r = selection_bias_experiment(df71, F, officer_weight=w, seed=0)
    rows.append({"심사역 인상 비중": w, "승인자 부도율": r["bad_rate_accepted_dev"], "전체 부도율": r["bad_rate_all_dev"],
                 "AUROC(승인자 안)": r["auroc_on_accepted"], "AUROC(전체 신청자)": r["auroc_on_all"], "예측PD/실제(전체)": r["pd_underestimate_ratio"]})
print(pd.DataFrame(rows).round(3).to_string(index=False))
if pred_sel is not None:
    print("내 예측:", pred_sel)
'''
        ),
        md(
            """
- 승인자만 보면 **부도율이 낮아 보입니다**(표의 첫 두 열). 승인자 안의 AUROC 는 위험 범위가 잘려 전체 신청자 대상 값과 달라지며(범위 제한), 방향은 설정에 따라 다르므로 *승인자 안의 지표만으로 전체 성능을 판단할 수 없습니다*.
- 과거 승인이 *데이터에 있는 변수*만으로 결정되었다면(인상 비중 0) 모형이 승인자만으로 학습해도 전체 PD 를 거의 맞힙니다(마지막 열 ≈ 1). 과거 승인이 *데이터에 없는 정보*에 의존할수록 전체 신청자의 PD 를 **과소추정**합니다(마지막 열이 1 보다 훨씬 작아짐).
- 실제 데이터에서는 *심사역 인상의 크기를 알 수 없다*는 것이 문제입니다. 거절 추론 기법은 가정에 의존하므로, 이 실험은 **편향의 크기를 가늠하는 도구**이지 편향을 없애는 도구가 아닙니다. 가장 믿을 만한 방법은 소액·통제된 무작위 승인(실험)이나 외부 정보로 거절자의 이후 성과를 관찰하는 것입니다.
"""
        ),
        md("## 2. 모형과 등급 (OOT 구간 기준)"),
        code(
            '''
from creditlab.capstone import run_capstone

res = run_capstone(seed=SEED, n=15000, n_boot=100)       # 단계 16 의 파이프라인을 재사용 (개발/검증/OOT)
card, oot, dev = res["card"], res["splits"]["oot"], res["splits"]["dev"]
T = res["meta"]["target"]
print("변수:", res["selected"])
print(f"개발 {res['sizes']['dev']:,} / 검증 {res['sizes']['val']:,} / OOT {res['sizes']['oot']:,} | OOT AUROC {res['metrics'].loc['oot', 'auroc']:.3f}")
print(res["grade_tables"]["oot"][["grade", "n", "bad", "bad_rate"]].round(3).to_string(index=False))
'''
        ),
        md("## 3. 기대신용손실(ECL)"),
        code("pred_ecl = None  # ← PD 5%, LGD 45%, EAD 1,000 일 때 ECL 은? PD 를 절반으로 과소추정하면 충당금은?"),
        code(
            '''
from creditlab.governance import ecl_pd_bias_impact, expected_credit_loss, portfolio_ecl

print("단일 계약: ECL =", expected_credit_loss(0.05, 0.45, 1000.0))

oot_sc, oot_pd = res["score"]["oot"], res["pd"]["oot"]
from creditlab.scorecard import assign_grades
grades = assign_grades(oot_sc, res["grade_edges"])
g = pd.DataFrame({"grade": grades.to_numpy(), "pd": oot_pd, "y": oot[T].to_numpy()}).groupby("grade", observed=True).agg(n=("pd", "size"), pd=("pd", "mean"), observed=("y", "mean")).reset_index()
g["lgd"], g["ead"] = 0.45, 1000.0                      # 가정 (교육용)
pe = portfolio_ecl(g)
print(pe[["grade", "n", "pd", "observed", "lgd", "ead", "ecl"]].round(4).to_string(index=False))
print(f"\\n포트폴리오 ECL (모형 PD) = {pe.attrs['total_ecl']:,.0f}")
g2 = g.copy(); g2["pd"] = g2["observed"]
print(f"포트폴리오 ECL (실제 부도율) = {portfolio_ecl(g2).attrs['total_ecl']:,.0f}   ← OOT 실제와 비교(표본이 작으면 출렁임)")
b = ecl_pd_bias_impact(g, 0.5)
print(f"PD 를 절반으로 과소추정하면 ECL {b['base_ecl']:,.0f} → {b['biased_ecl']:,.0f} ({b['change_pct']:+.0f}%)")
if pred_ecl is not None:
    print("내 예측:", pred_ecl)
'''
        ),
        md("ECL 은 PD 에 **비례**하므로 PD 가 절반이면 충당금도 정확히 절반입니다. 레슨 5-2 의 보정이 *회계 숫자*로 이어지는 이유입니다. (IFRS 9 은 신용위험의 유의한 증가 여부에 따라 12개월/전체 만기 ECL 을 구분하며, 이 노트북의 PD 는 단순화한 값입니다.)"),
        md("## 4. 불이익 사유 후보 — 스코어카드 점수로"),
        code(
            '''
from creditlab.governance import reason_codes

worst3 = oot.loc[res["score"]["oot"].nsmallest(3).index]               # 점수가 가장 낮은 신청자 3명
rc = reason_codes(card, worst3, dev, top_n=3)
print(rc[["applicant", "rank", "variable", "points", "reference_points", "shortfall"]].round(2).to_string(index=False))

# 가산성 검산: (점수 − 기준 평균 점수) = Σ (변수 점수 − 변수별 기준 평균 점수)
ref_mean = card.points_frame(dev).mean()
for idx in worst3.index[:2]:
    pts = card.points_frame(oot.loc[[idx]]).iloc[0]
    print(f"신청자 {idx}: 점수 차이 {card.score(oot.loc[[idx]]).iloc[0] - card.score(dev).mean():+.3f} = 변수별 차이 합 {float((pts - ref_mean).sum()):+.3f}")
'''
        ),
        md(
            """
- 점수가 가장 낮은 세 신청자는 같은 구간에 속해 사유 후보도 같게 나옵니다(구간화된 점수표의 특성). 서로 다른 신청자의 사유가 갈리는 모습은 과제 3에서 확인합니다.
- 사유 후보는 *기준집단 평균 대비 점수를 가장 많이 깎은 변수*입니다. 점수표는 가산 구조라 합이 정확히 맞습니다(검산).
- **상관된 변수**가 있으면 기여가 변수 사이에서 나뉘어 사유가 불안정해질 수 있습니다. 기준집단·산출 방식을 고정하고 검증합니다.
- 고객에게 보이는 **문구·사유 개수·변수-문구 매핑**은 법무·소비자 보호 기준을 따릅니다. 이 노트북은 *후보 산출*만 보여 줍니다.
"""
        ),
        md("## 5. 공정성 점검 — 계산은 쉽고 해석은 어렵다"),
        code(
            '''
from creditlab.governance import approval_ratio, group_metrics

gm = res["group_metrics"]
print(gm.round(3).to_string(index=False))
rates = gm.set_index("group")["approval_rate"].to_dict()
print(f"\\n집단별 승인율 {({k: round(v, 3) for k, v in rates.items()})} | 최저/최고 = {min(rates.values()) / max(rates.values()):.3f}")

# 지역은 모형 변수가 아니지만, 점수와 연관돼 있나? (프록시 점검)
by_region = pd.DataFrame({"region": oot["region"].to_numpy(), "score": res["score"]["oot"].to_numpy()}).groupby("region")["score"].agg(["size", "mean"])
print("\\n지역별 평균 점수:")
print(by_region.round(1).to_string())
'''
        ),
        md(
            """
- 승인율 비율, 집단별 부도율·평균 PD·보정 차이·AUROC 는 **서로 다른 질문**에 답합니다. 여러 지표를 동시에 만족시킬 수 없는 경우가 있습니다(Kleinberg 외, 2016; Chouldechova, 2017).
- **작은 집단**(`small_sample`)은 부도 건수가 적어 지표가 불안정합니다. 한두 건 차이로 해석이 뒤집힙니다.
- 지역·직업·소비 패턴은 보호 속성의 **대리(프록시)** 가 될 수 있습니다. 모형에 *쓰지 않았다고* 안전한 것이 아니며, 변수 후보를 심사할 때 업무 타당성과 차별적 영향을 함께 봅니다.
- 승인율 비율의 0.8 기준('4/5 규칙')은 일부 국가 지침의 *경고 신호*일 뿐 법적 판단 기준이 아닙니다.
"""
        ),
        md("## 6. 모형 카드 — 빈칸을 드러내기"),
        code(
            '''
from creditlab.governance import build_model_card, check_model_card_completeness

draft = {"overview": {"model_name": "교육용 스코어카드 v0.1"}, "data": {"source": "합성데이터"}, "method": {}, "performance": {}, "monitoring": {}, "limits": {}, "reproducibility": {}}
chk = check_model_card_completeness(draft)
print(f"빠진 항목 {len(chk['missing'])}개, 한계 항목 충분: {chk['limitations_enough']}, 완성: {chk['complete']}")
print("처음 6개 빠진 항목:", chk["missing"][:6])
print()
print("\\n".join(build_model_card(draft).splitlines()[:12]))
'''
        ),
        md("모형 카드는 **빈칸이 빈칸으로 보이게** 만드는 것이 첫 번째 역할입니다(`(미기입)`). 항목이 채워졌다는 것이 내용이 맞다는 뜻은 아니므로, 독립 검증이 별도로 필요합니다. 이 노트북의 모형 카드 점검은 *형식* 점검일 뿐입니다."),
        md("## 7. ✏️ 실습 — ECL 과 승인율 비율 계산"),
        code(
            '''
# [exercise]
grades_ex = [  # (고객 수, PD, LGD, EAD)
    (2000, 0.02, 0.40, 500.0),
    (1000, 0.08, 0.40, 500.0),
    (400, 0.25, 0.40, 500.0),
]
approved = {"A": (120, 200), "B": (45, 100)}   # 집단: (승인 건수, 신청 건수)

ecl_total = None     # ← 포트폴리오 ECL 합계
ratio = None         # ← 승인율 비율 = 낮은 승인율 / 높은 승인율

if ecl_total is not None and ratio is not None:
    true_ecl = sum(n * p * l * e for n, p, l, e in grades_ex)
    ra = [a / n for a, n in approved.values()]
    true_ratio = min(ra) / max(ra)
    print(f"ECL {ecl_total:,.1f} (정답 {true_ecl:,.1f}) :", "맞음" if abs(ecl_total - true_ecl) < 1e-6 else "다시 계산")
    print(f"비율 {ratio:.3f} (정답 {true_ratio:.3f}) :", "맞음" if abs(ratio - true_ratio) < 1e-9 else "다시 계산")
else:
    print("ecl_total, ratio 를 채우고 다시 실행하세요.")
'''
        ),
        md(
            """
<details>
<summary>정답 예시 (먼저 직접 해 본 뒤 펼치세요)</summary>

```python
ecl_total = sum(n * p * l * e for n, p, l, e in grades_ex)
rates_ex = [a / n for a, n in approved.values()]
ratio = min(rates_ex) / max(rates_ex)
```
</details>
"""
        ),
        md("## 8. 퀴즈"),
        code(
            '''
from creditlab.learning import Question, check, grade, show

QUIZ = [
    Question("Q1", "선택 편향", "승인자만으로 학습한 모형이 전체 신청자의 PD 를 크게 과소추정하기 쉬운 상황은?",
             {"A": "과거 승인이 데이터에 있는 변수만으로 결정된 경우", "B": "과거 승인이 데이터에 없는 정보(심사역 판단 등)에 크게 의존하고 그 정보가 결과와 상관된 경우",
              "C": "표본이 클 때", "D": "변수가 적을 때"},
             "B", "데이터에 없는 정보로 선택되면 승인자의 결과가 전체를 대표하지 못합니다.", ("선택이 무엇에 의존했나요?",)),
    Question("Q2", "ECL", "PD 가 실제의 절반으로 과소추정되면 ECL 은(LGD·EAD 는 같다고 할 때)?",
             {"A": "변하지 않는다", "B": "절반이 된다", "C": "두 배가 된다", "D": "0 이 된다"},
             "B", "ECL = PD × LGD × EAD 라서 PD 에 비례합니다.", ("곱셈 구조를 보세요.",)),
    Question("Q3", "사유 코드", "스코어카드 기반 사유 후보에서 상관된 변수가 있을 때 주의할 점은?",
             {"A": "문제없다", "B": "기여가 변수 사이에서 나뉘어 사유가 불안정해질 수 있으므로 산출 방법을 고정하고 검증한다",
              "C": "사유는 항상 가장 큰 계수의 변수이다", "D": "사유가 필요 없다"},
             "B", "상관된 변수는 기여를 나눠 가지므로 사유 순위가 흔들릴 수 있습니다.", ("한 정보를 두 변수가 나눠 가질 때를 생각하세요.",)),
    Question("Q4", "공정성", "모형에 지역 변수를 쓰지 않았습니다. 지역 간 격차 문제가 사라졌나요?",
             {"A": "예, 사용하지 않았으니 사라졌다", "B": "아니다. 다른 변수가 지역·보호 속성의 프록시일 수 있어 집단별 결과를 직접 점검해야 한다",
              "C": "점검은 불필요하다", "D": "지역 변수를 쓰는 것이 항상 공정하다"},
             "B", "변수를 빼도 다른 변수가 같은 정보를 전달할 수 있습니다.", ("프록시를 떠올려 보세요.",)),
    Question("Q5", "모형 카드", "모형 카드의 항목이 모두 채워졌다면?",
             {"A": "모형이 올바르다는 증거다", "B": "형식상 빈칸이 없다는 뜻일 뿐, 내용의 타당성은 독립 검증이 따로 필요하다", "C": "규제 승인을 받은 것이다", "D": "모니터링이 불필요하다"},
             "B", "문서의 완성도와 모형의 타당성은 다른 문제입니다.", ("누가 내용을 확인하나요?",)),
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
1. `selection_bias_experiment` 의 `accept_share` 를 0.3·0.6·0.9 로 바꿔 승인율이 높을수록 편향이 줄어드는지 표로 확인하고, 현실에서 *승인율을 높여* 편향을 줄이려면 어떤 비용이 드는지 쓰세요.
2. 등급별 PD 를 **Platt 보정 전/후**(레슨 5-2)로 바꿔 ECL 합계가 어떻게 달라지는지 비교하세요. 어느 쪽이 OOT 실제 부도율 기반 ECL 에 더 가깝습니까?
3. 사유 코드 상위 3개가 *기준집단을 바꿨을 때*(예: 검증 구간) 얼마나 달라지는지 확인하고, 사유 산출 방법을 문서에 어떻게 고정할지 한 단락으로 쓰세요.

**체크포인트**: 퀴즈 4/5 이상 · ECL·승인율 비율 실습 통과 · 공정성 지표를 해석할 때의 한계 세 가지를 말할 수 있음

**다음 레슨**: 레슨 7-2 — 종합 프로젝트(캡스톤): 지금까지의 모든 선택을 하나의 재현 가능한 흐름과 모형 카드로 묶습니다.

**참고자료**
- Mitchell, M. 외, "Model Cards for Model Reporting", FAT* (2019).
- Kleinberg, Mullainathan & Raghavan, "Inherent trade-offs in the fair determination of risk scores" (2016).
- Chouldechova, A., "Fair prediction with disparate impact", *Big Data* (2017).
- IFRS Foundation, IFRS 9 *Financial Instruments*.
- 이론: `docs/theory/stage_15.html`
"""
        ),
    ]
