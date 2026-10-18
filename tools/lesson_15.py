from nbkit import SETUP_CELL, code, md

TITLE = "레슨 7-2. 종합 프로젝트(캡스톤): 처음부터 끝까지 하나의 흐름과 모형 카드"


def cells():
    return [
        md(
            """
# 레슨 7-2. 종합 프로젝트(캡스톤)

**예상 시간** 120분 · **선행** 레슨 7-1 · **이론 페이지** `docs/theory/stage_16.html`

## 학습 목표
1. 시간순 분할 → 변수 선별 → WOE → 로지스틱 → 점수표 → 등급 → 평가 → 챌린저 → 모니터링 기준까지 **한 번에 재현**한다.
2. 자동으로 점검되는 항목과 **사람이 판단해야 하는 항목**을 구분한다.
3. 모형 카드의 판단 항목(용도·범위·한계·책임자…)을 직접 쓰고, 빈칸 점검을 통과시킨다.

> 이 캡스톤의 결과는 **합성데이터** 기준이며 실제 성능이 아닙니다. 실제 데이터에서는 부도 정의, 관측·성과 기간, 거절자 처리, 규제 요건이 다시 정해져야 합니다.
"""
        ),
        SETUP_CELL,
        md("## 1. 파이프라인 한 번에 실행"),
        code(
            '''
from creditlab.capstone import run_capstone

res = run_capstone(source="synthetic", seed=SEED, n=15000, n_boot=200)
print("사용 변수:", res["selected"])
print(f"개발 {res['sizes']['dev']:,} / 검증 {res['sizes']['val']:,} / OOT {res['sizes']['oot']:,} (시간순 분할: {res['time_split']})")
print(res["metrics"].round(4).to_string())
'''
        ),
        md(
            """
성능 지표는 **개발 → 검증 → OOT** 순서로 읽습니다. 개발 구간 성능은 모형이 이미 본 데이터라 낙관적이고, 설정은 개발·검증으로 정했으며 OOT 는 한 번만 봅니다(CLAUDE 원칙: 평가 데이터로 설정을 바꾸지 않는다).
"""
        ),
        md("## 2. 등급·보정·안정성·챌린저"),
        code(
            '''
print("[OOT 등급표]")
print(res["grade_tables"]["oot"][["grade", "n", "bad", "bad_rate"]].round(3).to_string(index=False))
cal = res["calibration"]
print(f"\\n[보정] ECE {cal['ece']:.4f} | 기울기 {cal['slope_intercept']['slope']:.3f} | 평균 PD {cal['mean_pd']:.3f} vs 실제 {cal['bad_rate']:.3f}")
print("[안정성] 점수 PSI(검증, OOT):", {k: round(v["score_psi"], 4) for k, v in res["psi"].items()})
print("[변수별 CSI 상위 3]"); print(res["csi_oot"].sort_values("psi", ascending=False).head(3).round(4).to_string())
d = res["challenger"]["delong"]
print(f"\\n[챌린저] 부스팅 OOT AUROC {res['challenger']['auroc']:.4f} vs 스코어카드 {d['auc2']:.4f} | DeLong p={d['p_value']:.3f} | 차이 95% 구간 [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]")
'''
        ),
        md(
            """
- 챌린저가 *점수는 높더라도* 신뢰구간이 0 을 포함하면 우열을 단정할 수 없습니다(레슨 6-2). 설명 가능성·운영 비용까지 보고 챔피언을 정합니다.
- PSI/CSI 는 *변화의 신호*이지 성능 하락의 증거가 아닙니다. 경보 기준과 조치 절차를 사전에 정해 둡니다(레슨 5-1).
"""
        ),
        md("## 3. 자동 점검표 — 기계가 확인할 수 있는 것만"),
        code(
            '''
rub = res["rubric_auto"]
print(rub.to_string(index=False))
print(f"\\n통과 {int(rub['통과'].sum())}/{len(rub)}")
'''
        ),
        md(
            """
자동 점검은 *누수 방지의 시간 순서, 점수표 검산, 신뢰구간 보고, 등급 단조성, 재현성 기록* 같은 **형식·일관성**만 확인합니다. 변수의 업무 타당성, 부도 정의, 의도된 용도에 대한 적합성은 사람이 판단해야 하고 이 표로는 알 수 없습니다. 모두 통과해도 모형이 올바르다는 뜻은 아닙니다.
"""
        ),
        md("## 4. ✏️ 실습 — 모형 카드의 판단 항목 쓰기"),
        code(
            '''
# [exercise]
judgment = {
    "intended_use": None,        # ← 이 모형을 어디에 쓰는지 (예: 무담보 개인 신용대출 신청 심사의 보조 점수)
    "out_of_scope": None,        # ← 쓰면 안 되는 곳
    "owner": None,               # ← 모니터링 책임 (역할명)
    "alert_thresholds": None,    # ← 경보 기준 (예: 점수 PSI 0.10 주의 / 0.25 조치)
    "population_gap": None,      # ← 개발 모집단과 실제 적용 모집단의 차이
    "quality_issues": None,      # ← 데이터 품질 이슈와 처리
    "assumptions": None,         # ← 가정 (합성데이터, 정의 등)
    "fairness_check": None,      # ← 어떤 집단 점검을 했고 한계는 무엇인지
    "known_limitations": None,   # ← 8자 이상 문장 3개 이상의 리스트
}
print("채운 항목:", [k for k, v in judgment.items() if v])
'''
        ),
        md(
            """
<details>
<summary>정답 예시 (먼저 직접 써 본 뒤 펼치세요)</summary>

```python
judgment = {
    "intended_use": "무담보 개인 신용대출 신청 심사의 보조 점수(교육용 예시)",
    "out_of_scope": "법인 대출, 담보 대출, 실제 승인 결정의 단독 근거",
    "owner": "모형 위험 관리 담당(역할 기준, 교육용 가정)",
    "alert_thresholds": "점수 PSI 0.10 주의 / 0.25 조치(교육용 가정, 기관 기준 우선)",
    "population_gap": "합성 신청자 모집단. 실제 승인자·거절자 구성과 다를 수 있음",
    "quality_issues": "합성데이터라 결측·이상치는 정의된 방식으로만 존재",
    "assumptions": "부도 정의와 성과 기간은 합성 설정에 따름. 실제 값은 기관 기준",
    "fairness_check": "지역 집단별 승인율·부도율·보정 점검. 표본이 작은 집단은 불안정",
    "known_limitations": ["합성데이터 기반이라 실제 성능을 대표하지 않는다",
                          "거절자 성과가 없어 선택 편향을 보정하지 못했다",
                          "경기 변화에 따른 재보정과 재개발 기준은 별도 정의가 필요하다"],
}
```
</details>
"""
        ),
        md("## 5. 모형 카드 조립과 점검"),
        code(
            '''
from creditlab.capstone import auto_model_card_info
from creditlab.governance import build_model_card, check_model_card_completeness

info = auto_model_card_info(res)
filled = {k: v for k, v in judgment.items() if v}
info["overview"].update({"model_name": "교육용 스코어카드 캡스톤", **{k: filled[k] for k in ("intended_use", "out_of_scope") if k in filled}})
info["data"].update({k: filled[k] for k in ("quality_issues", "population_gap") if k in filled})
info["monitoring"].update({k: filled[k] for k in ("alert_thresholds", "owner") if k in filled})
info["limits"].update({k: filled[k] for k in ("known_limitations", "assumptions", "fairness_check") if k in filled})

chk = check_model_card_completeness(info)
print(f"빠진 항목 {len(chk['missing'])}개:", chk["missing"])
print("한계 3개 이상:", chk["limitations_enough"], "| 완성:", chk["complete"])

from pathlib import Path
out = Path(PROJECT_ROOT) / "artifacts" / "capstone_report.md"
out.parent.mkdir(exist_ok=True)
out.write_text(build_model_card(info), encoding="utf-8")
print("\\n보고서 저장:", out.relative_to(PROJECT_ROOT), f"({out.stat().st_size:,} bytes, git 추적 제외)")
'''
        ),
        md("실습을 채우기 전에는 빠진 항목이 **그대로 드러납니다**. 채운 뒤에도 *완성 표시는 형식 점검일 뿐*입니다. 모형 카드는 독립 검증자와 승인 절차를 거쳐야 의미가 생깁니다."),
        md("## 6. 퀴즈"),
        code(
            '''
from creditlab.learning import Question, check, grade, show

QUIZ = [
    Question("Q1", "평가 데이터", "OOT 성능을 본 뒤 변수 선택을 바꾸고 다시 OOT 로 평가하면?",
             {"A": "정상적인 개선 절차다", "B": "OOT 정보가 설정에 스며들어 성능이 낙관적이 된다", "C": "아무 영향이 없다", "D": "성능이 반드시 나빠진다"},
             "B", "평가 데이터로 설정을 바꾸면 더 이상 독립 평가가 아닙니다.", ("누구의 데이터로 결정했나요?",)),
    Question("Q2", "자동 점검", "자동 점검표를 모두 통과했다는 것은?",
             {"A": "모형이 올바르다는 증거", "B": "형식·일관성 항목을 만족했다는 뜻일 뿐, 타당성·용도 적합성은 사람이 판단해야 한다", "C": "규제 승인", "D": "재검증이 필요 없다"},
             "B", "자동 점검은 기계가 확인 가능한 항목만 봅니다.", ("무엇을 점검했는지 보세요.",)),
    Question("Q3", "챌린저", "부스팅의 OOT AUROC 가 약간 높지만 DeLong 차이의 95% 구간이 0 을 포함할 때 결론은?",
             {"A": "부스팅이 확실히 우수하다", "B": "우열을 단정할 수 없어 설명 가능성·운영 비용 등을 함께 보고 정한다", "C": "스코어카드가 확실히 우수하다", "D": "검정이 틀렸다"},
             "B", "구간이 0 을 포함하면 차이가 우연일 수 있습니다.", ("신뢰구간의 의미를 떠올리세요.",)),
    Question("Q4", "합성데이터", "합성데이터로 얻은 캡스톤 성능을 보고할 때 올바른 표현은?",
             {"A": "이 모형의 실제 성능이다", "B": "합성데이터 기준 값으로 실제 성능을 대표하지 않으며 방법의 연습 결과다", "C": "업계 평균이다", "D": "실제보다 항상 낮다"},
             "B", "합성 분포는 실제 포트폴리오가 아닙니다.", ("데이터가 어디서 왔나요?",)),
    Question("Q5", "재현성", "재현성을 위해 최소한 기록할 것은?",
             {"A": "모형 점수만", "B": "코드 버전, 시드, 데이터 해시, 라이브러리 버전", "C": "작성자 이름만", "D": "기록은 필요 없다"},
             "B", "같은 입력과 환경을 복원할 수 있어야 재실행 결과를 비교할 수 있습니다.", ("무엇이 결과를 바꾸나요?",)),
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
## 7. 과제와 정리
**과제**
1. `seed` 를 5개 바꿔 `run_capstone` 을 반복하고 OOT AUROC·PSI 가 얼마나 흔들리는지 표로 정리하세요. 이 흔들림이 *성능 차이를 해석할 때* 어떤 의미를 갖는지 한 단락으로 쓰세요.
2. `source="uci"` 로 실제 공개 데이터를 시도하세요(네트워크가 필요하며, 실패하면 합성으로 대체됩니다). 결과가 합성 결과와 어떻게 다른지 비교하세요.
3. 직접 쓴 모형 카드를 동료(또는 다른 사람 역할)가 **독립 검증자**의 눈으로 읽고 근거가 부족한 문장에 표시하게 하세요.

**체크포인트**: 퀴즈 4/5 이상 · 모형 카드 빈칸 0 · 한계 3개 이상 · 성능을 *합성 기준*으로 설명할 수 있음

**참고자료**
- Mitchell, M. 외, "Model Cards for Model Reporting", FAT* (2019).
- Siddiqi, N., *Credit Risk Scorecards* (Wiley, 2006).
- 이론: `docs/theory/stage_16.html`
"""
        ),
    ]
