# Credit Model Mastery Lab

신용평가모형을 **처음부터 끝까지** 직접 만들 수 있게 하는 학습 프로그램.
규제형 로지스틱 CSS(스코어카드)와 ML(부스팅)을 같은 기준으로 비교하는 법까지 다룬다.

## 구성

| 위치 | 내용 |
|---|---|
| `docs/index.html` | 학습 로드맵(웹). 브라우저로 열면 된다 |
| `docs/theory/stage_XX.html` | **단계 0~16 이론 설명 페이지** (각 10개 항목 + 퀴즈) |
| `docs/theory/glossary.html` | 검색되는 용어 사전 |
| `docs/theory/all.html` | 이론 전체를 한 페이지로 |
| `notebooks/` | Colab 실습 노트북 (현재 단계 1~5: 노트북 01~05) |
| `src/creditlab/` | 실습에서 쓰는 파이썬 패키지 |
| `lessons/theory/*.md` | 이론 페이지의 원본(Markdown) |
| `docs/*_template.md` | 목표 정의서·데이터 사전·모형 카드 템플릿 |

## 학습 방법

1. 이론 페이지를 읽고 *손계산 예제를 직접 풀어 본다*.
2. 같은 단계의 노트북을 Colab 에서 실행한다 (예측 → 실행 → 비교).
3. 퀴즈로 확인하고 틀린 개념은 이론 페이지로 돌아간다.

## 로컬에서 실행

```bash
pip install -e ".[dev]"
python tools/run_tests.py            # 패키지 테스트
python tools/run_notebooks.py        # 노트북 검증(--solve 로 정답 셀 채움)
python tools/build_theory.py         # 이론 페이지 다시 만들기
python tools/check_theory_code.py    # 이론 페이지의 코드 블록 실행 검증
```

선택 패키지: `.[stats]`(statsmodels), `.[ml]`(xgboost, catboost, shap).

## 약속(전 단계 공통)

- target: 1 = 부도(bad), 0 = 정상(good)
- WOE = ln(%bad/%good) → 높을수록 위험 (Siddiqi 교재와 부호 반대, IV 는 동일)
- Score = Offset + Factor·ln(odds_good), Factor = PDO/ln2 (예: 600점@50:1, PDO 20)
- 모든 적합(비닝·대체값·모형)은 학습 구간에서만, 평가는 시간 기준 분할

## 현재 상태와 한계

- 완료: Phase 1(단계 1~3), Phase 2(단계 4~5: 변수 진단·비닝·WOE/IV) 실습 노트북, 단계 0~16 이론 페이지, 용어 사전.
- Phase 3 완료: 노트북 06(로지스틱 회귀 해석·규제), 07(스코어카드 변환·등급·컷오프) — `creditlab.models`, `creditlab.scorecard`.
- 예정: Phase 4~6 노트북(평가·PSI, 부스팅·보정, 캡스톤).
- 참고: 원 명세서는 비닝·WOE·IV 를 Phase 3 에 두었지만, 단계 순서에 맞춰 Phase 2 로 앞당겼다.
- 노트북은 일반 파이썬 실행기로 검증했다. 실제 Jupyter/Colab 커널에서의 실행은 처음 열어 확인해야 한다.
- UCI 신용카드 데이터 내려받기는 개발 환경의 네트워크 제한 때문에 실제로 확인하지 못했다. 실패하면 합성데이터로 대체되며 경고가 표시된다. 수동 업로드도 지원한다.
- 수식 표시(MathJax)와 일부 링크는 인터넷 연결이 필요하다. 서지·규제 문서는 학습 시점에 원문을 재확인할 것.
- 이 자료는 교육용이며 법률·규제 자문이 아니다. 교육용 합성데이터의 성능은 실제 성능이 아니다.
