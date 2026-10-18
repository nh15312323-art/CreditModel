# 에이전트 작업 지침 (Credit Model Mastery Lab)

## 원칙
- 한 번에 한 Phase 만 구현한다. 끝나면 테스트·검증 결과와 함께 보고하고 멈춘다.
- 모든 숫자·산식은 코드로 계산해 확인한 뒤 문서에 쓴다. 손계산만 믿지 않는다.
- 확인하지 못한 것(네트워크, 외부 링크, 커널 실행)은 *확인 못 했다고* 쓴다. 우회하지 않는다.
- 데이터 누수 방지: 분할 후 학습 구간에서만 적합. 평가 데이터로 설정을 바꾸지 않는다.

## 약속
- target 1=부도, WOE=ln(%bad/%good), Score=Offset+Factor·ln(odds_good).
- 원본 데이터는 수정하지 않고 복사본을 정제하며 cleaning_log 를 남긴다.
- 합성데이터 결과를 실제 성능처럼 말하지 않는다.

## 이론 페이지 규칙
- 각 단계는 10개 항목(왜 생겼는가 … 참고자료)과 퀴즈를 가진다.
- 코드 블록은 `tools/check_theory_code.py` 로 실행 검증된다. 설명용은 첫 줄에 `# [설명용]`, 선택 패키지는 `# [requires: pkg]`.
- 표 셀 안의 수식에 `|` 를 쓰지 않는다(`\mid` 사용).
- 숫자를 바꾸면 `tests/test_theory_numbers.py` 도 갱신한다.

## 검증 명령
```
python3 -B tools/run_tests.py
python3 -B tools/run_notebooks.py --solve
python3 -B tools/build_theory.py && python3 -B tools/check_theory_code.py
```

## 다음 작업 (Phase 4)
단계 9~10: `creditlab.evaluation`(AUROC·Gini·KS·보정·PSI/CSI) + 노트북 08~09(또는 명세 번호) + 테스트.
(Phase 3 완료: `creditlab.models`, `creditlab.scorecard`, 노트북 06~07)
