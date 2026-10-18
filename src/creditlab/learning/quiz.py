"""노트북에서 쓰는 가벼운 퀴즈·힌트 도구.

설계 의도
  * 정답을 input() 으로 받지 않고 변수에 적게 한다(헤드리스 실행·재실행이 쉽다).
  * 힌트는 3단계(개념 → 절차 → 코드 일부)로 열고, 마지막에 해설을 보여준다.
  * 틀린 문제의 concept 목록을 돌려줘서 다음 레슨에서 다시 출제(간격 반복)할 수 있게 한다.
점수는 학습 점검용이며 실무 자격이나 규제 승인을 의미하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    qid: str
    concept: str
    prompt: str
    choices: dict
    answer: str
    explanation: str
    hints: tuple = ()


def show(question: Question) -> None:
    """문제와 보기를 출력한다."""
    print(f"[{question.qid}] {question.prompt}")
    for key, text in question.choices.items():
        print(f"  {key}. {text}")


def hint(question: Question, level: int) -> None:
    """level 1~3 힌트를 출력한다. 힌트가 없으면 안내만 한다."""
    if not question.hints:
        print("이 문제에는 힌트가 없습니다. 개념 설명 페이지를 다시 읽어 보세요.")
        return
    if level < 1:
        raise ValueError("level 은 1 이상이어야 합니다.")
    idx = min(level, len(question.hints)) - 1
    print(f"힌트 {idx + 1}/{len(question.hints)}: {question.hints[idx]}")


def check(question: Question, answer) -> bool | None:
    """답을 채점해 정오를 출력한다. 답이 비어 있으면 None 을 돌려준다(채점 안 함)."""
    if answer is None or str(answer).strip() == "":
        print(f"[{question.qid}] 아직 답을 적지 않았습니다. 위 문제에 answer 를 적고 다시 실행하세요.")
        return None
    normalized = str(answer).strip().upper()
    if normalized not in {k.upper() for k in question.choices}:
        print(f"[{question.qid}] 보기에 없는 답입니다: {answer!r} (선택지: {', '.join(question.choices)})")
        return None
    correct = normalized == question.answer.upper()
    print(("정답입니다. " if correct else "오답입니다. ") + question.explanation)
    return correct


def grade(questions: list, answers: dict, pass_ratio: float = 0.75) -> dict:
    """여러 문제를 한 번에 채점한다.

    answers = {"Q1": "B", ...}. 답이 없거나 틀린 문제는 wrong 에 들어가고,
    그 문제들의 concept 가 concepts_to_review 가 된다.
    """
    if not questions:
        raise ValueError("questions 가 비어 있습니다.")
    correct_ids, wrong_ids = [], []
    for q in questions:
        given = answers.get(q.qid)
        ok = given is not None and str(given).strip().upper() == q.answer.upper()
        (correct_ids if ok else wrong_ids).append(q.qid)
    wrong_concepts = sorted({q.concept for q in questions if q.qid in wrong_ids})
    score = len(correct_ids)
    total = len(questions)
    return {
        "score": score,
        "total": total,
        "ratio": score / total,
        "passed": score / total >= pass_ratio,
        "wrong": wrong_ids,
        "concepts_to_review": wrong_concepts,
    }
