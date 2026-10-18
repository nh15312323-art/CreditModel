"""노트북 코드 셀을 위에서부터 실제로 실행해 오류가 없는지 검증한다.

Jupyter 커널이 없는 환경을 위한 간이 실행기다(마지막 식의 값은 Jupyter 처럼 출력한다).
  python tools/run_notebooks.py            # 학습자가 아직 아무것도 안 적은 상태로 실행
  python tools/run_notebooks.py --solve    # 퀴즈 정답·정제 실습 정답 예시를 넣고 실행
  python tools/run_notebooks.py -v         # 셀 출력도 보기
주의: 실제 Jupyter/Colab 에서의 렌더링(표·그림 표시)까지 검증하지는 않는다.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import os
import re
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]


def run_cell(source: str, ns: dict) -> str:
    """셀 하나를 실행하고 stdout(+마지막 식의 repr)을 돌려준다."""
    tree = ast.parse(source)
    last_expr = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = ast.Expression(tree.body.pop().value)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(tree, "<cell>", "exec"), ns)  # noqa: S102 (우리가 만든 노트북만 실행)
        if last_expr is not None:
            value = eval(compile(last_expr, "<cell>", "eval"), ns)  # noqa: S307
            if value is not None:
                print(repr(value) if not hasattr(value, "to_string") else value.to_string(max_rows=12))
    return buf.getvalue()


def solution_codes(cells: list[dict]) -> list[str]:
    """접힌 정답(<details>) 안의 python 블록들을 순서대로 모은다."""
    out = []
    for cell in cells:
        if cell["cell_type"] == "markdown":
            text = "".join(cell["source"])
            if "<details>" in text:
                m = re.search(r"```python\n(.*?)```", text, re.S)
                if m:
                    out.append(m.group(1))
    return out


def run_notebook(path: Path, solve: bool, verbose: bool) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    ns: dict = {"__name__": "__main__"}
    sols = solution_codes(nb["cells"]) if solve else []
    n_ex = 0
    ok = True
    n_code = 0
    cwd = os.getcwd()
    os.chdir(ROOT / "notebooks")  # 노트북은 notebooks/ 에서 열린다고 가정
    try:
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            n_code += 1
            src = "".join(cell["source"])
            try:
                out = run_cell(src, ns)
                if solve and ("cleaned = None" in src or "# [exercise]" in src):
                    if n_ex < len(sols):          # i 번째 실습 셀에는 i 번째 정답 블록을 넣는다
                        run_cell(sols[n_ex], ns)
                    n_ex += 1
                if solve and "answers = {" in src and "QUIZ" in ns:
                    ns["answers"] = {q.qid: q.answer for q in ns["QUIZ"]}
                if verbose and out.strip():
                    print(f"--- cell {i}\n{out.rstrip()[:900]}")
            except Exception:  # noqa: BLE001
                ok = False
                print(f"[FAIL] {path.name} cell {i}\n{src[:300]}\n{traceback.format_exc(limit=3)}")
                break
    finally:
        os.chdir(cwd)
    print(f"[{'OK' if ok else 'FAIL'}] {path.name}: 코드 셀 {n_code}개 {'모두 실행됨' if ok else '중 일부 실패'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    results = [run_notebook(p, args.solve, args.verbose) for p in sorted((ROOT / "notebooks").glob("*.ipynb"))]
    return 0 if results and all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
