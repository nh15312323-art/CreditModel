"""이론 페이지(lessons/theory/*.md)의 ```python 코드 블록을 실제로 실행해 오류를 찾는다.

규칙
  * 페이지마다 코드 블록을 위에서부터 같은 네임스페이스에서 실행한다(앞 블록의 변수를 뒤에서 쓸 수 있음).
  * 블록 첫 줄이  `# [설명용]`  이면 실행하지 않는다(경로·외부 데이터가 필요한 예시).
  * 블록 첫 줄이  `# [requires: xgboost]`  처럼 적혀 있고 그 패키지가 없으면 건너뛴다(건너뛴 개수를 보고한다).
사용법: python tools/check_theory_code.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import re
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FENCE = re.compile(r"^```python[ \t]*\n(.*?)^```", re.S | re.M)


def main() -> int:
    total = ran = skipped_doc = skipped_req = 0
    failures = []
    for path in sorted((ROOT / "lessons" / "theory").glob("*.md")):
        ns: dict = {"__name__": "__theory__"}
        for i, m in enumerate(FENCE.finditer(path.read_text(encoding="utf-8"))):
            code = m.group(1)
            total += 1
            first = code.lstrip().splitlines()[0] if code.strip() else ""
            if first.startswith("# [설명용]"):
                skipped_doc += 1
                continue
            req = re.match(r"# \[requires: ([\w,\s-]+)\]", first)
            if req and any(importlib.util.find_spec(r.strip().replace("-", "_")) is None for r in req.group(1).split(",")):
                skipped_req += 1
                continue
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(code, f"{path.name}#block{i}", "exec"), ns)  # noqa: S102
                ran += 1
            except Exception:  # noqa: BLE001
                failures.append((path.name, i, code[:200], traceback.format_exc(limit=2)))
    for name, i, snippet, tb in failures:
        print(f"[FAIL] {name} block {i}\n{snippet}\n{tb}")
    print(
        f"코드 블록 {total}개: 실행 성공 {ran}, 설명용 건너뜀 {skipped_doc}, "
        f"패키지 없음으로 건너뜀 {skipped_req}, 실패 {len(failures)}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
