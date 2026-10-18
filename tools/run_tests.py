"""pytest 가 없는 환경(제한된 네트워크 등)에서 tests/ 를 실행하는 대체 실행기.

사용법:  python tools/run_tests.py
pytest 를 설치할 수 있다면 `pytest -q` 를 쓰는 편이 낫다(같은 테스트를 그대로 실행한다).
이 실행기는 'test_' 로 시작하는 함수만 인자 없이 호출하며, fixture/parametrize 는 지원하지 않는다.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]


def main() -> int:
    passed, failed = 0, []
    started = time.time()
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:  # noqa: BLE001
            failed.append((f"{path.name} (import)", traceback.format_exc()))
            continue
        for name in sorted(n for n in dir(module) if n.startswith("test_")):
            func = getattr(module, name)
            if not callable(func):
                continue
            try:
                func()
                passed += 1
                print(f"PASS  {path.name}::{name}")
            except Exception:  # noqa: BLE001
                failed.append((f"{path.name}::{name}", traceback.format_exc()))
                print(f"FAIL  {path.name}::{name}")
    print("\n" + "=" * 60)
    for name, tb in failed:
        print(f"\n--- {name}\n{tb}")
    print(f"{passed} passed, {len(failed)} failed in {time.time() - started:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
