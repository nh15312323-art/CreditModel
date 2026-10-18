"""재현성 관련 도우미: 난수 시드 고정과 실행 환경 기록."""

from __future__ import annotations

import platform
import random
import sys
from datetime import datetime, timezone

import numpy as np

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """파이썬/NumPy 난수 시드를 고정하고 사용한 시드를 돌려준다."""
    random.seed(seed)
    np.random.seed(seed)
    return seed


def _version_of(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except ImportError:
        return "not installed"
    return getattr(module, "__version__", "unknown")


def environment_info(seed: int | None = None) -> dict:
    """보고서에 기록할 실행 환경 정보를 딕셔너리로 돌려준다."""
    from . import __version__

    packages = ["numpy", "pandas", "scipy", "sklearn", "matplotlib"]
    return {
        "creditlab": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: _version_of(name) for name in packages},
        "seed": seed,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }
