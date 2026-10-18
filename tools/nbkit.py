"""nbformat 없이 .ipynb(JSON, nbformat 4) 를 만드는 최소 도구와 공통 셀."""

from __future__ import annotations

import json
from pathlib import Path


def _lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def write_notebook(path, cells: list[dict], title: str) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "name": title},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    # nbformat 4.5 는 셀 id 가 필요하다
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"c{i:03d}"
    Path(path).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


SETUP_CELL = code(
    '''
# [환경 준비] 맨 위 셀부터 순서대로 실행하세요. (Colab/로컬 모두 동작)
import os
import subprocess
import sys
import warnings
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/nh15312323-art/CreditModel.git"


def find_or_fetch_project() -> Path:
    for base in (Path.cwd(), Path.cwd().parent, Path("/content/credit-model-mastery-lab")):
        if (base / "src" / "creditlab").exists():
            return base
    if not IN_COLAB:
        raise RuntimeError("프로젝트 폴더를 찾지 못했습니다. 저장소 루트(또는 notebooks/)에서 노트북을 여세요.")
    if REPO_URL:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "/content/credit-model-mastery-lab"], check=True)
    else:
        from google.colab import files  # Colab 전용

        print("credit-model-mastery-lab.zip 파일을 선택해 업로드하세요.")
        files.upload()
        subprocess.run(["unzip", "-q", "-o", "credit-model-mastery-lab.zip", "-d", "/content"], check=True)
    return Path("/content/credit-model-mastery-lab")


PROJECT_ROOT = find_or_fetch_project()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
if IN_COLAB:  # UCI 원본이 엑셀(.xls)이라 읽기 엔진이 필요합니다
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "xlrd", "openpyxl"], check=False)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import creditlab
from creditlab import environment_info, set_seed

SEED = set_seed(42)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 140)
print("프로젝트 위치:", PROJECT_ROOT)
print("creditlab", creditlab.__version__, "| 시드", SEED)
print({k: v for k, v in environment_info(SEED)["packages"].items()})
'''
)
