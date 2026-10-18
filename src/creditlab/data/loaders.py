"""데이터 읽기·다운로드·데이터 레지스트리 기록.

원칙
  * 공식 다운로드 주소만 사용한다. 로그인/약관/접근제한을 우회하지 않는다.
  * 다운로드가 실패하면 수동 업로드 방법을 안내한다.
  * 내려받은 파일은 SHA-256 해시와 수집일을 기록한다.
"""

from __future__ import annotations

import hashlib
import io
import urllib.error
import urllib.parse
import urllib.request
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .synthetic import ID_COL as SYNTHETIC_ID_COL
from .synthetic import TARGET as SYNTHETIC_TARGET
from .synthetic import make_synthetic_credit_data

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
UCI_PAGE_URL = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
UCI_TARGET_RAW_NAMES = ("default payment next month", "default.payment.next.month")

SUPPORTED_SUFFIXES = (".csv", ".tsv", ".parquet", ".xls", ".xlsx")


class DownloadError(RuntimeError):
    """자동 다운로드 실패. 메시지에 수동 대안이 들어 있다."""


def file_sha256(path) -> str:
    """파일의 SHA-256 해시(16진수)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_bytes_as_table(name: str, data: bytes, **kwargs) -> pd.DataFrame:
    suffix = Path(name).suffix.lower()
    buf = io.BytesIO(data)
    if suffix == ".csv":
        return pd.read_csv(buf, **kwargs)
    if suffix == ".tsv":
        return pd.read_csv(buf, sep="\t", **kwargs)
    if suffix == ".parquet":
        kwargs.pop("header", None)
        return pd.read_parquet(buf, **kwargs)
    if suffix in (".xls", ".xlsx"):
        try:
            return pd.read_excel(buf, **kwargs)
        except ImportError as exc:
            engine = "xlrd" if suffix == ".xls" else "openpyxl"
            raise ImportError(
                f"엑셀 파일({suffix})을 읽으려면 '{engine}' 패키지가 필요합니다. "
                f"Colab/터미널에서 `pip install {engine}` 를 실행한 뒤 다시 시도하세요. "
                "또는 엑셀을 CSV 로 저장해서 올려도 됩니다."
            ) from exc
    raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix} (지원: {', '.join(SUPPORTED_SUFFIXES)})")


def load_table(path, **read_kwargs) -> pd.DataFrame:
    """CSV/TSV/Parquet/Excel/ZIP(안에 위 형식 파일이 있는 경우) 을 DataFrame 으로 읽는다."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {p}\n"
            "확인 방법: 파일 이름·경로·확장자를 다시 보세요. Colab 이라면 왼쪽 폴더 아이콘에서 "
            "업로드가 끝났는지 확인하세요."
        )
    suffix = p.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(p) as zf:
            members = [m for m in zf.namelist() if Path(m).suffix.lower() in SUPPORTED_SUFFIXES]
            if not members:
                raise ValueError(f"zip 안에 읽을 수 있는 표 파일이 없습니다: {zf.namelist()}")
            return _read_bytes_as_table(members[0], zf.read(members[0]), **read_kwargs)
    return _read_bytes_as_table(p.name, p.read_bytes(), **read_kwargs)


def load_uci_default_credit(path) -> pd.DataFrame:
    """UCI 'Default of Credit Card Clients' 파일(xls/csv/zip)을 읽어 표준 형태로 만든다.

    원본 엑셀은 1행이 X1..X23/Y 라벨이고 실제 컬럼명은 2행에 있다(header=1).
    타깃 컬럼 'default payment next month' 는 'default' 로 바꾼다. 그 외 컬럼명은 그대로 둔다.
    """
    df = load_table(path, header=1)
    if "ID" not in df.columns:  # 1행에 컬럼명이 있는 CSV 판본
        df = load_table(path, header=0)
    df.columns = [str(c).strip() for c in df.columns]
    target_raw = next((c for c in df.columns if c in UCI_TARGET_RAW_NAMES), None)
    if target_raw is None:
        raise ValueError(
            "UCI 타깃 컬럼('default payment next month')을 찾지 못했습니다. "
            f"현재 컬럼: {list(df.columns)[:12]} ... 올바른 파일인지 확인하세요."
        )
    df = df.rename(columns={target_raw: "default"})
    df.attrs["source"] = "uci_default_credit_card"
    return df


def download_file(url: str, dest_dir, filename: str | None = None, timeout: int = 60) -> dict:
    """공식 주소에서 파일을 내려받고 해시·수집일을 기록해 돌려준다."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = filename or Path(urllib.parse.urlparse(url).path).name or "download.bin"
    dest = dest_dir / filename
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (공식 URL만 사용)
            data = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DownloadError(
            f"자동 다운로드에 실패했습니다: {url}\n원인: {exc}\n"
            f"수동 대안: 브라우저로 {UCI_PAGE_URL} 에서 파일을 내려받아 "
            f"'{dest_dir}' 폴더에 넣은 뒤 다시 실행하세요. (Colab 은 왼쪽 파일 패널에 업로드)"
        ) from exc
    dest.write_bytes(data)
    return {
        "path": str(dest),
        "url": url,
        "bytes": len(data),
        "sha256": file_sha256(dest),
        "collected_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }


def _find_local_uci_file(raw_dir: Path) -> Path | None:
    patterns = ["default of credit card clients*", "default+of+credit+card+clients*", "UCI_Credit_Card*"]
    for pattern in patterns:
        for candidate in sorted(raw_dir.glob(pattern)):
            if candidate.suffix.lower() in SUPPORTED_SUFFIXES + (".zip",):
                return candidate
    return None


def load_learning_data(
    dataset: str = "uci",
    raw_dir="data/raw",
    allow_download: bool = True,
    seed: int = 42,
    inject_quality_issues: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """학습용 데이터를 불러온다. 돌려주는 값: (DataFrame, 메타정보 dict).

    dataset="uci"        : UCI 실데이터. 로컬 파일 → 공식 다운로드 순으로 시도한다.
                           (다운로드·엑셀 엔진 문제로 실패하면 합성 데이터로 대체하고,
                           meta["source"] 와 meta["fallback_reason"] 에 분명히 남긴다.)
    dataset="synthetic"  : 합성 데이터(인터넷 불필요).
    """
    raw_dir = Path(raw_dir)
    loaded_on = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    fallback_reason = None

    if dataset == "uci":
        try:
            local = _find_local_uci_file(raw_dir) if raw_dir.exists() else None
            download_info = None
            if local is None and allow_download:
                download_info = download_file(UCI_ZIP_URL, raw_dir)
                local = Path(download_info["path"])
            if local is not None:
                df = load_uci_default_credit(local)
                meta = {
                    "dataset_id": "uci_default_credit_card",
                    "source": "uci",
                    "path": str(local),
                    "sha256": file_sha256(local),
                    "loaded_on": loaded_on,
                    "target": "default",
                    "id_col": "ID",
                    "license": "CC BY 4.0 (UCI, 2026-10-18 확인)",
                    "note": "대만 카드 고객 2005년 데이터. 한국 은행 실제 모집단을 대표하지 않는다.",
                }
                if download_info:
                    meta["downloaded_from"] = download_info["url"]
                return df, meta
            fallback_reason = "로컬에 UCI 파일이 없고 allow_download=False 입니다."
        except (DownloadError, ImportError) as exc:
            fallback_reason = str(exc).splitlines()[0]
        warnings.warn(
            f"UCI 데이터를 불러오지 못해 합성 데이터로 대체합니다. 이유: {fallback_reason}",
            stacklevel=2,
        )
    elif dataset != "synthetic":
        raise ValueError("dataset 은 'uci' 또는 'synthetic' 이어야 합니다.")

    df = make_synthetic_credit_data(seed=seed, inject_quality_issues=inject_quality_issues)
    meta = {
        "dataset_id": "synthetic_loan_v1",
        "source": "synthetic",
        "path": None,
        "sha256": None,
        "loaded_on": loaded_on,
        "target": SYNTHETIC_TARGET,
        "id_col": SYNTHETIC_ID_COL,
        "license": "프로젝트 내장 생성기 (실제 고객 데이터 아님)",
        "note": "알려진 확률식으로 만든 가짜 데이터. 성능 수치는 실제 모집단의 근거가 아니다.",
        "seed": seed,
    }
    if fallback_reason:
        meta["fallback_reason"] = fallback_reason
    return df, meta
