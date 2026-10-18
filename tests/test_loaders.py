import tempfile
import warnings
import zipfile
from pathlib import Path

import pandas as pd
from helpers import expect_raises

from creditlab.data.loaders import (
    DownloadError,
    download_file,
    file_sha256,
    load_learning_data,
    load_table,
    load_uci_default_credit,
)

# 원본 UCI 엑셀처럼 1행=X1..Y 라벨, 2행=실제 컬럼명 인 가짜 CSV
UCI_LIKE_TWO_HEADERS = (
    "X0,X1,X2,Y\n"
    "ID,LIMIT_BAL,PAY_0,default payment next month\n"
    "1,20000,2,1\n"
    "2,120000,-1,0\n"
    "3,90000,0,0\n"
)
UCI_LIKE_ONE_HEADER = "ID,LIMIT_BAL,PAY_0,default.payment.next.month\n1,20000,2,1\n2,120000,-1,0\n"


def _tmp():
    return tempfile.TemporaryDirectory()


def test_sha256_known_value():
    with _tmp() as d:
        p = Path(d) / "abc.txt"
        p.write_bytes(b"abc")
        assert file_sha256(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_load_csv_and_zip_roundtrip():
    with _tmp() as d:
        csv = Path(d) / "t.csv"
        csv.write_text("a,b\n1,x\n2,y\n", encoding="utf-8")
        df = load_table(csv)
        assert list(df.columns) == ["a", "b"] and len(df) == 2
        z = Path(d) / "t.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.write(csv, "inner/t.csv")
        pd.testing.assert_frame_equal(load_table(z), df)


def test_load_table_errors_are_helpful():
    with expect_raises(FileNotFoundError, "찾을 수 없습니다"):
        load_table("/no/such/file.csv")
    with _tmp() as d:
        p = Path(d) / "x.txt"
        p.write_text("hi")
        with expect_raises(ValueError, "지원하지 않는"):
            load_table(p)
        empty_zip = Path(d) / "e.zip"
        with zipfile.ZipFile(empty_zip, "w") as zf:
            zf.writestr("readme.md", "no table")
        with expect_raises(ValueError, "읽을 수 있는 표 파일이 없습니다"):
            load_table(empty_zip)


def test_load_uci_two_header_rows_and_rename_target():
    with _tmp() as d:
        p = Path(d) / "uci.csv"
        p.write_text(UCI_LIKE_TWO_HEADERS, encoding="utf-8")
        df = load_uci_default_credit(p)
        assert list(df.columns) == ["ID", "LIMIT_BAL", "PAY_0", "default"]
        assert list(df["default"]) == [1, 0, 0]
        assert df.attrs["source"] == "uci_default_credit_card"


def test_load_uci_single_header_variant():
    with _tmp() as d:
        p = Path(d) / "kaggle.csv"
        p.write_text(UCI_LIKE_ONE_HEADER, encoding="utf-8")
        df = load_uci_default_credit(p)
        assert "default" in df.columns and len(df) == 2


def test_load_uci_wrong_file_is_explained():
    with _tmp() as d:
        p = Path(d) / "other.csv"
        p.write_text("ID,x\n1,2\n", encoding="utf-8")
        with expect_raises(ValueError, "타깃 컬럼"):
            load_uci_default_credit(p)


def test_load_learning_data_synthetic():
    df, meta = load_learning_data("synthetic", seed=5)
    assert meta["source"] == "synthetic" and meta["target"] == "default"
    assert meta["id_col"] in df.columns and len(df) == 5000


def test_load_learning_data_unknown_dataset():
    with expect_raises(ValueError):
        load_learning_data("nope")


def test_load_learning_data_local_uci_file_records_hash():
    with _tmp() as d:
        p = Path(d) / "UCI_Credit_Card.csv"
        p.write_text(UCI_LIKE_ONE_HEADER, encoding="utf-8")
        df, meta = load_learning_data("uci", raw_dir=d, allow_download=False)
        assert meta["source"] == "uci" and meta["sha256"] == file_sha256(p)
        assert "default" in df.columns


def test_uci_fallback_is_loud_and_recorded():
    with _tmp() as d:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df, meta = load_learning_data("uci", raw_dir=d, allow_download=False)
        assert meta["source"] == "synthetic"
        assert "fallback_reason" in meta
        assert any("합성 데이터로 대체" in str(w.message) for w in caught)


def test_download_failure_gives_manual_alternative():
    with _tmp() as d:
        with expect_raises(DownloadError, "수동 대안"):
            download_file("http://127.0.0.1:9/none.zip", d, timeout=2)
