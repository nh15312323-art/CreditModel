# 단계 0. Python과 데이터 분석 기초

> 신용모형 개발에 필요한 최소한의 Python·pandas 도구를 익히고, 작은 대출 데이터를 읽어 품질 점검까지 할 수 있게 된다.

**연결 실습**: 노트북 `01~03` 에서 이 도구들을 바로 사용한다. 이 페이지의 코드는 복사해서 Colab 셀에 붙여 실행할 수 있다.

## 학습 목표

- 변수·자료형·조건문·반복문·함수·예외 처리를 읽고 쓸 수 있다.
- 리스트·딕셔너리·집합·튜플을 용도에 맞게 고른다.
- NumPy 배열과 pandas `Series`/`DataFrame` 을 만들고 인덱싱·필터링·그룹화·병합한다.
- 결측치·자료형·날짜를 다루고 파일을 읽고 쓴다.
- 작은 함수를 만들고 `assert` 로 검증한다.

## 1. 왜 생겼는가

엑셀로도 표를 다룰 수 있지만, 모형 개발은 같은 분석을 **수십 번 반복**하고, 수만~수백만 행을 다루며, 결과를 **그대로 다시 재현**해야 한다. 클릭으로 한 작업은 재현이 어렵고 실수가 보이지 않는다. 코드는 작업 과정 자체가 기록이 된다. Python 은 문법이 쉽고, 데이터 분석(pandas·NumPy)과 통계·머신러닝(SciPy·scikit-learn) 도구가 한 언어에 모여 있어 모형 개발의 표준 도구가 되었다.

## 2. 왜 필요한가

신용평가모형 개발의 거의 모든 의사결정(표본 선정, 제외, 결측 처리, 비닝, 분할, 평가)은 결국 **표 형태 데이터를 조작하는 코드**로 구현된다. 코드를 읽지 못하면 결과가 왜 그렇게 나왔는지 설명할 수 없고, 설명할 수 없는 모형은 검증받기 어렵다.

## 3. 무엇을 하는가

| 도구 | 하는 일 | 모형 개발에서의 쓰임 |
|---|---|---|
| Python 기본 문법 | 계산·조건·반복·함수 | 자동화, 검증 로직 |
| 컬렉션(list/dict/set/tuple) | 값을 묶어 보관 | 변수 목록, 설정, 매핑 |
| NumPy | 숫자 배열 연산 | 확률·로그·벡터 연산 |
| pandas | 표 데이터 처리 | 읽기, 필터, 그룹, 병합 |
| matplotlib | 그림 | 분포·부도율 그래프 |
| `assert` / 테스트 | 계산 검증 | 지표 함수의 정확성 확인 |

## 4. 어떻게 작동하는가

### 4-1. 값과 자료형

변수는 값에 붙인 이름이다. 값에는 종류(자료형)가 있고, 종류에 따라 가능한 연산이 다르다.

```python
n_total = 10_000          # int(정수)
n_default = 450
rate = n_default / n_total  # float(실수)  → 0.045
name = "개인신용"           # str(문자열)
is_default = rate > 0.04    # bool(참/거짓)
print(rate, type(rate).__name__, is_default)
assert abs(rate - 0.045) < 1e-12
```

> 직관: `/` 는 항상 실수를 돌려주고 `//` 는 몫, `%` 는 나머지다. 부도율처럼 비율을 구할 때는 분모가 0이 아닌지 먼저 확인한다(0으로 나누면 `ZeroDivisionError`).

### 4-2. 조건문·반복문·함수·예외 처리

```python
def default_rate(n_default, n_total):
    """부도율 = 부도 건수 / 전체 건수. 분모가 0 이하면 계산하지 않고 멈춘다."""
    if n_total <= 0:
        raise ValueError("전체 건수는 1 이상이어야 합니다.")
    return n_default / n_total

for d, n in [(45, 1000), (3, 50), (0, 200)]:
    print(d, n, f"{default_rate(d, n):.1%}")

try:
    default_rate(1, 0)
except ValueError as e:
    print("오류를 잡았습니다:", e)
```

함수는 *입력 → 출력* 이 분명해야 한다. 이 프로그램의 `creditlab` 함수들은 모두 이 원칙으로 만들어져 있다. 예외는 숨기지 말고, 무엇이 문제인지 알려 주며 멈추는 것이 안전하다.

### 4-3. 컬렉션 네 가지

| 자료형 | 모양 | 특징 | 신용모형 예 |
|---|---|---|---|
| list | `[1, 2, 3]` | 순서 있음, 변경 가능 | 변수 목록 |
| dict | `{"age": 35}` | 이름→값, 변경 가능 | 변수 설명, 설정 |
| set | `{1, 2, 3}` | 중복 없음, 순서 없음 | 두 변수 목록의 차이 |
| tuple | `(18, 100)` | 변경 불가 | (하한, 상한) 범위 규칙 |

```python
features_dev = ["age", "income", "utilization"]
features_prod = ["age", "income", "tenure"]
only_in_dev = set(features_dev) - set(features_prod)     # 개발에만 있는 변수
only_in_prod = set(features_prod) - set(features_dev)
print(sorted(only_in_dev), sorted(only_in_prod))
assert only_in_dev == {"utilization"}
```

### 4-4. NumPy: 숫자 배열

```python
import numpy as np

p = np.array([0.01, 0.05, 0.10, 0.50])
odds = p / (1 - p)           # 원소별 계산 (반복문 없이)
logit = np.log(odds)
print(np.round(odds, 4), np.round(logit, 4))
assert np.allclose(logit[1], np.log(0.05 / 0.95))
```

한 번에 배열 전체를 계산하는 것을 **벡터화**라 한다. 반복문보다 짧고 빠르다. 로지스틱 회귀의 `odds`, `logit` 계산이 바로 이 형태다(단계 6).

### 4-5. pandas: 표 데이터

```python
import pandas as pd

df = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6],
    "age": [25, 40, 33, 58, None, 47],
    "region": ["A", "B", "A", "C", "B", "A"],
    "default": [0, 1, 0, 0, 1, 0],
})
print(df.shape)                      # (행, 열)
print(df["age"].isna().sum())        # 결측 건수
print(df.loc[df["age"] > 30, "id"].tolist())    # 조건 필터 (결측은 False 로 취급됨)
by_region = df.groupby("region")["default"].agg(n="size", rate="mean")
print(by_region)
assert by_region.loc["A", "n"] == 3 and abs(by_region.loc["A", "rate"]) < 1e-12
```

핵심 동작 네 가지만 기억하자.

- **선택**: `df["col"]`, `df.loc[행조건, 열]`, `df.iloc[위치]`
- **필터**: `df[df["age"] > 30]` — 결측(`NaN`)은 비교가 항상 `False`가 되어 조용히 빠진다. 결측이 있는 열로 필터링할 때는 *빠진 행이 얼마인지* 확인한다.
- **그룹화**: `groupby(...).agg(...)` 으로 부도율·건수를 구한다(단계 5의 비닝 표가 이것이다).
- **병합**: `pd.merge(left, right, on="id", how="left", validate="one_to_one")`. `validate` 를 쓰면 *중복 키로 행이 늘어나는 사고*를 막는다.

```python
accounts = pd.DataFrame({"id": [1, 2, 3], "limit": [500, 800, 300]})
customers = pd.DataFrame({"id": [1, 2, 2, 3], "age": [30, 40, 40, 50]})   # id=2 중복
try:
    pd.merge(accounts, customers, on="id", how="left", validate="one_to_one")
except pd.errors.MergeError as e:
    print("병합 검증이 중복 키를 잡았습니다")
```

### 4-6. 날짜와 파일 입출력

```python
dates = pd.to_datetime(pd.Series(["2024-01-31", "2024-02-29", "2024-03-15"]))
print(dates.dt.to_period("M").astype(str).tolist())    # 월 단위로 묶기
assert dates.dt.month.tolist() == [1, 2, 3]

# 파일 읽기/쓰기 (설명용: 실제 경로에 맞게 바꿔 사용)
# df = pd.read_csv("data/raw/file.csv")
# df.to_csv("data/processed/out.csv", index=False)
```

날짜는 문자열로 두지 말고 `datetime` 으로 바꿔야 *시간 기준 분할*(단계 8)을 할 수 있다.

### 4-7. 그림과 테스트

```python
import matplotlib
matplotlib.use("Agg")      # 화면 없는 환경용. Colab 에서는 생략해도 된다
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 3))
by_region["rate"].plot.bar(ax=ax)
ax.set_ylabel("default rate")
ax.set_title("Default rate by region")
plt.close(fig)
```

`assert 조건, "메시지"` 는 조건이 거짓이면 즉시 멈춘다. 지표 함수를 만들면 **손으로 계산 가능한 작은 예**로 `assert` 를 달아 둔다. 이 프로그램의 `tests/` 가 그 모음이다.

## 5. 가정

- 데이터는 **행 = 관측, 열 = 변수** 인 *정돈된(tidy)* 표라고 가정한다.
- 같은 열의 값은 같은 의미·단위·자료형이어야 한다 (단위가 섞이면 코드가 정상이어도 결과는 틀린다).
- 코드를 *위에서 아래로 순서대로* 실행한다고 가정한다. 노트북에서 셀을 건너뛰어 실행하면 변수 상태가 달라져 재현되지 않는다.

## 6. 한계와 오용

!!! warning "자주 하는 실수"
    - **원본을 덮어쓰기**: `df["age"] = ...` 는 원본을 바꾼다. 정제는 `df.copy()` 로 복사본에서 한다.
    - **결측 필터의 함정**: `df[df["x"] != 3]` 은 `x` 가 결측인 행을 *포함*하고, `df[df["x"] == 3]` 은 *제외*한다. 의도와 맞는지 건수를 확인한다.
    - **정수 나눗셈·형 변환**: 문자로 읽힌 숫자("1,000")는 계산이 안 되거나 이어붙기가 된다.
    - **병합 후 행 수가 늘어남**: 키 중복 때문이다. `validate=` 로 막는다.
    - **노트북 상태 의존**: 실행 순서가 뒤섞이면 어제 결과가 오늘 재현되지 않는다. 마지막엔 *런타임 재시작 후 모두 실행*으로 확인한다.

## 7. 대안 및 비교

| 대안 | 장점 | 한계 |
|---|---|---|
| 엑셀 | 즉시 눈으로 확인 | 재현·대용량·자동화 어려움 |
| SQL | 큰 DB 에서 추출·집계에 강함 | 통계·모형 학습은 별도 도구 필요 |
| R | 통계 패키지 풍부 | 팀·시스템 연동 환경에 따라 선택 |
| SAS | 금융권 레거시·규제 이력 | 라이선스·확장성, 최신 ML 생태계 |

실무에서는 *SQL 로 추출 → Python 으로 분석·모형 → 문서 자동화* 조합이 흔하다. 어떤 도구든 **재현 가능한 스크립트**라는 원칙은 같다.

## 8. 실무 적용

- 모든 분석은 노트북/스크립트로 남기고, **난수 시드·패키지 버전·데이터 버전**을 기록한다 (`creditlab.environment_info`).
- 데이터 이동은 로컬/사내망 안에서 한다. 실제 고객 데이터를 외부 서비스에 올리지 않는다.
- 변수명·단위·정의를 **데이터 사전**으로 남긴다 (`creditlab.data.data_dictionary`).
- 코드 리뷰와 검증팀 재현을 위해 *함수 단위*로 쪼개고 테스트한다.

## 9. 지식 확인

**Q1.** `df[df["age"] > 30]` 을 실행했더니 6행 중 4행이 남았다. 사라진 2행에 대해 반드시 확인할 것은?

<details markdown="1"><summary>정답 보기</summary>

`age` 가 30 이하인 행과 **`age` 가 결측인 행**이 함께 빠진다. 결측 때문에 빠진 행이 몇 건인지 확인하고, 의도한 필터인지 판단한다.

</details>

**Q2.** 고객 테이블과 계좌 테이블을 `id` 로 병합했더니 행이 늘었다. 원인과 예방법은?

<details markdown="1"><summary>정답 보기</summary>

한쪽(또는 양쪽)의 키가 중복되어 행이 곱해졌다. `pd.merge(..., validate="one_to_one")`(또는 `many_to_one`)로 병합 전에 구조를 검증한다.

</details>

**Q3 (코드 읽기).** 다음 코드의 출력은?

```python
d = {"a": 1, "b": 2}
s = {1, 2, 2, 3}
t = (5, 6)
print(len(d), len(s), t[0])
```

<details markdown="1"><summary>정답 보기</summary>

`2 3 5`. 딕셔너리는 키 2개, 집합은 중복이 사라져 원소 3개, 튜플 첫 원소는 5.

</details>

**실습 과제**: 작은 대출 CSV(5~10행)를 직접 만들어 읽고, ① 결측률 ② 지역별 부도율 ③ 데이터 사전(열 이름·자료형·설명) 표를 만들어 보라. 노트북 `03_missing_and_quality.ipynb` 의 함수를 참고해도 된다.

## 10. 참고자료

- pandas 공식 문서: <https://pandas.pydata.org/docs/>
- NumPy 공식 문서: <https://numpy.org/doc/>
- Python 공식 튜토리얼: <https://docs.python.org/ko/3/tutorial/>
- matplotlib 공식 문서: <https://matplotlib.org/stable/>
- 확인일 2026-10-18 (링크 유효성은 학습 시점에 다시 확인할 것)
