from nbkit import SETUP_CELL, code, md

TITLE = "레슨 2-2. 비닝·WOE·IV"


def cells():
    return [
        md(
            """
# 레슨 2-2. 비닝·WOE·IV

**예상 시간** 80분 · **선행** 레슨 2-1 · **이론 페이지** `docs/theory/stage_05.html`

## 학습 목표
1. 연속형 변수를 **구간(bin)** 으로 나누고 구간별 부도율을 읽는다.
2. **WOE = ln(%bad/%good)** 와 **IV** 를 손으로 계산하고 코드 결과와 맞춘다.
3. **0 건 bin 과 스무딩**을 이해한다.
4. bin 수·방법에 따라 IV 가 달라지고, **개발 구간에서 만든 bin 이 새 데이터에서는 덜 맞는** 과적합을 확인한다.
5. 범주형 희소 범주(OTHER), 결측(MISSING), 학습에 없던 값의 처리 규칙을 안다.
6. **경계는 개발 구간에서만** 정하고, 테스트에는 그대로 적용한다.

> **부호 약속**: 이 프로그램에서 WOE 는 `ln(%bad/%good)` 입니다. **값이 클수록 위험**하고, 로지스틱 계수가 대체로 **양(+)** 이 됩니다. (Siddiqi 교재는 반대 부호를 쓰지만 IV 는 같습니다. 다른 자료를 볼 때 부호 정의를 꼭 확인하세요.)
"""
        ),
        SETUP_CELL,
        md(
            """
## 1. 개념

| 용어 | 뜻 |
|---|---|
| 비닝 | 연속형 변수를 구간으로 묶는 일. 구간이 너무 많으면 우연을 학습하고, 너무 적으면 정보를 잃는다 |
| WOE | 구간의 `ln(%bad/%good)`. %bad = 그 구간의 부도 ÷ 전체 부도, %good = 그 구간의 정상 ÷ 전체 정상 |
| IV | `Σ (%bad − %good) × WOE`. 변수 전체가 부도와 정상을 얼마나 가르는지의 요약(0 이상) |
| 스무딩 | 각 구간 건수에 작은 값(a)을 더해 0 건 문제를 피하는 방법 |

### 손계산 — 구간이 두 개뿐인 작은 예
`hi` 구간: 부도 8건, 정상 2건 / `lo` 구간: 부도 2건, 정상 8건 (총 부도 10, 총 정상 10)
"""
        ),
        code("pred_woe_hi = None  # ← 'hi' 구간의 WOE 는 양수일까 음수일까요? 크기는 대략? (예: 1.4)"),
        code(
            '''
from creditlab.features import woe_iv_from_counts

t = woe_iv_from_counts([8, 2], [2, 8], smoothing=0).assign(bin=["hi", "lo"])
print(t[["bin", "pct_bad", "pct_good", "woe", "iv"]].round(4).to_string(index=False))
print(f"\\n직접 계산: hi 의 %bad = 8/10 = 0.8, %good = 2/10 = 0.2 → WOE = ln(0.8/0.2) = ln 4 = {np.log(4):.4f}")
print(f"IV = (0.8−0.2)·ln4 + (0.2−0.8)·ln(1/4) = 1.2·ln4 = {1.2 * np.log(4):.4f}  (코드: {t['iv'].sum():.4f})")
if pred_woe_hi is not None:
    print("내 예측:", pred_woe_hi)
'''
        ),
        md("`hi`(위험한 구간)의 WOE 가 **양수**입니다. 위험할수록 큰 값 — 이것이 이 프로그램의 부호 약속입니다."),
        md("## 2. 데이터와 개발·테스트 구간\n앞 70%(개발)에서만 경계를 정하고, 뒤 30%(테스트)에는 **적용만** 합니다."),
        code(
            '''
from creditlab.data import load_learning_data
from creditlab.features import WoeBinner, monotonic_direction, plot_woe_table

df_all, meta = load_learning_data("synthetic", seed=SEED)
TARGET = meta["target"]
df_all = df_all.sort_values("application_date").reset_index(drop=True)
cut = int(len(df_all) * 0.7)
dev, test = df_all.iloc[:cut].copy(), df_all.iloc[cut:].copy()
print(f"개발 {len(dev):,}행 (부도 {int(dev[TARGET].sum())}건) | 테스트 {len(test):,}행 (부도 {int(test[TARGET].sum())}건)")
'''
        ),
        md("## 3. 비닝 표 만들기 — `card_utilization`(카드 사용률)"),
        code("pred_trend = None  # ← 사용률이 높을수록 부도율은: \"increasing\" / \"decreasing\" / \"none\""),
        code(
            '''
b = WoeBinner(method="quantile", max_bins=5).fit(dev["card_utilization"], dev[TARGET])
show_cols = ["bin", "n", "pct", "bad", "good", "bad_rate", "bad_rate_lo", "bad_rate_hi", "woe", "iv"]
print(b.table_[show_cols].round(3).to_string(index=False))
trend = monotonic_direction(b.table_["bad_rate"])
print(f"\\nIV 합계: {b.iv_:.4f} | 부도율 추세: {trend}")
if pred_trend:
    print("내 예측:", pred_trend, "→", "맞음" if pred_trend == trend else "틀림")
plot_woe_table(b.table_, "card_utilization: share and default rate (dev)")
plt.tight_layout()
plt.show()
'''
        ),
        md(
            """
**읽는 법**
- `bad_rate` 가 구간별로 올라가면 WOE 도 같이 올라갑니다(위험할수록 큰 값).
- `bad_rate_lo~hi`(95% 구간)가 이웃 구간과 많이 겹치면 구간을 나눈 근거가 약합니다.
- 구간은 오른쪽 닫힘 `(a, b]` — 경계값과 같은 값은 **아래쪽** 구간입니다.
"""
        ),
        md("## 4. bin 수와 방법에 따라 IV 가 달라진다 — 그리고 새 데이터에서는?"),
        md(
            """
개발 구간에서 정한 **경계와 WOE 값**을 테스트 구간에 그대로 적용했을 때의 *분리 정도*를 `테스트 적용 IV` 로 비교합니다.

`테스트 적용 IV` = Σ (테스트 %bad − 테스트 %good) × **개발 WOE** (= 테스트에서 부도 그룹의 평균 WOE − 정상 그룹의 평균 WOE)

개발 IV 는 같은 식을 개발 데이터에 적용한 값입니다. 두 값의 차이가 클수록 *개발 데이터의 우연*을 외운 정도가 큽니다.
"""
        ),
        code("pred_overfit = None  # ← 소득(income_10k)처럼 부도와 거의 무관한 변수에서 bin 을 40개로 늘리면 개발 IV 는 (오른다/내려간다), 테스트 적용 IV 는 (0 근처/크게 오른다)"),
        code(
            '''
def applied_iv(data, binner, var):
    """binner(개발에서 학습)의 경계·WOE 를 data 에 그대로 적용한 IV. data 자신의 %bad, %good 에 개발 WOE 를 곱해 합한다."""
    bins = binner.transform_bins(data[var])
    ct = pd.crosstab(bins, data[TARGET]).reindex(binner.table_["bin"]).fillna(0)
    pb, pg = ct[1] / ct[1].sum(), ct[0] / ct[0].sum()
    return float(((pb - pg) * binner.table_["woe"].to_numpy()).sum())

rows = []
for var in ["card_utilization", "income_10k"]:
    for method in ["quantile", "tree"]:
        for k in [3, 5, 10, 20, 40]:
            bb = WoeBinner(method=method, max_bins=k, min_bin_pct=0.0, min_events=0).fit(dev[var], dev[TARGET])
            rows.append({"변수": var, "방법": method, "max_bins": k, "실제 bin 수": len(bb.table_),
                         "개발 IV": bb.iv_, "테스트 적용 IV": applied_iv(test, bb, var)})
res = pd.DataFrame(rows)
res["차이(개발−테스트)"] = res["개발 IV"] - res["테스트 적용 IV"]
print(res.round(3).to_string(index=False))
if pred_overfit:
    print("\\n내 예측:", pred_overfit)
'''
        ),
        md(
            """
**읽는 법**
- 소득처럼 부도와 거의 무관한 변수에서도 bin 을 잘게 나누면 **개발 IV 가 계속 오릅니다.** 구간이 많을수록 우연히 갈린 차이를 담기 때문입니다. 그런데 같은 경계·WOE 를 테스트에 적용하면 **0 근처(때로는 음수)** 입니다 — 정보가 없었다는 뜻입니다.
- 정보가 있는 변수(카드 사용률)도 나무 기반 bin 을 20~40개로 늘리면 개발 IV 만 크게 오르고 테스트 적용 IV 는 오르지 않습니다. 이것이 **타깃을 보고 최적화한 bin 의 과적합**입니다.
- 반대로 bin 이 적을 때는 테스트 적용 IV 가 개발 IV 보다 *더 크게* 나오기도 합니다. 테스트에는 부도가 몇백 건뿐이라 *우연한 변동*이 있기 때문입니다. **작은 차이는 해석하지 마세요.** 큰 방향(개발만 오르는가)을 봅니다.
- IV 는 변수가 미래에도 예측력이 있다는 증명이 아닙니다.
"""
        ),
        md("### 구간별 부도율이 개발과 테스트에서 같은 모양인가?"),
        code(
            '''
b5 = WoeBinner(method="quantile", max_bins=5).fit(dev["card_utilization"], dev[TARGET])
bins_t = b5.transform_bins(test["card_utilization"])
t_rate = test.groupby(bins_t.to_numpy())[TARGET].agg(["count", "mean"]).reindex(b5.table_["bin"])
cmp_tbl = pd.DataFrame({"bin": b5.table_["bin"], "개발 부도율": b5.table_["bad_rate"].to_numpy(),
                        "테스트 부도율": t_rate["mean"].to_numpy(), "테스트 건수": t_rate["count"].to_numpy()})
print(cmp_tbl.round(3).to_string(index=False))
print("\\n개발 추세:", monotonic_direction(cmp_tbl["개발 부도율"]), "| 테스트 추세:", monotonic_direction(cmp_tbl["테스트 부도율"]))
'''
        ),
        md("## 5. 단조성 — 부도율이 올라갔다 내려가는 구간을 합치기"),
        code(
            '''
var = "income_10k"
free = WoeBinner(method="quantile", max_bins=8, min_events=1).fit(dev[var], dev[TARGET])
mono = WoeBinner(method="quantile", max_bins=8, min_events=1, monotonic="auto").fit(dev[var], dev[TARGET])
print(f"[{var}] 제약 없음: {len(free.table_)}개 bin, 추세 {monotonic_direction(free.table_['bad_rate'])}, IV {free.iv_:.4f}")
print(free.table_[["bin", "n", "bad_rate", "woe"]].round(3).to_string(index=False))
print(f"\\n[{var}] monotonic='auto'(방향: {mono.direction_}): {len(mono.table_)}개 bin, 추세 {monotonic_direction(mono.table_['bad_rate'])}, IV {mono.iv_:.4f}")
print(mono.table_[["bin", "n", "bad_rate", "woe"]].round(3).to_string(index=False))
'''
        ),
        md(
            """
단조 제약은 **업무 논리가 단조일 때**(예: 연체 횟수가 늘수록 위험) 써야 합니다. 데이터의 우연한 출렁임을 합쳐 *설명 가능성과 안정성*을 얻는 대신 IV 가 조금 줄 수 있습니다. 업무적으로 U자형이 타당한 변수(예: 매우 낮거나 높은 값이 모두 위험)에는 단조 제약을 걸지 않습니다.
"""
        ),
        md("## 6. 범주형·결측·처음 보는 값"),
        code(
            '''
bc = WoeBinner(min_bin_pct=0.10).fit(dev["housing_type"], dev[TARGET])
print("[housing_type] 비중 10% 미만 범주는 OTHER 로 묶임")
print(bc.table_[["bin", "n", "pct", "bad", "bad_rate", "woe"]].round(3).to_string(index=False))

new_values = pd.Series(["rent", "own", "dormitory", None])
print("\\n테스트/운영에서 처음 보는 값:")
print(pd.DataFrame({"값": new_values, "bin": bc.transform_bins(new_values), "WOE": bc.transform(new_values)}).to_string(index=False))
print("→ 처음 보는 범주는 OTHER, 결측은 MISSING 규칙을 따르고, 학습에 해당 bin 이 없으면 WOE 0(중립)입니다.")
'''
        ),
        code(
            '''
# 결측이 있는 데이터(오류 주입 합성 데이터)로 MISSING bin 확인 — 이 레슨에서는 income_10k 의 결측만 봅니다
raw, _ = load_learning_data("synthetic", seed=SEED, inject_quality_issues=True)
raw = raw.drop_duplicates()
rcut = int(len(raw) * 0.7)
rdev = raw.sort_values("application_date").iloc[:rcut]
bm = WoeBinner(method="quantile", max_bins=4, min_events=1).fit(rdev["income_10k"], rdev[TARGET])
print(bm.table_[["bin", "kind", "n", "pct", "bad", "bad_rate", "bad_rate_lo", "bad_rate_hi", "woe"]].round(3).to_string(index=False))
'''
        ),
        md(
            """
`MISSING` bin 은 따로 계산됩니다. 건수가 적으면 부도율 구간이 넓으므로 WOE 를 과신하지 마세요. 그리고 결측의 **이유가 운영에서도 같은지**(레슨 1-3)를 확인해야 합니다.
"""
        ),
        md("## 7. 0 건 bin 과 스무딩"),
        code(
            '''
try:
    woe_iv_from_counts([0, 10], [5, 5], smoothing=0)
except ValueError as e:
    print("smoothing=0 :", e)

for a in [0.1, 0.5, 1.0, 5.0]:
    w = woe_iv_from_counts([0, 10], [5, 5], smoothing=a)
    print(f"smoothing={a:>3}: 첫 bin WOE = {w['woe'][0]:+.3f}")
'''
        ),
        md("스무딩 값(a)이 클수록 WOE 가 0 쪽으로 당겨집니다. 0 건 bin 을 **버리지 않고** 유한한 값으로 만드는 대신, a 의 선택이 결과에 영향을 줍니다. 그래서 a 를 **문서화**하고, 가능하면 0 건 bin 은 이웃과 합칩니다."),
        md(
            """
## 8. ✏️ 실습 — 대안 경계 비교
`employment_years`(근속연수)에 대해 **업무적으로 설명할 수 있는 경계**를 직접 정해(`my_edges`) 아래 세 방식을 비교하세요.
1. 내 경계(manual) 2. 등빈도 5구간 3. 나무 기반(최대 5구간)

비교 항목: 구간 수 · 개발 IV · 테스트 적용 IV · 부도율 추세. 어느 쪽을 쓰겠습니까? 이유를 한 문장으로 적으세요.
"""
        ),
        code(
            '''
# [exercise]
my_edges = None   # ← 예: [1, 3, 5, 10]  (근속 1년 이하 / 1~3 / 3~5 / 5~10 / 10년 초과)
my_choice = ""    # ← 어느 방식을 쓰겠습니까? 이유는?
# 힌트1: 신입(1년 미만), 단기, 중기, 장기처럼 업무에서 쓰는 말로 나눠 보세요.
# 힌트2: 구간마다 부도가 최소 수십 건은 되는지 table_ 의 bad 열로 확인하세요.

if my_edges:
    var = "employment_years"
    options = {
        "내 경계": WoeBinner(method="manual", edges=my_edges),
        "등빈도 5": WoeBinner(method="quantile", max_bins=5),
        "나무 5": WoeBinner(method="tree", max_bins=5),
    }
    out = []
    for name, bb in options.items():
        bb.fit(dev[var], dev[TARGET])
        out.append({"방식": name, "bin 수": len(bb.table_), "최소 bin 부도 건수": int(bb.table_["bad"].min()),
                    "개발 IV": bb.iv_, "테스트 적용 IV": applied_iv(test, bb, var),
                    "추세": monotonic_direction(bb.table_.loc[bb.table_["kind"] == "range", "bad_rate"])})
    print(pd.DataFrame(out).round(4).to_string(index=False))
    print("\\n내 선택:", my_choice if my_choice else "(아직 적지 않았습니다)")
else:
    print("my_edges 를 정하고 다시 실행하세요.")
'''
        ),
        md(
            """
<details>
<summary>정답 예시 (먼저 직접 해 본 뒤 펼치세요)</summary>

```python
my_edges = [1, 3, 5, 10]
my_choice = "내 경계: 신입/단기/중기/장기로 설명이 쉽고, 개발·테스트 IV 차이가 작으면 채택. 추세가 단조인지도 확인."
```
</details>
"""
        ),
        md(
            """
## 9. 모형에 넣을 WOE 열 만들기 — 누수 없이
경계와 WOE 값은 **개발 구간에서 한 번** 정하고, 테스트에는 `transform` 만 합니다.
"""
        ),
        code(
            '''
from creditlab.features import fit_binners, transform_woe_frame, iv_summary
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CANDS = ["age", "income_10k", "loan_amount_10k", "employment_years", "card_utilization", "num_delinq_12m", "inquiries_6m", "housing_type", "region"]
rank = iv_summary(dev, CANDS, TARGET, method="quantile", max_bins=5, min_events=10)
print(rank[["variable", "kind", "n_bins", "iv", "iv_label", "bad_rate_trend"]].round(4).to_string(index=False))

binners = fit_binners(dev, CANDS, TARGET, method="quantile", max_bins=5, min_events=10)
Wdev = transform_woe_frame(dev, binners)
Wtest = transform_woe_frame(test, binners)      # test 에는 transform 만 — 경계·WOE 는 개발 때 그대로
print("\\n변환 결과:", Wdev.shape, Wtest.shape, "| 결측:", int(Wdev.isna().sum().sum() + Wtest.isna().sum().sum()))
'''
        ),
        code(
            '''
# 부호 약속 점검: WOE = ln(%bad/%good) 이므로, 로지스틱 회귀 계수는 대체로 양(+) 이어야 정상입니다.
lr = LogisticRegression(C=np.inf, max_iter=2000).fit(Wdev, dev[TARGET])
coefs = pd.Series(lr.coef_[0], index=Wdev.columns).round(3)
print(coefs.to_string())
print("\\n음수 계수:", [c for c, v in coefs.items() if v < 0] or "없음")
auc_test = roc_auc_score(test[TARGET], lr.predict_proba(Wtest)[:, 1])
print(f"\\n테스트 AUROC(참고용): {auc_test:.3f}")
print("※ 이 숫자는 비닝 코드가 동작하는지 확인하는 용도입니다. 모형 개발과 성능 보고는 다음 Phase(로지스틱·스코어카드)에서 제대로 합니다.")
'''
        ),
        md(
            """
계수가 **음수**로 나오면 WOE 방향이 뒤집혔거나, 변수 간 공선성이 있거나, bin 이 불안정하다는 신호입니다. 부호가 이상하면 이유를 확인하기 전에는 모형에 쓰지 않습니다.

또 하나의 점검: WOE 를 입력으로 쓰면 변수 하나만 넣었을 때 계수가 **1 근처**가 되는 것이 이론적으로 자연스럽습니다(WOE 가 이미 로그오즈 단위이기 때문). 여러 변수를 함께 넣으면 변수 간 겹침으로 달라지지만, **계수가 1 에서 크게 벗어나는 변수**는 WOE 가 정보량에 비해 불안정하지 않은지(예: 부도와 거의 무관한 변수의 잡음) 확인해 보세요.
"""
        ),
        md("## 10. 퀴즈"),
        code(
            '''
from creditlab.learning import Question, check, grade, show

QUIZ = [
    Question("Q1", "WOE 부호", "이 프로그램에서 어떤 구간의 WOE 가 +0.8 입니다. 의미는?",
             {"A": "그 구간은 전체 평균보다 안전하다", "B": "그 구간은 정상 대비 부도 비중이 높아 위험하다 (ln(%bad/%good) > 0)",
              "C": "그 구간의 부도율이 80% 이다", "D": "그 구간은 삭제해야 한다"},
             "B", "WOE=ln(%bad/%good). 양수면 부도 비중이 정상 비중보다 큰 위험 구간입니다.", ("hi 구간 예제를 떠올리세요.",)),
    Question("Q2", "과적합", "bin 을 40개로 늘렸더니 개발 IV 가 크게 올랐습니다. 올바른 판단은?",
             {"A": "예측력이 좋아졌으니 채택한다", "B": "우연한 차이까지 담았을 수 있다. 같은 경계·WOE 를 새 데이터에 적용해 구간별 부도율의 방향과 적용 IV 를 확인한다",
              "C": "IV 는 항상 클수록 좋다", "D": "테스트 데이터로 경계를 다시 정한다"},
             "B", "개발 구간에서 타깃을 보고 만든 bin 은 개발 데이터에 과적합될 수 있습니다.", ("개발 IV 와 테스트 적용 IV 의 차이 열을 보세요.",)),
    Question("Q3", "누수", "테스트 데이터까지 포함해 분위수 경계를 만들면 어떤 문제가 있나요?",
             {"A": "문제 없다", "B": "테스트 정보가 경계에 스며들어 성능이 낙관적으로 나올 수 있다(누수). 경계는 개발 구간에서만 정한다",
              "C": "계산이 느려진다", "D": "WOE 부호가 바뀐다"},
             "B", "검증·테스트 데이터는 '미래의 처음 보는 데이터' 역할을 해야 합니다.", ("운영 때 미래 데이터의 분포를 지금 알 수 있을까요?",)),
    Question("Q4", "스무딩", "어떤 bin 에 부도가 0건입니다. 스무딩 없이 WOE 는?",
             {"A": "0", "B": "ln(0) 이 되어 정의되지 않는다(−∞). 스무딩을 하거나 이웃 bin 과 합친다",
              "C": "−1", "D": "그 bin 은 자동 삭제된다"},
             "B", "%bad=0 이면 ln(0). 스무딩 값은 결과에 영향을 주므로 문서화합니다.", ("ln(0) 은 얼마일까요?",)),
    Question("Q5", "IV 해석", "IV 가 0.6 으로 매우 높게 나온 변수가 있습니다. 가장 먼저 할 일은?",
             {"A": "가장 좋은 변수이니 바로 채택한다", "B": "누수(결과 이후 정보)나 과적합, 데이터 오류를 먼저 의심하고 점검한다",
              "C": "IV 는 틀릴 수 없다", "D": "IV 0.3 이하로 줄이도록 bin 을 합친다"},
             "B", "신용 데이터에서 비정상적으로 높은 IV 는 대개 누수·정의 오류의 신호입니다. IV 구간 기준은 관행일 뿐입니다.",
             ("레슨 2-1 의 score_v2 를 떠올려 보세요.",)),
]

answers = {"Q1": None, "Q2": None, "Q3": None, "Q4": None, "Q5": None}  # ← "A"~"D"

for q in QUIZ:
    show(q)
    print()
'''
        ),
        code(
            '''
for q in QUIZ:
    check(q, answers[q.qid])
if all(v is not None for v in answers.values()):
    r = grade(QUIZ, answers)
    print("\\n점수:", r["score"], "/", r["total"], "| 통과(75% 기준):", r["passed"], "| 다시 볼 개념:", r["concepts_to_review"])
'''
        ),
        md(
            """
## 11. 과제와 정리
**과제**
1. UCI 데이터(`load_learning_data("uci")`)의 `PAY_0`(최근 월 상환 상태)와 `LIMIT_BAL`(한도)에 대해 비닝 표와 WOE/IV 를 만드세요. `PAY_0` 처럼 값이 몇 개 안 되는 변수는 **연속형처럼 비닝할지, 범주형으로 볼지** 정하고 이유를 쓰세요.
2. 한 변수에서 bin 수를 3·5·10·20 으로 바꿔 **개발 IV 와 테스트 적용 IV 의 차이**를 그래프로 그리고 한 문단으로 해석하세요.
3. WOE 부호가 이 프로그램과 반대인 교재의 표를 읽을 때 어떻게 변환할지 설명하세요.

**체크포인트**: 퀴즈 4/5 이상 · 대안 경계 비교표 작성 · 로지스틱 계수 부호 점검

**다음 Phase**: 로지스틱 회귀와 스코어카드 — WOE 열로 모형을 만들고 점수로 바꿉니다.

**참고자료**
- Siddiqi, N., *Credit Risk Scorecards* (Wiley, 2006) — WOE/IV 와 스코어카드(부호 정의 확인).
- Thomas, L., *Consumer Credit Models* (Oxford, 2009).
- 이론: `docs/theory/stage_05.html`
"""
        ),
    ]
