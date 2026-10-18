from nbkit import SETUP_CELL, code, md

TITLE = "레슨 5-2. 튜닝·앙상블·보정·컷오프"


def cells():
    return [
        md(
            """
# 레슨 5-2. 튜닝·앙상블·보정·컷오프

**예상 시간** 100분 · **선행** 레슨 5-1 · **이론 페이지** `docs/theory/stage_12.html`

## 학습 목표
1. 데이터를 **적합 / 보정 / 테스트**로 시간순 분리하고, 각 구간의 역할을 지킨다.
2. 시간순 교차검증으로 튜닝하고, **CV 점수는 성능이 아니라 설정을 고르는 도구**임을 본다(최고점 선택 편향, 작은 초기 폴드).
3. 순위 평균과 **스태킹**(표본 외 예측으로 메타 모형 학습)을 구현해 비교한다.
4. **Platt·Isotonic·절편 보정**을 적합하고, 순위(AUROC)는 그대로이고 확률만 고쳐짐을 확인한다.
5. 보정 데이터를 학습에도 쓰면 안 되는 이유를 숫자로 본다.
6. **컷오프는 정책**이며 이론 최적 `m/(m+L)` 은 *확률이 보정되어 있을 때만* 성립함을 이해한다.

> **순서 규칙**: 튜닝·보정·컷오프 결정은 모두 개발 구간 안에서 끝내고, 테스트는 **마지막에 한 번** 봅니다.
> 합성데이터 결과는 실제 성능이 아닙니다. 비용·이익 숫자(m, L)는 **가정**입니다.
"""
        ),
        SETUP_CELL,
        md(
            """
## 1. 구간의 역할
```
시간 →  [ 적합(fit) 구간 ][ 보정(cal) 구간 ][   테스트(봉인)   ]
         모형 학습·튜닝      확률 보정·컷오프     최종 성능 1회
```
테스트를 보고 설정을 바꾸면 **그 테스트는 더 이상 테스트가 아닙니다.** 보정 구간은 모형 학습에 쓰지 않은 구간이어야 합니다.
"""
        ),
        code(
            '''
from creditlab.data import load_learning_data
from creditlab.evaluation import (auroc, brier_score, calibration_slope_intercept, calibration_table, compare_models,
                                  expected_calibration_error, paired_bootstrap_diff)
from creditlab.challengers import MLFeatureEncoder, fit_gbm, tune_gbm_timecv
from creditlab.features import fit_binners, transform_woe_frame
from creditlab.models import fit_logistic, fit_regularized, logit, sigmoid
from creditlab.scorecard import ScoreScaling, Scorecard

df_all, meta = load_learning_data("synthetic", seed=SEED)
TARGET = meta["target"]
df_all = df_all.sort_values("application_date").reset_index(drop=True)
n = len(df_all)
dev, test = df_all.iloc[: int(n * 0.7)].copy(), df_all.iloc[int(n * 0.7):].copy()
fit_part, cal_part = dev.iloc[: int(len(dev) * 0.75)], dev.iloc[int(len(dev) * 0.75):]

NUM = ["age", "income_10k", "loan_amount_10k", "employment_years", "card_utilization", "num_delinq_12m", "inquiries_6m"]
CAT = ["housing_type", "region"]
enc = MLFeatureEncoder(NUM, CAT).fit(dev)
Xd, Xt = enc.transform(dev), enc.transform(test)
Xf, Xc = Xd.iloc[: len(fit_part)], Xd.iloc[len(fit_part):]
yd, yt = dev[TARGET].reset_index(drop=True), test[TARGET].reset_index(drop=True)
yf, yc = yd.iloc[: len(fit_part)], yd.iloc[len(fit_part):].reset_index(drop=True)
print(f"적합 {len(fit_part):,} | 보정 {len(cal_part):,} (부도 {int(yc.sum())}) | 테스트 {len(test):,} (부도 {int(yt.sum())})")
'''
        ),
        md("## 2. 시간순 교차검증 튜닝과 선택 편향"),
        code("pred_cv_vs_test = None  # ← CV 최고 점수와 봉인된 테스트 점수는 같은 값일까요? 어느 쪽이 높을지 예측하고, 그 이유(최고점 선택, 폴드의 학습 표본 크기)를 생각해 보세요."),
        code(
            '''
grid = {"max_depth": [2, 3, 5], "learning_rate": [0.03, 0.1], "max_iter": [100, 300]}
tab = tune_gbm_timecv(Xd, yd, grid, enc.cat_mask, n_splits=4, min_samples_leaf=50)
print(tab.round(4).to_string(index=False))

best = tab.iloc[0]
final = fit_gbm(Xd, yd, enc.cat_mask, max_depth=int(best["max_depth"]), learning_rate=float(best["learning_rate"]),
                max_iter=int(best["max_iter"]), min_samples_leaf=50)
test_auc = auroc(yt, final.predict_proba(Xt)[:, 1])
print(f"\\nCV 최고 {best['mean_auc']:.4f} (표준편차 {best['std_auc']:.4f}) | 봉인된 테스트(1회) {test_auc:.4f}")
print(f"설정 12개 중 CV 점수 범위: {tab['mean_auc'].min():.4f} ~ {tab['mean_auc'].max():.4f}")
if pred_cv_vs_test is not None:
    print("내 예측:", pred_cv_vs_test)
'''
        ),
        md(
            """
CV 점수와 테스트 점수는 **같은 값이 아닙니다.** 두 가지 힘이 섞입니다.
- *최고점 선택 편향*: 여러 설정 중 최댓값을 골랐으므로 CV 점수는 위로 치우칩니다.
- *폴드의 학습 표본 크기*: 시간순 CV 는 앞쪽 폴드를 작은 데이터로 학습하므로 평균이 **낮아질** 수 있습니다(최종 모형은 개발 전체로 학습).

위 결과에서는 후자가 커서 테스트 점수가 더 높게 나왔을 수 있습니다. 어느 쪽으로 나오든 **CV 점수를 최종 성능으로 보고하지 않고**, 설정을 고른 뒤 봉인된 테스트를 한 번만 봅니다. 또 설정 간 CV 점수 차이가 폴드 표준편차와 비슷하면 *최고 설정이 진짜 더 좋은 것인지* 알 수 없습니다. 테스트를 본 뒤 설정을 바꾸면 테스트가 오염됩니다.
"""
        ),
        md("## 3. 앙상블과 스태킹"),
        code(
            '''
from scipy.stats import rankdata
from creditlab.calibration import StackedModel

# 챔피언: 스코어카드
VARS = ["num_delinq_12m", "card_utilization", "age", "inquiries_6m", "employment_years", "loan_amount_10k"]
binners = fit_binners(dev, VARS, TARGET, method="quantile", max_bins=5, min_events=10)
card = Scorecard.from_fit(binners, fit_logistic(transform_woe_frame(dev, binners), dev[TARGET]), scaling=ScoreScaling(600, 50, 20))
p_card = card.predict_pd(test)
p_gbm = final.predict_proba(Xt)[:, 1]

# 순위 평균(확률이 아님!)
rank_avg = (rankdata(p_card) + rankdata(p_gbm)) / 2 / len(p_card)

class LRWrap:                        # 스태킹의 기본 모형 ① : 규제 로지스틱(원자료)
    def __init__(self, X, y):
        self.mu, self.sd = X.mean(), X.std().replace(0, 1)
        self.m = fit_regularized(((X - self.mu) / self.sd).fillna(0), y, penalty="l2", C=1.0)
    def predict_proba(self, X):
        return self.m.predict_proba(((X - self.mu) / self.sd).fillna(0))

Xd_num = Xd[NUM]; Xt_num = Xt[NUM]
stack = StackedModel({"lr": lambda X, y: LRWrap(X, y),
                      "gbm": lambda X, y: fit_gbm(X, y, max_depth=3, learning_rate=0.05, max_iter=140, min_samples_leaf=50)},
                     n_splits=4).fit(Xd_num, yd)
p_stack = stack.predict_proba(Xt_num)[:, 1]
print("메타 모형 가중치(로짓 단위):", {k: round(v, 3) for k, v in stack.meta_weights_.items()}, "| 절편", round(stack.meta_intercept_, 3))
print(f"메타 모형 학습에 쓰인 표본 수(표본 외 예측이 있는 행) = {stack.n_meta_:,} / {len(Xd_num):,}")

res = pd.DataFrame({"AUROC": {"스코어카드": auroc(yt, p_card), "부스팅(튜닝)": auroc(yt, p_gbm),
                              "순위 평균": auroc(yt, rank_avg), "스태킹": auroc(yt, p_stack)}})
print()
print(res.round(4).to_string())
d = paired_bootstrap_diff(auroc, yt, p_stack, p_card, n_boot=500, seed=SEED)
print(f"\\n스태킹 − 스코어카드: {d['diff']:+.4f}  95% 구간 [{d['lo']:+.4f}, {d['hi']:+.4f}]  0 포함: {d['includes_zero']}")
'''
        ),
        md(
            """
- **순위 평균은 확률이 아닙니다.** 순위만 좋아질 뿐 PD 로 쓰려면 보정이 따로 필요합니다.
- 스태킹의 메타 모형은 기본 모형이 **보지 못한(표본 외) 예측**으로 학습합니다. 시간순 분할에서는 가장 앞 구간의 표본 외 예측이 없어 *그 행은 메타 학습에서 빠집니다*(위 표본 수 참고).
- 앙상블의 이득은 **모형의 오류가 덜 상관될 때** 큽니다. 이득이 구간 안(0 포함)이면 복잡성만 늘리는 것일 수 있습니다.
- 메타 가중치가 **음수**로 나오는 것은 기본 모형 예측이 서로 매우 비슷(공선)해 가중치가 불안정하다는 신호일 수 있습니다. 이 노트북에서도 스태킹이 단독 부스팅보다 낫다는 근거는 나오지 않았습니다. 복잡한 앙상블은 *이기는 것이 확인될 때만* 씁니다.
"""
        ),
        md("## 4. 보정: Platt · Isotonic · 절편"),
        code(
            '''
# 일부러 과신하기 쉬운 설정(깊고 오래 학습): 보정이 필요한 상황을 만든다.
risky = fit_gbm(Xf, yf, enc.cat_mask, max_depth=5, learning_rate=0.1, max_iter=400, min_samples_leaf=20)
s_fit = risky.predict_proba(Xf)[:, 1]            # 학습 구간 예측(과신)
s_cal = risky.predict_proba(Xc)[:, 1]            # 보정 구간 예측(모형이 못 본 구간)
s_te = risky.predict_proba(Xt)[:, 1]
print(f"과신 모형  보정 구간 AUROC {auroc(yc, s_cal):.4f} | 학습 구간 AUROC {auroc(yf, s_fit):.4f} | 테스트 AUROC {auroc(yt, s_te):.4f}")
print(f"평균 예측 — 학습 {s_fit.mean():.4f} / 보정 {s_cal.mean():.4f} / 테스트 {s_te.mean():.4f}   실제 부도율 — 학습 {yf.mean():.4f} / 보정 {yc.mean():.4f} / 테스트 {yt.mean():.4f}")
'''
        ),
        code("pred_cal = None  # ← Platt 보정 후 AUROC 는 변할까요? Isotonic 은?"),
        code(
            '''
from creditlab.calibration import InterceptCalibrator, IsotonicCalibrator, PlattCalibrator

platt = PlattCalibrator().fit(s_cal, yc)
iso = IsotonicCalibrator().fit(s_cal, yc)
icpt = InterceptCalibrator().fit(s_cal, yc)
bad_platt = PlattCalibrator().fit(s_fit, yf)       # 잘못된 방법: 모형이 학습한 구간으로 보정기를 적합

preds = {"보정 전": s_te, "Platt": platt.predict(s_te), "Isotonic": iso.predict(s_te), "절편만": icpt.predict(s_te),
         "Platt(학습 구간으로 적합, 잘못)": bad_platt.predict(s_te)}
tab = compare_models(yt, preds)
print(tab[["auroc", "brier", "log_loss", "ece", "cal_slope", "cal_intercept", "mean_pd"]].round(4).to_string())
print(f"\\nPlatt 계수: 기울기 a = {platt.a_:.3f}, 절편 b = {platt.b_:.3f}  (a < 1 이면 원래 예측이 과신이었다는 뜻)")
print(f"고유 확률 값 개수 — 보정 전 {len(np.unique(s_te))}, Platt {len(np.unique(platt.predict(s_te)))}, Isotonic {len(np.unique(iso.predict(s_te)))}")
print(f"Isotonic 예측 확률 범위: {iso.predict(s_te).min():.4f} ~ {iso.predict(s_te).max():.4f}")
if pred_cal is not None:
    print("내 예측:", pred_cal)
'''
        ),
        md(
            """
- **Platt·절편 보정은 단조 변환**이라 AUROC 가 변하지 않습니다. 보정은 확률의 *값*을 고칩니다.
- **Isotonic** 은 계단함수라 동점이 생겨 AUROC 가 조금 변할 수 있고, 보정 표본이 작으면 과적합하기 쉽습니다(고유 확률 값 개수 참고).
- **절편만** 맞추면 평균 수준은 맞지만 과신(기울기 문제)은 그대로라 Brier·ECE 가 좋아지지 않습니다. 이 모형의 문제는 수준이 아니라 *극단적인 기울기*였습니다.
- **학습 구간으로 보정기를 적합하면** 그 구간의 예측은 이미 과신(학습 AUROC 1.0)이라 보정기가 *잘못된 것을 배웁니다.* 표의 마지막 행처럼 보정 전보다 **오히려 나빠집니다.** 보정은 *모형이 보지 못한* 구간에서 합니다.
- Isotonic 의 log loss 가 Platt 보다 나쁜 것은 예측 확률 범위(위 출력)에서 보이듯 0 에 가까운 계단 때문일 수 있습니다. 확률이 0 이면 한 건의 부도가 log loss 를 크게 올립니다.
- 보정은 테스트를 보고 고르지 않습니다. 보정 방법 선택도 보정/개발 구간 안에서 합니다.
"""
        ),
        code(
            '''
fig, ax = plt.subplots(figsize=(5, 4))
for name, c in [("보정 전", "#999999"), ("Platt", "#3d6fc8"), ("Isotonic", "#c8553d")]:
    ct = calibration_table(yt, preds[name], n_bins=6)
    ax.plot(ct["mean_pred"], ct["obs_rate"], "o-", label={"보정 전": "raw", "Platt": "Platt", "Isotonic": "isotonic"}[name], color=c)
lim = max(preds["보정 전"].max(), 0.4)
ax.plot([0, lim], [0, lim], "--", color="k", lw=0.8)
ax.set_xlabel("mean predicted PD"); ax.set_ylabel("observed default rate"); ax.legend(); ax.set_title("reliability (test)")
plt.tight_layout(); plt.show()
'''
        ),
        md("### 전체 부도율이 이동했을 때 (사전확률 변화)"),
        code(
            '''
# 시뮬레이션: 운영 중 전반적 부도율이 높아졌다고 '가정'하고(로짓 +0.7) 새 라벨을 만든다. 기울기는 같다.
rng = np.random.default_rng(SEED)
y_shift = pd.Series((rng.random(len(yt)) < sigmoid(logit(platt.predict(s_te)) + 0.7)).astype(int))
print(f"가정된 새 부도율 {y_shift.mean():.4f} (기존 테스트 {yt.mean():.4f}) | 보정된 모형의 평균 PD {platt.predict(s_te).mean():.4f}")

# 새 부도율을 안다고 가정하고(예: 최근 성과 관찰) 절편만 다시 맞춘다 — 순위·기울기는 그대로
fix = InterceptCalibrator(target_rate=float(y_shift.mean())).fit(platt.predict(s_te))
for name, p in [("재보정 전", platt.predict(s_te)), ("절편 재보정 후", fix.predict(platt.predict(s_te)))]:
    cs = calibration_slope_intercept(y_shift, p)
    print(f"{name}: 평균 PD {p.mean():.4f} | AUROC {auroc(y_shift, p):.4f} | 보정 절편 {cs['intercept']:+.3f} | 기울기 {cs['slope']:.3f}")
'''
        ),
        md("절편 재보정은 *전체 수준*(평균 PD)만 고치며 순위(AUROC)와 기울기는 그대로입니다. 평균을 맞추는 것과 보정 절편이 정확히 0 이 되는 것은 같지 않습니다(기울기가 1 이 아니기 때문). **어떤 부도율에 맞출지는 판단**이므로(일시적 변동인가 구조적 변화인가) 근거와 함께 문서화합니다."),
        md("## 5. 컷오프는 정책이다 — 그리고 보정이 전제다"),
        code(
            '''
from creditlab.calibration import profit_by_cutoff, theoretical_cutoff

M_GAIN, LOSS = 1.0, 8.0          # 가정: 정상 1건 이익 1, 부도 1건 손실 8 (실제 값은 상품·경기·비용 구조에 따라 다르다)
t_star = theoretical_cutoff(M_GAIN, LOSS)
print(f"이론 컷오프 m/(m+L) = {t_star:.4f}  → PD 가 {t_star:.1%} 미만인 고객만 승인")

cands = np.linspace(0.03, 0.40, 38)
res = {}
for name, f_cal, f_te in [("보정 전", s_cal, s_te), ("Platt 보정", platt.predict(s_cal), platt.predict(s_te))]:
    # 컷오프는 '보정 구간'의 기대이익으로 고르고, 테스트에서는 실현 이익만 확인한다
    pc = profit_by_cutoff(f_cal, M_GAIN, LOSS, cands, yc)
    t_pick = float(pc.loc[pc["expected_profit"].idxmax(), "cutoff"])
    tt = profit_by_cutoff(f_te, M_GAIN, LOSS, [t_pick], yt).iloc[0]
    res[name] = {"보정구간 기대이익 최대 컷오프": t_pick, "테스트 승인율": tt["approval_rate"], "테스트 기대이익": tt["expected_profit"], "테스트 실현이익": tt["realized_profit"]}
print(pd.DataFrame(res).T.round(3).to_string())
'''
        ),
        md(
            """
- 컷오프 **숫자**는 두 경우 모두 이론값 근처로 나올 수 있지만, 보정 전 확률은 과신이라 *같은 숫자가 서로 다른 승인 집합*(테스트 승인율 비교)을 뜻하고, 계산상 **기대이익이 실현이익보다 크게 부풀려집니다.**
- 보정된 확률에서는 기대이익과 실현이익이 훨씬 가깝습니다. 기대이익으로 정책을 설계하려면 **보정이 전제**인 이유입니다.
- 이 숫자는 가정(m, L)과 합성데이터에 의존하며 **승인 정책을 정하는 근거로 쓰이지 않습니다.** 실제 정책은 수익·손실·규제·고객 영향을 함께 봅니다.
"""
        ),
        md("## 6. ✏️ 실습 — 이론 컷오프 계산"),
        code(
            '''
# [exercise]
m_ex, L_ex = 2.0, 30.0           # 정상 1건 이익 2, 부도 1건 손실 30
customers = {"고객1": 0.04, "고객2": 0.0625, "고객3": 0.09}      # (보정된) 예측 PD

cut_mine = None      # ← 이론 컷오프 m/(m+L)
approve_mine = None  # ← {"고객1": True/False, ...}  p < 컷오프 이면 승인
profit_c3 = None     # ← 고객3 한 명을 승인했을 때의 기대이익 (1−p)·m − p·L

if None not in (cut_mine, approve_mine, profit_c3):
    cut = theoretical_cutoff(m_ex, L_ex)
    ok_ap = {k: v < cut for k, v in customers.items()}
    print(f"컷오프 {cut_mine:.4f} (정답 {cut:.4f}) :", "맞음" if abs(cut_mine - cut) < 1e-3 else "다시 계산")
    print("승인 판단 :", "맞음" if approve_mine == ok_ap else f"다시 확인 (정답 {ok_ap})")
    exp3 = (1 - 0.09) * m_ex - 0.09 * L_ex
    print(f"고객3 기대이익 {profit_c3:.3f} (정답 {exp3:.3f}) :", "맞음" if abs(profit_c3 - exp3) < 1e-3 else "다시 계산")
else:
    print("세 값을 모두 채우고 다시 실행하세요.")
'''
        ),
        md(
            """
<details>
<summary>정답 예시 (먼저 직접 해 본 뒤 펼치세요)</summary>

```python
cut_mine = m_ex / (m_ex + L_ex)
approve_mine = {k: p < cut_mine for k, p in customers.items()}
profit_c3 = (1 - customers["고객3"]) * m_ex - customers["고객3"] * L_ex
```
</details>
"""
        ),
        md("## 7. 퀴즈"),
        code(
            '''
from creditlab.learning import Question, check, grade, show

QUIZ = [
    Question("Q1", "테스트 오염", "튜닝 후 테스트 AUROC 가 마음에 들지 않아 학습률을 바꿔 같은 테스트로 다시 평가했습니다. 문제는?",
             {"A": "문제없다", "B": "테스트를 보고 설정을 바꿨으므로 그 테스트는 더 이상 독립적인 평가가 아니다",
              "C": "학습률은 테스트와 무관하다", "D": "AUROC 가 변해서 문제다"},
             "B", "테스트는 마지막에 한 번 봅니다. 바꾸려면 새 구간(또는 OOT)이 필요합니다.", ("테스트 결과가 설정 선택에 영향을 줬나요?",)),
    Question("Q2", "CV 최고점", "시간순 CV 로 설정 12개 중 최고 AUROC 0.74 인 설정을 골랐습니다. 이 0.74 를 최종 성능으로 보고해도 될까요?",
             {"A": "된다. CV 는 가장 믿을 만한 성능이다", "B": "안 된다. 최댓값 선택 편향과 폴드 구성의 영향이 있으므로, 설정 선택 후 봉인 테스트를 한 번 평가한 값을 보고한다",
              "C": "테스트 점수가 CV 와 같아야 정상이다", "D": "CV 를 반복하면 보고해도 된다"},
             "B", "CV 점수는 설정을 고르는 도구입니다. 선택 편향과 작은 초기 폴드 때문에 테스트와 같은 값이 아닙니다.", ("몇 개 중에서 골랐나요?",)),
    Question("Q3", "스태킹 누수", "스태킹에서 메타 모형은 어떤 예측으로 학습해야 하나요?",
             {"A": "기본 모형이 학습한 데이터에 대한 예측", "B": "기본 모형이 보지 못한 데이터에 대한 표본 외(OOF) 예측",
              "C": "테스트 예측", "D": "임의의 난수"},
             "B", "학습 데이터에 대한 예측은 과신이라 메타 모형이 기본 모형을 과신하게 됩니다.", ("누수가 어디서 생길까요?",)),
    Question("Q4", "보정 방법", "Platt 보정 후 AUROC 는?",
             {"A": "크게 오른다", "B": "변하지 않는다(단조 변환)", "C": "크게 내린다", "D": "항상 0.5 가 된다"},
             "B", "Platt 은 로짓의 선형변환이라 순위가 같습니다. Isotonic 은 동점 때문에 약간 변할 수 있습니다.", ("순위가 바뀌나요?",)),
    Question("Q5", "컷오프 공식", "p < m/(m+L) 로 승인하는 규칙이 이론적으로 최적이려면 무엇이 필요한가요?",
             {"A": "AUROC 가 0.8 이상", "B": "예측 확률이 잘 보정되어 있고 m, L 가정이 맞음", "C": "표본이 1만 건 이상", "D": "변수가 10개 이상"},
             "B", "기대이익 공식은 p 가 실제 부도확률이라는 전제 위에 있습니다.", ("p 가 틀리면 기대이익도 틀립니다.",)),
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
## 8. 과제와 정리
**과제**
1. 격자(`max_depth`, `learning_rate`, `max_iter`)에 값을 더 넣어 설정 수를 12 → 36 으로 늘리고, **CV 최고점과 테스트 점수의 차이**가 어떻게 변하는지 보세요. 왜 그런지 설명하세요. (설정을 바꾼 뒤 테스트를 다시 보면 안 됩니다. 한 번만 보세요.)
2. 보정 구간을 개발의 마지막 25% → 10% → 40% 로 바꿔 Platt/Isotonic 의 테스트 Brier 가 어떻게 변하는지 비교하고, 표본이 작을수록 Isotonic 이 불리한 이유를 쓰세요.
3. (m, L) 을 (1, 4), (1, 8), (1, 16) 으로 바꿔 이론 컷오프와 보정 구간에서 고른 컷오프를 비교하세요. 정책 변화가 승인율에 미치는 영향을 정리하세요.

**체크포인트**: 퀴즈 4/5 이상 · 컷오프 실습 통과 · 보정 후 AUROC 변화 여부를 설명할 수 있음

**다음 Phase**: 6 — 단계 13~16(합성데이터, 통계 검정, 거버넌스·보고, 캡스톤).

**참고자료**
- Platt, J., "Probabilistic outputs for support vector machines…" (1999).
- Niculescu-Mizil & Caruana, "Predicting good probabilities with supervised learning", ICML (2005).
- Wolpert, D., "Stacked generalization", *Neural Networks* (1992).
- 이론: `docs/theory/stage_12.html`
"""
        ),
    ]
