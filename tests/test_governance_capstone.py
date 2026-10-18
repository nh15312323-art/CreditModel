"""거버넌스(ECL·사유·공정성·선택편향·모형카드)와 캡스톤 파이프라인 테스트."""
import numpy as np
import pandas as pd

from creditlab.capstone import auto_model_card_info, run_capstone
from creditlab.data import make_synthetic_credit_data
from creditlab.features import fit_binners, transform_woe_frame
from creditlab.governance import (PLACEHOLDER, approval_ratio, build_model_card, check_model_card_completeness, ecl_pd_bias_impact,
                                  expected_credit_loss, group_metrics, portfolio_ecl, reason_codes, selection_bias_experiment)
from creditlab.models import fit_logistic
from creditlab.scorecard import ScoreScaling, Scorecard
from helpers import close, expect_raises


def test_ecl_theory_numbers_and_validation():
    assert close(expected_credit_loss(0.05, 0.45, 1000.0), 22.5)
    t = pd.DataFrame({"n": [1000, 500, 100], "pd": [0.01, 0.05, 0.20], "lgd": [0.45] * 3, "ead": [1000.0] * 3})
    assert close(portfolio_ecl(t).attrs["total_ecl"], 24750.0)
    half = ecl_pd_bias_impact(t, 0.5)
    assert close(half["change_pct"], -50.0) and close(half["biased_ecl"], 24750 / 2)
    with expect_raises(ValueError, "PD"):
        expected_credit_loss(1.2, 0.4, 100)
    with expect_raises(ValueError, "EAD"):
        expected_credit_loss(0.1, 0.4, -5)


def test_approval_ratio_and_group_metrics():
    rates, ratio = approval_ratio([1] * 6 + [0] * 4 + [1] * 8 + [0] * 2, ["A"] * 10 + ["B"] * 10)
    assert close(ratio, 0.75) and close(rates["A"], 0.6)
    rng = np.random.default_rng(0)
    g = np.array(["x"] * 300 + ["y"] * 20)
    p = rng.uniform(0.02, 0.3, 320)
    y = (rng.random(320) < p).astype(int)
    gm = group_metrics(y, p, (p < 0.2).astype(int), g, min_n=100)
    assert gm.set_index("group").loc["y", "small_sample"] and not gm.set_index("group").loc["x", "small_sample"]
    assert close(gm["n"].sum(), 320)
    with expect_raises(ValueError):
        group_metrics(y, p[:5], np.ones(320), g)


def test_selection_bias_depends_on_hidden_information():
    df = make_synthetic_credit_data(n=20000, seed=71)
    feats = ["age", "income_10k", "loan_amount_10k", "employment_years", "card_utilization", "num_delinq_12m", "inquiries_6m"]
    obs = selection_bias_experiment(df, feats, officer_weight=0.0, seed=0)
    hid = selection_bias_experiment(df, feats, officer_weight=1.5, seed=0)
    for r in (obs, hid):
        assert r["bad_rate_accepted_dev"] < r["bad_rate_all_dev"] and r["auroc_on_accepted"] < r["auroc_on_all"]
    assert abs(obs["pd_underestimate_ratio"] - 1) < 0.1                   # 관측 변수만으로 선택 → PD 거의 정확
    assert hid["pd_underestimate_ratio"] < 0.6                             # 데이터에 없는 정보에 의존 → 크게 과소추정
    with expect_raises(ValueError):
        selection_bias_experiment(df, feats, accept_share=1.5)
    with expect_raises(ValueError, "열"):
        selection_bias_experiment(df, ["nope"])


def test_reason_codes_additivity():
    df = make_synthetic_credit_data(n=5000, seed=5).sort_values("application_date").reset_index(drop=True)
    dev = df.iloc[:3500]
    cols = ["age", "card_utilization", "num_delinq_12m", "inquiries_6m"]
    bs = fit_binners(dev, cols, "default", min_events=10)
    card = Scorecard.from_fit(bs, fit_logistic(transform_woe_frame(dev, bs), dev["default"]), scaling=ScoreScaling(600, 50, 20))
    app = df.iloc[[3600, 3601, 3602]]
    rc = reason_codes(card, app, dev, top_n=2)
    assert len(rc) == 6 and rc["rank"].tolist() == [1, 2] * 3 and (rc.groupby("applicant")["shortfall"].apply(lambda s: s.is_monotonic_decreasing)).all()
    ref_mean = card.points_frame(dev).mean()
    for idx in app.index:
        diff = card.points_frame(app.loc[[idx]]).iloc[0] - ref_mean
        assert close(diff.sum(), float(card.score(app.loc[[idx]]).iloc[0] - card.score(dev).mean()), 1e-8)   # 가산성
    with expect_raises(ValueError):
        reason_codes(card, app, dev, top_n=0)


def test_model_card_completeness_and_rendering():
    empty = {}
    chk = check_model_card_completeness(empty)
    assert not chk["complete"] and len(chk["missing"]) > 20 and not chk["limitations_enough"]
    md = build_model_card(empty)
    assert md.count(PLACEHOLDER) > 20 and md.startswith("# 모형 카드")
    full = {
        "overview": {"model_name": "demo v1", "intended_use": "교육용 데모", "out_of_scope": "실제 심사에 사용 금지"},
        "data": {k: "내용 있음 기록" for k in ["source", "period", "default_definition", "quality_issues", "population_gap"]},
        "method": {k: "내용 있음 기록" for k in ["split", "variable_selection", "model_type", "score_formula"]},
        "performance": {"metrics_table": pd.DataFrame({"a": [1]}), "calibration": "ok ok", "grade_table": pd.DataFrame({"g": ["A"]}), "challenger": "비교 완료"},
        "monitoring": {"psi_plan": "월별 PSI", "alert_thresholds": "0.10/0.25", "owner": "모형팀"},
        "limits": {"known_limitations": ["합성데이터라 실제 성능 아님", "시간 구간이 짧음", "공정성 점검이 제한적임"], "assumptions": ["독립 관측"], "fairness_check": "지역별 점검"},
        "reproducibility": {"code_version": "abc", "seed": 42, "data_hash": "deadbeef", "library_versions": "numpy 2"},
    }
    ok = check_model_card_completeness(full)
    assert ok["complete"] and ok["missing"] == []
    full["limits"]["known_limitations"] = ["너무 짧음", "짧음", "x"]
    assert not check_model_card_completeness(full)["limitations_enough"]
    full["data"]["source"] = PLACEHOLDER
    assert "data.source" in check_model_card_completeness(full)["missing"]


def test_capstone_runs_and_passes_automatic_rubric():
    r = run_capstone(seed=7, n=9000, n_boot=60)
    assert r["sizes"]["dev"] + r["sizes"]["val"] + r["sizes"]["oot"] == 9000 and r["time_split"]
    assert r["rubric_auto"]["통과"].all(), r["rubric_auto"]
    assert max(r["verify"].values()) < 1e-8
    # 시간 순서와 설정 개발 구간 고정
    sp = r["splits"]
    assert sp["dev"]["application_date"].max() <= sp["val"]["application_date"].min() <= sp["oot"]["application_date"].min()
    assert 0.6 < r["metrics"].loc["oot", "auroc"] < 0.9 and len(r["grade_tables"]["oot"]) == 5
    assert r["challenger"]["delong"]["n_bad"] == r["bads"]["oot"]
    info = auto_model_card_info(r)
    chk = check_model_card_completeness(info)
    assert "method.model_type" not in chk["missing"] and "reproducibility.data_hash" not in chk["missing"]
    assert "overview.model_name" in chk["missing"] and "limits.known_limitations" in chk["missing"]      # 판단 항목은 사람이 채운다
    r2 = run_capstone(seed=7, n=9000, n_boot=60)                                                         # 같은 시드 → 같은 결과
    assert r["reproducibility"]["data_hash"] == r2["reproducibility"]["data_hash"]
    assert close(r["metrics"].loc["oot", "auroc"], r2["metrics"].loc["oot", "auroc"], 1e-12)
