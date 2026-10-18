"""종합 프로젝트(단계 16): 정의 → 품질 → 분할 → 비닝 → 모형 → 점수 → 등급 → 평가 → 보정 → PSI → 챌린저 → 공정성 점검.

모든 적합(비닝 경계·모형·등급 경계·챌린저 설정)은 **개발 구간에서만** 한다. 검증/OOT 구간은 적용·평가에만 쓴다.
이 함수는 *교육용 축소 파이프라인*이다. 실제 프로젝트의 각 단계는 별도 모듈·테스트·승인 절차를 가진다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import PlattCalibrator  # noqa: F401  (노트북에서 재사용)
from .challengers import MLFeatureEncoder, fit_gbm_early_stop
from .data import load_learning_data, make_synthetic_credit_data, quality_summary
from .evaluation import (auroc, bootstrap_ci, brier_score, calibration_slope_intercept, calibration_table, csi_table,
                         discrimination_summary, expected_calibration_error, psi_from_bins, psi_numeric)
from .features import fit_binners, iv_summary, transform_woe_frame
from .governance import group_metrics, reproducibility_info
from .models import fit_logistic
from .scorecard import (ScoreScaling, Scorecard, assign_grades, grade_monotonic, grade_table, make_grade_edges)
from .stats import delong_test


def _split_time(df: pd.DataFrame, date_col: str | None, fr=(0.6, 0.2, 0.2)):
    if date_col and date_col in df.columns:
        d = df.sort_values(date_col).reset_index(drop=True)
        time_split = True
    else:
        d = df.reset_index(drop=True)
        time_split = False                      # 시간 열이 없으면 행 순서(진짜 시간 분할이 아님)
    n = len(d)
    a, b = int(n * fr[0]), int(n * (fr[0] + fr[1]))
    return d.iloc[:a].copy(), d.iloc[a:b].copy(), d.iloc[b:].copy(), time_split


def run_capstone(source: str = "synthetic", seed: int = 42, n: int = 15000, iv_min: float = 0.02,
                 scaling: ScoreScaling | None = None, n_grades: int = 5, group_col: str = "region", n_boot: int = 300) -> dict:
    scaling = scaling or ScoreScaling(600, 50, 20)
    if source == "synthetic":
        df = make_synthetic_credit_data(n=n, seed=seed)
        meta = {"source": "synthetic", "target": "default", "id_col": "application_id", "date_col": "application_date"}
    else:
        df, m = load_learning_data(source, seed=seed)
        meta = {"source": m["source"], "target": m["target"], "id_col": m.get("id_col"),
                "date_col": "application_date" if "application_date" in df.columns else None}
    T, id_col, date_col = meta["target"], meta["id_col"], meta["date_col"]
    raw_hash = reproducibility_info(df, seed)["data_hash"]

    dev, val, oot, time_split = _split_time(df, date_col)
    out: dict = {"meta": meta, "seed": seed, "n_rows": len(df), "time_split": time_split,
                 "sizes": {"dev": len(dev), "val": len(val), "oot": len(oot)},
                 "bads": {k: int(d[T].sum()) for k, d in [("dev", dev), ("val", val), ("oot", oot)]},
                 "quality": quality_summary(df, id_col=id_col if id_col in df.columns else None)}

    # --- 변수 후보와 IV (개발 구간)
    drop = {T, id_col, date_col}
    cands = [c for c in df.columns if c not in drop]
    iv = iv_summary(dev, cands, T, method="quantile", max_bins=5, min_events=10)
    selected = iv.loc[iv["iv"] >= iv_min, "variable"].tolist()
    out["iv_table"], out["selected"] = iv, selected

    # --- 비닝·WOE·로지스틱·점수표
    binners = fit_binners(dev, selected, T, method="quantile", max_bins=5, min_events=10)
    fit = fit_logistic(transform_woe_frame(dev, binners), dev[T])
    card = Scorecard.from_fit(binners, fit, scaling=scaling)
    out.update({"binners": binners, "fit": fit, "card": card, "coef_table": fit.summary()})
    out["verify"] = {k: card.verify(d)["max_abs_diff"] for k, d in [("dev", dev), ("val", val), ("oot", oot)]}

    sets = {"dev": dev, "val": val, "oot": oot}
    pdp = {k: card.predict_pd(d) for k, d in sets.items()}
    scr = {k: card.score(d) for k, d in sets.items()}
    out["splits"], out["pd"], out["score"] = sets, pdp, scr

    # --- 등급(경계는 개발 구간에서만)
    edges = make_grade_edges(scr["dev"], n_grades, "quantile")
    out["grade_edges"] = edges
    out["grade_tables"] = {k: grade_table(scr[k], sets[k][T], edges) for k in sets}

    # --- 평가
    metrics = {k: discrimination_summary(sets[k][T], pdp[k]) for k in sets}
    out["metrics"] = pd.DataFrame(metrics).T
    out["auroc_ci"] = {k: bootstrap_ci(auroc, sets[k][T], pdp[k], n_boot=n_boot, seed=seed) for k in ["val", "oot"]}
    cal = calibration_table(oot[T], pdp["oot"], n_bins=5)
    out["calibration"] = {"table": cal, "ece": expected_calibration_error(cal), "brier": brier_score(oot[T], pdp["oot"]),
                          "slope_intercept": calibration_slope_intercept(oot[T], pdp["oot"]),
                          "mean_pd": float(pdp["oot"].mean()), "bad_rate": float(oot[T].mean())}

    # --- PSI/CSI (기준 = 개발)
    out["psi"] = {k: {"score_psi": psi_numeric(scr["dev"], scr[k])["psi"],
                      "grade_psi": psi_from_bins(assign_grades(scr["dev"], edges), assign_grades(scr[k], edges), order=list("ABCDE"[:n_grades]) if n_grades <= 5 else None)["psi"]}
                  for k in ["val", "oot"]}
    out["csi_oot"] = csi_table(dev, oot, binners)

    # --- 챌린저(단조 제약 부스팅) + DeLong (OOT)
    num = [c for c in cands if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in cands if c not in num]
    enc = MLFeatureEncoder(num, cat).fit(dev)
    mono = {c: 1 for c in ["num_delinq_12m", "card_utilization"] if c in num}
    gbm, curve, best_it = fit_gbm_early_stop(enc.transform(dev), dev[T], enc.cat_mask, monotonic=mono,
                                             learning_rate=0.05, max_depth=3, min_samples_leaf=50, max_iter=500)
    p_gbm = gbm.predict_proba(enc.transform(oot))[:, 1]
    out["challenger"] = {"best_iteration": best_it, "auroc": auroc(oot[T], p_gbm), "delong": delong_test(oot[T], p_gbm, pdp["oot"]),
                         "monotone": mono}

    # --- 공정성 점검(그룹별) : 컷오프는 개발 점수 하위 30% 거절이라고 가정
    cutoff = float(np.quantile(scr["dev"], 0.30))
    out["cutoff"] = cutoff
    if group_col in df.columns:
        approved = (scr["oot"].to_numpy() >= cutoff).astype(int)
        out["group_metrics"] = group_metrics(oot[T], pdp["oot"], approved, oot[group_col].to_numpy())

    # --- 재현성과 자동 점검표
    out["reproducibility"] = reproducibility_info(df, seed)
    out["rubric_auto"] = rubric_checks(out, dev, val, oot, date_col, T)
    out["data_hash_raw"] = raw_hash
    return out


def rubric_checks(res: dict, dev, val, oot, date_col, T) -> pd.DataFrame:
    """자동으로 확인 가능한 루브릭 항목만 점검한다. 사람이 판단해야 하는 항목(정의·한계·문서)은 포함하지 않는다."""
    rows = []

    def add(area, item, ok, detail):
        rows.append({"영역": area, "항목": item, "통과": bool(ok), "근거": detail})

    ordered = bool(date_col) and dev[date_col].max() <= val[date_col].min() <= oot[date_col].min() if date_col else False
    add("누수 방지", "개발<검증<OOT 시간 순서", ordered and res["time_split"], f"시간 분할={res['time_split']}")
    fit_dev = all(int(b.table_["n"].sum()) == len(dev) for b in res["binners"].values())
    add("누수 방지", "비닝이 개발 구간 표본으로만 적합됨", fit_dev, f"개발 {len(dev):,}행")
    aucs = {c: auroc(dev[T], dev[c].fillna(dev[c].median())) for c in res["selected"] if pd.api.types.is_numeric_dtype(dev[c])}
    worst = max((max(a, 1 - a) for a in aucs.values()), default=0.5)
    add("누수 방지", "단일 변수 AUROC 가 비정상적으로 높지 않음(>0.9 의심)", worst < 0.9, f"최대 {worst:.3f}")
    signs = np.asarray(res["fit"].slopes, dtype=float)
    add("모형", "WOE 계수가 모두 양(+)", bool((signs > 0).all()), f"최소 계수 {signs.min():.3f}")
    add("모형", "점수표 합계와 모형 일치(검산)", max(res["verify"].values()) < 1e-8, f"최대 차이 {max(res['verify'].values()):.1e}")
    add("평가", "AUROC 신뢰구간 보고", all(not np.isnan(v["lo"]) for v in res["auroc_ci"].values()), "검증·OOT")
    add("평가", "보정 점검(ECE·기울기)", not np.isnan(res["calibration"]["ece"]), f"ECE {res['calibration']['ece']:.4f}")
    add("평가", "개발 구간 등급별 부도율이 단조 증가", grade_monotonic(res["grade_tables"]["dev"]) == "increasing", grade_monotonic(res["grade_tables"]["dev"]))
    add("운영", "PSI 산출(검증·OOT)", all(np.isfinite(v["score_psi"]) for v in res["psi"].values()), f"OOT 점수 PSI {res['psi']['oot']['score_psi']:.4f}")
    add("재현성", "시드·데이터 해시 기록", bool(res["reproducibility"]["data_hash"]) and res["reproducibility"]["seed"] is not None, res["reproducibility"]["data_hash"][:12])
    return pd.DataFrame(rows)


def auto_model_card_info(res: dict) -> dict:
    """결과에서 자동으로 채울 수 있는 모형 카드 항목만 채운다. 용도·한계·책임자 등 *판단*은 비워 둔다(사람이 쓴다)."""
    m = res["metrics"].copy()
    m.insert(0, "split", m.index)
    m = m[["split", "n", "bads", "bad_rate", "auroc", "gini", "ks"]].reset_index(drop=True)
    gt = res["grade_tables"]["oot"][["grade", "n", "bad", "bad_rate"]]
    ch = res["challenger"]
    d = ch["delong"]
    cal = res["calibration"]
    return {
        "overview": {}, 
        "data": {"source": f"{res['meta']['source']} (교육용 합성데이터)" if res["meta"]["source"] == "synthetic" else res["meta"]["source"],
                 "period": "application_date 기준 시간순 분할" if res["time_split"] else "행 순서(시간 열 없음)",
                 "default_definition": "target=1 부도(합성 정의). 실제 정의는 기관 기준을 따른다"},
        "method": {"split": f"개발 {res['sizes']['dev']:,} / 검증 {res['sizes']['val']:,} / OOT {res['sizes']['oot']:,} (시간순)",
                   "variable_selection": f"개발 구간 IV ≥ 0.02 로 선별 후 변수 {len(res['selected'])}개: {', '.join(res['selected'])}",
                   "model_type": "WOE 로지스틱 회귀 → 스코어카드(점수 = Offset + Factor·ln(odds_good))",
                   "score_formula": f"PDO={res['card'].scaling.pdo}, 기준 {res['card'].scaling.base_score}점@{res['card'].scaling.base_odds}:1"},
        "performance": {"metrics_table": m,
                        "calibration": f"OOT ECE {cal['ece']:.4f}, 보정 기울기 {cal['slope_intercept']['slope']:.3f}, 평균 PD {cal['mean_pd']:.3f} vs 실제 {cal['bad_rate']:.3f}",
                        "grade_table": gt,
                        "challenger": f"단조 제약 부스팅 OOT AUROC {ch['auroc']:.4f} vs 스코어카드 {d['auc2']:.4f}, DeLong p={d['p_value']:.3f}, 차이 95% 구간 [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]"},
        "monitoring": {"psi_plan": f"기준=개발 구간. OOT 점수 PSI {res['psi']['oot']['score_psi']:.4f}, 변수별 최대 CSI {res['csi_oot']['psi'].max():.4f}"},
        "limits": {},
        "reproducibility": dict(res["reproducibility"]),
    }
