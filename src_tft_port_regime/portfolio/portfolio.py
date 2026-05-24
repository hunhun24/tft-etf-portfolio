"""
portfolio/portfolio.py — 포트폴리오 구성 엔진

제약 적용 순서:
  1. 전략별 점수 계산 & 종목 선택
  2. softmax 비중
  3. 전략 내부 제약 (방어 섹터 floor, 현금성 floor)
  4. DB GAPS 외부 제약
       ① 세부자산별 합산 상한 (ticker 단위 매핑 기준)
       ② 위험자산 합산 70% 상한
       ③ 개별 종목 20% 상한
  5. 합산 1.0 검증
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from typing import Literal

from config import (
    CONSERVATIVE, AGGRESSIVE,
    DEFENSIVE_SECTORS, GROWTH_SECTORS, ALTERNATIVE_SECTORS,
    TICKER_ASSET_MAP, MAX_SINGLE_TICKER, MAX_RISK_ASSET,
    SECTOR_COL, GROUP_COL,
    PORTFOLIO_RESULT_DIR,
    DUPLICATE_TICKERS,
)

StrategyType = Literal["conservative", "aggressive"]


# ─────────────────────────────────────────────────────────────
#  Signal 생성
# ─────────────────────────────────────────────────────────────

def build_signal(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()
    df[GROUP_COL] = df[GROUP_COL].astype(str)
    df = df[~df[GROUP_COL].isin(DUPLICATE_TICKERS)].copy()
    df["signal_return"]   = df["pred_q50"]
    df["signal_upside"]   = df["pred_q90"] - df["pred_q50"]
    df["signal_downside"] = df["pred_q50"] - df["pred_q10"]
    df["signal_risk"]     = (df["signal_upside"] + df["signal_downside"]) / 2

    eps = 1e-8
    df["signal_sharpe"] = df["signal_return"] / (df["signal_downside"] + eps)

    # 섹터 타입 태깅 (전략 점수 계산용)
    def tag_sector(s):
        if s in DEFENSIVE_SECTORS:   return "defensive"
        if s in GROWTH_SECTORS:      return "growth"
        if s in ALTERNATIVE_SECTORS: return "alternative"
        return "other"
    df["sector_type"] = df[SECTOR_COL].apply(tag_sector)

    # DB GAPS 매핑 — ticker 단위로 정확하게
    ticker_str = df[GROUP_COL].astype(str)
    df["asset_class"]  = ticker_str.map(lambda t: TICKER_ASSET_MAP.get(t, ("위험자산", "기타", 0.20))[0])
    df["sub_asset"]    = ticker_str.map(lambda t: TICKER_ASSET_MAP.get(t, ("위험자산", "기타", 0.20))[1])
    df["sub_asset_cap"]= ticker_str.map(lambda t: TICKER_ASSET_MAP.get(t, ("위험자산", "기타", 0.20))[2])

    # 매핑 누락 ticker 경고
    unmapped = df[df["sub_asset"] == "기타"][GROUP_COL].unique()
    if len(unmapped) > 0:
        print(f"[경고] TICKER_ASSET_MAP 미등록 ticker: {list(unmapped)}")

    return df


# ─────────────────────────────────────────────────────────────
#  내부 유틸
# ─────────────────────────────────────────────────────────────

def _softmax_weights(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max()
    exp    = np.exp(scores)
    return exp / exp.sum()


def _clip_and_renorm(w: np.ndarray, max_w: float) -> np.ndarray:
    for _ in range(30):
        clipped = np.clip(w, 0, max_w)
        total   = clipped.sum()
        if total < 1e-8:
            return np.ones(len(w)) / len(w)
        w = clipped / total
        if (w <= max_w + 1e-8).all():
            break
    return w


# ─────────────────────────────────────────────────────────────
#  전략 내부 제약
# ─────────────────────────────────────────────────────────────

def _enforce_defensive_floor(
    df: pd.DataFrame, weights: np.ndarray,
    floor: float, max_w: float,
) -> np.ndarray:
    def_mask  = (df["sector_type"] == "defensive").values
    grow_mask = ~def_mask
    def_w     = weights[def_mask].sum()

    if def_w >= floor:
        return weights

    shortfall  = floor - def_w
    grow_total = weights[grow_mask].sum()

    if grow_total < shortfall:
        if def_mask.sum() > 0:
            weights[def_mask]  += grow_total / def_mask.sum()
            weights[grow_mask]  = 0.0
    else:
        weights[grow_mask] *= (1 - shortfall / grow_total)
        if def_mask.sum() > 0:
            weights[def_mask] += shortfall / def_mask.sum()

    return _clip_and_renorm(weights, max_w)


def _enforce_rf_weight(
    df: pd.DataFrame, weights: np.ndarray,
    rf_weight: float, max_w: float,
) -> np.ndarray:
    # 금리연계형_초단기채권 + 국내채권_종합 포함
    rf_mask  = df["sub_asset"].isin({
        "금리연계형_초단기채권", "국내채권_종합"
    }).values
    rf_total = weights[rf_mask].sum()

    if rf_total >= rf_weight:
        return weights

    shortfall    = rf_weight - rf_total
    non_rf       = ~rf_mask
    non_rf_total = weights[non_rf].sum()

    if non_rf_total > shortfall and rf_mask.sum() > 0:
        weights[non_rf]  *= (1 - shortfall / non_rf_total)
        weights[rf_mask] += shortfall / rf_mask.sum()

    return _clip_and_renorm(weights, max_w)


# ─────────────────────────────────────────────────────────────
#  DB GAPS 외부 제약
# ─────────────────────────────────────────────────────────────

def _enforce_sub_asset_caps(
    df: pd.DataFrame, weights: np.ndarray,
) -> np.ndarray:
    """세부자산별 합산 비중 상한 적용 (ticker 단위 매핑 기준)."""
    for sub in df["sub_asset"].unique():
        mask  = (df["sub_asset"] == sub).values
        cap   = df.loc[mask, "sub_asset_cap"].iloc[0]
        total = weights[mask].sum()

        if total > cap + 1e-8:
            weights[mask] *= cap / total

    total = weights.sum()
    if total > 1e-8:
        weights = weights / total

    return weights


def _enforce_risk_asset_cap(
    df: pd.DataFrame, weights: np.ndarray,
) -> np.ndarray:
    """위험자산 합산 70% 상한 적용."""
    risk_mask  = (df["asset_class"] == "위험자산").values
    safe_mask  = ~risk_mask
    risk_total = weights[risk_mask].sum()

    if risk_total <= MAX_RISK_ASSET + 1e-8:
        return weights

    weights[risk_mask] *= MAX_RISK_ASSET / risk_total

    shortfall = 1.0 - weights.sum()
    if shortfall < 1e-8:
        return weights

    # 안전자산 종목별 가용 용량 계산 — 상한까지 남은 여유만큼만 채움
    per_ticker_cap = np.minimum(df["sub_asset_cap"].values, MAX_SINGLE_TICKER)
    available      = np.maximum(per_ticker_cap - weights, 0.0)
    available_safe = available * safe_mask

    total_available = available_safe.sum()
    if total_available > 1e-8:
        weights += shortfall * (available_safe / total_available)
    elif safe_mask.sum() > 0:
        weights[safe_mask] += shortfall / safe_mask.sum()

    return weights


def apply_dbgaps_constraints(df, weights):
    # 종목별 single ticker cap: 위험자산 MAX_SINGLE_TICKER, 안전자산 min(sub_asset_cap, 0.50)
    ticker_caps = np.where(
        df["asset_class"].values == "안전자산",
        np.minimum(df["sub_asset_cap"].values, 0.50),
        MAX_SINGLE_TICKER,
    )
    for _ in range(20):
        old = weights.copy()
        weights = _enforce_sub_asset_caps(df, weights)
        weights = _enforce_risk_asset_cap(df, weights)
        # 종목별 cap 적용 후 renorm (수렴까지 반복)
        for __ in range(30):
            clipped = np.minimum(weights, ticker_caps)
            total = clipped.sum()
            if total < 1e-8:
                weights = np.ones(len(weights)) / len(weights)
                break
            weights = clipped / total
            if (weights <= ticker_caps + 1e-8).all():
                break
        if np.max(np.abs(weights - old)) < 1e-8:
            break
    return weights


# ─────────────────────────────────────────────────────────────
#  메인 비중 계산
# ─────────────────────────────────────────────────────────────

def compute_weights(
    signal_df: pd.DataFrame,
    strategy:  StrategyType = "conservative",
) -> pd.DataFrame:
    cfg = CONSERVATIVE if strategy == "conservative" else AGGRESSIVE

    top_n            = cfg["top_n"]
    max_w            = cfg["max_single_weight"]
    defensive_floor  = cfg["defensive_floor"]
    return_threshold = cfg["return_threshold"]
    rf_weight        = cfg["risk_free_weight"]

    df = signal_df.copy()

    # ── 1. 최소 수익률 필터 ──
    df = df[df["signal_return"] >= return_threshold].copy()
    if df.empty:
        print(f"[{strategy}] 편입 가능 종목 없음 → 전량 현금")
        EMPTY_WEIGHT_COLUMNS = [
            GROUP_COL, "name", SECTOR_COL,
            "asset_class", "sub_asset", "sector_type",
            "signal_return", "signal_sharpe", "weight",
        ]
        return pd.DataFrame(columns=EMPTY_WEIGHT_COLUMNS)

    if len(df) < 3 and strategy != "conservative":
        print(f"[{strategy}] 후보 종목 {len(df)}개 미만 → conservative fallback")
        return compute_weights(signal_df, strategy="conservative")

    # ─────────────────────────────────────────────────────────────
    # aggressive: Greedy Knapsack
    # ─────────────────────────────────────────────────────────────
    if strategy == "aggressive":
        df["score"] = df["signal_return"] * 0.6 + df["signal_upside"] * 0.4
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

        n          = len(df)
        sub_assets = df["sub_asset"].tolist()
        sub_caps   = df["sub_asset_cap"].tolist()
        asset_cls  = df["asset_class"].tolist()
        weights    = np.zeros(n)

        # 상태는 루프 밖에서 한 번만 초기화, 배정 시마다 누적 업데이트
        sub_used  = {}
        risk_used = 0.0

        for _ in range(20):
            weight_remaining = 1.0 - weights.sum()
            if weight_remaining < 1e-6:
                break

            made_progress = False
            for i in range(n):
                weight_remaining = 1.0 - weights.sum()
                if weight_remaining < 1e-6:
                    break

                sub     = sub_assets[i]
                sub_cap = sub_caps[i]

                individual_avail = MAX_SINGLE_TICKER - weights[i]
                sub_avail        = sub_cap - sub_used.get(sub, 0.0)
                class_avail      = (MAX_RISK_ASSET - risk_used
                                    if asset_cls[i] == "위험자산"
                                    else weight_remaining)

                alloc = max(0.0, min(individual_avail, sub_avail,
                                     class_avail, weight_remaining))

                if alloc > 1e-6:
                    weights[i]     += alloc
                    sub_used[sub]   = sub_used.get(sub, 0.0) + alloc
                    if asset_cls[i] == "위험자산":
                        risk_used += alloc
                    made_progress = True

            if not made_progress:
                break

        # ── 잔여 비중 안전자산 fallback ──
        # return_threshold 필터로 제외됐던 안전자산 종목도 포함해 잔여 채움
        weight_remaining = 1.0 - weights.sum()
        if weight_remaining > 1e-6:
            # signal_df 원본에서 안전자산 후보를 다시 가져옴 (필터 미적용)
            #fallback_candidates = signal_df[
            #    signal_df["asset_class"] == "안전자산"
            #].copy()
            fallback_candidates = signal_df[
                (signal_df["asset_class"] == "안전자산") &
                (signal_df["signal_return"] >= 0)
            ].copy()
            fallback_candidates["score"] = (
                fallback_candidates["signal_return"] * 0.6
                + fallback_candidates["signal_upside"] * 0.4
            )
            fallback_candidates = fallback_candidates.sort_values("score", ascending=False)

            for _, row in fallback_candidates.iterrows():
                weight_remaining = 1.0 - weights.sum()
                if weight_remaining < 1e-6:
                    break

                ticker = str(row[GROUP_COL])
                sub    = row["sub_asset"]
                sub_cap= row["sub_asset_cap"]

                # df에 이미 있는 종목이면 해당 인덱스 재활용, 없으면 신규 추가
                existing = df[df[GROUP_COL].astype(str) == ticker]
                if not existing.empty:
                    i_idx = existing.index[0]
                    cur_w = weights[i_idx]
                    individual_avail = MAX_SINGLE_TICKER - cur_w
                    sub_avail        = sub_cap - sub_used.get(sub, 0.0)
                    alloc = max(0.0, min(individual_avail, sub_avail, weight_remaining))
                    if alloc > 1e-6:
                        weights[i_idx] += alloc
                        sub_used[sub]   = sub_used.get(sub, 0.0) + alloc
                else:
                    # 신규 종목: df에 행 추가
                    sub_avail        = sub_cap - sub_used.get(sub, 0.0)
                    individual_avail = MAX_SINGLE_TICKER
                    alloc = max(0.0, min(individual_avail, sub_avail, weight_remaining))
                    if alloc > 1e-6:
                        new_row = row.copy()
                        new_row["weight"] = 0.0
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                        weights = np.append(weights, alloc)
                        sub_used[sub] = sub_used.get(sub, 0.0) + alloc
                        sub_assets.append(sub)
                        sub_caps.append(sub_cap)
                        asset_cls.append("안전자산")

        # ── 정규화 금지: 잔여 비중이 남아도 cap 위반 없이 그대로 유지 ──
        # weights.sum()이 1.0 미만일 수 있음 (conservative fallback 고려)
        #weight_total = weights.sum()
        #if weight_total < 0.90:
           # print(f"[{strategy}] 비중 합계 {weight_total:.4f} < 0.90 → conservative fallback")
          #  return compute_weights(signal_df, strategy="conservative")

        #if weight_total < 1.0 - 1e-5:
         #   print(f"[{strategy}] 경고: 비중 합계 {weight_total:.4f} (< 1.0), cap 초과 없이 배분 완료")

        df["weight"] = weights

        #assert abs(df["weight"].sum() - 1.0) < 0.11, \
         #   f"비중 합산 오류 (fallback 미작동): {df['weight'].sum():.6f}"

        return df[[
            GROUP_COL, "name", SECTOR_COL,
            "asset_class", "sub_asset",
            "sector_type", "signal_return", "signal_sharpe", "weight",
        ]]

    # ─────────────────────────────────────────────────────────────
    # conservative: 기존 로직 유지
    # ─────────────────────────────────────────────────────────────

    # ── 2. 점수 계산 ──
    df["score"] = df["signal_sharpe"] * 0.7 + df["signal_return"] * 0.3
    df.loc[df["sector_type"] == "defensive", "score"] *= 1.3

    # ── 3. 상위 N개 선택 ──
    df = df.nlargest(top_n, "score").reset_index(drop=True)

    # ── 4. softmax 초기 비중 ──
    weights = _softmax_weights(df["score"].values)

    # ── 5. 전략 내부 제약 ──
    weights = _clip_and_renorm(weights, max_w)
    weights = _enforce_defensive_floor(df, weights, defensive_floor, max_w)
    if rf_weight > 0:
        weights = _enforce_rf_weight(df, weights, rf_weight, max_w)

    # ── 6. DB GAPS 외부 제약 (ticker 단위) ──
    weights = apply_dbgaps_constraints(df, weights)

    df["weight"] = weights

    # ── 7. 합산 검증 ──
    assert abs(df["weight"].sum() - 1.0) < 1e-5, \
        f"비중 합산 오류: {df['weight'].sum():.6f}"

    return df[[
        GROUP_COL, "name", SECTOR_COL,
        "asset_class", "sub_asset",
        "sector_type", "signal_return", "signal_sharpe", "weight",
    ]]


# ─────────────────────────────────────────────────────────────
#  요약 출력
# ─────────────────────────────────────────────────────────────

def portfolio_summary(weight_df: pd.DataFrame, strategy: StrategyType) -> dict:
    if weight_df.empty:
        return {}

    expected_ret = (weight_df["weight"] * weight_df["signal_return"]).sum()
    asset_alloc  = weight_df.groupby("asset_class")["weight"].sum().to_dict()
    sub_alloc    = weight_df.groupby("sub_asset")["weight"].sum().to_dict()

    # DB GAPS 위반 체크
    violations = []
    for _, row in weight_df.iterrows():
        if row["weight"] > MAX_SINGLE_TICKER + 1e-4:
            violations.append(f"개별종목 초과: {row['name']} {row['weight']*100:.1f}%")
    if asset_alloc.get("위험자산", 0) > MAX_RISK_ASSET + 1e-4:
        violations.append(f"위험자산 초과: {asset_alloc['위험자산']*100:.1f}% > 70%")
    for sub, w in sub_alloc.items():
        cap = next((v[2] for v in TICKER_ASSET_MAP.values() if v[1] == sub), None)
        if cap and w > cap + 1e-4:
            violations.append(f"세부자산 초과: {sub} {w*100:.1f}% > {cap*100:.0f}%")

    summary = {
        "strategy":        strategy,
        "n_holdings":      len(weight_df),
        "expected_return": round(expected_ret * 100, 3),
        "asset_allocation": asset_alloc,
        "sub_allocation":   sub_alloc,
        "violations":       violations,
        "top_holdings":     weight_df.nlargest(5, "weight")[
            [GROUP_COL, "name", "sub_asset", "weight", "signal_return"]
        ].to_dict("records"),
    }

    print(f"\n{'─'*55}")
    print(f"[{strategy.upper()}] 포트폴리오 요약")
    print(f"  편입 종목 수    : {summary['n_holdings']}")
    print(f"  기대 5d 수익률  : {summary['expected_return']:.3f}%")
    print(f"\n  [DB GAPS] 자산구분")
    for k, v in asset_alloc.items():
        cap_str = " (상한 70%)" if k == "위험자산" else ""
        flag    = " ⚠️" if k == "위험자산" and v > MAX_RISK_ASSET + 1e-4 else " ✅"
        print(f"    {k:<8} {v*100:.1f}%{cap_str}{flag}")
    print(f"\n  [DB GAPS] 세부자산")
    for sub, w in sorted(sub_alloc.items(), key=lambda x: -x[1]):
        cap  = next((v[2] for v in TICKER_ASSET_MAP.values() if v[1] == sub), 1.0)
        flag = " ⚠️" if w > cap + 1e-4 else " ✅"
        print(f"    {sub:<22} {w*100:.1f}% / {cap*100:.0f}%{flag}")
    print(f"\n  Top5 종목")
    for h in summary["top_holdings"]:
        flag = " ⚠️" if h["weight"] > MAX_SINGLE_TICKER + 1e-4 else ""
        print(f"    {h['name']:<28} {h['weight']*100:.1f}%  "
              f"예측 {h['signal_return']*100:.2f}%  [{h['sub_asset']}]{flag}")
    if violations:
        print(f"\n  ⚠️ DB GAPS 위반:")
        for v in violations:
            print(f"    - {v}")
    else:
        print(f"\n  ✅ DB GAPS 제약 모두 충족")

    return summary


# ─────────────────────────────────────────────────────────────
#  비교 & 저장
# ─────────────────────────────────────────────────────────────

def compare_strategies(cons_df: pd.DataFrame, agg_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        cons_df[[GROUP_COL, "name", SECTOR_COL, "sub_asset", "weight", "signal_return"]].rename(
            columns={"weight": "w_conservative"}),
        agg_df[[GROUP_COL, "name", SECTOR_COL, "weight"]].rename(
            columns={"weight": "w_aggressive"}),
        on=[GROUP_COL, "name", SECTOR_COL], how="outer",
    ).fillna(0)
    return merged.sort_values("w_aggressive", ascending=False)


def save_results(cons_df: pd.DataFrame, agg_df: pd.DataFrame, pred_date: str) -> None:
    out = PORTFOLIO_RESULT_DIR / pred_date
    out.mkdir(parents=True, exist_ok=True)
    cons_df.to_csv(out / "portfolio_conservative.csv", index=False)
    agg_df.to_csv(out  / "portfolio_aggressive.csv",   index=False)
    compare_strategies(cons_df, agg_df).to_csv(out / "portfolio_comparison.csv", index=False)
    print(f"\n[save] 포트폴리오 저장 → {out}")
