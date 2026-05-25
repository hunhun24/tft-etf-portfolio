"""
xgboost_backtest.py — XGBoost top-10 포트폴리오 백테스트 및 TFT 비교

실행: python baseline/xgboost_backtest.py

입력:
  - baseline/xgboost_predictions.csv        (xgboost_baseline.py 실행 후 생성)
  - src_tft_port_regime/outputs/portfolio/backtest_regime/backtest_returns.csv

XGBoost 포트폴리오:
  2025-01-02 이후 주 첫 거래일 리밸런싱 (TFT와 동일 구간).
  pred > 0 필터 후 상위 10개 equal weight → DB GAPS 제약 적용.

DB GAPS 제약:
  src_tft_port_regime/config.py, portfolio/portfolio.py 에서 직접 import.
  개별 종목(위험자산) 20%, 세부자산 sub_asset_cap, 위험자산 합계 70%,
  안전자산 개별 min(sub_cap, 50%).

출력 CSV 컬럼:
  strategy, cum_return_%, ann_return_%, ann_vol_%, sharpe, MDD_%, hit_rate_%, n_periods
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── 경로 ─────────────────────────────────────────────────────────
BASELINE_DIR = Path(__file__).resolve().parent        # geonho/baseline/
PROJECT_ROOT = BASELINE_DIR.parent.parent             # Financial_project/
TFT_SRC      = PROJECT_ROOT / "src_tft_port_regime"

# runtime import: Pylance 미해석은 정상 (sys.path 동적 추가)
sys.path.insert(0, str(TFT_SRC))
from config import (                                   # noqa: E402
    TICKER_ASSET_MAP, MAX_SINGLE_TICKER, MAX_RISK_ASSET,
    TEST_START_DATE, GROUP_COL, DUPLICATE_TICKERS,
)
from portfolio.portfolio import apply_dbgaps_constraints  # noqa: E402

PRED_PATH = BASELINE_DIR / "xgboost_predictions.csv"
TFT_PATH  = (
    TFT_SRC / "outputs" / "portfolio" / "backtest_regime" / "backtest_returns.csv"
)
OUT_PATH  = BASELINE_DIR / "backtest_comparison.csv"

TOP_N = 10
SEP   = "-" * 65

# 출력 CSV 컬럼 순서
SUMMARY_COLS = [
    "strategy", "cum_return_%", "ann_return_%", "ann_vol_%",
    "sharpe", "MDD_%", "hit_rate_%", "n_periods",
]


# ── 로드 ─────────────────────────────────────────────────────────

def load_xgb_preds() -> pd.DataFrame:
    if not PRED_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost 예측 파일 없음: {PRED_PATH}\n"
            "xgboost_baseline.py를 먼저 실행하세요."
        )
    df = pd.read_csv(PRED_PATH)
    df["date"]   = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    # 중복 ticker 제거 (SARIMAX·TFT와 동일하게 universe 정렬)
    # 157450: A 접두사 없는 잘못된 row (rate_cash로 잘못 매핑됨, 정상 ticker는 A157450)
    df = df[~df["ticker"].isin(DUPLICATE_TICKERS)]
    # target_5d NaN 제거 후 TFT와 동일 구간 필터
    df = df.dropna(subset=["target_5d"])
    df = df[df["date"] >= pd.Timestamp(TEST_START_DATE)].reset_index(drop=True)
    return df


def load_tft_returns() -> pd.DataFrame:
    if not TFT_PATH.exists():
        raise FileNotFoundError(f"TFT backtest_returns.csv 없음: {TFT_PATH}")
    df = pd.read_csv(TFT_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def resolve_tft_columns(tft: pd.DataFrame) -> tuple[str, str]:
    """
    TFT CSV 컬럼명을 robust하게 매칭.
    backtest_regime.py 버전에 따라 컬럼명이 `*_return` 또는 `*_return_5d` 일 수 있음.
    """
    eq_candidates = ("equal_weight_return", "equal_weight_return_5d")
    rg_candidates = ("regime_return", "final_return_5d", "final_return")

    eq_col = next((c for c in eq_candidates if c in tft.columns), None)
    rg_col = next((c for c in rg_candidates if c in tft.columns), None)

    if eq_col is None or rg_col is None:
        raise ValueError(
            f"TFT CSV 컬럼 매칭 실패. "
            f"available={tft.columns.tolist()}  "
            f"eq_candidates={eq_candidates}  rg_candidates={rg_candidates}"
        )
    return eq_col, rg_col


# ── DB GAPS 자산 정보 ─────────────────────────────────────────────

def build_asset_df(tickers: list[str]) -> pd.DataFrame:
    """TICKER_ASSET_MAP 기준으로 ticker별 asset_class / sub_asset / sub_asset_cap 조회."""
    rows = []
    for t in tickers:
        asset_class, sub_asset, sub_cap = TICKER_ASSET_MAP.get(
            t, ("위험자산", "기타", 0.20)
        )
        rows.append({
            GROUP_COL:       t,
            "asset_class":   asset_class,
            "sub_asset":     sub_asset,
            "sub_asset_cap": sub_cap,
        })
    return pd.DataFrame(rows)


# ── 리밸런싱 날짜 ────────────────────────────────────────────────

def make_rebal_dates(preds: pd.DataFrame) -> list[pd.Timestamp]:
    """target_5d.notna() 날짜 기준 주 첫 거래일."""
    dates = preds[["date"]].drop_duplicates().copy()
    dates["week"] = dates["date"].dt.to_period("W")
    return sorted(dates.groupby("week")["date"].min().tolist())


# ── 포트폴리오 백테스트 ──────────────────────────────────────────

def xgb_backtest(preds: pd.DataFrame, rebal_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """
    각 리밸런싱일:
      1. pred > 0 필터 후 pred 상위 TOP_N 선택
      2. 초기 비중 1/N 균등
      3. apply_dbgaps_constraints() 로 DB GAPS 제약 적용
         - 위험자산 개별 ≤ 20%, 세부자산 합 ≤ sub_asset_cap
         - 위험자산 합계 ≤ 70%, 안전자산 개별 ≤ min(sub_cap, 50%)
      4. realized_return = Σ(weight_i × target_5d_i)
    """
    records = []
    for rd in rebal_dates:
        day = preds[preds["date"] == rd].copy()
        if day.empty:
            continue

        candidates = (
            day[day["pred"] > 0]
            .dropna(subset=["target_5d"])
            .nlargest(TOP_N, "pred")
            .reset_index(drop=True)
        )
        n = len(candidates)

        if n == 0:
            records.append({"date": rd, "n_holdings": 0, "realized_return": 0.0})
            continue

        tickers   = candidates["ticker"].tolist()
        asset_df  = build_asset_df(tickers)
        weights   = np.ones(n) / n

        # DB GAPS 제약 적용
        weights = apply_dbgaps_constraints(asset_df, weights)

        # 사후 검증: top-N에 안전자산이 없으면 apply_dbgaps_constraints 내부의
        # _enforce_risk_asset_cap이 redistribute 못해서 70% cap이 깨질 수 있음.
        # → 위험자산 70% 초과 시 강제로 잘라내고 잔여분은 implicit cash (0% 수익).
        # 정규화하지 않음 → cash 잔여 = 1.0 - weights.sum() 이 0% 수익으로 자동 처리.
        risk_mask  = (asset_df["asset_class"] == "위험자산").values
        risk_total = weights[risk_mask].sum()
        if risk_total > MAX_RISK_ASSET + 1e-6:
            weights[risk_mask] *= MAX_RISK_ASSET / risk_total

        realized = float((weights * candidates["target_5d"].values).sum())

        records.append({
            "date":            rd,
            "n_holdings":      n,
            "realized_return": realized,
        })

    return pd.DataFrame(records)


# ── 성과 분석 ────────────────────────────────────────────────────

def analyze_performance(rets: pd.Series, label: str) -> dict:
    """backtest_regime.py::analyze_performance와 동일한 지표."""
    rets = rets.dropna().sort_index()
    n    = len(rets)
    if n == 0:
        return {c: (label if c == "strategy" else 0) for c in SUMMARY_COLS}

    cum_ret   = float((1 + rets).prod() - 1)
    ann_ret   = float((1 + cum_ret) ** (52 / n) - 1)
    ann_vol   = float(rets.std() * np.sqrt(52))
    sharpe    = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0
    cum_curve = (1 + rets).cumprod()
    mdd       = float((cum_curve / cum_curve.cummax() - 1).min())
    hit_rate  = float((rets > 0).mean())

    return {
        "strategy":     label,
        "cum_return_%": round(cum_ret  * 100, 2),
        "ann_return_%": round(ann_ret  * 100, 2),
        "ann_vol_%":    round(ann_vol  * 100, 2),
        "sharpe":       round(sharpe,          3),
        "MDD_%":        round(mdd      * 100, 2),
        "hit_rate_%":   round(hit_rate * 100, 1),
        "n_periods":    n,
    }


def print_row(row: dict) -> None:
    if not row or row.get("n_periods", 0) == 0:
        print(f"  [{row.get('strategy', '?')}] 데이터 없음")
        return
    print(f"  전략            : {row['strategy']}")
    print(f"  기간(주)        : {row['n_periods']}")
    print(f"  누적 수익률     : {row['cum_return_%']:+.2f}%")
    print(f"  연환산 수익률   : {row['ann_return_%']:+.2f}%")
    print(f"  연환산 변동성   : {row['ann_vol_%']:.2f}%")
    print(f"  Sharpe          : {row['sharpe']:.3f}")
    print(f"  MDD             : {row['MDD_%']:.2f}%")
    print(f"  Hit Rate        : {row['hit_rate_%']:.1f}%")


# ── main ─────────────────────────────────────────────────────────

def main() -> None:
    print(SEP)
    print(f"XGBoost top-{TOP_N} baseline vs TFT portfolio strategies")
    print(f"비교 성격: XGBoost top-{TOP_N} baseline vs TFT portfolio strategies "
          f"(베이스라인 대비 부가가치 검증)")
    print(SEP)

    # ── 경로 ──
    print("\n[데이터 경로]")
    print(f"  XGBoost 예측 : {PRED_PATH}")
    print(f"  TFT 결과     : {TFT_PATH}")

    # ── 로드 ──
    print("\n1. 데이터 로드")
    preds = load_xgb_preds()
    tft   = load_tft_returns()
    print(f"  XGBoost prediction date range : "
          f"{preds['date'].min().date()} ~ {preds['date'].max().date()}  "
          f"({len(preds):,}행)")
    print(f"  TFT backtest date range       : "
          f"{tft.index.min().date()} ~ {tft.index.max().date()}  "
          f"({len(tft):,}행)")

    # TFT 컬럼 선택 (버전 차이 흡수)
    eq_col, rg_col = resolve_tft_columns(tft)
    print(f"  TFT columns resolved : equal_weight='{eq_col}', regime='{rg_col}'")

    # ── XGBoost 백테스트 ──
    print(f"\n2. XGBoost top-{TOP_N} + DB GAPS 백테스트")
    print(f"  XGBoost 적용 제약: DB GAPS only (개별 20%, sub_asset_cap, 위험자산 70%)")
    rebal_dates = make_rebal_dates(preds)
    print(f"  XGBoost 리밸런싱 날짜 수: {len(rebal_dates)}회  "
          f"({rebal_dates[0].date()} ~ {rebal_dates[-1].date()})")

    xgb_result = xgb_backtest(preds, rebal_dates)
    xgb_rets   = xgb_result.set_index("date")["realized_return"].sort_index()

    # ── 공통 날짜 교집합 ──
    common_dates = xgb_rets.index.intersection(tft.index)
    n_common     = len(common_dates)

    if n_common == 0:
        print(f"\n3. 공통 날짜 없음")
        print(f"  XGBoost 기간: {xgb_rets.index.min().date()} ~ {xgb_rets.index.max().date()}")
        print(f"  TFT 기간    : {tft.index.min().date()} ~ {tft.index.max().date()}")
        return

    print(f"\n3. 공통 날짜 교집합")
    print(f"  common date range : "
          f"{common_dates.min().date()} ~ {common_dates.max().date()}  "
          f"(common n_periods = {n_common}주)")
    print(f"  TFT regime 적용 제약: DB GAPS + 내부 제약 "
          f"(signal_sharpe, 방어섹터 floor, 현금성 floor, 레짐 판단)")

    xgb_c  = xgb_rets.loc[common_dates]
    rg_c   = tft.loc[common_dates, rg_col]
    eq_c   = tft.loc[common_dates, eq_col]

    # ── 스케일 검증 (산식 일관성 확인) ──
    print(f"\n4. 스케일 검증 (target_5d=log return, realized_return=Σw·target_5d, (1+r).prod() 공통)")
    t5d_sample = preds["target_5d"].dropna().head(5).values
    xgb_sample = xgb_c.head(5).values
    rg_sample  = rg_c.head(5).values
    print(f"  target_5d 샘플 5개     : {[round(v, 6) for v in t5d_sample]}")
    print(f"  XGBoost realized_return: {[round(v, 6) for v in xgb_sample]}")
    print(f"  TFT regime_return      : {[round(v, 6) for v in rg_sample]}")

    # ── 성과 계산 ──
    rows = [
        analyze_performance(xgb_c, "xgboost_top10"),
        analyze_performance(rg_c,  "tft_regime"),
        analyze_performance(eq_c,  "tft_equal_weight"),
    ]

    # ── 개별 출력 ──
    labels = [f"XGBoost top-{TOP_N} (DB GAPS)", "TFT regime", "TFT equal_weight"]
    for label, row in zip(labels, rows):
        print(f"\n{SEP}")
        print(f"[{label}]")
        print(SEP)
        print_row(row)

    # ── 비교표 ──
    summary_df = pd.DataFrame(rows)[SUMMARY_COLS]
    print(f"\n{SEP}")
    print("성과 비교 요약")
    print(SEP)
    print(summary_df.to_string(index=False))

    # ── 저장 ──
    summary_df.to_csv(OUT_PATH, index=False)
    print(f"\n[save] {OUT_PATH}")


if __name__ == "__main__":
    main()
