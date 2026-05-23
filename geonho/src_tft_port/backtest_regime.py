"""
backtest_regime.py — regime-switching 롤링 백테스트
실행:
python geonho/src_tft_port/backtest_regime.py \
  --ckpt geonho/src_tft_port/outputs/checkpoints/tft-no_end_macro_sentiment-epoch=06-val_loss=0.0125.ckpt

기존 backtest.py의 TFT inference / 성과 요약 흐름을 유지하고,
포트폴리오 선택만 regime-switch wrapper로 수행한다.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import numpy as np
import pandas as pd
import torch
import torchmetrics
from pytorch_forecasting import TemporalFusionTransformer

from backtest import (
    analyze_performance,
    collect_rolling_predictions,
    find_latest_checkpoint,
)
from config import (
    PROCESSED_DATA_PATH, CHECKPOINT_DIR, DATE_COL, GROUP_COL,
    TEST_START_DATE, PORTFOLIO_RESULT_DIR,
)
from data.dataset import make_datasets
from portfolio.portfolio_regime import compute_regime_portfolio


def _realized_return(
    weight_df: pd.DataFrame,
    actual: pd.DataFrame,
    date_str: str,
) -> float:
    """기존 backtest.py와 동일하게 target_5d를 merge하고 결측 수익률은 0으로 계산."""
    if weight_df.empty:
        return np.nan

    act_day = actual[actual[DATE_COL] == pd.Timestamp(date_str)]
    merged = weight_df.merge(
        act_day[[GROUP_COL, "target_5d"]],
        on=GROUP_COL,
        how="left",
    )
    return float((merged["weight"] * merged["target_5d"].fillna(0)).sum())


def _flatten_regime_info(date_str: str, regime_info: dict) -> dict:
    """regime_info를 regime_log_all.csv 저장용 1-row dict로 변환."""
    row = {"date": date_str}
    for key, value in regime_info.items():
        if key == "triggered_conditions":
            continue
        if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
            row[key] = value
    for key, value in regime_info.get("triggered_conditions", {}).items():
        row[f"cond_{key}"] = value
    return row


def _make_summary_input(result_df: pd.DataFrame) -> pd.DataFrame:
    """wide regime return table을 analyze_performance()용 long format으로 변환."""
    mappings = {
        "regime_switch": "final_return_5d",
        "conservative_candidate": "conservative_candidate_return_5d",
        "aggressive_candidate": "aggressive_candidate_return_5d",
        "equal_weight": "equal_weight_return_5d",
    }

    frames = []
    for strategy, col in mappings.items():
        part = result_df[["date", col]].rename(columns={col: "realized_return"}).copy()
        part["strategy"] = strategy
        part["n_holdings"] = np.nan
        part["expected_return"] = np.nan
        frames.append(part)

    return pd.concat(frames, ignore_index=True)[[
        "date", "strategy", "n_holdings", "expected_return", "realized_return",
    ]]


def run_backtest_regime(
    ckpt_path: Path,
    start_date: str = TEST_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    print(f"[backtest regime] checkpoint: {ckpt_path}")

    # ── torchmetrics.Metric._apply 패치 ───────────────────────────────────────
    # 기존 backtest.py와 동일한 checkpoint loading 원칙 유지.
    _orig_metric_apply = torchmetrics.Metric._apply

    def _cuda_safe_metric_apply(self, fn, *args, **kwargs):
        if hasattr(self, '_device') and 'cuda' in str(self._device):
            self._device = torch.device('cpu')
        return _orig_metric_apply(self, fn, *args, **kwargs)

    torchmetrics.Metric._apply = _cuda_safe_metric_apply
    try:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = TemporalFusionTransformer.load_from_checkpoint(
            str(ckpt_path),
            map_location="cpu",
            logging_metrics=[],
        )
        model = model.to(device)
        model.eval()
    finally:
        torchmetrics.Metric._apply = _orig_metric_apply
    # ─────────────────────────────────────────────────────────────────────────

    # 기존 backtest.py와 동일하게 make_datasets()로 reference dataset 획득
    train_ds, _, _, _ = make_datasets()

    # 전처리 완료 데이터 로드 (inference 슬라이스 및 실제 수익률 조회용)
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=[DATE_COL])
    df[GROUP_COL] = df[GROUP_COL].astype(str)

    # 리밸런싱 날짜: test 구간 내 주 첫 거래일
    test_dates = df[df[DATE_COL] >= pd.Timestamp(start_date)][[DATE_COL]].drop_duplicates()
    test_dates["week"] = test_dates[DATE_COL].dt.to_period("W")
    rebal_dates = test_dates.groupby("week")[DATE_COL].min().tolist()

    print(f"[backtest regime] 리밸런싱 횟수: {len(rebal_dates)}회  "
          f"({rebal_dates[0].date() if rebal_dates else '—'} ~ "
          f"{rebal_dates[-1].date() if rebal_dates else '—'})")

    # ── 예측 수집 ──
    preds_by_date = collect_rolling_predictions(model, df, train_ds, rebal_dates)

    # ── 실제 수익률 lookup ──
    actual = df[[DATE_COL, GROUP_COL, "target_5d"]].copy()
    actual[GROUP_COL] = actual[GROUP_COL].astype(str)

    records = []
    regime_logs = []

    for date_str, pred_df in preds_by_date.items():
        try:
            result = compute_regime_portfolio(
                pred_df=pred_df,
                pred_date=date_str,
                verbose=False,
            )

            final_df = result["final"]
            cons_df = result["conservative"]
            agg_df = result["aggressive"]
            regime_info = result["regime_info"]

            final_return = _realized_return(final_df, actual, date_str)
            cons_return = _realized_return(cons_df, actual, date_str)
            agg_return = _realized_return(agg_df, actual, date_str)

            all_tickers = pred_df[GROUP_COL].unique()
            act_day = actual[actual[DATE_COL] == pd.Timestamp(date_str)]
            bm_df = act_day[act_day[GROUP_COL].isin(all_tickers)]
            bm_ret = bm_df["target_5d"].mean() if not bm_df.empty else 0.0

            records.append({
                "date": date_str,
                "regime": regime_info.get("regime"),
                "selected_mode": regime_info.get("selected_mode"),
                "regime_score": regime_info.get("regime_score"),
                "fallback_reason": "",
                "final_return_5d": final_return,
                "conservative_candidate_return_5d": cons_return,
                "aggressive_candidate_return_5d": agg_return,
                "equal_weight_return_5d": bm_ret,
                "n_final_assets": len(final_df),
                "final_weight_sum": float(final_df["weight"].sum()) if not final_df.empty else 0.0,
                "avg_pred_return": regime_info.get("avg_pred_return"),
                "top_pred_return": regime_info.get("top_pred_return"),
                "downside_risk": regime_info.get("downside_risk"),
                "forecast_uncertainty": regime_info.get("forecast_uncertainty"),
                "pred_dispersion": regime_info.get("pred_dispersion"),
            })
            regime_logs.append(_flatten_regime_info(date_str, regime_info))

        except Exception as e:
            print(f"  [skip] {date_str}: {e}")
            records.append({
                "date": date_str,
                "regime": np.nan,
                "selected_mode": np.nan,
                "regime_score": np.nan,
                "fallback_reason": str(e),
                "final_return_5d": np.nan,
                "conservative_candidate_return_5d": np.nan,
                "aggressive_candidate_return_5d": np.nan,
                "equal_weight_return_5d": np.nan,
                "n_final_assets": 0,
                "final_weight_sum": 0.0,
                "avg_pred_return": np.nan,
                "top_pred_return": np.nan,
                "downside_risk": np.nan,
                "forecast_uncertainty": np.nan,
                "pred_dispersion": np.nan,
            })
            regime_logs.append({
                "date": date_str,
                "error": str(e),
            })

    result_df = pd.DataFrame(records)
    regime_log_df = pd.DataFrame(regime_logs)
    summary_input = _make_summary_input(result_df)
    summary_df = analyze_performance(summary_input)

    return result_df, regime_log_df, summary_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",  type=str, default=None)
    p.add_argument("--start", type=str, default=TEST_START_DATE)
    args = p.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else find_latest_checkpoint(Path(CHECKPOINT_DIR))

    result_df, regime_log_df, summary_df = run_backtest_regime(
        ckpt,
        start_date=args.start,
    )

    out_dir = PORTFOLIO_RESULT_DIR / "backtest_regime"
    out_dir.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(out_dir / "regime_backtest_returns.csv", index=False)
    regime_log_df.to_csv(out_dir / "regime_log_all.csv", index=False)
    summary_df.to_csv(out_dir / "regime_backtest_summary.csv", index=False)

    print(f"\n[save] → {out_dir}")


if __name__ == "__main__":
    main()
