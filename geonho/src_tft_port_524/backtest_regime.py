"""
backtest_regime.py — regime-switching 롤링 백테스트

실행:
python geonho/src_tft_port_524/backtest_regime.py \
  --ckpt geonho/src_tft_port_524/outputs/checkpoints/tft-no_end_macro_sentiment-epoch=06-val_loss=0.0125.ckpt \
  --start 2025-01-02

checkpoint dataset_parameters 기준으로 TimeSeriesDataSet 구성.
매 리밸런싱 날짜마다 determine_regime()으로 전략을 동적 선택.
비교 전략: equal_weight / conservative / aggressive / regime(동적)
저장 경로: outputs/portfolio/backtest_regime/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import numpy as np
import pandas as pd
import torch
import torchmetrics
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from config import (
    PROCESSED_DATA_PATH, CHECKPOINT_DIR, DATE_COL, GROUP_COL,
    SECTOR_COL, TEST_START_DATE, PORTFOLIO_RESULT_DIR,
    BATCH_SIZE, NUM_WORKERS,
)
from backtest import (
    collect_rolling_predictions, analyze_performance, find_latest_checkpoint,
)
from portfolio.portfolio import build_signal, compute_weights
from portfolio.regime_rule import determine_regime


# ─────────────────────────────────────────────────────────────
#  Checkpoint 로드
# ─────────────────────────────────────────────────────────────

def load_cuda_safe_tft_checkpoint(ckpt_path: Path) -> TemporalFusionTransformer:
    _orig = torchmetrics.Metric._apply

    def _safe(self, fn, *args, **kwargs):
        if hasattr(self, "_device") and "cuda" in str(self._device):
            self._device = torch.device("cpu")
        return _orig(self, fn, *args, **kwargs)

    torchmetrics.Metric._apply = _safe
    try:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = TemporalFusionTransformer.load_from_checkpoint(
            str(ckpt_path), map_location="cpu", logging_metrics=[],
        )
        model = model.to(device)
        model.eval()
        return model
    finally:
        torchmetrics.Metric._apply = _orig


# ─────────────────────────────────────────────────────────────
#  리밸런싱 날짜 생성 (target_5d 결측 방지)
# ─────────────────────────────────────────────────────────────

def make_rebal_dates(df: pd.DataFrame, start_date: str) -> list[pd.Timestamp]:
    test_dates = df[
        (df[DATE_COL] >= pd.Timestamp(start_date)) &
        (df["target_5d"].notna())
    ][[DATE_COL]].drop_duplicates()
    test_dates["week"] = test_dates[DATE_COL].dt.to_period("W")
    return [pd.Timestamp(d) for d in test_dates.groupby("week")[DATE_COL].min().tolist()]


# ─────────────────────────────────────────────────────────────
#  백테스트 루프
# ─────────────────────────────────────────────────────────────

def _realized_return(w_df: pd.DataFrame, actual: pd.DataFrame, date_str: str) -> float:
    if w_df.empty:
        return np.nan
    act_day = actual[actual[DATE_COL] == pd.Timestamp(date_str)]
    merged = w_df.merge(act_day[[GROUP_COL, "target_5d"]], on=GROUP_COL, how="left")
    return float((merged["weight"] * merged["target_5d"].fillna(0)).sum())


def run_backtest_regime(
    ckpt_path: Path,
    start_date: str = TEST_START_DATE,
    sentiment_threshold: float = -0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # ── 검증 로그 ──
    print(f"[backtest regime] checkpoint: {ckpt_path}")
    print(f"[backtest regime] sentiment_threshold: {sentiment_threshold}")

    model = load_cuda_safe_tft_checkpoint(ckpt_path)

    print("[backtest regime] ckpt x_reals:")
    for i, col in enumerate(model.hparams.x_reals):
        print(f"  {i}: {col}")

    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=[DATE_COL])
    df[GROUP_COL] = df[GROUP_COL].astype(str)

    # checkpoint dataset_parameters 기준으로 reference dataset 구성
    dataset_params = model.hparams.dataset_parameters
    dataset_df = df.dropna(subset=["target_5d"]).copy()
    print(
        f"[backtest regime] dataset rows after target_5d dropna: "
        f"{len(df)} -> {len(dataset_df)}"
    )

    train_ds = TimeSeriesDataSet.from_parameters(
        dataset_params, dataset_df, predict=False, stop_randomization=True,
    )

    rebal_dates = make_rebal_dates(df, start_date)
    print(
        f"[backtest regime] 리밸런싱 횟수: {len(rebal_dates)}회  "
        f"({rebal_dates[0].date() if rebal_dates else '—'} ~ "
        f"{rebal_dates[-1].date() if rebal_dates else '—'})"
    )

    out_dir = PORTFOLIO_RESULT_DIR / "backtest_regime"
    print(f"[backtest regime] save dir: {out_dir}")

    preds_by_date = collect_rolling_predictions(model, df, train_ds, rebal_dates)

    actual = df[[DATE_COL, GROUP_COL, "target_5d"]].copy()
    actual[GROUP_COL] = actual[GROUP_COL].astype(str)

    records    = []
    regime_log = []

    for date_str, pred_df in preds_by_date.items():
        signal_df = build_signal(pred_df)

        # 진단 지표 — determine_regime() 호출 전에 계산
        market_return   = signal_df["signal_return"].mean()
        market_upside   = signal_df["signal_upside"].mean()
        market_downside = signal_df["signal_downside"].mean()
        _recent = df[df["date"] <= pd.Timestamp(date_str)].tail(
            5 * len(signal_df["ticker"].unique())
        )
        recent_sentiment = _recent.groupby("date")["domestic_mean"].mean().tail(5).mean()

        regime, reason = determine_regime(
            signal_df, df, date_str, sentiment_threshold,
        )

        regime_w = compute_weights(signal_df, strategy=regime)
        cons_w   = compute_weights(signal_df, strategy="conservative")
        agg_w    = compute_weights(signal_df, strategy="aggressive")

        all_tickers = pred_df[GROUP_COL].unique()
        act_day = actual[actual[DATE_COL] == pd.Timestamp(date_str)]
        bm_df   = act_day[act_day[GROUP_COL].isin(all_tickers)]
        bm_ret  = bm_df["target_5d"].mean() if not bm_df.empty else 0.0

        records.append({
            "date":                date_str,
            "regime":              regime,
            "reason":              reason,
            "market_return":       market_return,
            "market_upside":       market_upside,
            "market_downside":     market_downside,
            "recent_sentiment":    recent_sentiment,
            "regime_return":       _realized_return(regime_w, actual, date_str),
            "conservative_return": _realized_return(cons_w,   actual, date_str),
            "aggressive_return":   _realized_return(agg_w,    actual, date_str),
            "equal_weight_return": bm_ret,
        })
        regime_log.append({"date": date_str, "regime": regime, "reason": reason})

        print(f"  [{date_str}] regime={regime} ({reason})")

    result_df    = pd.DataFrame(records)
    regime_log_df = pd.DataFrame(regime_log)

    # analyze_performance용 long-format 변환
    frames = []
    for strategy, col in [
        ("regime",       "regime_return"),
        ("conservative", "conservative_return"),
        ("aggressive",   "aggressive_return"),
        ("equal_weight", "equal_weight_return"),
    ]:
        part = result_df[["date", col]].rename(columns={col: "realized_return"}).copy()
        part["strategy"]        = strategy
        part["n_holdings"]      = np.nan
        part["expected_return"] = np.nan
        frames.append(part)

    summary_input = pd.concat(frames, ignore_index=True)[[
        "date", "strategy", "n_holdings", "expected_return", "realized_return",
    ]]
    summary_df = analyze_performance(summary_input)

    return result_df, regime_log_df, summary_df


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",                type=str,   default=None)
    p.add_argument("--start",               type=str,   default=TEST_START_DATE)
    p.add_argument("--sentiment_threshold", type=float, default=-0.5)
    args = p.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else find_latest_checkpoint(Path(CHECKPOINT_DIR))

    result_df, regime_log_df, summary_df = run_backtest_regime(
        ckpt,
        start_date=args.start,
        sentiment_threshold=args.sentiment_threshold,
    )

    out_dir = PORTFOLIO_RESULT_DIR / "backtest_regime"
    out_dir.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(out_dir / "backtest_returns.csv",   index=False)
    regime_log_df.to_csv(out_dir / "regime_log.csv",     index=False)
    summary_df.to_csv(out_dir / "backtest_summary.csv",  index=False)

    print(f"\n[save] → {out_dir}")


if __name__ == "__main__":
    main()
