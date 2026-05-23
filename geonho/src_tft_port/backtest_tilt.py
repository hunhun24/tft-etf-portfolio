"""
backtest_tilt.py — Equal Weight + TFT ranking tilt 전용 롤링 백테스트

실행:
python geonho/src_tft_port/backtest_tilt.py \
  --ckpt geonho/src_tft_port/outputs/checkpoints/tft-no_end_macro_sentiment-epoch=06-val_loss=0.0125.ckpt \
  --alpha 0.3
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

from backtest import (
    analyze_performance,
    collect_rolling_predictions,
    find_latest_checkpoint,
)
from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATE_COL,
    GROUP_COL,
    PORTFOLIO_RESULT_DIR,
    PROCESSED_DATA_PATH,
    TEST_START_DATE,
)
from portfolio.portfolio import build_signal, compute_weights


def format_alpha_for_dir(alpha: float) -> str:
    text = f"{alpha:g}".replace("-", "m").replace(".", "")
    return f"alpha{text}"


def load_cuda_safe_tft_checkpoint(ckpt_path: Path) -> TemporalFusionTransformer:
    """
    Colab CUDA checkpoint를 CUDA 없는 Mac 로컬에서도 로드 가능하게 처리.
    ranking_check.py와 같은 원칙으로 map_location과 torchmetrics 패치를 적용한다.
    """
    _orig_metric_apply = torchmetrics.Metric._apply

    def _cuda_safe_metric_apply(self, fn, *args, **kwargs):
        if hasattr(self, "_device") and "cuda" in str(self._device):
            self._device = torch.device("cpu")
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
        return model
    finally:
        torchmetrics.Metric._apply = _orig_metric_apply


def make_rebal_dates(df: pd.DataFrame, start_date: str) -> list[pd.Timestamp]:
    test_dates = df[
        (df[DATE_COL] >= pd.Timestamp(start_date)) &
        (df["target_5d"].notna())
    ][[DATE_COL]].drop_duplicates()
    test_dates["week"] = test_dates[DATE_COL].dt.to_period("W")
    return [pd.Timestamp(d) for d in test_dates.groupby("week")[DATE_COL].min().tolist()]


def make_checkpoint_reference_dataset(
    model: TemporalFusionTransformer,
    df: pd.DataFrame,
) -> TimeSeriesDataSet:
    dataset_params = model.hparams.dataset_parameters
    dataset_df = df.dropna(subset=["target_5d"]).copy()
    print(
        f"[backtest tilt] dataset rows after target_5d dropna: "
        f"{len(df)} -> {len(dataset_df)}"
    )

    return TimeSeriesDataSet.from_parameters(
        dataset_params,
        dataset_df,
        predict=False,
        stop_randomization=True,
    )


def append_weighted_result(
    records: list[dict],
    actual: pd.DataFrame,
    date_str: str,
    strategy: str,
    w_df: pd.DataFrame,
) -> None:
    if w_df.empty:
        return

    act_day = actual[actual[DATE_COL] == pd.Timestamp(date_str)]
    w_df = w_df.merge(
        act_day[[GROUP_COL, "target_5d"]],
        on=GROUP_COL,
        how="left",
    )
    realized = (w_df["weight"] * w_df["target_5d"].fillna(0)).sum()

    records.append({
        "date": date_str,
        "strategy": strategy,
        "n_holdings": len(w_df),
        "expected_return": (w_df["weight"] * w_df["signal_return"]).sum(),
        "realized_return": realized,
    })


def run_backtest_tilt(
    ckpt_path: Path,
    alpha: float = 0.3,
    start_date: str = TEST_START_DATE,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    print(f"[backtest tilt] checkpoint: {ckpt_path}")
    print(f"[backtest tilt] alpha: {alpha}")
    if out_dir is not None:
        print(f"[backtest tilt] save dir: {out_dir}")

    model = load_cuda_safe_tft_checkpoint(ckpt_path)

    print("[backtest tilt] ckpt x_reals:")
    for i, col in enumerate(model.hparams.x_reals):
        print(f"  {i}: {col}")

    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=[DATE_COL])
    df[GROUP_COL] = df[GROUP_COL].astype(str)

    train_ds = make_checkpoint_reference_dataset(model, df)

    rebal_dates = make_rebal_dates(df, start_date)
    print(
        f"[backtest tilt] 리밸런싱 횟수: {len(rebal_dates)}회  "
        f"({rebal_dates[0].date() if rebal_dates else '—'} ~ "
        f"{rebal_dates[-1].date() if rebal_dates else '—'})"
    )

    preds_by_date = collect_rolling_predictions(
        model=model,
        df=df,
        train_ds=train_ds,
        rebal_dates=rebal_dates,
    )

    actual = df[[DATE_COL, GROUP_COL, "target_5d"]].copy()
    actual[GROUP_COL] = actual[GROUP_COL].astype(str)

    records = []

    for date_str, pred_df in preds_by_date.items():
        signal_df = build_signal(pred_df)

        for strategy in ("conservative", "aggressive"):
            w_df = compute_weights(signal_df, strategy=strategy)
            append_weighted_result(records, actual, date_str, strategy, w_df)

        tilt_df = compute_weights(signal_df, strategy="tilt", tilt_alpha=alpha)
        append_weighted_result(records, actual, date_str, "tilt", tilt_df)

        all_tickers = pred_df[GROUP_COL].unique()
        act_day = actual[actual[DATE_COL] == pd.Timestamp(date_str)]
        bm_df = act_day[act_day[GROUP_COL].isin(all_tickers)]
        bm_ret = bm_df["target_5d"].mean() if not bm_df.empty else 0.0
        records.append({
            "date": date_str,
            "strategy": "equal_weight",
            "n_holdings": len(all_tickers),
            "expected_return": np.nan,
            "realized_return": bm_ret,
        })

    return pd.DataFrame(records)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--start", type=str, default=TEST_START_DATE)
    parser.add_argument("--alpha", type=float, default=0.3)
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt = Path(args.ckpt) if args.ckpt else find_latest_checkpoint(Path(CHECKPOINT_DIR))
    out_dir = PORTFOLIO_RESULT_DIR / f"backtest_tilt_{format_alpha_for_dir(args.alpha)}"

    result_df = run_backtest_tilt(
        ckpt_path=ckpt,
        alpha=args.alpha,
        start_date=args.start,
        out_dir=out_dir,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_dir / "backtest_returns.csv", index=False)

    summary = analyze_performance(result_df)
    summary.to_csv(out_dir / "backtest_summary.csv", index=False)

    print(f"\n[save] → {out_dir}")


if __name__ == "__main__":
    main()
