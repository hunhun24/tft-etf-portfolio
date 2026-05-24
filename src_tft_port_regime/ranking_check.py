# src_tft_port_regime/ranking_check.py
'''
python src_tft_port_regime/ranking_check.py \
  --ckpt src_tft_port_regime/outputs/checkpoints/tft-no_end_macro_sentiment-epoch=06-val_loss=0.0125.ckpt
'''
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import pandas as pd
import numpy as np
import torch
import torchmetrics
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from config import (
    PROCESSED_DATA_PATH,
    CHECKPOINT_DIR,
    DATE_COL,
    GROUP_COL,
    TEST_START_DATE,
    PORTFOLIO_RESULT_DIR,
)
from backtest import collect_rolling_predictions, find_latest_checkpoint


TOP_K = 5


def load_cuda_safe_tft_checkpoint(ckpt_path: Path) -> TemporalFusionTransformer:
    """
    Colab CUDA checkpoint를 CUDA 없는 Mac 로컬에서도 로드 가능하게 처리.
    predict.py/backtest.py의 checkpoint loading 로직과 동일하게 유지.
    """
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
        return model
    finally:
        torchmetrics.Metric._apply = _orig_metric_apply


def make_rebal_dates(df: pd.DataFrame, start_date: str) -> list[pd.Timestamp]:
    """
    target_5d가 존재하는 날짜만 대상으로 주별 첫 거래일을 리밸런싱 날짜로 선택.
    마지막 구간 target_5d NA 문제 방지용.
    """
    test_dates = df[
        (df[DATE_COL] >= pd.Timestamp(start_date)) &
        (df["target_5d"].notna())
    ][[DATE_COL]].drop_duplicates()

    test_dates["week"] = test_dates[DATE_COL].dt.to_period("W")
    rebal_dates = test_dates.groupby("week")[DATE_COL].min().tolist()

    return [pd.Timestamp(d) for d in rebal_dates]


def evaluate_one_date(
    pred_df: pd.DataFrame,
    df: pd.DataFrame,
    date_str: str,
    top_k: int = 5,
):
    pred = pred_df.copy()
    pred[DATE_COL] = pd.to_datetime(pred[DATE_COL])
    pred[GROUP_COL] = pred[GROUP_COL].astype(str)

    actual = df[[DATE_COL, GROUP_COL, "target_5d"]].copy()
    actual[DATE_COL] = pd.to_datetime(actual[DATE_COL])
    actual[GROUP_COL] = actual[GROUP_COL].astype(str)

    merged = pred.merge(
        actual,
        on=[DATE_COL, GROUP_COL],
        how="left",
    )

    merged = merged.dropna(subset=["pred_q50", "target_5d"]).copy()

    if len(merged) < top_k * 2:
        print(f"  skip: too few valid assets ({len(merged)})")
        return None

    merged = merged.sort_values("pred_q50", ascending=False)

    top = merged.head(top_k)
    bottom = merged.tail(top_k)

    top_realized = top["target_5d"].mean()
    bottom_realized = bottom["target_5d"].mean()
    equal_realized = merged["target_5d"].mean()

    spearman_ic = merged[["pred_q50", "target_5d"]].corr(method="spearman").iloc[0, 1]

    return {
        "date": date_str,
        "n_assets": len(merged),
        "top_k": top_k,
        "top5_realized": top_realized,
        "equal_realized": equal_realized,
        "bottom5_realized": bottom_realized,
        "top_minus_equal": top_realized - equal_realized,
        "top_minus_bottom": top_realized - bottom_realized,
        "top_win_equal": int(top_realized > equal_realized),
        "spearman_ic": spearman_ic,
        "top_names": ", ".join(top["name"].astype(str).tolist()) if "name" in top.columns else "",
        "bottom_names": ", ".join(bottom["name"].astype(str).tolist()) if "name" in bottom.columns else "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--start", type=str, default=TEST_START_DATE)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt) if args.ckpt else find_latest_checkpoint(Path(CHECKPOINT_DIR))
    print(f"[ranking check] checkpoint: {ckpt_path}")

    model = load_cuda_safe_tft_checkpoint(ckpt_path)

    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=[DATE_COL])
    df[GROUP_COL] = df[GROUP_COL].astype(str)

    # Use the dataset parameters saved inside the checkpoint, not the current config.py.
    # This prevents feature-order / feature-count mismatches when old experiments are evaluated locally.
    dataset_params = model.hparams.dataset_parameters

    # TimeSeriesDataSet cannot be built with NA target values. The last few dates have
    # missing target_5d because future 5-day returns are not observable, so exclude them
    # only for the reference dataset construction. Keep the original df for evaluation.
    dataset_df = df.dropna(subset=["target_5d"]).copy()
    print(
        f"[ranking check] dataset rows after target_5d dropna: "
        f"{len(df)} -> {len(dataset_df)}"
    )

    train_ds = TimeSeriesDataSet.from_parameters(
        dataset_params,
        dataset_df,
        predict=False,
        stop_randomization=True,
    )

    print("[ranking check] ckpt x_reals:")
    for i, c in enumerate(model.hparams.x_reals):
        print(f"  {i}: {c}")

    rebal_dates = make_rebal_dates(df, args.start)

    print(f"[ranking check] 리밸런싱 날짜 수: {len(rebal_dates)}")
    print(
        f"[ranking check] date range: "
        f"{rebal_dates[0].date() if rebal_dates else '—'} ~ "
        f"{rebal_dates[-1].date() if rebal_dates else '—'}"
    )

    # 핵심: collect_rolling_predictions는 날짜 하나씩이 아니라 전체 날짜 리스트를 한 번에 받음
    preds_by_date = collect_rolling_predictions(
        model=model,
        df=df,
        train_ds=train_ds,
        rebal_dates=rebal_dates,
    )

    records = []

    for date_str, pred_df in preds_by_date.items():
        print(f"\n[ranking check] {date_str}")

        result = evaluate_one_date(
            pred_df=pred_df,
            df=df,
            date_str=date_str,
            top_k=TOP_K,
        )

        if result is None:
            continue

        records.append(result)

        print(
            f"  top5={result['top5_realized']:.4%}, "
            f"equal={result['equal_realized']:.4%}, "
            f"bottom5={result['bottom5_realized']:.4%}, "
            f"top-equal={result['top_minus_equal']:.4%}, "
            f"IC={result['spearman_ic']:.3f}"
        )

    ranking_df = pd.DataFrame(records)

    if ranking_df.empty:
        print("No valid ranking records.")
        return

    print("\n" + "=" * 70)
    print("Ranking Check Summary")
    print("=" * 70)

    summary = {
        "n_periods": len(ranking_df),
        "top5_realized_mean": ranking_df["top5_realized"].mean(),
        "equal_realized_mean": ranking_df["equal_realized"].mean(),
        "bottom5_realized_mean": ranking_df["bottom5_realized"].mean(),
        "top_minus_equal_mean": ranking_df["top_minus_equal"].mean(),
        "top_minus_bottom_mean": ranking_df["top_minus_bottom"].mean(),
        "top_win_equal_rate": ranking_df["top_win_equal"].mean(),
        "spearman_ic_mean": ranking_df["spearman_ic"].mean(),
        "spearman_ic_positive_rate": (ranking_df["spearman_ic"] > 0).mean(),
    }

    for k, v in summary.items():
        if k == "n_periods":
            print(f"{k}: {v}")
        elif "rate" in k:
            print(f"{k}: {v:.2%}")
        else:
            print(f"{k}: {v:.4%}")

    out_dir = PORTFOLIO_RESULT_DIR / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "ranking_check.csv"
    ranking_df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
