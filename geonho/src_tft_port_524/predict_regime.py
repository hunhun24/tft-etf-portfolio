"""
predict_regime.py — TFT 추론 + regime-switch 포트폴리오 구성

실행:
python geonho/src_tft_port_524/predict_regime.py \
    --ckpt geonho/src_tft_port_524/outputs/checkpoints/tft-no_end_macro_sentiment-epoch=06-val_loss=0.0125.ckpt \
    --date 2025-12-30

checkpoint dataset_parameters 기준으로 TimeSeriesDataSet 구성 (ranking_check.py 방식).
레짐 판단 후 해당 전략으로 포트폴리오 구성.
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
    SECTOR_COL, BATCH_SIZE, NUM_WORKERS, PORTFOLIO_RESULT_DIR,
)
from backtest import find_latest_checkpoint
from portfolio.portfolio import (
    build_signal, compute_weights, portfolio_summary, save_results,
)
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
#  추론
# ─────────────────────────────────────────────────────────────

def run_inference(
    model: TemporalFusionTransformer,
    df: pd.DataFrame,
    pred_date: str,
) -> pd.DataFrame:
    # checkpoint dataset_parameters 기준으로 reference dataset 구성
    dataset_params = model.hparams.dataset_parameters
    dataset_df = df.dropna(subset=["target_5d"]).copy()
    print(
        f"[inference] dataset rows after target_5d dropna: "
        f"{len(df)} -> {len(dataset_df)}"
    )

    train_ds = TimeSeriesDataSet.from_parameters(
        dataset_params, dataset_df, predict=False, stop_randomization=True,
    )

    pred_date_ts = pd.Timestamp(pred_date)
    valid_tickers = df.loc[df[DATE_COL] == pred_date_ts, GROUP_COL].astype(str).unique()
    missing = set(df[GROUP_COL].astype(str).unique()) - set(valid_tickers)
    if missing:
        print(f"[WARN] pred_date 데이터 없는 ticker {len(missing)}개 제외")

    inf_df = df[df[DATE_COL] <= pred_date_ts].copy()
    inf_df[GROUP_COL] = inf_df[GROUP_COL].astype(str)
    inf_df = inf_df[inf_df[GROUP_COL].isin(valid_tickers)].copy()
    inf_df = inf_df.dropna(subset=["target_5d"]).copy()

    print(f"[inference] pred_date={pred_date_ts.date()}, valid tickers={len(valid_tickers)}")

    inf_ds = TimeSeriesDataSet.from_dataset(
        train_ds, inf_df, predict=True, stop_randomization=True,
    )
    inf_loader = inf_ds.to_dataloader(
        train=False, batch_size=BATCH_SIZE * 2, num_workers=NUM_WORKERS,
    )

    with torch.no_grad():
        raw = model.predict(
            inf_loader, mode="quantiles", return_index=True, return_x=False,
        )

    preds_np = raw.output[:, 0, :].cpu().numpy()
    pred_df = raw.index.copy()
    pred_df["pred_q10"] = preds_np[:, 0]
    pred_df["pred_q50"] = preds_np[:, 1]
    pred_df["pred_q90"] = preds_np[:, 2]

    meta = (
        df[[GROUP_COL, "name", SECTOR_COL]]
        .drop_duplicates(GROUP_COL)
        .assign(**{GROUP_COL: lambda x: x[GROUP_COL].astype(str)})
    )
    pred_df[GROUP_COL] = pred_df[GROUP_COL].astype(str)
    pred_df = pred_df.merge(meta, on=GROUP_COL, how="left")
    pred_df["date"] = pred_date

    q = pred_df[["pred_q10", "pred_q50", "pred_q90"]].values
    pred_df[["pred_q10", "pred_q50", "pred_q90"]] = np.sort(q, axis=1)

    print(pred_df[["pred_q10", "pred_q50", "pred_q90"]].describe())
    print(f"[inference] 예측 완료: {len(pred_df)}건 / valid tickers: {len(valid_tickers)}개")
    return pred_df


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",                type=str,   default=None)
    p.add_argument("--date",                type=str,   default=None)
    p.add_argument("--sentiment_threshold", type=float, default=-0.5)
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt) if args.ckpt else find_latest_checkpoint(Path(CHECKPOINT_DIR))

    # ── 검증 로그 ──
    print(f"[predict_regime] checkpoint: {ckpt_path}")
    print(f"[predict_regime] sentiment_threshold: {args.sentiment_threshold}")

    model = load_cuda_safe_tft_checkpoint(ckpt_path)

    print("[predict_regime] ckpt x_reals:")
    for i, col in enumerate(model.hparams.x_reals):
        print(f"  {i}: {col}")

    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=[DATE_COL])
    df[GROUP_COL] = df[GROUP_COL].astype(str)

    pred_date = args.date or df[DATE_COL].max().strftime("%Y-%m-%d")
    print(f"[predict_regime] 예측 기준일: {pred_date}")

    pred_df   = run_inference(model, df, pred_date)
    signal_df = build_signal(pred_df)

    regime, reason = determine_regime(
        signal_df, df, pred_date,
        sentiment_threshold=args.sentiment_threshold,
    )
    print(f"\n[regime] 판단: {regime}  (이유: {reason})")

    # 레짐 전략 포트폴리오
    w_df = compute_weights(signal_df, strategy=regime)
    portfolio_summary(w_df, regime)

    # 비교용 두 전략도 저장
    cons_df = compute_weights(signal_df, strategy="conservative")
    agg_df  = compute_weights(signal_df, strategy="aggressive")
    save_results(cons_df, agg_df, pred_date)

    out = PORTFOLIO_RESULT_DIR / pred_date
    out.mkdir(parents=True, exist_ok=True)
    w_df.to_csv(out / f"portfolio_regime_{regime}.csv", index=False)
    pd.DataFrame([{"pred_date": pred_date, "regime": regime, "reason": reason}]).to_csv(
        out / "regime_log.csv", index=False,
    )

    print(f"\n[save] → {out}")
    print(f"  portfolio_regime_{regime}.csv")
    print(f"  regime_log.csv")
    print(f"  portfolio_conservative.csv")
    print(f"  portfolio_aggressive.csv")


if __name__ == "__main__":
    main()
