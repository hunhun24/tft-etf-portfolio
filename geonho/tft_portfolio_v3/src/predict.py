"""
predict.py — TFT 추론 + 포트폴리오 구성
실행: python src/predict.py [--ckpt PATH] [--date YYYY-MM-DD]

  --ckpt  : 체크포인트 경로 (생략 시 outputs/checkpoints/ 최신 파일 자동 검색)
  --date  : 예측 기준일 (생략 시 데이터 최신 날짜 사용)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer

from config import (
    PREPROCESSED_CSV, CHECKPOINT_DIR, DATE_COL, GROUP_COL,
    SECTOR_COL, TIME_IDX_COL, ENCODER_LENGTH, BATCH_SIZE, NUM_WORKERS,
    QUANTILES,
)
from data.dataset    import load_preprocessed, build_datasets
from portfolio.portfolio import (
    build_signal, compute_weights,
    portfolio_summary, save_results,
)


# ─────────────────────────────────────────────────────────────
#  Checkpoint 탐색
# ─────────────────────────────────────────────────────────────

def find_latest_checkpoint(ckpt_dir: Path) -> Path:
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"체크포인트를 찾을 수 없음: {ckpt_dir}")
    return ckpts[-1]


# ─────────────────────────────────────────────────────────────
#  추론
# ─────────────────────────────────────────────────────────────

def run_inference(
    ckpt_path: Path,
    pred_date: str | None = None,
) -> pd.DataFrame:
    """
    Returns:
        pred_df: ticker별 분위수 예측값 DataFrame
          columns: date, ticker, name, sector,
                   pred_q10, pred_q50, pred_q90
    """
    print(f"[load] checkpoint: {ckpt_path}")
    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_path))
    model.eval()

    df = load_preprocessed()

    # 예측 기준일: 지정 없으면 최신 날짜
    if pred_date is None:
        pred_date = df[DATE_COL].max().strftime("%Y-%m-%d")
    print(f"[inference] 예측 기준일: {pred_date}")

    # 예측 기준일 포함 ENCODER_LENGTH일 데이터 슬라이스
    # (TFT는 train_dataset 기준으로 inference용 DataLoader를 따로 만들어야 함)
    train_ds, _ = build_datasets(df)

    # ── inference dataset: pred_date 전후 ──
    inf_df = df[df[DATE_COL] <= pd.Timestamp(pred_date)].copy()
    from pytorch_forecasting import TimeSeriesDataSet

    inf_dataset = TimeSeriesDataSet.from_dataset(
        train_ds,
        inf_df,
        predict            = True,
        stop_randomization = True,
    )
    inf_loader = inf_dataset.to_dataloader(
        train=False,
        batch_size  = BATCH_SIZE * 2,
        num_workers = NUM_WORKERS,
    )

    # ── 예측 수행 ──
    with torch.no_grad():
        raw_preds = model.predict(
            inf_loader,
            mode          = "quantiles",   # (n_samples, n_quantiles)
            return_index  = True,
            return_x      = False,
        )

    preds, index = raw_preds.output, raw_preds.index

    # preds shape: (n_samples, max_pred_len=1, n_quantiles=3)
    preds_np = preds[:, 0, :].cpu().numpy()    # → (n_samples, 3)

    pred_df = index.copy()
    pred_df["pred_q10"] = preds_np[:, 0]
    pred_df["pred_q50"] = preds_np[:, 1]
    pred_df["pred_q90"] = preds_np[:, 2]

    # ticker별 메타 정보 조인
    meta = (df[[GROUP_COL, "name", SECTOR_COL]]
            .drop_duplicates(subset=[GROUP_COL])
            .assign(**{GROUP_COL: lambda x: x[GROUP_COL].astype(str)}))
    pred_df[GROUP_COL] = pred_df[GROUP_COL].astype(str)
    pred_df = pred_df.merge(meta, on=GROUP_COL, how="left")
    pred_df["date"] = pred_date

    print(f"[inference] 예측 완료: {len(pred_df)} 건")
    print(pred_df[["name", "pred_q10", "pred_q50", "pred_q90"]].head(10).to_string())
    return pred_df


# ─────────────────────────────────────────────────────────────
#  포트폴리오 구성
# ─────────────────────────────────────────────────────────────

def build_portfolio(pred_df: pd.DataFrame, pred_date: str) -> dict:
    """
    예측값으로 안정형 / 수익추구형 포트폴리오 구성.
    """
    signal_df = build_signal(pred_df)

    cons_w = compute_weights(signal_df, strategy="conservative")
    agg_w  = compute_weights(signal_df, strategy="aggressive")

    cons_summary = portfolio_summary(cons_w, "conservative")
    agg_summary  = portfolio_summary(agg_w,  "aggressive")

    save_results(cons_w, agg_w, pred_date)

    return {
        "conservative": {"weights": cons_w, "summary": cons_summary},
        "aggressive":   {"weights": agg_w,  "summary": agg_summary},
    }


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TFT Inference + Portfolio")
    p.add_argument("--ckpt", type=str, default=None,
                   help="체크포인트 경로 (생략 시 자동 탐색)")
    p.add_argument("--date", type=str, default=None,
                   help="예측 기준일 YYYY-MM-DD (생략 시 데이터 최신)")
    return p.parse_args()


def main():
    args = parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else find_latest_checkpoint(CHECKPOINT_DIR)
    pred_df   = run_inference(ckpt, pred_date=args.date)
    portfolio = build_portfolio(pred_df, pred_date=args.date or pred_df["date"].iloc[0])

    print("\n" + "=" * 60)
    print("포트폴리오 구성 완료")
    print(f"  outputs/portfolio/{args.date or pred_df['date'].iloc[0]}/")
    print("  ├── portfolio_conservative.csv")
    print("  ├── portfolio_aggressive.csv")
    print("  └── portfolio_comparison.csv")
    return portfolio


if __name__ == "__main__":
    main()
