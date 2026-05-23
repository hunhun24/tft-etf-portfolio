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
import numpy as np
import pandas as pd
import torch
import torchmetrics
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from config import (
    PROCESSED_DATA_PATH, CHECKPOINT_DIR, DATE_COL, GROUP_COL,
    SECTOR_COL, BATCH_SIZE, NUM_WORKERS,
)
from data.dataset import make_datasets
from portfolio.portfolio import (
    build_signal, compute_weights,
    portfolio_summary, save_results,
)

QUANTILES = [0.1, 0.5, 0.9]


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

    # ── torchmetrics.Metric._apply 패치 ───────────────────────────────────────
    # 원인: Metric._apply 내부에서 fn(torch.zeros(1, device=self.device)) 호출 시
    #        self._device 가 "cuda"인 채로 남아 있으면 CUDA 미설치 환경에서 터짐.
    #        self._device 는 tensor/buffer 가 아닌 Python attribute 이므로
    #        map_location="cpu" 나 load_state_dict 로도 변경되지 않음.
    # 해결: load 전후로만 _apply 를 패치해 _device 가 "cuda" 이면 강제로 "cpu" 초기화.
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
        torchmetrics.Metric._apply = _orig_metric_apply  # 패치 복원
    # ─────────────────────────────────────────────────────────────────────────

    # src 방식: make_datasets()로 train_ds 획득
    train_ds, _, _, _ = make_datasets()

    # 전처리 완료 데이터 로드 (inference 슬라이스용)
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=[DATE_COL])
    df[GROUP_COL] = df[GROUP_COL].astype(str)

    # 예측 기준일: 지정 없으면 최신 날짜
    if pred_date is None:
        pred_date = df[DATE_COL].max().strftime("%Y-%m-%d")
    print(f"[inference] 예측 기준일: {pred_date}")

    # inference dataset: pred_date 당일 데이터가 있는 ticker만 포함
    pred_date_ts = pd.Timestamp(pred_date)
    valid_tickers = (
        df.loc[df[DATE_COL] == pred_date_ts, GROUP_COL]
          .astype(str)
          .unique()
    )
    all_tickers = df[GROUP_COL].astype(str).unique()
    missing = set(all_tickers) - set(valid_tickers)
    if missing:
        print(f"[WARN] pred_date 데이터 없는 ticker {len(missing)}개 제외")

    inf_df = df[df[DATE_COL] <= pred_date_ts].copy()
    inf_df[GROUP_COL] = inf_df[GROUP_COL].astype(str)
    inf_df = inf_df[inf_df[GROUP_COL].isin(valid_tickers)].copy()
    inf_df = inf_df.dropna(subset=["target_5d"]).copy()

    print(f"[inference] pred_date={pred_date_ts.date()}, valid tickers={len(valid_tickers)}")

    inf_dataset = TimeSeriesDataSet.from_dataset(
        train_ds, inf_df,
        predict=True, stop_randomization=True,
    )
    inf_loader = inf_dataset.to_dataloader(
        train=False, batch_size=BATCH_SIZE * 2, num_workers=NUM_WORKERS,
    )

    # ── 예측 수행 ──
    with torch.no_grad():
        raw_preds = model.predict(
            inf_loader,
            mode="quantiles",
            return_index=True,
            return_x=False,
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

    # quantile crossing 방어
    q = pred_df[["pred_q10", "pred_q50", "pred_q90"]].values
    q_sorted = np.sort(q, axis=1)
    pred_df[["pred_q10", "pred_q50", "pred_q90"]] = q_sorted

    print(pred_df[["pred_q10", "pred_q50", "pred_q90"]].describe())
    print(f"[inference] 예측 완료: {len(pred_df)}건 / valid tickers: {len(valid_tickers)}개")
    print(pred_df[["name", "pred_q10", "pred_q50", "pred_q90"]].head(10).to_string())
    return pred_df


# ─────────────────────────────────────────────────────────────
#  포트폴리오 구성
# ─────────────────────────────────────────────────────────────

def build_portfolio(pred_df: pd.DataFrame, pred_date: str) -> dict:
    """예측값으로 안정형 / 수익추구형 포트폴리오 구성."""
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

    ckpt = Path(args.ckpt) if args.ckpt else find_latest_checkpoint(Path(CHECKPOINT_DIR))
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
