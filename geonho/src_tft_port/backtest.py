"""
backtest.py — 롤링 백테스트
실행: 
python geonho/src_tft_port/backtest.py \
  --ckpt geonho/src_tft_port/outputs/checkpoints/tft-no_end_macro_sentiment-epoch=06-val_loss=0.0125.ckpt

  매 주 (월요일) 포트폴리오를 리밸런싱하고,
  5일 후 실제 target_5d 기준 수익률을 기록.
  안정형 / 수익추구형 / 동일가중 벤치마크를 비교.
"""

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
    PROCESSED_DATA_PATH, CHECKPOINT_DIR, DATE_COL, GROUP_COL,
    SECTOR_COL, TEST_START_DATE, PORTFOLIO_RESULT_DIR,
    BATCH_SIZE, NUM_WORKERS,
)
from data.dataset import make_datasets
from portfolio.portfolio import build_signal, compute_weights


# ─────────────────────────────────────────────────────────────
#  롤링 예측 수집
# ─────────────────────────────────────────────────────────────

def collect_rolling_predictions(
    model:      TemporalFusionTransformer,
    df:         pd.DataFrame,
    train_ds,
    rebal_dates: list[pd.Timestamp],
) -> dict[str, pd.DataFrame]:
    """각 리밸런싱 날짜별로 inference 수행 → dict[date_str → pred_df]"""
    results = {}
    for rd in rebal_dates:
        inf_df = df[df[DATE_COL] <= rd].copy()
        if inf_df.empty:
            continue
        try:
            inf_ds = TimeSeriesDataSet.from_dataset(
                train_ds, inf_df,
                predict=True, stop_randomization=True,
            )
            inf_loader = inf_ds.to_dataloader(
                train=False, batch_size=BATCH_SIZE * 2, num_workers=NUM_WORKERS,
            )
            with torch.no_grad():
                raw = model.predict(inf_loader, mode="quantiles",
                                    return_index=True, return_x=False)
            preds  = raw.output[:, 0, :].cpu().numpy()
            index  = raw.index.copy()
            index["pred_q10"] = preds[:, 0]
            index["pred_q50"] = preds[:, 1]
            index["pred_q90"] = preds[:, 2]

            # quantile crossing 방어
            q = index[["pred_q10", "pred_q50", "pred_q90"]].values
            q_sorted = np.sort(q, axis=1)
            index[["pred_q10", "pred_q50", "pred_q90"]] = q_sorted

            meta = (df[[GROUP_COL, "name", SECTOR_COL]]
                    .drop_duplicates(GROUP_COL)
                    .assign(**{GROUP_COL: lambda x: x[GROUP_COL].astype(str)}))
            index[GROUP_COL] = index[GROUP_COL].astype(str)
            index = index.merge(meta, on=GROUP_COL, how="left")
            index["date"] = rd.strftime("%Y-%m-%d")

            results[rd.strftime("%Y-%m-%d")] = index
            print(f"  [inference] {rd.date()} — {len(index)} tickers")
        except Exception as e:
            print(f"  [skip] {rd.date()}: {e}")

    return results


# ─────────────────────────────────────────────────────────────
#  백테스트 루프
# ─────────────────────────────────────────────────────────────

def run_backtest(
    ckpt_path:  Path,
    start_date: str = TEST_START_DATE,
) -> pd.DataFrame:

    print(f"[backtest] checkpoint: {ckpt_path}")

    # ── torchmetrics.Metric._apply 패치 ───────────────────────────────────────
    # predict.py 와 동일한 이유로 패치 적용 (상세 설명은 predict.py 참고)
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

    # 전처리 완료 데이터 로드 (inference 슬라이스 및 실제 수익률 조회용)
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=[DATE_COL])
    df[GROUP_COL] = df[GROUP_COL].astype(str)

    # 리밸런싱 날짜: test 구간 내 주 첫 거래일
    test_dates = df[df[DATE_COL] >= pd.Timestamp(start_date)][[DATE_COL]].drop_duplicates()
    test_dates["week"] = test_dates[DATE_COL].dt.to_period("W")
    rebal_dates = test_dates.groupby("week")[DATE_COL].min().tolist()

    print(f"[backtest] 리밸런싱 횟수: {len(rebal_dates)}회  "
          f"({rebal_dates[0].date() if rebal_dates else '—'} ~ "
          f"{rebal_dates[-1].date() if rebal_dates else '—'})")

    # ── 예측 수집 ──
    preds_by_date = collect_rolling_predictions(model, df, train_ds, rebal_dates)

    # ── 실제 수익률 lookup ──
    actual = df[[DATE_COL, GROUP_COL, "target_5d"]].copy()
    actual[GROUP_COL] = actual[GROUP_COL].astype(str)

    records = []

    for date_str, pred_df in preds_by_date.items():
        signal_df = build_signal(pred_df)

        for strategy in ("conservative", "aggressive"):
            w_df = compute_weights(signal_df, strategy=strategy)
            if w_df.empty:
                continue

            # 해당 날짜의 실제 target_5d 조인
            act_day = actual[actual[DATE_COL] == pd.Timestamp(date_str)]
            w_df = w_df.merge(
                act_day[[GROUP_COL, "target_5d"]],
                on=GROUP_COL, how="left",
            )
            realized = (w_df["weight"] * w_df["target_5d"].fillna(0)).sum()

            records.append({
                "date":            date_str,
                "strategy":        strategy,
                "n_holdings":      len(w_df),
                "expected_return": (w_df["weight"] * w_df["signal_return"]).sum(),
                "realized_return": realized,
            })

        # 동일가중 벤치마크
        all_tickers = pred_df[GROUP_COL].unique()
        act_day = actual[actual[DATE_COL] == pd.Timestamp(date_str)]
        bm_df   = act_day[act_day[GROUP_COL].isin(all_tickers)]
        bm_ret  = bm_df["target_5d"].mean() if not bm_df.empty else 0.0
        records.append({
            "date":            date_str,
            "strategy":        "equal_weight",
            "n_holdings":      len(all_tickers),
            "expected_return": np.nan,
            "realized_return": bm_ret,
        })

    result_df = pd.DataFrame(records)
    return result_df


# ─────────────────────────────────────────────────────────────
#  성과 분석
# ─────────────────────────────────────────────────────────────

def analyze_performance(result_df: pd.DataFrame) -> pd.DataFrame:
    """전략별 누적 수익률, Sharpe, MDD 계산."""
    summary_rows = []

    for strategy, grp in result_df.groupby("strategy"):
        rets = grp.set_index("date")["realized_return"].sort_index()

        cum_ret   = (1 + rets).prod() - 1
        ann_ret   = (1 + cum_ret) ** (52 / len(rets)) - 1   # 주간 → 연환산
        ann_vol   = rets.std() * np.sqrt(52)
        sharpe    = ann_ret / ann_vol if ann_vol > 0 else 0.0
        cum_curve = (1 + rets).cumprod()
        mdd       = (cum_curve / cum_curve.cummax() - 1).min()
        hit_rate  = (rets > 0).mean()

        summary_rows.append({
            "strategy":       strategy,
            "cum_return_%":   round(cum_ret  * 100, 2),
            "ann_return_%":   round(ann_ret  * 100, 2),
            "ann_vol_%":      round(ann_vol  * 100, 2),
            "sharpe":         round(sharpe, 3),
            "MDD_%":          round(mdd    * 100, 2),
            "hit_rate_%":     round(hit_rate * 100, 1),
            "n_periods":      len(rets),
        })

    summary = pd.DataFrame(summary_rows).sort_values("sharpe", ascending=False)
    print("\n" + "=" * 65)
    print("백테스트 성과 요약")
    print(summary.to_string(index=False))
    return summary


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def find_latest_checkpoint(ckpt_dir: Path) -> Path:
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_dir}")
    return ckpts[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",  type=str, default=None)
    p.add_argument("--start", type=str, default=TEST_START_DATE)
    args = p.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else find_latest_checkpoint(Path(CHECKPOINT_DIR))

    result_df = run_backtest(ckpt, start_date=args.start)

    out_dir = PORTFOLIO_RESULT_DIR / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_dir / "backtest_returns.csv", index=False)

    summary = analyze_performance(result_df)
    summary.to_csv(out_dir / "backtest_summary.csv", index=False)

    print(f"\n[save] → {out_dir}")


if __name__ == "__main__":
    main()
