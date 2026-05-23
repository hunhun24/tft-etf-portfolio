"""
backtest.py — 롤링 백테스트
실행: python src/backtest.py [--ckpt PATH] [--start YYYY-MM-DD]

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
from pathlib import Path
from pytorch_forecasting import TemporalFusionTransformer

from config import (
    PREPROCESSED_CSV, CHECKPOINT_DIR, DATE_COL, GROUP_COL,
    SECTOR_COL, TEST_START_DATE, PORTFOLIO_RESULT_DIR,
    BATCH_SIZE, NUM_WORKERS,
)
from data.dataset    import load_preprocessed, build_datasets
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
    """
    각 리밸런싱 날짜별로 inference 수행 → dict[date_str → pred_df]
    """
    from pytorch_forecasting import TimeSeriesDataSet

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
    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_path))
    model.eval()

    df        = load_preprocessed()
    train_ds, _ = build_datasets(df)

    # 리밸런싱 날짜: test 구간 내 월요일 (주 1회)
    test_dates  = df[df[DATE_COL] >= pd.Timestamp(start_date)][DATE_COL].sort_values().unique()
    # 월요일(weekday=0)만, 또는 해당 주 첫 거래일
    rebal_dates = pd.to_datetime(test_dates)
    rebal_dates = [d for d in rebal_dates if d.weekday() == 0]   # 월요일

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
            "expected_return": bm_ret,
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

    ckpt = Path(args.ckpt) if args.ckpt else find_latest_checkpoint(CHECKPOINT_DIR)

    result_df = run_backtest(ckpt, start_date=args.start)

    out_dir = PORTFOLIO_RESULT_DIR / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_dir / "backtest_returns.csv", index=False)

    summary = analyze_performance(result_df)
    summary.to_csv(out_dir / "backtest_summary.csv", index=False)

    print(f"\n[save] → {out_dir}")


if __name__ == "__main__":
    main()
