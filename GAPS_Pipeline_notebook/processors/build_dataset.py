"""
processors/build_dataset.py
새 데이터를 final_dataset.csv 뒤에 append
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from config import FINAL_DATASET, RAW_DIR, MERGED_DIR

SENT_CSV  = MERGED_DIR / "sentiment_daily.csv"
MACRO_CSV = RAW_DIR    / "macro.csv"
PRICE_CSV = RAW_DIR    / "price_raw.csv"

SENT_COLS  = ["domestic_mean","domestic_std","domestic_count",
              "global_mean","global_std","global_count"]
FINAL_COLS = ["date","ticker","name","end","AUM","target_5d","sector",
              "domestic_mean","domestic_std","domestic_count",
              "global_mean","global_std","global_count",
              "usd_krw","vix_close","us_10y","kr_10y","rate_spread","time_idx"]


def append_new_data() -> pd.DataFrame:
    print("[데이터셋] 기존 파일 로드...")
    if not FINAL_DATASET.exists():
        raise FileNotFoundError(f"final_dataset.csv 없음: {FINAL_DATASET}")

    existing  = pd.read_csv(FINAL_DATASET, parse_dates=["date"])
    last_date = existing["date"].max()
    print(f"  기존: {existing['date'].min().date()} ~ {last_date.date()} ({len(existing):,}행)")

    # ── 가격 로드 ──────────────────────────────────────────────
    price = pd.read_csv(PRICE_CSV, parse_dates=["date"])
    price = price[price["date"] > last_date].copy()
    price = price[price["date"].dt.weekday < 5]

    if price.empty:
        print("  신규 가격 데이터 없음")
        return existing

    print(f"  신규 가격: {price['date'].min().date()} ~ {price['date'].max().date()} ({len(price):,}행)")

    # ── 거시 병합 ──────────────────────────────────────────────
    macro = pd.read_csv(MACRO_CSV, parse_dates=["date"])
    all_dates = pd.DataFrame({"date": price["date"].unique()}).sort_values("date")
    macro_m   = all_dates.merge(macro, on="date", how="left")
    for col in ["vix_close","usd_krw","us_10y","kr_10y","rate_spread"]:
        if col in macro_m.columns:
            macro_m[col] = macro_m[col].ffill().bfill()

    price = price.merge(
        macro_m[["date","vix_close","usd_krw","us_10y","kr_10y","rate_spread"]],
        on="date", how="left"
    )

    # ── 감성 병합 ──────────────────────────────────────────────
    if SENT_CSV.exists():
        sent  = pd.read_csv(SENT_CSV)
        sent["date"] = sent["date"].astype(str)
        price["date_str"] = price["date"].dt.strftime("%Y-%m-%d")
        price = price.merge(
            sent[["date","sector"]+SENT_COLS].rename(columns={"date":"date_str"}),
            on=["date_str","sector"], how="left"
        ).drop(columns=["date_str"])
    for col in SENT_COLS:
        if col not in price.columns:
            price[col] = 0.0
    price[SENT_COLS] = price[SENT_COLS].fillna(0)

    # ── target_5d 계산 ─────────────────────────────────────────
    # 기존 + 신규 합쳐서 계산해야 5일 shift가 올바름
    combined_for_calc = pd.concat([
        existing[["ticker","date","end"]],
        price[["ticker","date","end"]].rename(columns={"ticker":"ticker"})
    ], ignore_index=True).sort_values(["ticker","date"])

    # ticker별 target_5d 재계산 (신규 구간만)
    new_targets = {}
    for ticker, g in combined_for_calc.groupby("ticker"):
        g = g.sort_values("date").copy()
        g["target_5d"] = g["end"].pct_change(5).shift(-5)
        # 신규 날짜의 target_5d만
        new_mask = g["date"] > last_date
        new_targets[ticker] = g[new_mask][["date","target_5d"]]

    price["target_5d"] = 0.0
    for ticker, df_t in new_targets.items():
        tmask = price["ticker"] == ticker
        price.loc[tmask, "target_5d"] = price.loc[tmask, "date"].map(
            df_t.set_index("date")["target_5d"]
        )

    # ── time_idx 이어서 부여 ───────────────────────────────────
    last_idx = existing.groupby("ticker")["time_idx"].max().to_dict()
    parts = []
    for ticker, g in price.groupby("ticker"):
        g = g.sort_values("date").copy()
        offset = last_idx.get(ticker, -1) + 1
        g["time_idx"] = range(offset, offset + len(g))
        parts.append(g)
    price = pd.concat(parts, ignore_index=True)

    # ── 컬럼 정리 & append ─────────────────────────────────────
    for col in FINAL_COLS:
        if col not in price.columns:
            price[col] = np.nan

    new_rows = price[FINAL_COLS]
    final    = pd.concat([existing, new_rows], ignore_index=True)
    final    = final.drop_duplicates(["date","ticker"], keep="last")
    final    = final.sort_values(["ticker","date"]).reset_index(drop=True)

    final.to_csv(FINAL_DATASET, index=False)
    print(f"  ✅ 갱신: {len(final):,}행 (+{len(final)-len(existing):,})")
    print(f"     기간: {final['date'].min().date()} ~ {final['date'].max().date()}")
    return final
