"""
collectors/collect_macro.py
가격 데이터 + 거시지표 수집 (주간 증분)
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pykrx import stock as krx

from config import (
    RAW_DIR, FINAL_DATASET, BOK_API_KEY, BOK_STAT, BOK_ITEM,
    YF_TICKERS,
)

MACRO_CSV = RAW_DIR / "macro.csv"
PRICE_CSV = RAW_DIR / "price_raw.csv"


# ── 가격 수집 ─────────────────────────────────────────────────
def fetch_price(code: str, start: str, end: str) -> pd.DataFrame:
    try:
        df = krx.get_market_ohlcv_by_date(start, end, code)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = df.columns.str.strip()
        df = df.rename(columns={"날짜":"date","종가":"end"})
        df["code"] = code
        return df[["date","code","end"]].copy()
    except Exception as e:
        print(f"    [SKIP] {code}: {e}")
        return pd.DataFrame()


def collect_prices(etf_list: pd.DataFrame, incremental=True) -> pd.DataFrame:
    today = datetime.today().strftime("%Y%m%d")

    if incremental and PRICE_CSV.exists():
        existing = pd.read_csv(PRICE_CSV, parse_dates=["date"])
        last     = existing["date"].max()
        start    = (last - timedelta(days=10)).strftime("%Y%m%d")
        print(f"[가격] 증분: {start} ~ {today}")
    else:
        start    = "20190101"
        existing = pd.DataFrame()
        print(f"[가격] 전체: 20190101 ~ {today}")

    tickers = etf_list["code"].tolist()
    batches = []
    for i, code in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {code}", end="\r")
        df = fetch_price(code, start, today)
        if not df.empty:
            batches.append(df)
        time.sleep(0.3)
    print()

    if not batches:
        return existing

    new = pd.concat(batches, ignore_index=True)
    new["date"] = pd.to_datetime(new["date"])
    new = new[new["date"].dt.weekday < 5]

    meta = etf_list[["code","ticker","name","sector","aum"]].rename(columns={"aum":"AUM"})
    new  = new.merge(meta, on="code", how="left")

    combined = pd.concat([existing, new]).drop_duplicates(["date","code"], keep="last")
    combined = combined.sort_values(["code","date"]).reset_index(drop=True)
    combined.to_csv(PRICE_CSV, index=False)
    print(f"[가격] 저장: {combined['code'].nunique()}종목, {len(combined):,}행")
    return combined


# ── 거시 수집 ─────────────────────────────────────────────────
def _fetch_bok_kr10y(start: str, end: str) -> pd.Series:
    if not BOK_API_KEY or BOK_API_KEY == "YOUR_BOK_API_KEY":
        return pd.Series(dtype=float)
    try:
        url = (
            f"https://ecos.bok.or.kr/api/StatisticSearch"
            f"/{BOK_API_KEY}/json/kr/1/9999"
            f"/{BOK_STAT}/D/{start.replace('-','')}/{end.replace('-','')}/{BOK_ITEM}"
        )
        rows = requests.get(url, timeout=15).json()["StatisticSearch"]["row"]
        df   = pd.DataFrame(rows)[["TIME","DATA_VALUE"]]
        df.columns  = ["date","kr_10y"]
        df["date"]  = pd.to_datetime(df["date"], format="%Y%m%d")
        df["kr_10y"] = pd.to_numeric(df["kr_10y"], errors="coerce")
        return df.set_index("date")["kr_10y"]
    except:
        return pd.Series(dtype=float)


def collect_macro(incremental=True) -> pd.DataFrame:
    today = datetime.today()

    if incremental and MACRO_CSV.exists():
        existing = pd.read_csv(MACRO_CSV, parse_dates=["date"])
        last     = existing["date"].max()
        start    = (last - timedelta(days=10)).strftime("%Y-%m-%d")
        print(f"[거시] 증분: {start} ~ {today.strftime('%Y-%m-%d')}")
    else:
        start    = "2019-01-01"
        existing = pd.DataFrame()
        print(f"[거시] 전체: 2019-01-01 ~ {today.strftime('%Y-%m-%d')}")

    end = today.strftime("%Y-%m-%d")

    # Yahoo Finance
    tickers = list(YF_TICKERS.values())
    raw     = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    macro   = raw["Close"].rename(columns={v:k for k,v in YF_TICKERS.items()})
    macro.index.name = "date"
    macro   = macro.reset_index()
    macro["date"] = pd.to_datetime(macro["date"])

    # 한국 10년물
    kr = _fetch_bok_kr10y(start, end)
    if not kr.empty:
        kr_df = kr.reset_index()
        kr_df.columns = ["date","kr_10y"]
        macro = macro.merge(kr_df, on="date", how="left")
        print("  한국 10Y: BOK ECOS ✅")
    else:
        macro["kr_10y"] = np.nan
        print("  ⚠️  한국 10Y NaN")

    # 영업일 뼈대 + ffill
    base  = pd.DataFrame({"date": pd.date_range(start, end, freq="B")})
    macro = base.merge(macro, on="date", how="left")
    for col in ["vix_close","usd_krw","us_10y","kr_10y"]:
        if col in macro.columns:
            macro[col] = macro[col].ffill().bfill()

    macro["rate_spread"] = macro["kr_10y"] - macro["us_10y"]

    if not existing.empty:
        combined = pd.concat([existing, macro]).drop_duplicates("date", keep="last")
        combined = combined.sort_values("date").reset_index(drop=True)
    else:
        combined = macro

    combined.to_csv(MACRO_CSV, index=False)
    r = combined.iloc[-1]
    print(f"  최신 ({pd.Timestamp(r['date']).date()}):"
          f" VIX={r.get('vix_close',0):.2f}"
          f" KRW={r.get('usd_krw',0):.1f}"
          f" US10Y={r.get('us_10y',0):.3f}%"
          f" KR10Y={r.get('kr_10y',0):.3f}%"
          f" spread={r.get('rate_spread',0):.3f}%p")
    return combined
