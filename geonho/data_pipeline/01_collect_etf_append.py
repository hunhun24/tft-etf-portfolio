"""
2026 ETF append price/AUM pipeline.

기존 geonho/ ETF 수집 파일은 수정하지 않고, 기존 산출물의 ticker universe와
메타데이터를 기준으로 append 기간의 ETF 가격 데이터를 수집한다.

Example:
python data_pipeline/01_collect_etf_append.py \
  --start-date 2026-01-01 \
  --end-date 2026-05-24
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pykrx import stock

warnings.filterwarnings("ignore")


GEONHO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = GEONHO_ROOT / "etf_data"
OUTPUT_DIR = GEONHO_ROOT / "data_pipeline" / "outputs" / "etf"

REQUEST_SLEEP = 0.3

OUTPUT_COLUMNS = [
    "Date",
    "티커",
    "ETF명",
    "구분1",
    "구분2",
    "sector_main",
    "sector_sub",
    "기초지수",
    "AUM(억원)",
    "종가",
    "일별수익률",
]

META_COLUMNS = [
    "티커",
    "ETF명",
    "구분1",
    "구분2",
    "sector_main",
    "sector_sub",
    "기초지수",
    "AUM(억원)",
]


@dataclass(frozen=True)
class CollectionSpec:
    name: str
    source_file: Path
    output_file: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect append ETF price data into data_pipeline/outputs/etf."
    )
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--request-sleep", type=float, default=REQUEST_SLEEP)
    return parser.parse_args()


def ymd(date_text: str) -> str:
    return pd.Timestamp(date_text).strftime("%Y%m%d")


def load_existing_universe(source_file: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    기존 geonho/etf_data 산출물에서 ticker universe와 마지막 메타데이터를 읽는다.

    AUM(억원)은 기존 수집 스크립트에서도 엑셀 universe의 정적 값이었으므로,
    append 파일에서도 동일 의미를 유지하기 위해 기존 산출물의 최신 메타값을 사용한다.
    """
    if not source_file.exists():
        raise FileNotFoundError(f"기존 ETF 산출물을 찾을 수 없음: {source_file}")

    df = pd.read_csv(source_file, encoding="utf-8-sig")
    missing = [col for col in OUTPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{source_file.name} 컬럼 누락: {missing}")

    df["Date"] = pd.to_datetime(df["Date"])
    df["티커"] = df["티커"].astype(str)
    df = df.sort_values(["티커", "Date"]).reset_index(drop=True)

    meta = df.groupby("티커", as_index=False).tail(1)[META_COLUMNS].copy()
    meta = meta.sort_values("티커").reset_index(drop=True)
    previous_close = (
        df.groupby("티커", as_index=True).tail(1).set_index("티커")["종가"].to_dict()
    )

    return meta, previous_close


def standardize_price_frame(
    df_ohlcv: pd.DataFrame,
    meta_row: pd.Series,
    previous_close: float | None = None,
) -> pd.DataFrame:
    """pykrx OHLCV 결과를 기존 ETF 산출물과 같은 스키마로 변환한다."""
    df_ohlcv = df_ohlcv.reset_index().copy()
    df_ohlcv.columns = df_ohlcv.columns.str.strip()

    date_candidates = [
        c for c in df_ohlcv.columns
        if "날짜" in c or "Date" in c or "date" in c
    ]
    if date_candidates:
        df_ohlcv = df_ohlcv.rename(columns={date_candidates[0]: "Date"})
    else:
        df_ohlcv = df_ohlcv.rename(columns={df_ohlcv.columns[0]: "Date"})

    close_candidates = [
        c for c in df_ohlcv.columns
        if "종가" in c or "Close" in c or "close" in c
    ]
    if not close_candidates:
        raise ValueError("종가 컬럼을 찾지 못했습니다.")

    close_col = close_candidates[0]
    df_out = df_ohlcv[["Date", close_col]].copy()
    df_out = df_out.rename(columns={close_col: "종가"})

    df_out["Date"] = pd.to_datetime(df_out["Date"]).dt.normalize()
    for col in META_COLUMNS:
        df_out[col] = meta_row[col]

    df_out = df_out.sort_values("Date").reset_index(drop=True)
    close_for_return = df_out["종가"].copy()
    if previous_close is not None and len(close_for_return) > 0:
        close_for_return = pd.concat(
            [pd.Series([previous_close]), close_for_return],
            ignore_index=True,
        )
        df_out["일별수익률"] = close_for_return.pct_change().iloc[1:].to_numpy()
    else:
        df_out["일별수익률"] = close_for_return.pct_change()

    return df_out[OUTPUT_COLUMNS]


def collect_one_group(
    spec: CollectionSpec,
    start_date: str,
    end_date: str,
    request_sleep: float,
) -> pd.DataFrame:
    meta_df, previous_close = load_existing_universe(spec.source_file)

    print("\n" + "=" * 70)
    print(f"[{spec.name}] ETF append 수집")
    print(f"  source : {spec.source_file}")
    print(f"  output : {spec.output_file}")
    print(f"  period : {start_date} ~ {end_date}")
    print(f"  tickers: {len(meta_df)}")
    print("=" * 70)

    collected = []
    start_ymd = ymd(start_date)
    end_ymd = ymd(end_date)

    for idx, row in meta_df.iterrows():
        ticker = str(row["티커"])
        ticker_krx = ticker.replace("A", "", 1)
        name = row["ETF명"]

        try:
            df_ohlcv = stock.get_market_ohlcv_by_date(start_ymd, end_ymd, ticker_krx)
            if df_ohlcv.empty:
                print(f"  [WARN] [{idx+1}/{len(meta_df)}] {ticker} {name}: 데이터 없음")
                continue

            df_out = standardize_price_frame(
                df_ohlcv=df_ohlcv,
                meta_row=row,
                previous_close=previous_close.get(ticker),
            )
            collected.append(df_out)
            print(f"  [OK]   [{idx+1}/{len(meta_df)}] {ticker} {name}: {len(df_out)} rows")
            time.sleep(request_sleep)

        except Exception as e:
            print(f"  [WARN] [{idx+1}/{len(meta_df)}] {ticker} {name}: {e}")
            continue

    if not collected:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df_all = pd.concat(collected, ignore_index=True)
    df_all = df_all.sort_values(["티커", "Date"]).reset_index(drop=True)
    return df_all[OUTPUT_COLUMNS]


def print_output_summary(path: Path, df: pd.DataFrame) -> None:
    print("\n" + "-" * 70)
    print(f"Saved: {path}")
    print(f"  rows       : {len(df):,}")
    print(f"  tickers    : {df['티커'].nunique() if not df.empty else 0}")
    if df.empty:
        print("  date range : -")
    else:
        print(f"  date range : {df['Date'].min().date()} ~ {df['Date'].max().date()}")


def main() -> None:
    args = parse_args()
    start_date = pd.Timestamp(args.start_date).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(args.end_date).strftime("%Y-%m-%d")

    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise ValueError(f"start-date가 end-date보다 늦음: {start_date} > {end_date}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    specs = [
        CollectionSpec(
            name="domestic_bond",
            source_file=SOURCE_DIR / "etf_domestic_bond_close.csv",
            output_file=OUTPUT_DIR / "etf_domestic_bond_close_2026_append.csv",
        ),
        CollectionSpec(
            name="global_bond",
            source_file=SOURCE_DIR / "etf_global_bond_close.csv",
            output_file=OUTPUT_DIR / "etf_global_bond_close_2026_append.csv",
        ),
        CollectionSpec(
            name="fx_commodity",
            source_file=SOURCE_DIR / "etf_fx_commodity_close.csv",
            output_file=OUTPUT_DIR / "etf_fx_commodity_close_2026_append.csv",
        ),
    ]

    print("ETF append collection config")
    print(f"  source dir : {SOURCE_DIR}")
    print(f"  output dir : {OUTPUT_DIR}")
    print(f"  period     : {start_date} ~ {end_date}")

    for spec in specs:
        df = collect_one_group(
            spec=spec,
            start_date=start_date,
            end_date=end_date,
            request_sleep=args.request_sleep,
        )
        df.to_csv(spec.output_file, index=False, encoding="utf-8-sig")
        print_output_summary(spec.output_file, df)


if __name__ == "__main__":
    main()
