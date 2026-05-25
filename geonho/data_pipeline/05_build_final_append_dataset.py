"""
Build 2026 append final TFT datasets.

기존 geonho/merge/merge.py와 기존 final_dataset_for_tft_*.csv는 수정하지 않고,
geonho/data_pipeline/outputs 아래의 ETF/sentiment append 파일만 병합한다.

주의:
append 파일만으로 계산한 target_5d는 전체 데이터 기준 target_5d와 달라질 수 있다.
최종 전체 데이터 병합 후 target_5d와 time_idx는 전체 ticker별 시계열 기준으로
재계산하는 것을 권장한다.

Example:
python data_pipeline/05_build_final_append_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


GEONHO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_OUTPUT_DIR = GEONHO_ROOT / "data_pipeline" / "outputs"
ETF_DIR = PIPELINE_OUTPUT_DIR / "etf"
SENTIMENT_DIR = PIPELINE_OUTPUT_DIR / "sentiment"
FINAL_DIR = PIPELINE_OUTPUT_DIR / "final"

ETF_DOMESTIC_BOND = ETF_DIR / "etf_domestic_bond_close_2026_append.csv"
ETF_FX_COMMODITY = ETF_DIR / "etf_fx_commodity_close_2026_append.csv"
ETF_GLOBAL_BOND = ETF_DIR / "etf_global_bond_close_2026_append.csv"
MK_SENTIMENT = SENTIMENT_DIR / "mk_sentiment_scored_2026_append.csv"
NYT_SENTIMENT = SENTIMENT_DIR / "nyt_sentiment_scored_2026_append.csv"

SECTOR_ETF = {
    "sovereign_kr": ["A157450", "A148070", "A439870"],
    "sovereign_us": ["A329750", "A305080", "A453850"],
    "real_assets": ["A411060", "A144600", "A261220", "A261240"],
}

TICKER_TO_SECTOR = {
    ticker: sector
    for sector, tickers in SECTOR_ETF.items()
    for ticker in tickers
}

DOMESTIC_SENTIMENT_MAP = {
    "sovereign_kr": "domestic_bond",
    "sovereign_us": "domestic_bond",
    "real_assets": "real_assets",
}

NYT_SENTIMENT_MAP = {
    "sovereign_kr": "overseas_bond",
    "sovereign_us": "overseas_bond",
    "real_assets": "real_assets",
}

FINAL_COLS = [
    "date",
    "ticker",
    "name",
    "end",
    "AUM",
    "target_5d",
    "sector",
    "domestic_mean",
    "domestic_std",
    "domestic_count",
    "global_mean",
    "global_std",
    "global_count",
]

OUTPUT_FILES = {
    "sovereign_kr": "final_dataset_for_tft_sovereign_kr_2026_append.csv",
    "sovereign_us": "final_dataset_for_tft_sovereign_us_2026_append.csv",
    "real_assets": "final_dataset_for_tft_realassets_2026_append.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 2026 append final TFT datasets from pipeline ETF/sentiment outputs."
    )
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument(
        "--save-combined",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save combined final_dataset_for_tft_geonho_2026_append.csv.",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"required input file not found: {path}")


def load_etf_file(path: Path) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = ["Date", "티커", "ETF명", "AUM(억원)", "종가"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    return df


def build_etf_panel(start_date: str | None, end_date: str | None) -> pd.DataFrame:
    etf_dom = load_etf_file(ETF_DOMESTIC_BOND)
    etf_fx = load_etf_file(ETF_FX_COMMODITY)
    etf_glob = load_etf_file(ETF_GLOBAL_BOND)

    etf = pd.concat([etf_dom, etf_fx, etf_glob], ignore_index=True)
    etf = etf.rename(columns={
        "티커": "ticker",
        "ETF명": "name",
        "AUM(억원)": "AUM",
        "종가": "end",
        "Date": "date",
    })
    etf["date"] = pd.to_datetime(etf["date"], errors="coerce", format="mixed")
    etf["ticker"] = etf["ticker"].astype(str)
    etf = etf.dropna(subset=["date"]).reset_index(drop=True)

    all_tickers = [ticker for tickers in SECTOR_ETF.values() for ticker in tickers]
    etf = etf[etf["ticker"].isin(all_tickers)].copy()
    etf["sector"] = etf["ticker"].map(TICKER_TO_SECTOR)

    if start_date is not None:
        etf = etf[etf["date"] >= pd.Timestamp(start_date)].copy()
    if end_date is not None:
        etf = etf[etf["date"] <= pd.Timestamp(end_date)].copy()

    etf = etf.drop_duplicates(subset=["ticker", "date"], keep="last").reset_index(drop=True)
    etf = etf.sort_values(["ticker", "date"]).reset_index(drop=True)

    # append 단독 기준 임시 target_5d. 전체 병합 후 재계산 권장.
    etf["target_5d"] = etf.groupby("ticker")["end"].transform(
        lambda x: np.log(x.shift(-5) / x)
    )
    return etf


def load_sentiment(path: Path) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = ["date", "section", "sentiment_score"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    df["section"] = df["section"].fillna("").astype(str).str.strip()
    return df


def aggregate_sentiment(
    df: pd.DataFrame,
    section: str,
    prefix: str,
) -> pd.DataFrame:
    sub = df[df["section"] == section].copy()
    if sub.empty:
        return pd.DataFrame(columns=[
            "date",
            f"{prefix}_mean",
            f"{prefix}_std",
            f"{prefix}_count",
        ])

    return (
        sub.groupby("date")["sentiment_score"]
        .agg(
            **{
                f"{prefix}_mean": "mean",
                f"{prefix}_std": "std",
                f"{prefix}_count": "count",
            }
        )
        .reset_index()
    )


def summarize_panel(sector: str, panel: pd.DataFrame, out_path: Path) -> None:
    print(f"\n[{sector}]")
    print(f"  shape               : {panel.shape}")
    if panel.empty:
        print("  date range          : -")
        print("  ticker count        : 0")
        print("  target_5d NaN       : -")
        print("  domestic_mean NaN   : -")
        print("  global_mean NaN     : -")
    else:
        print(f"  date range          : {panel['date'].min().date()} ~ {panel['date'].max().date()}")
        print(f"  ticker count        : {panel['ticker'].nunique()}")
        print(f"  target_5d NaN       : {panel['target_5d'].isna().mean() * 100:.1f}%")
        print(f"  domestic_mean NaN   : {panel['domestic_mean'].isna().mean() * 100:.1f}%")
        print(f"  global_mean NaN     : {panel['global_mean'].isna().mean() * 100:.1f}%")
    print(f"  saved               : {out_path}")


def main() -> None:
    args = parse_args()
    start_date = pd.Timestamp(args.start_date).strftime("%Y-%m-%d") if args.start_date else None
    end_date = pd.Timestamp(args.end_date).strftime("%Y-%m-%d") if args.end_date else None
    if start_date and end_date and pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise ValueError(f"start-date가 end-date보다 늦음: {start_date} > {end_date}")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Building 2026 append final datasets")
    print(f"  ETF dir       : {ETF_DIR}")
    print(f"  sentiment dir : {SENTIMENT_DIR}")
    print(f"  output dir    : {FINAL_DIR}")
    print(f"  date filter   : {start_date or '-'} ~ {end_date or '-'}")
    print(
        "  NOTE: 최종 전체 데이터 병합 후 target_5d와 time_idx는 "
        "전체 ticker별 시계열 기준으로 재계산하는 것을 권장한다."
    )

    etf = build_etf_panel(start_date=start_date, end_date=end_date)
    mk = load_sentiment(MK_SENTIMENT)
    nyt = load_sentiment(NYT_SENTIMENT)

    panels = []
    for sector, out_file in OUTPUT_FILES.items():
        panel = etf[etf["sector"] == sector].copy()

        dom_sent = aggregate_sentiment(
            df=mk,
            section=DOMESTIC_SENTIMENT_MAP[sector],
            prefix="domestic",
        )
        panel = panel.merge(dom_sent, on="date", how="left")

        glob_sent = aggregate_sentiment(
            df=nyt,
            section=NYT_SENTIMENT_MAP[sector],
            prefix="global",
        )
        panel = panel.merge(glob_sent, on="date", how="left")

        panel = panel[FINAL_COLS].sort_values(["date", "ticker"]).reset_index(drop=True)
        out_path = FINAL_DIR / out_file
        panel.to_csv(out_path, index=False, encoding="utf-8-sig")
        summarize_panel(sector, panel, out_path)
        panels.append(panel)

    if args.save_combined:
        combined = pd.concat(panels, ignore_index=True)
        combined = combined.sort_values(["date", "sector", "ticker"]).reset_index(drop=True)
        combined_path = FINAL_DIR / "final_dataset_for_tft_geonho_2026_append.csv"
        combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
        summarize_panel("combined", combined, combined_path)


if __name__ == "__main__":
    main()
