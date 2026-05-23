"""
add_calendar_features.py
────────────────────────
final_dataset_tft_ready.csv에 calendar/event feature를 추가하는 스크립트.
전처리(스케일링) 이전 단계에서 실행한다.

생성 feature
────────────
[known reals]
  days_to_fomc        : 다음 FOMC 날짜까지의 영업일 수 (0 = 당일)
  holiday_gap_days    : 다음 한국 공휴일(또는 NYSE 휴장일)까지의 일수

[known categoricals]
  is_fomc_week        : 해당 주에 FOMC가 있으면 1
  is_fomc_day         : FOMC 당일이면 1
  is_post_holiday     : 공휴일 다음 첫 번째 영업일이면 1
  is_month_end        : 월 마지막 영업일이면 1
  is_rebalancing_day  : 분기 마지막 월(3/6/9/12월)의 마지막 영업일이면 1

사용법
──────
Colab 셀에 붙여넣기 또는 독립 스크립트로 실행:
  python add_calendar_features.py

입력  : RAW_DATA_PATH   (final_dataset_tft_ready.csv)
출력  : OUTPUT_PATH     (final_dataset_tft_calendar.csv)
"""

import pandas as pd
import numpy as np

# ── 경로 (Colab 환경에 맞게 수정) ────────────────────────────────────────────
RAW_DATA_PATH = "/content/drive/MyDrive/tft_data/final_dataset_tft_ready.csv"
OUTPUT_PATH   = "/content/drive/MyDrive/tft_data/final_dataset_tft_calendar.csv"

# ── FOMC 실제 날짜 하드코딩 (2019~2026, 금리 결정 당일 기준) ────────────────
# 출처: federalreserve.gov press release dates
FOMC_DATES = sorted(pd.to_datetime([
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05",
    "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026 (예정)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
]))

# ── 한국 공휴일 + NYSE 주요 휴장일 하드코딩 (2019~2026) ──────────────────────
# ETF는 한국 거래소 기준이므로 KR 공휴일 중심, NYSE 휴장은 보조
KR_HOLIDAYS = sorted(pd.to_datetime([
    # 2019
    "2019-01-01","2019-02-04","2019-02-05","2019-02-06",
    "2019-03-01","2019-05-06","2019-06-06","2019-08-15",
    "2019-09-12","2019-09-13","2019-10-03","2019-10-09",
    "2019-12-25",
    # 2020
    "2020-01-01","2020-01-24","2020-01-27",
    "2020-03-01","2020-04-15","2020-05-05","2020-09-30",
    "2020-10-01","2020-10-02","2020-10-09","2020-12-25",
    # 2021
    "2021-01-01","2021-02-11","2021-02-12",
    "2021-03-01","2021-05-05","2021-09-20","2021-09-21","2021-09-22",
    "2021-10-04","2021-12-25","2021-12-31",
    # 2022
    "2022-01-01","2022-01-31","2022-02-01","2022-02-02",
    "2022-03-01","2022-03-09","2022-05-05","2022-06-01",
    "2022-06-06","2022-08-15","2022-09-09","2022-09-12",
    "2022-10-03","2022-10-10","2022-12-25","2022-12-26",
    # 2023
    "2023-01-23","2023-01-24","2023-01-25",
    "2023-03-01","2023-05-05","2023-06-06",
    "2023-08-15","2023-09-28","2023-09-29","2023-10-02","2023-10-03",
    "2023-10-09","2023-12-25",
    # 2024
    "2024-01-01","2024-02-09","2024-02-12",
    "2024-03-01","2024-04-10","2024-05-06","2024-05-15",
    "2024-06-06","2024-08-15","2024-09-16","2024-09-17","2024-09-18",
    "2024-10-03","2024-10-09","2024-12-25",
    # 2025
    "2025-01-01","2025-01-28","2025-01-29","2025-01-30",
    "2025-03-01","2025-03-03","2025-05-05","2025-05-06",
    "2025-06-06","2025-08-15","2025-10-05","2025-10-06","2025-10-07",
    "2025-10-08","2025-10-09","2025-12-25",
    # 2026
    "2026-01-01","2026-02-17","2026-02-18","2026-02-19",
    "2026-03-01","2026-05-05","2026-06-06","2026-08-17",
    "2026-09-24","2026-09-25","2026-09-26","2026-10-09","2026-12-25",
]))

KR_HOLIDAY_SET = set(KR_HOLIDAYS)

# ─────────────────────────────────────────────────────────────────────────────

def next_fomc_bdays(date: pd.Timestamp, fomc_series: pd.DatetimeIndex) -> int:
    """date 기준 다음(또는 당일) FOMC까지 영업일 수 (max 30으로 클리핑)."""
    future = fomc_series[fomc_series >= date]
    if len(future) == 0:
        return 30
    target = future[0]
    bdays = pd.bdate_range(date, target)
    return min(len(bdays) - 1, 30)  # 당일이면 0


def next_holiday_days(date: pd.Timestamp, holiday_set: set) -> int:
    """date 이후 최초 공휴일까지의 캘린더 일수 (max 30 클리핑)."""
    for i in range(1, 31):
        candidate = date + pd.Timedelta(days=i)
        if candidate in holiday_set:
            return i
    return 30


def is_business_month_end(date: pd.Timestamp) -> bool:
    """월 마지막 영업일 여부."""
    next_bday = date + pd.offsets.BDay(1)
    return next_bday.month != date.month


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dates = df["date"]

    fomc_idx = pd.DatetimeIndex(FOMC_DATES)
    fomc_set = set(FOMC_DATES)

    print("Computing days_to_fomc ...")
    df["days_to_fomc"] = dates.apply(
        lambda d: next_fomc_bdays(d, fomc_idx)
    ).astype(float)

    print("Computing holiday_gap_days ...")
    df["holiday_gap_days"] = dates.apply(
        lambda d: next_holiday_days(d, KR_HOLIDAY_SET)
    ).astype(float)

    print("Computing is_fomc_week / is_fomc_day ...")
    df["is_fomc_day"] = dates.apply(lambda d: int(d in fomc_set))

    week_starts = dates.apply(lambda d: d - pd.Timedelta(days=d.dayofweek))
    week_ends   = week_starts + pd.Timedelta(days=4)
    df["is_fomc_week"] = [
        int(any((ws <= f <= we) for f in FOMC_DATES))
        for ws, we in zip(week_starts, week_ends)
    ]

    print("Computing is_post_holiday ...")
    def _is_post_holiday(d):
        prev = d - pd.Timedelta(days=1)
        while prev.weekday() >= 5:          # 주말이면 더 뒤로
            prev -= pd.Timedelta(days=1)
        return int(prev in KR_HOLIDAY_SET)

    df["is_post_holiday"] = dates.apply(_is_post_holiday)

    print("Computing is_month_end ...")
    df["is_month_end"] = dates.apply(lambda d: int(is_business_month_end(d)))

    print("Computing is_rebalancing_day ...")
    df["is_rebalancing_day"] = dates.apply(
        lambda d: int(d.month in (3, 6, 9, 12) and is_business_month_end(d))
    )

    # categorical feature는 string으로 변환 (TFT TimeSeriesDataSet 요구사항)
    cat_cols = [
        "is_fomc_week", "is_fomc_day", "is_post_holiday",
        "is_month_end", "is_rebalancing_day",
    ]
    for col in cat_cols:
        df[col] = df[col].astype(str)

    return df


# ── 실행 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading: {RAW_DATA_PATH}")
    df_raw = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
    print(f"  Shape: {df_raw.shape}")

    df_cal = add_calendar_features(df_raw)

    # 생성된 feature 분포 확인
    new_cols = [
        "days_to_fomc", "holiday_gap_days",
        "is_fomc_week", "is_fomc_day", "is_post_holiday",
        "is_month_end", "is_rebalancing_day",
    ]
    print("\n── Sanity Check ─────────────────────────────")
    print(df_cal[new_cols].describe(include="all").to_string())

    print("\n── Sample (first 5 rows) ────────────────────")
    print(df_cal[["date"] + new_cols].head(10).to_string(index=False))

    df_cal.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}")