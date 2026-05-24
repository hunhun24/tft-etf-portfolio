"""
dataset.py
----------
TimeSeriesDataSet 선언 및 train/val DataLoader 생성.
전처리 완료 파일(final_dataset_preprocessed.csv)을 입력으로 받음.

컬럼 구조 (전처리 후):
    date, ticker, name, end, target_5d, sector,
    domestic_mean, domestic_std, global_mean, global_std,
    usd_krw, vix_close, time_idx,
    AUM_log, domestic_count_log, global_count_log
"""

import os
import sys

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    PROCESSED_DATA_PATH,
    TRAIN_END_DATE,
    VAL_START_DATE,
    TARGET_COL,
    GROUP_COL,
    SECTOR_COL,
    TIME_IDX_COL,
    TIME_VARYING_UNKNOWN_REALS,
    TIME_VARYING_KNOWN_REALS,
    TIME_VARYING_KNOWN_CATEGORICALS,
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    BATCH_SIZE,
)

STATIC_CATEGORICALS = [GROUP_COL, SECTOR_COL]  # ticker, sector


def load_data(csv_path: str) -> pd.DataFrame:
    """전처리 완료 CSV 로드 및 타입 정리."""
    df = pd.read_csv(csv_path, parse_dates=["date"])

    # target NaN (shift(-5) 끝부분) 제거
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # categorical 타입 명시
    for col in STATIC_CATEGORICALS:
        df[col] = df[col].astype(str)

    df = df.sort_values([GROUP_COL, "date"]).reset_index(drop=True)

    return df


def make_datasets(
    csv_path: str = PROCESSED_DATA_PATH,
    batch_size: int = BATCH_SIZE,
):
    """
    Returns
    -------
    train_ds, val_ds, train_loader, val_loader
    """
    df = load_data(csv_path)

    # ── train/val split (날짜 기준, ratio 아님) ───────────────────────────────
    
    train_end = pd.Timestamp(TRAIN_END_DATE)
    val_start = pd.Timestamp(VAL_START_DATE)
    train_df = df[df["date"] <= train_end].copy()

    # val 시작 time_idx (cutoff 이후 첫 번째 time_idx)
    val_min_idx = df.loc[df["date"] >= val_start, TIME_IDX_COL].min()

    # ── Train DataSet ─────────────────────────────────────────────────────────
    train_ds = TimeSeriesDataSet(
        train_df,
        time_idx=TIME_IDX_COL,
        target=TARGET_COL,
        group_ids=[GROUP_COL],

        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,

        static_categoricals=STATIC_CATEGORICALS,
        static_reals=[],

        time_varying_known_categoricals=TIME_VARYING_KNOWN_CATEGORICALS,
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,

        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,

        # target: ticker별 동적 정규화
        target_normalizer=GroupNormalizer(
            groups=[GROUP_COL],
            transformation=None,   # log return은 추가 변환 불필요
        ),

        # AUM_log: ticker별 정규화 (종목 간 절대값 차이 제거)
        scalers={
            "AUM_log": GroupNormalizer(groups=[GROUP_COL], transformation=None),
        },

        allow_missing_timesteps=True,   # ticker별 상장일 차이 및 일부 결측 허용
        add_relative_time_idx=True,     # 상대적 시간 위치 피처 자동 추가
        add_target_scales=True,         # target scale 정보 자동 추가
        add_encoder_length=True,        # encoder 실제 길이 자동 추가
    )

    # ── Val DataSet (train 파라미터 그대로 상속) ───────────────────────────────
    val_ds = TimeSeriesDataSet.from_dataset(
        train_ds,
        df,                          # 전체 df: encoder가 train 끝부분 참조
        predict=False,       # True=그룹당 마지막 1개(추론용), False=전체 val 샘플
        stop_randomization=True,
        min_prediction_idx=val_min_idx,
    )

    # ── DataLoader ────────────────────────────────────────────────────────────
    train_loader = train_ds.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_loader = val_ds.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=0
    )
    print(f"  Train date range : {train_df['date'].min().date()} ~ {train_df['date'].max().date()}")
    print(f"  Val start date   : {df[df['date'] >= val_start]['date'].min().date()}")
    print(f"  val_min_idx      : {val_min_idx}")
    print(f"  Train rows       : {len(train_df)}")
    print(f"  Val samples      : {len(val_ds)}")
    
    return train_ds, val_ds, train_loader, val_loader


# ── sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_ds, val_ds, train_loader, val_loader = make_datasets()

    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}")

    x, y = next(iter(train_loader))
    print(f"Batch x keys  : {list(x.keys())}")
    print(f"Batch y shape : {y[0].shape}")  # (batch, prediction_length)
