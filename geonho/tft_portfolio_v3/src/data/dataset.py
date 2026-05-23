"""
data/dataset.py — TimeSeriesDataSet 선언 + DataLoader 생성
직접 실행 시 sanity check 수행
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from torch.utils.data import DataLoader

from config import (
    PREPROCESSED_CSV, DATE_COL, GROUP_COL, SECTOR_COL,
    TARGET_COL, TIME_IDX_COL,
    TRAIN_END_DATE, VAL_START_DATE,
    ENCODER_LENGTH, MAX_PREDICTION_LENGTH,
    TIME_VARYING_UNKNOWN_REALS, TIME_VARYING_KNOWN_REALS,
    STATIC_CATS, BATCH_SIZE, NUM_WORKERS,
)


def load_preprocessed() -> pd.DataFrame:
    df = pd.read_csv(PREPROCESSED_CSV, parse_dates=[DATE_COL])
    # z-score / scaled 컬럼으로 교체
    col_map = {
        "domestic_mean": "domestic_mean_z",
        "domestic_std":  "domestic_std_z",
        "global_mean":   "global_mean_z",
        "global_std":    "global_std_z",
        "AUM":           "AUM_log",
        "usd_krw":       "usd_krw_scaled",
        "vix_close":     "vix_close_scaled",
    }
    for orig, new in col_map.items():
        if new in df.columns and orig in df.columns:
            df = df.drop(columns=[orig])
            df = df.rename(columns={new: orig})
    df[GROUP_COL] = df[GROUP_COL].astype(str)
    df[SECTOR_COL] = df[SECTOR_COL].astype(str)
    return df


def build_datasets(df: pd.DataFrame):
    """
    Returns:
        train_dataset, val_dataset
    """
    train_df = df[df[DATE_COL] <= pd.Timestamp(TRAIN_END_DATE)].copy()
    val_df   = df[df[DATE_COL] >= pd.Timestamp(VAL_START_DATE)].copy()

    # ── time_varying_unknown_reals: 실제로 df에 있는 컬럼만 사용 ──
    avail_unk = [c for c in TIME_VARYING_UNKNOWN_REALS if c in df.columns]
    avail_kno = [c for c in TIME_VARYING_KNOWN_REALS   if c in df.columns]

    train_dataset = TimeSeriesDataSet(
        train_df,
        time_idx                  = TIME_IDX_COL,
        target                    = TARGET_COL,
        group_ids                 = [GROUP_COL],
        min_encoder_length        = ENCODER_LENGTH // 2,   # 최소 30일 (결측 허용)
        max_encoder_length        = ENCODER_LENGTH,
        min_prediction_length     = MAX_PREDICTION_LENGTH,
        max_prediction_length     = MAX_PREDICTION_LENGTH,
        static_categoricals       = STATIC_CATS,
        static_reals              = [],
        time_varying_known_categoricals   = [],
        time_varying_known_reals          = avail_kno,
        time_varying_unknown_categoricals = [],
        time_varying_unknown_reals        = avail_unk,
        target_normalizer         = GroupNormalizer(
            groups=[GROUP_COL],
            transformation="softplus",
        ),
        add_relative_time_idx     = True,
        add_target_scales         = True,
        add_encoder_length        = True,
        allow_missing_timesteps   = True,
        categorical_encoders      = {
            SECTOR_COL: NaNLabelEncoder(add_nan=True),
            GROUP_COL:  NaNLabelEncoder(add_nan=True),
        },
    )

    val_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset,
        val_df,
        predict        = True,
        stop_randomization = True,
    )

    return train_dataset, val_dataset


def build_dataloaders(train_dataset, val_dataset):
    train_loader = train_dataset.to_dataloader(
        train=True, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = val_dataset.to_dataloader(
        train=False, batch_size=BATCH_SIZE * 2,
        num_workers=NUM_WORKERS,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    print("=== DataSet Sanity Check ===")
    df = load_preprocessed()
    print(f"Loaded: {len(df):,} rows")

    train_ds, val_ds = build_datasets(df)
    print(f"Train samples : {len(train_ds):,}")
    print(f"Val   samples : {len(val_ds):,}")

    train_loader, val_loader = build_dataloaders(train_ds, val_ds)
    x, y = next(iter(train_loader))
    print(f"\nBatch encoder shape : {x['encoder_cont'].shape}")
    print(f"Batch target shape  : {y[0].shape}")
    print("Sanity check PASSED ✓")
