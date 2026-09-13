"""
preprocess.py
-------------
전처리 파이프라인. 반드시 config.py의 TRAIN_CUTOFF_DATE 기준으로
train 통계만 fit하고 전체에 transform.

실행:
    python src_tft_port_regime/data/preprocess.py
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    SCALER_DIR,
    TRAIN_END_DATE,
    LOG1P_COLS,
    SECTOR_ZSCORE_COLS,
    MACRO_SCALE_COLS,
    SECTOR_COL,
)


def load_and_sort(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def apply_log1p(df: pd.DataFrame) -> pd.DataFrame:
    """AUM, domestic_count, global_count → log1p 변환 후 _log 컬럼으로 추가."""
    for col in LOG1P_COLS:
        df[f"{col}_log"] = np.log1p(df[col])
    return df


def fit_sector_zscore(train_df: pd.DataFrame) -> dict:
    """
    train 구간에서 sector별 mean/std 계산.
    반환: {col: {sector: {"mean": float, "std": float}}}
    """
    stats = {}
    for col in SECTOR_ZSCORE_COLS:
        stats[col] = {}
        for sector, grp in train_df.groupby(SECTOR_COL):
            m = grp[col].mean()
            s = grp[col].std()
            s = s if s > 0 else 1.0   # std=0 방어
            stats[col][sector] = {"mean": m, "std": s}
    return stats


def apply_sector_zscore(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """train에서 계산한 sector 통계로 전체 df transform (inplace)."""
    for col, sector_stats in stats.items():
        def _zscore(row):
            s = sector_stats.get(row[SECTOR_COL])
            if s is None:
                return row[col]
            return (row[col] - s["mean"]) / s["std"]
        df[col] = df.apply(_zscore, axis=1)
    return df


def fit_macro_scaler(train_df: pd.DataFrame) -> StandardScaler:
    """train 구간 macro 변수 StandardScaler fit."""
    scaler = StandardScaler()
    scaler.fit(train_df[MACRO_SCALE_COLS])
    return scaler


def apply_macro_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """전체 df에 macro scaler transform."""
    df[MACRO_SCALE_COLS] = scaler.transform(df[MACRO_SCALE_COLS])
    return df


def save_scalers(stats: dict, macro_scaler: StandardScaler, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sector_zscore_stats.pkl"), "wb") as f:
        pickle.dump(stats, f)
    with open(os.path.join(out_dir, "macro_scaler.pkl"), "wb") as f:
        pickle.dump(macro_scaler, f)
    print(f"Scalers saved to {out_dir}")


def main():
    # 1. 로드 및 정렬
    print("Loading data...")
    df = load_and_sort(RAW_DATA_PATH)
    print(f"  Shape: {df.shape}")

    # 2. log1p 변환
    print("Applying log1p...")
    df = apply_log1p(df)

    # 3. train/val 분리 (fit용)
    train_mask = df["date"] <= pd.Timestamp(TRAIN_END_DATE)
    train_df = df[train_mask]
    print(f"  Train rows: {train_mask.sum()}, Val rows: {(~train_mask).sum()}")

    # 4. sector z-score: train 기준 fit → 전체 transform
    print("Fitting sector z-score stats (train only)...")
    sector_stats = fit_sector_zscore(train_df)
    df = apply_sector_zscore(df, sector_stats)

    # 5. macro StandardScaler: train 기준 fit → 전체 transform
    print("Fitting macro StandardScaler (train only)...")
    macro_scaler = fit_macro_scaler(train_df)
    df = apply_macro_scaler(df, macro_scaler)

    # 6. 원본 LOG1P_COLS 제거 (log 버전으로 대체)
    df = df.drop(columns=LOG1P_COLS)

    # 7. 저장
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Preprocessed data saved to {PROCESSED_DATA_PATH}")

    # 8. scaler 저장 (추론 시 재사용)
    save_scalers(sector_stats, macro_scaler, SCALER_DIR)

    # 9. 간단한 sanity check
    print("\n── Sanity Check ──")
    print(f"  Columns: {list(df.columns)}")
    print(f"  NaN counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    val_df = df[~train_mask.values]
    print(f"  Scaler fit 기준 max date: {train_df['date'].max().date()}")
    print(f"  Val date range: {val_df['date'].min()} ~ {val_df['date'].max()}")


if __name__ == "__main__":
    main()