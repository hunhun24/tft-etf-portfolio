"""
data/preprocess.py — 원본 CSV 전처리
실행: python src/data/preprocess.py

수행 작업:
  1. AUM, domestic_count, global_count → log1p
  2. sentiment 4개 → train 구간 기준 sector별 z-score (leakage 방지)
  3. usd_krw, vix_close → train 구간 기준 전체 StandardScaler
  4. scaler 통계를 .pkl 저장 (predict.py 재사용)
  5. time_idx 검증 (글로벌 날짜 인덱스 방식)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

from config import (
    RAW_CSV, PREPROCESSED_CSV, SCALER_DIR,
    LOG1P_COLS, SENTIMENT_COLS, STANDARD_SCALE_COLS,
    TRAIN_END_DATE, DATE_COL, SECTOR_COL, GROUP_COL,
)


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    print(f"[load] {len(df):,} rows  |  {df[GROUP_COL].nunique()} tickers  "
          f"|  {df[DATE_COL].min().date()} ~ {df[DATE_COL].max().date()}")
    return df


def fix_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    """글로벌 날짜 인덱스 (ticker별 리셋 아님)."""
    date_to_idx = {d: i for i, d in enumerate(sorted(df[DATE_COL].unique()))}
    df["time_idx"] = df[DATE_COL].map(date_to_idx)
    return df


def apply_log1p(df: pd.DataFrame) -> pd.DataFrame:
    for col in LOG1P_COLS:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
            print(f"  [log1p] {col} → {col}_log")
    return df


def apply_sector_zscore(df: pd.DataFrame, train_mask: pd.Series) -> tuple[pd.DataFrame, dict]:
    """sector별 z-score. train 구간 통계로 fit → 전체 transform."""
    scalers = {}
    train_df = df[train_mask].copy()

    for col in SENTIMENT_COLS:
        if col not in df.columns:
            continue
        df[f"{col}_z"] = np.nan
        stats = train_df.groupby(SECTOR_COL)[col].agg(["mean", "std"]).rename(
            columns={"mean": f"{col}_mean", "std": f"{col}_std"}
        )
        scalers[f"sector_{col}"] = stats
        for sector, row in stats.iterrows():
            mask = df[SECTOR_COL] == sector
            std = row[f"{col}_std"] if row[f"{col}_std"] > 1e-8 else 1.0
            df.loc[mask, f"{col}_z"] = (df.loc[mask, col] - row[f"{col}_mean"]) / std
        # fillna: 훈련에 없는 섹터는 0
        df[f"{col}_z"] = df[f"{col}_z"].fillna(0.0)
        print(f"  [sector z-score] {col} → {col}_z")

    return df, scalers


def apply_standard_scaler(df: pd.DataFrame, train_mask: pd.Series) -> tuple[pd.DataFrame, dict]:
    scalers = {}
    for col in STANDARD_SCALE_COLS:
        if col not in df.columns:
            continue
        scaler = StandardScaler()
        scaler.fit(df.loc[train_mask, col].values.reshape(-1, 1))
        df[f"{col}_scaled"] = scaler.transform(df[col].values.reshape(-1, 1)).flatten()
        scalers[col] = scaler
        print(f"  [StandardScaler] {col} → {col}_scaled")
    return df, scalers


def save_scalers(scalers: dict, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name, obj in scalers.items():
        fpath = path / f"{name}.pkl"
        with open(fpath, "wb") as f:
            pickle.dump(obj, f)
    print(f"[save] {len(scalers)} scaler(s) → {path}")


def main():
    df = load_raw(RAW_CSV)
    df = fix_time_idx(df)

    train_mask = df[DATE_COL] <= pd.Timestamp(TRAIN_END_DATE)
    print(f"[split] train: {train_mask.sum():,} rows  |  "
          f"val+test: {(~train_mask).sum():,} rows")

    # 전처리
    df = apply_log1p(df)
    df, sector_scalers  = apply_sector_zscore(df, train_mask)
    df, global_scalers  = apply_standard_scaler(df, train_mask)

    all_scalers = {**sector_scalers, **global_scalers}
    save_scalers(all_scalers, SCALER_DIR)

    # 결측 처리 (target 결측은 유지 — TFT가 내부 처리)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(method="ffill").fillna(0.0)

    df.to_csv(PREPROCESSED_CSV, index=False)
    print(f"[done] 저장 완료 → {PREPROCESSED_CSV}")
    print(df.describe())


if __name__ == "__main__":
    main()
