"""
XGBoost Baseline + Optuna 하이퍼파라미터 튜닝
Data: FINAL_DATA/tft_data/final_dataset_tft_ready.csv

Split 기준: TFT config와 동일
  train : date <= TRAIN_END_DATE ("2024-07-23")
  val   : date >= VAL_START_DATE ("2024-07-30")
  5거래일 gap은 target_5d 계산 시 look-ahead leakage 방지용.
  test set 없음 (모의투자대회 목적, val로 평가 충분).

Rolling 피처:
  ticker별 과거 가격 동향을 반영하기 위해 5/20일 rolling 수익률·변동성 추가.
  XGBoost는 시계열 순서를 모르므로 rolling으로 순서 정보를 명시적으로 주입.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

# TFT config에서 split 날짜 재사용 (날짜 일관성 보장)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src_tft_port_524"))
from config import TRAIN_END_DATE, VAL_START_DATE

# DUPLICATE_TICKERS: backtest universe와 일관성 유지용.
# src_tft_port_524/config.py 에 정의되어 있으면 import, 없으면 fallback 하드코딩
# (src_tft_port_regime/config.py 의 값과 동기화).
try:
    from config import DUPLICATE_TICKERS
except ImportError:
    DUPLICATE_TICKERS = {"157450"}
    print(f"[INFO] src_tft_port_524/config.py 에 DUPLICATE_TICKERS 없음 → "
          f"하드코딩 {DUPLICATE_TICKERS} 사용")

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 50

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DATA_PATH = os.path.join(ROOT, "Financial_project-main/FINAL_DATA/tft_data/final_dataset_tft_ready.csv")

SEP = "-" * 60


# ── 1. 로드 & 전처리 ──────────────────────────────────────────────
def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["target_5d"]).reset_index(drop=True)
    df = df.drop(columns=["name", "time_idx"])
    return df


# ── 2. Rolling 피처 생성 (ticker별) ──────────────────────────────
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """ticker별 5/20일 rolling 수익률·변동성 추가.

    pandas 2.x 호환: 기존 groupby().apply(_roll) 패턴은 pandas 2.x 에서
    groupby key('ticker')를 결과 컬럼에서 자동 제거하는 동작 변경이 있음.
    이를 회피하기 위해 groupby().transform() 체인으로 refactor.
    transform은 원본 DataFrame의 index/columns를 보존하므로 ticker 컬럼이 유지됨.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    grp = df.groupby("ticker", sort=False)

    # 가격 기반 rolling
    df["end_ma5"]  = grp["end"].transform(lambda x: x.rolling(5).mean())
    df["end_ma20"] = grp["end"].transform(lambda x: x.rolling(20).mean())

    # 수익률 기반 rolling (pct_change → rolling std)
    df["_ret_tmp"]   = grp["end"].transform(lambda x: x.pct_change())
    grp2 = df.groupby("ticker", sort=False)
    df["end_vol5"]   = grp2["_ret_tmp"].transform(lambda x: x.rolling(5).std())
    df["end_vol20"]  = grp2["_ret_tmp"].transform(lambda x: x.rolling(20).std())
    df = df.drop(columns=["_ret_tmp"])

    return df.dropna().reset_index(drop=True)


# ── 3. Label Encoding ─────────────────────────────────────────────
def label_encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["ticker", "sector"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


# ── 4. Train/Val split ────────────────────────────────────────────
def split(df: pd.DataFrame):
    # TFT config 날짜 기준: train ≤ TRAIN_END_DATE, val ≥ VAL_START_DATE
    train = df[df["date"] <= TRAIN_END_DATE]
    val   = df[df["date"] >= VAL_START_DATE]

    for name, sub in [("train", train), ("val", val)]:
        print(f"  {name:5s}: {len(sub):>6,}행  "
              f"date {sub['date'].min().date()} ~ {sub['date'].max().date()}")

    feature_cols = [c for c in df.columns if c not in ("target_5d", "date")]

    def xy(sub):
        return sub[feature_cols].values, sub["target_5d"].values

    return xy(train), xy(val), feature_cols


# ── 5. Optuna 튜닝 ───────────────────────────────────────────────
def tune(X_train, y_train, X_val, y_val) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     500,
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 4),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "random_state":     42,
            "n_jobs":           -1,
            "tree_method":      "hist",
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        pred = model.predict(X_val)
        return mean_squared_error(y_val, pred) ** 0.5

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n  Best val RMSE : {study.best_value:.6f}")
    print(f"  Best params   :")
    for k, v in study.best_params.items():
        print(f"    {k:<22s} = {v}")

    return study.best_params


# ── 6. 평가 ──────────────────────────────────────────────────────
def evaluate(model, X, y, label: str):
    pred = model.predict(X)
    mae  = mean_absolute_error(y, pred)
    rmse = mean_squared_error(y, pred) ** 0.5
    print(f"  {label:5s} | MAE={mae:.6f}  RMSE={rmse:.6f}")


# ── 7. Naive Baseline ─────────────────────────────────────────────
def naive_baseline(df_raw: pd.DataFrame) -> None:
    """과거 target_5d 평균으로 당일 target_5d를 예측하는 naive 기준선.

    leakage 방지:
      target_5d(t) = log(end_{t+5}/end_t) 이므로 target_5d(t-1)은 end_{t+4}를 포함한다.
      시점 t 에서 leakage-free 하게 쓸 수 있는 가장 최근 target은 target_5d(t-5)
      (= log(end_t/end_{t-5}), 모든 가격이 t 이하에서 관측됨).
      따라서 shift(1) 이 아니라 shift(5) 이상이어야 함 → 본 코드는 shift(5).rolling(5).mean()
      으로 [t-5, t-6, ..., t-9] 5개의 과거 target 평균을 사용한다.
    """
    df = df_raw.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["naive_pred"] = (
        df.groupby("ticker")["target_5d"]
        .transform(lambda x: x.shift(5).rolling(5).mean())
    )
    df = df.dropna(subset=["naive_pred"]).reset_index(drop=True)

    val = df[df["date"] >= VAL_START_DATE]
    if val.empty:
        print("  [WARN] naive val 구간 데이터 없음")
        return

    mae  = mean_absolute_error(val["target_5d"], val["naive_pred"])
    rmse = mean_squared_error(val["target_5d"], val["naive_pred"]) ** 0.5
    print(f"  val   | MAE={mae:.6f}  RMSE={rmse:.6f}")
    print(f"  (val rows: {len(val):,}  "
          f"date {val['date'].min().date()} ~ {val['date'].max().date()})")


# ── main ──────────────────────────────────────────────────────────
def main():
    print(SEP)
    print("1. 로드 & 전처리")
    print(SEP)
    df_raw = load(DATA_PATH)
    print(f"  로드 완료: {df_raw.shape}")

    # 중복 ticker 제거 (backtest universe와 일관성 유지)
    n_before = len(df_raw)
    df_raw = df_raw[~df_raw["ticker"].astype(str).isin(DUPLICATE_TICKERS)].reset_index(drop=True)
    removed = n_before - len(df_raw)
    if removed > 0:
        print(f"  DUPLICATE_TICKERS 필터: {removed} rows 제거 → {df_raw.shape}  "
              f"(제외 ticker: {sorted(DUPLICATE_TICKERS)})")

    print("\n2. Rolling 피처 생성")
    df = add_rolling_features(df_raw)
    print(f"  Rolling 후 shape: {df.shape}")

    # label_encode 전에 원본 ticker/date 보존 (예측값 저장용)
    val_df = (
        df[df["date"] >= VAL_START_DATE][["date", "ticker", "target_5d"]]
        .copy()
        .reset_index(drop=True)
    )

    print("\n3. Label Encoding (ticker, sector)")
    df = label_encode(df)

    print(f"\n{SEP}")
    print("4. Train/Val split")
    print(f"   train: date <= {TRAIN_END_DATE}  |  val: date >= {VAL_START_DATE}")
    print(SEP)
    (X_train, y_train), (X_val, y_val), feature_cols = split(df)

    print(f"\n{SEP}")
    print(f"5. Optuna 하이퍼파라미터 튜닝 (n_trials={N_TRIALS})")
    print(SEP)
    best_params = tune(X_train, y_train, X_val, y_val)

    print(f"\n{SEP}")
    print("6. 최적 파라미터로 최종 모델 학습")
    print(SEP)
    final_model = xgb.XGBRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        **best_params,
    )
    final_model.fit(X_train, y_train, verbose=100)

    print(f"\n{SEP}")
    print("7. Feature Importance (상위 15개)")
    print(SEP)
    importance = pd.Series(final_model.feature_importances_, index=feature_cols)
    for feat, score in importance.sort_values(ascending=False).head(15).items():
        print(f"  {feat:<25s} {score:.4f}")

    print(f"\n{SEP}")
    print("8. 최종 평가 (MAE / RMSE)")
    print(SEP)
    evaluate(final_model, X_train, y_train, "train")
    evaluate(final_model, X_val,   y_val,   "val")
    print(f"  [NOTE] val 은 5단계 Optuna 튜닝에 사용된 set 이므로 위 val 점수는")
    print(f"         낙관적 추정치임. baseline 공정 비교 시 이 점 명시 필요.")
    print(SEP)

    print(f"\n{SEP}")
    print("9. 예측값 저장 (백테스트용)")
    print(SEP)
    val_pred = final_model.predict(X_val)
    val_df["pred"] = val_pred
    pred_save_path = Path(__file__).resolve().parent / "xgboost_predictions.csv"
    val_df[["date", "ticker", "pred", "target_5d"]].to_csv(pred_save_path, index=False)
    print(f"  [save] {pred_save_path}")
    print(SEP)

    print(f"\n{SEP}")
    print("10. Naive Baseline (이전 5일 target_5d 평균 → val 예측)")
    print(SEP)
    naive_baseline(df_raw)
    print(SEP)


if __name__ == "__main__":
    main()
