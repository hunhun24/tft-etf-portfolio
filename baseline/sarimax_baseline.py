"""
sarimax_baseline.py — ticker별 ARIMAX (pmdarima auto_arima) val 예측

실행:
  pip install pmdarima  # 최초 1회
  python baseline/sarimax_baseline.py

데이터: FINAL_DATA/tft_data/final_dataset_tft_ready.csv
Split:  train ≤ 2024-07-23  /  val ≥ 2024-07-30  (TFT·XGBoost와 동일)

[baseline 성격]
  SARIMAX는 ticker별로 따로 학습하는 전통 시계열 baseline이다.
  TFT처럼 29개 ticker를 하나의 패널 모델로 동시에 학습하지 않는다.
  따라서 이 결과는 "패널 기반 포트폴리오 모델"이 아니라
  "ticker별 점예측 baseline"으로 해석해야 한다.

auto_arima 사용 이유:
  XGBoost Optuna 튜닝과 동일한 공정 비교 원칙
  — 사람이 수동으로 order를 지정하지 않고 데이터 기반 탐색

d=0 고정 이유:
  log return(target_5d = log(end_{t+5}/end_t))은 이미 정상성을 만족하므로
  차분 없이 ARIMA 적합 가능

고정 모델 다중스텝 예측 방식:
  target_5d(t) = log(end_{t+5}/end_t) 이므로 시점 t의 target_5d는
  t+5 가격을 알아야 확정됨. 즉 val_date[i]의 target_5d를
  model.update()로 주입하면 val_date[i+1] 예측 시 미래 가격(end_{t+5})이
  AR 항을 통해 누출(leakage)됨.
  → model.update() 를 완전 제거하고, train fit 모델로
    model.predict(n_periods=len(val), exogenous=X_val) 단일 호출.
    AR 항은 자체 예측값을 재귀 참조(open-loop). exog는 당일 관측값이므로 leakage 없음.

exog 선택 기준:
  domestic_mean / global_mean : 감성 방향성
  domestic_std               : 감성 불확실성
  usd_krw                    : FX (원달러 환율)
  vix_close                  : 시장 변동성
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    from pmdarima import auto_arima
except ImportError:
    raise ImportError(
        "pmdarima가 설치되지 않았습니다. 먼저 실행하세요:\n"
        "  pip install pmdarima"
    )

warnings.filterwarnings("ignore")

# ── 경로 ─────────────────────────────────────────────────────────
BASELINE_DIR = Path(__file__).resolve().parent        # geonho/baseline/
PROJECT_ROOT = BASELINE_DIR.parent.parent             # Financial_project/

sys.path.insert(0, str(BASELINE_DIR.parent / "src_tft_port_524"))
from config import TRAIN_END_DATE, VAL_START_DATE      # noqa: E402

# DUPLICATE_TICKERS: backtest universe와 일관성 유지용.
# src_tft_port_524/config.py 에 정의되어 있으면 import, 없으면 fallback 하드코딩
# (src_tft_port_regime/config.py 의 값과 동기화).
try:
    from config import DUPLICATE_TICKERS              # noqa: E402
except ImportError:
    DUPLICATE_TICKERS = {"157450"}
    print(f"[INFO] src_tft_port_524/config.py 에 DUPLICATE_TICKERS 없음 → "
          f"하드코딩 {DUPLICATE_TICKERS} 사용")

DATA_PATH  = PROJECT_ROOT / "FINAL_DATA" / "tft_data" / "final_dataset_tft_ready.csv"
PRED_SAVE  = BASELINE_DIR / "sarimax_predictions.csv"
ORDER_SAVE = BASELINE_DIR / "sarimax_order_summary.csv"

EXOG_COLS = ["domestic_mean", "domestic_std", "global_mean", "usd_krw", "vix_close"]
MIN_TRAIN = 30    # auto_arima 최소 요구 샘플 수
SEP       = "-" * 60

# XGBoost val 성능 참고값
XGB_MAE  = 0.019672
XGB_RMSE = 0.031045


# ── 1. 로드 ──────────────────────────────────────────────────────
def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["target_5d"]).reset_index(drop=True)
    return df


# ── 2. ticker별 전처리 ────────────────────────────────────────────
def prepare_ticker(
    g: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    반환: (train_df, val_df) — 정렬·ffill·dropna·exog scaling 완료.
    train 샘플 부족 또는 val 없으면 None.
    """
    g = g.sort_values("date").copy()

    # exog ffill(limit=3) → dropna
    g[EXOG_COLS] = g[EXOG_COLS].ffill(limit=3)
    g = g.dropna(subset=EXOG_COLS + ["target_5d"]).reset_index(drop=True)

    train = g[g["date"] <= TRAIN_END_DATE].reset_index(drop=True)
    val   = g[g["date"] >= VAL_START_DATE].reset_index(drop=True)

    if len(train) < MIN_TRAIN or val.empty:
        return None

    # ticker별 StandardScaler: train fit → train/val transform
    scaler = StandardScaler()
    train = train.copy()
    val   = val.copy()
    train[EXOG_COLS] = scaler.fit_transform(train[EXOG_COLS])
    val[EXOG_COLS]   = scaler.transform(val[EXOG_COLS])

    return train, val


# ── 3. auto_arima + leakage-free open-loop forecast ──────────────────
def fit_and_forecast(
    ticker: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> tuple[list[float], tuple[int, int, int]] | None:
    """
    반환: (preds, (p, d, q)).  fit 실패 시 None, 개별 step 실패 시 0.0 대체.
    """
    y_train = train["target_5d"].values
    X_train = train[EXOG_COLS].values
    y_val   = val["target_5d"].values
    X_val   = val[EXOG_COLS].values

    # ── auto_arima fit ──
    try:
        model = auto_arima(
            y_train,
            exogenous             = X_train,
            start_p               = 0,
            max_p                 = 3,
            start_q               = 0,
            max_q                 = 3,
            d                     = 0,       # log return은 이미 정상성 만족
            seasonal              = False,
            information_criterion = "aic",
            stepwise              = True,    # 속도 최적화
            error_action          = "ignore",
            suppress_warnings     = True,
        )
    except Exception as exc:
        print(f"  [WARN] {ticker}: auto_arima 실패 → skip  ({exc})")
        return None

    order = model.order  # (p, 0, q)

    # ── 고정 모델 다중스텝 예측 (leakage 방지) ──
    # model.update(target_5d)는 end_{t+5} 가격 정보를 AR 항에 누출하므로 사용 금지.
    # train fit 상태 그대로 val 전체를 단일 호출로 예측 (open-loop recursive forecast).
    try:
        fc    = model.predict(n_periods=len(y_val), exogenous=X_val)
        preds = [float(v) for v in fc]
    except Exception as exc:
        print(f"  [WARN] {ticker}: predict 실패 → skip  ({exc})")
        return None

    return preds, order


# ── main ──────────────────────────────────────────────────────────
def main() -> None:
    print(SEP)
    print("SARIMAX Baseline (auto_arima, fixed-model multi-step forecast, leakage-free)")
    print(f"데이터: {DATA_PATH}")
    print(SEP)

    print("\n1. 로드 & 전처리")
    df_raw = load(DATA_PATH)

    # 중복 ticker 제거 (backtest universe와 일관성 유지)
    n_before = len(df_raw)
    df_raw = df_raw[~df_raw["ticker"].astype(str).isin(DUPLICATE_TICKERS)].reset_index(drop=True)
    removed = n_before - len(df_raw)
    if removed > 0:
        print(f"  DUPLICATE_TICKERS 필터: {removed} rows 제거 "
              f"(제외 ticker: {sorted(DUPLICATE_TICKERS)})")

    tickers = sorted(df_raw["ticker"].unique().tolist())
    n_train_total = int((df_raw["date"] <= TRAIN_END_DATE).sum())
    n_val_total   = int((df_raw["date"] >= VAL_START_DATE).sum())
    print(f"  shape={df_raw.shape}  "
          f"({df_raw['date'].min().date()} ~ {df_raw['date'].max().date()})")
    print(f"  ticker 수       : {len(tickers)}")
    print(f"  TRAIN_END_DATE  : {TRAIN_END_DATE}  (train rows: {n_train_total:,})")
    print(f"  VAL_START_DATE  : {VAL_START_DATE}  (val rows:   {n_val_total:,})")

    print(f"\n2. ticker별 auto_arima fit → leakage-free open-loop forecast")
    print(f"   (총 {len(tickers)}개 ticker — 시간 소요 예상)\n")

    all_preds  : list[dict]        = []
    order_rows : list[dict]        = []
    perf_rows  : list[dict]        = []
    skipped    : list[dict]        = []   # {"ticker": ..., "reason": ...}

    for idx, ticker in enumerate(tickers, 1):
        g        = df_raw[df_raw["ticker"] == ticker].copy()
        prepared = prepare_ticker(g)

        if prepared is None:
            reason = "train 부족 또는 val 없음"
            print(f"  [{idx:>2d}/{len(tickers)}] {ticker:<12s}  SKIP: {reason}")
            skipped.append({"ticker": ticker, "reason": reason})
            continue

        train, val = prepared
        result = fit_and_forecast(ticker, train, val)

        if result is None:
            skipped.append({"ticker": ticker, "reason": "auto_arima/predict 실패"})
            continue   # ← unpack 전에 다음 ticker로 넘어가야 TypeError 방지

        preds, (p, d, q) = result

        y_true = val["target_5d"].values
        mae    = mean_absolute_error(y_true, preds)
        rmse   = mean_squared_error(y_true, preds) ** 0.5
        print(f"  [{idx:>2d}/{len(tickers)}] {ticker:<12s}  "
              f"order=({p},{d},{q})  "
              f"MAE={mae:.6f}  RMSE={rmse:.6f}  "
              f"(val n={len(val)})")

        dates = val["date"].values
        for date, pred_val, true_val in zip(dates, preds, y_true):
            all_preds.append({
                "date":      pd.Timestamp(date).strftime("%Y-%m-%d"),
                "ticker":    ticker,
                "pred":      pred_val,
                "target_5d": true_val,
            })

        order_rows.append({"ticker": ticker, "p": p, "d": d, "q": q})
        perf_rows.append({"ticker": ticker, "mae": mae, "rmse": rmse})

    # ── order 분포 ──
    print(f"\n{SEP}")
    print("3. 선택된 ARIMA order 분포")
    print(SEP)
    if order_rows:
        order_labels = [f"({r['p']},{r['d']},{r['q']})" for r in order_rows]
        order_dist = pd.Series(order_labels).value_counts().sort_index()
        for label, cnt in order_dist.items():
            print(f"  order {label}: {cnt:>3d}개")

    # ── 전체 평균 성능 ──
    print(f"\n{SEP}")
    print("4. 전체 평균 MAE / RMSE  (ticker 단순 평균)")
    print(SEP)
    if perf_rows:
        perf_df   = pd.DataFrame(perf_rows).sort_values("rmse").reset_index(drop=True)
        mean_mae  = float(perf_df["mae"].mean())
        mean_rmse = float(perf_df["rmse"].mean())
        print(f"  SARIMAX val  | MAE={mean_mae:.6f}  RMSE={mean_rmse:.6f}  "
              f"(ticker {len(perf_df)}개)")
        print(f"  XGBoost val  | MAE={XGB_MAE:.6f}  RMSE={XGB_RMSE:.6f}  (참고)")

        print(f"\n  RMSE 상위 5 (낮은 순):")
        print(perf_df.head(5).to_string(index=False))
        print(f"\n  RMSE 하위 5 (높은 순):")
        print(perf_df.tail(5).to_string(index=False))

    # ── 저장 ──
    print(f"\n{SEP}")
    print("5. 저장 전 검증")
    print(SEP)

    pred_df = pd.DataFrame(all_preds)[["date", "ticker", "pred", "target_5d"]]
    pred_df["date"] = pd.to_datetime(pred_df["date"])

    n_dup      = int(pred_df.duplicated(subset=["ticker", "date"]).sum())
    n_t5d_nan  = int(pred_df["target_5d"].isna().sum())
    n_pred_nan = int(pred_df["pred"].isna().sum())

    print(f"  prediction date range     : "
          f"{pred_df['date'].min().date()} ~ {pred_df['date'].max().date()}")
    print(f"  prediction row 수         : {len(pred_df):,}")
    print(f"  ticker 수                 : {pred_df['ticker'].nunique()}")
    print(f"  ticker-date duplicate 수  : {n_dup}  "
          f"{'← 문제 있음' if n_dup > 0 else '(정상)'}")
    print(f"  target_5d NaN 수          : {n_t5d_nan}")
    print(f"  pred NaN 수               : {n_pred_nan}")

    print(f"\n{SEP}")
    print("6. 저장")
    print(SEP)

    pred_df.to_csv(PRED_SAVE, index=False)
    print(f"  [save] {PRED_SAVE}  ({len(pred_df):,}행)")

    if order_rows:
        order_df = pd.DataFrame(order_rows)[["ticker", "p", "d", "q"]]
        order_df.to_csv(ORDER_SAVE, index=False)
        print(f"  [save] {ORDER_SAVE}  ({len(order_df)}개 ticker)")

    # ── skip 요약 ──
    print(f"\n{SEP}")
    print("7. 실행 요약")
    print(SEP)
    print(f"  성공 ticker 수  : {len(perf_rows)}")
    print(f"  skip ticker 수  : {len(skipped)}")
    if skipped:
        print(f"  skip 목록:")
        for s in skipped:
            print(f"    {s['ticker']:<12s}  사유: {s['reason']}")

    print(SEP)


if __name__ == "__main__":
    main()
