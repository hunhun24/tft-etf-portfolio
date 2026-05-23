"""
regime_switch.py — TFT 예측 기반 risk_on / risk_off 판단 유틸.

이 모듈은 포트폴리오 비중을 계산하지 않고, pred_q10/pred_q50/pred_q90
예측 분포만 사용해 conservative/aggressive 선택 신호를 만든다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


REGIME_CONFIG = {
    "top_n": 3,
    "avg_return_threshold": 0.001,
    "top_return_threshold": 0.003,
    "downside_threshold": -0.03,
    "uncertainty_threshold": 0.06,
    "risk_on_score_threshold": 3,
}

REQUIRED_PRED_COLUMNS = ["pred_q10", "pred_q50", "pred_q90"]


def validate_regime_input(df: pd.DataFrame) -> None:
    """Regime 판단에 필요한 예측 컬럼과 결측 여부를 엄격히 검증한다."""
    missing = [col for col in REQUIRED_PRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"regime input missing required columns: {missing}")

    if df.empty:
        raise ValueError("regime input is empty")

    na_counts = df[REQUIRED_PRED_COLUMNS].isna().sum()
    na_counts = na_counts[na_counts > 0]
    if not na_counts.empty:
        raise ValueError(
            "regime input contains NA values in prediction columns: "
            f"{na_counts.to_dict()}"
        )


def summarize_regime_features(
    df: pd.DataFrame,
    top_n: int = 3,
) -> dict[str, float]:
    """TFT 분위수 예측값에서 regime 판단용 요약 지표를 계산한다."""
    validate_regime_input(df)

    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")

    pred = df[REQUIRED_PRED_COLUMNS].copy()
    effective_top_n = min(top_n, len(pred))

    return {
        "avg_pred_return": float(pred["pred_q50"].mean()),
        "top_pred_return": float(
            pred["pred_q50"].nlargest(effective_top_n).mean()
        ),
        "downside_risk": float(pred["pred_q10"].mean()),
        "forecast_uncertainty": float(
            (pred["pred_q90"] - pred["pred_q10"]).mean()
        ),
        "pred_dispersion": float(pred["pred_q50"].std(ddof=0)),
    }


def check_regime(
    df: pd.DataFrame,
    config: dict[str, Any] = REGIME_CONFIG,
) -> dict[str, Any]:
    """
    TFT 예측값을 기준으로 risk_on/risk_off regime을 판단한다.

    risk_on이면 aggressive 후보를, risk_off이면 conservative 후보를 선택하도록
    selected_mode를 함께 반환한다.
    """
    cfg = REGIME_CONFIG.copy()
    cfg.update(config or {})

    features = summarize_regime_features(df, top_n=int(cfg["top_n"]))

    triggered_conditions = {
        "avg_return_positive": (
            features["avg_pred_return"] > cfg["avg_return_threshold"]
        ),
        "top_return_strong": (
            features["top_pred_return"] > cfg["top_return_threshold"]
        ),
        "downside_acceptable": (
            features["downside_risk"] > cfg["downside_threshold"]
        ),
        "uncertainty_low": (
            features["forecast_uncertainty"] < cfg["uncertainty_threshold"]
        ),
    }
    regime_score = int(sum(triggered_conditions.values()))

    if regime_score >= cfg["risk_on_score_threshold"]:
        regime = "risk_on"
        selected_mode = "aggressive"
    else:
        regime = "risk_off"
        selected_mode = "conservative"

    return {
        "regime": regime,
        "selected_mode": selected_mode,
        "regime_score": regime_score,
        **features,
        "triggered_conditions": triggered_conditions,
    }
