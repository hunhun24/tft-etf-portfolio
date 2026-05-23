"""
portfolio_regime.py — conservative/aggressive 후보 위의 regime-switch wrapper.

기존 portfolio.py의 build_signal/compute_weights 로직은 그대로 호출하고,
TFT 예측 분포 기반 regime 판단 결과에 따라 최종 포트폴리오 하나를 선택한다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
for _path in (SRC_DIR, CURRENT_DIR):
    _path_str = str(_path)
    if _path_str in sys.path:
        sys.path.remove(_path_str)
    sys.path.insert(0, _path_str)

try:
    from portfolio import build_signal, compute_weights, portfolio_summary
    from regime_switch import REGIME_CONFIG, check_regime
except ImportError:
    from .portfolio import build_signal, compute_weights, portfolio_summary
    from .regime_switch import REGIME_CONFIG, check_regime

from config import DATE_COL, PORTFOLIO_RESULT_DIR


def compute_regime_portfolio(
    pred_df: pd.DataFrame,
    pred_date: str | None = None,
    config: dict[str, Any] = REGIME_CONFIG,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    TFT prediction DataFrame으로 후보 포트폴리오 2개와 최종 regime 포트폴리오를 만든다.

    실행 순서:
    1. build_signal(pred_df)
    2. compute_weights(..., strategy="conservative")
    3. compute_weights(..., strategy="aggressive")
    4. check_regime(pred_df)
    5. selected_mode에 따라 최종 포트폴리오 선택
    """
    pred = pred_df.copy()

    signal_df = build_signal(pred)
    cons_df = compute_weights(signal_df, strategy="conservative")
    agg_df = compute_weights(signal_df, strategy="aggressive")
    regime_info = check_regime(pred, config=config)

    selected_mode = regime_info["selected_mode"]
    fallback_reason = None

    if selected_mode == "aggressive":
        if not agg_df.empty:
            final_df = agg_df.copy()
        elif not cons_df.empty:
            final_df = cons_df.copy()
            selected_mode = "conservative"
            fallback_reason = "aggressive_empty_fallback_to_conservative"
        else:
            final_df = agg_df.copy()
            fallback_reason = "both_candidates_empty"
    else:
        if not cons_df.empty:
            final_df = cons_df.copy()
        elif not agg_df.empty:
            final_df = agg_df.copy()
            selected_mode = "aggressive"
            fallback_reason = "conservative_empty_fallback_to_aggressive"
        else:
            final_df = cons_df.copy()
            fallback_reason = "both_candidates_empty"

    regime_info["selected_mode"] = selected_mode
    regime_info["fallback_reason"] = fallback_reason

    if pred_date is None and DATE_COL in pred.columns and not pred.empty:
        pred_date = str(pd.to_datetime(pred[DATE_COL].iloc[0]).date())

    final_df["pred_date"] = pred_date
    final_df["selected_mode"] = selected_mode
    final_df["regime"] = regime_info["regime"]
    final_df["regime_score"] = regime_info["regime_score"]
    final_df.attrs["conservative_candidate"] = cons_df
    final_df.attrs["aggressive_candidate"] = agg_df

    if verbose:
        portfolio_summary(cons_df, "conservative")
        portfolio_summary(agg_df, "aggressive")
        print("\n" + "-" * 55)
        print("[REGIME] 판단 결과")
        print(f"  regime        : {regime_info['regime']}")
        print(f"  selected_mode : {selected_mode}")
        print(f"  regime_score  : {regime_info['regime_score']}")
        if regime_info.get("fallback_reason"):
            print(f"  fallback      : {regime_info['fallback_reason']}")
        print("  triggered_conditions:")
        for key, value in regime_info["triggered_conditions"].items():
            print(f"    {key}: {value}")

    return {
        "final": final_df,
        "regime_info": regime_info,
        "conservative": cons_df,
        "aggressive": agg_df,
    }


def save_regime_results(
    final_df: pd.DataFrame,
    regime_info: dict[str, Any],
    pred_date: str,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Regime-switch 결과를 기존 save_results와 별도 파일명으로 저장한다.

    output_dir가 없으면 PORTFOLIO_RESULT_DIR / pred_date 아래에 저장한다.
    candidate DataFrame은 final_df.attrs에서 선택적으로 읽는다.
    """
    out_dir = Path(output_dir) if output_dir is not None else PORTFOLIO_RESULT_DIR / pred_date
    out_dir.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(out_dir / "portfolio_regime_final.csv", index=False)

    regime_row = {
        key: value
        for key, value in regime_info.items()
        if key != "triggered_conditions"
    }
    for key, value in regime_info.get("triggered_conditions", {}).items():
        regime_row[f"condition_{key}"] = value
    pd.DataFrame([regime_row]).to_csv(out_dir / "regime_log.csv", index=False)

    conservative = final_df.attrs.get("conservative_candidate")
    aggressive = final_df.attrs.get("aggressive_candidate")
    if isinstance(conservative, pd.DataFrame):
        conservative.to_csv(out_dir / "portfolio_conservative_candidate.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "portfolio_conservative_candidate.csv", index=False)

    if isinstance(aggressive, pd.DataFrame):
        aggressive.to_csv(out_dir / "portfolio_aggressive_candidate.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "portfolio_aggressive_candidate.csv", index=False)

    print(f"\n[save] regime portfolio 저장 → {out_dir}")
    return out_dir
