import pandas as pd


def determine_regime(
    signal_df: pd.DataFrame,
    recent_df: pd.DataFrame,
    pred_date: str,
    sentiment_threshold: float = -0.5,
) -> tuple[str, str]:
    market_return   = signal_df["signal_return"].mean()
    market_downside = signal_df["signal_downside"].mean()

    recent = recent_df[recent_df["date"] <= pd.Timestamp(pred_date)].tail(
        5 * len(signal_df["ticker"].unique())
    )
    recent_sentiment = recent.groupby("date")["domestic_mean"].mean().tail(5).mean()

    if recent_sentiment < sentiment_threshold:
        return "conservative", "sentiment_stoploss"

    if market_return > 0.003 and market_downside < 0.032:
        return "aggressive", "momentum"

    return "conservative", "defensive"
