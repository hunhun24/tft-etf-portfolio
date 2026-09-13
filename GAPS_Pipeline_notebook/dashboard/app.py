"""
dashboard/app.py
Streamlit 대시보드 — 포트폴리오 현황 + 예측 + 감성 + 거시지표

실행:
    streamlit run dashboard/app.py
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import glob
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
from config import TFT_SRC_DIR, FINAL_DATASET, RAW_DIR, MERGED_DIR

st.set_page_config(
    page_title="GAPS ETF 파이프라인 대시보드",
    page_icon="📊",
    layout="wide"
)

# ── 데이터 로드 헬퍼 ─────────────────────────────────────────
@st.cache_data(ttl=300)
def load_portfolio(strategy="conservative"):
    files = sorted(glob.glob(str(TFT_SRC_DIR / "outputs/portfolio/*/portfolio_*.csv")))
    # strategy 포함된 파일만
    files = [f for f in files if strategy in f]
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[-1])

@st.cache_data(ttl=300)
def load_predictions():
    files = sorted(glob.glob(str(TFT_SRC_DIR / "outputs/portfolio/*/predictions.csv")))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[-1])

@st.cache_data(ttl=300)
def load_dataset(n_days=90):
    if not FINAL_DATASET.exists():
        return pd.DataFrame()
    df = pd.read_csv(FINAL_DATASET, parse_dates=["date"])
    cutoff = df["date"].max() - timedelta(days=n_days)
    return df[df["date"] >= cutoff]

@st.cache_data(ttl=300)
def load_macro():
    macro_path = RAW_DIR / "macro.csv"
    if not macro_path.exists():
        return pd.DataFrame()
    return pd.read_csv(macro_path, parse_dates=["date"]).tail(252)

@st.cache_data(ttl=300)
def load_sentiment():
    sent_path = MERGED_DIR / "sentiment_daily.csv"
    if not sent_path.exists():
        return pd.DataFrame()
    return pd.read_csv(sent_path, parse_dates=["date"]).tail(252)

def get_latest_portfolio_date():
    dirs = sorted(glob.glob(str(TFT_SRC_DIR / "outputs/portfolio/*")))
    return Path(dirs[-1]).name if dirs else "없음"


# ══════════════════════════════════════════════════════════════
# 헤더
# ══════════════════════════════════════════════════════════════
st.title("📊 GAPS ETF 주간 파이프라인 대시보드")
latest_date = get_latest_portfolio_date()
st.caption(f"최근 예측 기준일: **{latest_date}** | 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# 새로고침 버튼
if st.button("🔄 데이터 새로고침"):
    st.cache_data.clear()
    st.rerun()

st.divider()

# ══════════════════════════════════════════════════════════════
# 탭 구성
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 포트폴리오", "🔮 예측", "💬 감성 트렌드", "🌍 거시지표", "📋 데이터 현황"
])


# ── Tab 1: 포트폴리오 ─────────────────────────────────────────
with tab1:
    st.subheader("이번 주 추천 포트폴리오")

    col1, col2 = st.columns(2)

    for col, strategy, label in [
        (col1, "conservative", "🛡️ Conservative (안정형)"),
        (col2, "aggressive",   "⚡ Aggressive (수익추구형)"),
    ]:
        with col:
            st.markdown(f"### {label}")
            port = load_portfolio(strategy)
            if port.empty:
                st.info("포트폴리오 데이터 없음")
                continue

            # 파이차트
            fig = px.pie(
                port.head(10), values="weight", names="name",
                title=f"자산 배분 (상위 10개)",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

            # 테이블
            display = port[["ticker","name","sector","weight"]].copy()
            display["비중"] = display["weight"].map("{:.1%}".format)
            display = display.drop(columns=["weight"]).rename(
                columns={"ticker":"티커","name":"ETF명","sector":"섹터"}
            )
            st.dataframe(display, use_container_width=True, hide_index=True)

            # 섹터별 배분
            sector_w = port.groupby("sector")["weight"].sum().reset_index()
            sector_w["비중"] = sector_w["weight"].map("{:.1%}".format)
            st.markdown("**섹터별 배분**")
            fig2 = px.bar(
                sector_w.sort_values("weight", ascending=True),
                x="weight", y="sector", orientation="h",
                labels={"weight":"비중","sector":"섹터"},
                color="weight",
                color_continuous_scale="Blues"
            )
            fig2.update_layout(height=300, showlegend=False,
                               xaxis_tickformat=".0%")
            st.plotly_chart(fig2, use_container_width=True)


# ── Tab 2: 예측 ───────────────────────────────────────────────
with tab2:
    st.subheader("5거래일 수익률 예측")
    pred = load_predictions()

    if pred.empty:
        st.info("예측 데이터 없음")
    else:
        # 상위/하위 종목
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📈 상위 10 종목 (예측 수익률)**")
            top10 = pred.nlargest(10, "signal_return")[
                ["ticker","name","sector","signal_return","signal_sharpe"]
            ].copy()
            top10["수익률"] = top10["signal_return"].map("{:+.3%}".format)
            top10["샤프"]   = top10["signal_sharpe"].map("{:.3f}".format)
            st.dataframe(
                top10[["ticker","name","sector","수익률","샤프"]].rename(
                    columns={"ticker":"티커","name":"ETF명","sector":"섹터"}
                ),
                use_container_width=True, hide_index=True
            )

        with col2:
            st.markdown("**📉 하위 10 종목 (예측 수익률)**")
            bot10 = pred.nsmallest(10, "signal_return")[
                ["ticker","name","sector","signal_return","signal_sharpe"]
            ].copy()
            bot10["수익률"] = bot10["signal_return"].map("{:+.3%}".format)
            bot10["샤프"]   = bot10["signal_sharpe"].map("{:.3f}".format)
            st.dataframe(
                bot10[["ticker","name","sector","수익률","샤프"]].rename(
                    columns={"ticker":"티커","name":"ETF명","sector":"섹터"}
                ),
                use_container_width=True, hide_index=True
            )

        # 섹터별 평균 예측
        st.markdown("**섹터별 평균 예측 수익률**")
        sector_pred = pred.groupby("sector")["signal_return"].mean().reset_index()
        sector_pred = sector_pred.sort_values("signal_return")
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in sector_pred["signal_return"]]
        fig = go.Figure(go.Bar(
            x=sector_pred["signal_return"],
            y=sector_pred["sector"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2%}" for v in sector_pred["signal_return"]],
            textposition="outside"
        ))
        fig.update_layout(height=350, xaxis_tickformat=".1%",
                          xaxis_title="예측 수익률", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        # 분위수 불확실성
        if "pred_q10" in pred.columns:
            st.markdown("**예측 구간 (q10 ~ q90) — 상위 20개**")
            top20 = pred.nlargest(20, "signal_return").copy()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=top20["name"], y=top20["pred_q90"],
                mode="markers", name="Q90", marker=dict(color="green", size=6)
            ))
            fig2.add_trace(go.Scatter(
                x=top20["name"], y=top20["pred_q50"],
                mode="markers", name="Q50", marker=dict(color="blue", size=8)
            ))
            fig2.add_trace(go.Scatter(
                x=top20["name"], y=top20["pred_q10"],
                mode="markers", name="Q10", marker=dict(color="red", size=6)
            ))
            fig2.update_layout(height=350, xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)


# ── Tab 3: 감성 트렌드 ────────────────────────────────────────
with tab3:
    st.subheader("섹터별 감성 트렌드")
    sent = load_sentiment()

    if sent.empty:
        st.info("감성 데이터 없음")
    else:
        sent["date"] = pd.to_datetime(sent["date"])
        sectors = sorted(sent["sector"].unique())

        sel_sectors = st.multiselect(
            "섹터 선택", sectors, default=sectors[:4]
        )
        metric = st.radio("지표", ["domestic_mean","global_mean","domestic_count"], horizontal=True)

        if sel_sectors:
            fig = go.Figure()
            for sector in sel_sectors:
                sub = sent[sent["sector"] == sector].sort_values("date")
                fig.add_trace(go.Scatter(
                    x=sub["date"], y=sub[metric],
                    mode="lines", name=sector
                ))
            fig.update_layout(
                height=400,
                title=f"{metric} 추이",
                xaxis_title="날짜",
                yaxis_title=metric,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

        # 최근 1주 감성 히트맵
        st.markdown("**최근 1주 감성 현황**")
        recent = sent[sent["date"] >= sent["date"].max() - pd.Timedelta(days=7)]
        pivot  = recent.groupby("sector")[["domestic_mean","global_mean"]].mean().round(3)
        st.dataframe(
            pivot.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
            use_container_width=True
        )


# ── Tab 4: 거시지표 ───────────────────────────────────────────
with tab4:
    st.subheader("거시지표 모니터링")
    macro = load_macro()

    if macro.empty:
        st.info("거시 데이터 없음")
    else:
        macro["date"] = pd.to_datetime(macro["date"])
        latest = macro.iloc[-1]

        # 최신 수치 카드
        m1, m2, m3, m4, m5 = st.columns(5)
        prev = macro.iloc[-2] if len(macro) > 1 else latest
        for col, label, fmt, val, pval in [
            (m1, "VIX",       "{:.2f}",  latest.get("vix_close",0),   prev.get("vix_close",0)),
            (m2, "USD/KRW",   "{:.1f}원", latest.get("usd_krw",0),    prev.get("usd_krw",0)),
            (m3, "미국 10Y",   "{:.3f}%",  latest.get("us_10y",0),     prev.get("us_10y",0)),
            (m4, "한국 10Y",   "{:.3f}%",  latest.get("kr_10y",0),     prev.get("kr_10y",0)),
            (m5, "금리차(KR-US)","{:.3f}%p",latest.get("rate_spread",0),prev.get("rate_spread",0)),
        ]:
            delta = float(val) - float(pval) if pval and not pd.isna(pval) else 0
            col.metric(label, fmt.format(float(val)), f"{delta:+.3f}")

        st.divider()

        # 차트
        indicators = {
            "VIX": "vix_close",
            "USD/KRW 환율": "usd_krw",
            "미국 10년물 금리": "us_10y",
            "금리차 (KR-US)": "rate_spread",
        }
        sel = st.selectbox("지표 선택", list(indicators.keys()))
        col_name = indicators[sel]

        if col_name in macro.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=macro["date"], y=macro[col_name],
                mode="lines", name=sel,
                fill="tozeroy",
                line=dict(color="#1f77b4", width=2)
            ))
            fig.update_layout(
                height=400, title=sel,
                xaxis_title="날짜", yaxis_title=sel,
                hovermode="x"
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Tab 5: 데이터 현황 ────────────────────────────────────────
with tab5:
    st.subheader("데이터 파이프라인 현황")

    df = load_dataset(n_days=30)

    if df.empty:
        st.info("데이터 없음")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 ticker", df["ticker"].nunique())
        c2.metric("최신 날짜", df["date"].max().strftime("%Y-%m-%d"))
        c3.metric("전체 행수", f"{len(df):,}")
        c4.metric("감성 채움률",
                  f"{(df['domestic_mean']!=0).mean():.1%}")

        # 섹터별 데이터 현황
        st.markdown("**섹터별 현황 (최근 30일)**")
        sector_stat = df.groupby("sector").agg(
            ticker수=("ticker","nunique"),
            최신날짜=("date","max"),
            감성채움률=("domestic_mean", lambda x: f"{(x!=0).mean():.0%}")
        ).reset_index()
        st.dataframe(sector_stat, use_container_width=True, hide_index=True)

        # 종목별 최신 가격
        st.markdown("**종목별 최신 데이터**")
        latest_price = df.sort_values("date").groupby("ticker").last().reset_index()
        st.dataframe(
            latest_price[["ticker","name","sector","end","date"]]
            .rename(columns={"ticker":"티커","name":"ETF명","sector":"섹터",
                             "end":"최신종가","date":"날짜"}),
            use_container_width=True, hide_index=True
        )
