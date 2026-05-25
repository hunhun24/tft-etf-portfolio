# ETF Nowcasting
**TFT 기반 5거래일 ETF 수익률 예측 · 레짐 스위치 포트폴리오**

---

## 프로젝트 개요

국내·해외 ETF 29개를 대상으로 Temporal Fusion Transformer(TFT)를 이용해 5거래일 수익률을 예측하고, 예측 분포(q10/q50/q90)를 기반으로 시장 레짐을 판단해 DB GAPS 제약 안에서 운용 가능한 ETF 포트폴리오를 구성하는 시스템입니다.

```
데이터 수집 → 감성 분석 → 패널 데이터 구성 → TFT 학습 → 예측 → 레짐 판단 → 포트폴리오 구성 → 백테스트
```

---

## 데이터

- **기간**: 2019-01-02 ~ 2025-12-22 (약 6년)
- **ETF 수**: 29개 (국내·해외 지수, 섹터, 채권, 원자재, FX)
- **Train / Val split**: Train ≤ 2024-07-23 / Val ≥ 2024-07-30 (5거래일 gap)
- **패널 구조**: (ETF, date) 단위 · 31,198 / 9,831 행

### 피처 구성

| 구분 | 피처 |
|------|------|
| Price | AUM_log, target_5d (5일 log return) |
| Macro | usd_krw, vix_close |
| Sentiment | domestic_{mean,std,count_log}, global_{mean,std,count_log} |

- **국내 감성**: 매일경제 크롤링 → KR-FinBERT-SC
- **해외 감성**: NYT 크롤링 → FinBERT
- 뉴스 감성 +1 영업일 shift 적용 (leakage 방지)
- 종가 `end` 포함 시 가격 수준 차이에 과도하게 의존 → 최종 피처셋에서 제외

### 8개 섹터

| 섹터 | 구성 예시 |
|------|-----------|
| Domestic Core | KOSPI 200, KOSDAQ 150 |
| Global Macro | S&P500, 나스닥100, 유로스탁스, 니케이 |
| Emerging & Asia | 차이나CSI300, 인도Nifty50, 베트남VN30 |
| Tech & Future | 필라반도체, AI인프라, 빅테크7, 2차전지 |
| Defensive | 고배당, 미배당다우, 헬스케어, 로우볼, REITs |
| Rates & Cash | CD금리, KOFR, MMF, 초단기채권 |
| Sovereign Debt | 한국 국고채(단기/장기), 미국채(10·30년) |
| Real Assets & FX | 금현물, 은선물, WTI원유, USD 선물/인버스 |

---

## 모델

### Temporal Fusion Transformer (TFT)

- **라이브러리**: pytorch-forecasting
- **입력**: Past 60days (AUM_log, usd_krw, vix_close, 감성 피처) + Static (ticker, sector)
- **출력**: Quantile Loss → q10 / q50 / q90 · target_5d


---

## 포트폴리오 전략

### Signal 정의

```python
signal_return   = pred_q50
signal_upside   = pred_q90 - pred_q50
signal_downside = pred_q50 - pred_q10
signal_sharpe   = signal_return / signal_downside
```

### 레짐 판단 (매 리밸런싱마다)

```
① 최근 5일 감성 < -0.5          →  Conservative  (sentiment stoploss)
② 시장 수익률 > 0.3%
   AND 하방위험 < 0.032           →  Aggressive    (momentum)
③ 그 외                          →  Conservative  (defensive)
```

### Conservative (Sharpe 가중)
- score = 0.7 × signal_sharpe + 0.3 × signal_return
- 방어섹터 ×1.3 가중 · softmax 비중
- 방어자산 floor 50% · 현금성 floor 10% · DB GAPS 사후 적용

### Aggressive (Greedy Knapsack)
- score = 0.6 × signal_return + 0.4 × signal_upside
- score 기준 정렬 후 DB GAPS cap을 처음부터 반영하며 배정
- 정규화 금지 (기존 softmax cap 순환 문제 해결)

### DB GAPS 편입비중 상한

| 자산 | 상한 | 세부자산 | 상한 |
|------|------|----------|------|
| 위험자산 합계 | 70% | 국내주식_지수 | 30% |
| | | 해외주식_지수 | 30% |
| | | 국내주식_섹터 | 15% |
| | | 해외주식_섹터 | 10% |
| | | FX 및 원자재 | 20% |
| 안전자산 합계 | 100% | 국내채권_종합 | 50% |
| | | 해외채권_종합 | 50% |
| | | 금리연계형/초단기채권 | 50% |
| 개별 종목 | 20% | | |

---

## 백테스트 결과

**기간**: 2025-01-02 ~ 2025-12-22 · 52주 weekly 리밸런싱

| 구분 | 전략 | 누적수익 | Sharpe | MDD |
|------|------|----------|--------|-----|
| Baseline | SARIMAX top-10 | +24.96% | 1.977 | -12.47% |
| Baseline | XGBoost top-10 | +25.03% | 1.597 | -14.83% |
| TFT ablation | TFT equal_weight | +26.39% | 2.367 | -10.32% |
| **Final ★** | **TFT regime** | **+31.67%** | **2.584** | **-7.58%** |

> TFT regime은 누적수익·Sharpe·MDD 모든 지표에서 baseline 우위.
> 수익률 격차(+6%p)는 예측 정확도뿐 아니라 quantile 기반 risk-aware 포트폴리오 로직의 효과.
> MDD −7.58%는 SARIMAX/XGBoost baseline 대비 약 40~50% 낮은 수준.

### 2025년 레짐 분포 (52회)

| 레짐 | 횟수 | 비율 |
|------|------|------|
| conservative (defensive) | 30 | 57.7% |
| aggressive (momentum) | 18 | 34.6% |
| conservative (sentiment stoploss) | 4 | 7.7% |

---

## 베이스라인 비교

### 예측 정확도 (val 기준)

| 모델 | MAE | RMSE |
|------|-----|------|
| Naive (shift 5) | ~0.026 | ~0.041 |
| SARIMAX (auto_arima) | 0.019677 | 0.031029 |
| XGBoost (Optuna) | 0.020410 | 0.031587 |
| **TFT (final)** | **0.01887** | **0.02569** |

> XGBoost val 성능은 Optuna가 val에서 직접 튜닝한 낙관적 추정치임.

---

## 프로젝트 구조

```
Financial_project/
├── FINAL_DATA/
│   └── tft_data/
│       └── final_dataset_tft_ready.csv      # 최종 학습 데이터
├── src_tft_port_regime/
│   ├── config.py                            # 하이퍼파라미터, DB GAPS 설정
│   ├── train.py                             # TFT 학습
│   ├── predict_regime.py                    # 레짐 판단 + 포트폴리오 구성
│   ├── backtest_regime.py                   # 백테스트
│   ├── portfolio/
│   │   ├── portfolio.py                     # 포트폴리오 구성 엔진
│   │   └── regime_rule.py                   # 레짐 판단 공통 모듈
│   └── outputs/
│       ├── checkpoints/                     # 모델 체크포인트
│       └── portfolio/
│           ├── {date}/                      # 날짜별 포트폴리오 결과
│           └── backtest_regime/
│               ├── backtest_returns.csv     # 전략별 주간 수익률
│               └── backtest_summary.csv     # 성과 요약
├── baseline/
│   ├── sarimax_baseline.py                  # SARIMAX 예측
│   ├── sarimax_backtest.py                  # SARIMAX 백테스트
│   ├── xgboost_baseline.py                  # XGBoost 예측
│   └── xgboost_backtest.py                  # XGBoost 백테스트
└── dashboard_v2_real.html                    # 대시보드 (단일 HTML)
```

---

## 실행 방법

### 환경 설정

```bash
pip install pytorch-forecasting pytorch-lightning pmdarima xgboost optuna
```

### TFT 학습

```bash
python src_tft_port_regime/train.py
```

### 예측 및 포트폴리오 구성

```bash
python src_tft_port_regime/predict_regime.py \
  --ckpt src_tft_port_regime/outputs/checkpoints/tft-no_end_macro_sentiment-epoch=06-val_loss=0.0125.ckpt \
  --date 2025-05-19
```

### 백테스트

```bash
python src_tft_port_regime/backtest_regime.py \
  --ckpt src_tft_port_regime/outputs/checkpoints/tft-no_end_macro_sentiment-epoch=06-val_loss=0.0125.ckpt \
  --start 2025-01-02
```

### 2026년 OOS 검증 (모델 재학습 없음)

```bash
# 1. final_dataset_tft_ready.csv에 2026년 데이터 append
# 2. 전처리 재실행
python src_tft_port_regime/data/preprocess.py

# 3. OOS 백테스트
python src_tft_port_regime/backtest_regime.py \
  --ckpt src_tft_port_regime/outputs/checkpoints/tft-no_end_macro_sentiment-epoch=06-val_loss=0.0125.ckpt \
  --start 2026-01-02
```

> scaler는 TRAIN_END_DATE(2024-07-23) 이전 구간으로만 fit — 2026 데이터 추가 시 leakage 없음.

---

## VSN 해석

| 피처 | 중요도 |
|------|--------|
| usd_krw | 41.8% |
| vix_close | 17.5% |
| AUM_log | 12.6% |
| global_mean | 8.7% |
| domestic_mean | 7.5% |
| global_std | 5.0% |
| domestic_count_log | 3.7% |

- `usd_krw`가 압도적 1위 — train-val 구간에서 PSI 9.79로 가장 큰 covariate shift 확인
- Attention은 최근 10거래일에 집중 (단기 모멘텀 구조 학습)

---

## 참고

- **모델**: [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)
- **라이브러리**: [pytorch-forecasting](https://pytorch-forecasting.readthedocs.io/)
- **감성 모델**: [snunlp/KR-FinBert-SC](https://huggingface.co/snunlp/KR-FinBert-SC)