# TFT 기반 ETF 포트폴리오 파이프라인

## 프로젝트 구조

```
Financial_project/
├── FINAL_DATA/
│   └── tft_data/
│       └── final_dataset_tft_ready.csv      ← 원본 데이터
└── geonho/
    └── src/
        ├── config.py                         ← 전역 상수 (경로, 컬럼, 하이퍼파라미터)
        ├── train.py                          ← 학습 진입점
        ├── predict.py                        ← 추론 + 포트폴리오 구성
        ├── backtest.py                       ← 롤링 백테스트
        ├── data/
        │   ├── preprocess.py                 ← 전처리 (log1p, z-score, scaler)
        │   └── dataset.py                    ← TimeSeriesDataSet + DataLoader
        ├── model/
        │   └── tft.py                        ← TFT 모델 + Trainer
        └── portfolio/
            └── portfolio.py                  ← 포트폴리오 엔진
```

---

## 실행 순서

```bash
pip install -r requirements.txt

# 1. 전처리 (단독 실행 가능, train.py에서도 자동 실행)
python src/data/preprocess.py

# 2. (선택) DataSet sanity check
python src/data/dataset.py

# 3. 학습
python src/train.py

# 4. 예측 + 포트폴리오 구성 (최신 날짜 기준)
python src/predict.py

# 4-1. 특정 날짜 예측
python src/predict.py --date 2025-06-02

# 4-2. 특정 체크포인트 지정
python src/predict.py --ckpt outputs/checkpoints/tft_v1_epoch30_0.0123.ckpt

# 5. 백테스트
python src/backtest.py --start 2025-01-02
```

---

## TFT 모델 설계

| 항목 | 값 |
|---|---|
| Encoder length (lookback) | 60일 |
| Prediction length | 1 (target_5d = 이미 5일 forward return) |
| Loss | QuantileLoss [q10, q50, q90] |
| hidden_size | 64 |
| attention_head_size | 4 |
| dropout | 0.1 |
| lr | 3e-4 |
| EarlyStopping patience | 10 |

---

## 포트폴리오 전략

### 공통 파이프라인

```
TFT 예측값 (q10, q50, q90)
    ↓
Signal 생성
  · signal_return   = q50           (기대 수익률)
  · signal_upside   = q90 - q50     (상방 잠재력)
  · signal_downside = q50 - q10     (하방 리스크)
  · signal_sharpe   = return / risk  (위험조정 수익)
    ↓
전략별 종목 선택 + 비중 계산
    ↓
포트폴리오 저장 (outputs/portfolio/YYYY-MM-DD/)
```

### 안정형 (Conservative)

- **목표**: 안정적 수익, 낮은 변동성
- 점수 = `sharpe × 0.7 + return × 0.3`
- 방어 섹터(defensive, rate_cash, sovereign) 보너스 ×1.3
- 방어 섹터 최소 비중 **50%** 강제
- 현금성 종목 최소 **10%** 유지
- 단일 종목 최대 **15%**

| 섹터 타입 | 배분 방향 |
|---|---|
| defensive / rate_cash / sovereign | ≥ 50% |
| growth (반도체, BigTech 등) | 나머지 |
| real_assets | 소량 허용 |

### 수익추구형 (Aggressive)

- **목표**: 초과 수익 추구
- 점수 = `return × 0.6 + upside × 0.4`
- 예측 수익률 **0.5%** 미만 종목 제외
- 성장 섹터(Semiconductor, BigTech, India 등) 집중
- 방어 섹터 최소 **10%**만 유지
- 단일 종목 최대 **25%**

---

## 산출물

```
outputs/
├── data/
│   └── final_dataset_preprocessed.csv
├── scalers/
│   ├── sector_domestic_mean.pkl
│   ├── usd_krw.pkl
│   └── ...
├── checkpoints/
│   └── tft_v1_epoch30_0.0123.ckpt
├── logs/
│   └── tft_v1/metrics.csv
└── portfolio/
    ├── 2025-06-02/
    │   ├── portfolio_conservative.csv   ← 안정형 비중
    │   ├── portfolio_aggressive.csv     ← 수익추구형 비중
    │   └── portfolio_comparison.csv     ← 두 전략 비교
    └── backtest/
        ├── backtest_returns.csv         ← 주별 수익률 기록
        └── backtest_summary.csv         ← 전략별 성과 (Sharpe, MDD 등)
```

---

## config.py 주요 파라미터 수정 가이드

```python
# 학습 기간 조정
TRAIN_END_DATE  = "2024-07-23"
VAL_START_DATE  = "2024-07-30"
TEST_START_DATE = "2025-01-02"

# 모델 크기 조정 (GPU 메모리 부족 시 hidden_size 줄이기)
HIDDEN_SIZE = 64    # 32 / 64 / 128

# 포트폴리오 파라미터 (config.py CONSERVATIVE / AGGRESSIVE dict)
CONSERVATIVE["top_n"] = 10          # 편입 종목 수
CONSERVATIVE["defensive_floor"] = 0.5  # 방어 섹터 최소 비중
```
