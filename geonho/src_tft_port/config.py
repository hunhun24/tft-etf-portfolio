"""
config.py
---------
프로젝트 전역 상수. preprocess.py / dataset.py / train.py 모두 여기서 import.

디렉토리 구조:
    Financial_project/
    ├── FINAL_DATA/
    │   └── tft_data/
    │       └── final_dataset_tft_ready.csv   ← 원본 데이터
    └── geonho/
        └── src/
            ├── config.py
            ├── train.py
            ├── predict.py
            ├── data/
            │   ├── preprocess.py
            │   └── dataset.py
            ├── model/
            │   └── tft.py
            └── outputs/
                ├── checkpoints/
                ├── logs/
                ├── scalers/
                └── data/
                    └── final_dataset_preprocessed.csv
"""

import os

# ── 기준 경로 ──────────────────────────────────────────────────────────────────
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))       # src/
_ROOT_DIR = os.path.join(_SRC_DIR, "../..")        # Financial_project/
_OUT_DIR = os.path.join(_SRC_DIR, "outputs")                # src/outputs/

# ── 데이터 경로 ────────────────────────────────────────────────────────────────
RAW_DATA_PATH       = os.path.join(_ROOT_DIR, "FINAL_DATA", "tft_data", "final_dataset_tft_ready.csv")
PROCESSED_DATA_PATH = os.path.join(_OUT_DIR, "data", "final_dataset_preprocessed.csv")
SCALER_DIR          = os.path.join(_OUT_DIR, "scalers")
CHECKPOINT_DIR      = os.path.join(_OUT_DIR, "checkpoints")
LOG_DIR             = os.path.join(_OUT_DIR, "logs")

# ── Train/Val Split ────────────────────────────────────────────────────────────
# target_5d(t) = log(end_{t+5} / end_t) 이므로 train 마지막 row의 target이
# validation 구간 가격을 참조하지 않도록 TRAIN_END_DATE와 VAL_START_DATE를 분리.
#
# TRAIN_END_DATE = "2024-07-23"  → train feature row 마지막 날짜
# VAL_START_DATE = "2024-07-30"  → validation prediction 시작 날짜
#
# 본 모델은 t일 장 마감 이후 관측 가능한 가격, AUM, 뉴스 감성, 매크로 변수를
# 이용하여 t+5 거래일 누적 로그수익률을 예측한다.
# → preprocess.py, dataset.py 모두 이 두 날짜만 참조할 것

TRAIN_END_DATE = "2024-07-23"   # train feature row 마지막 날짜
VAL_START_DATE = "2024-07-30"   # validation prediction 시작 날짜

# ── 컬럼 정의 ──────────────────────────────────────────────────────────────────
TARGET_COL   = "target_5d"
GROUP_COL    = "ticker"
SECTOR_COL   = "sector"
TIME_IDX_COL = "time_idx"

# log1p 변환 대상 (원본 컬럼명)
LOG1P_COLS = ["AUM", "domestic_count", "global_count"]

# sector별 z-score 대상 (log1p 변환 후 컬럼명 포함)
SECTOR_ZSCORE_COLS = [
    "domestic_mean",
    "domestic_std",
    "domestic_count_log",
    "global_mean",
    "global_std",
    "global_count_log",
]

# 전체 StandardScaler 대상 (모든 ticker 공통 macro 변수)
MACRO_SCALE_COLS = ["usd_krw", "vix_close"]

# 실험 feature 버전 (변경 시 이 값만 수정)
FEATURE_VERSION = "price_macro_sentiment"

# TimeSeriesDataSet time-varying (known은 현재 미사용, 추후 calendar features 추가 예정)
TIME_VARYING_KNOWN_REALS        = []
TIME_VARYING_KNOWN_CATEGORICALS = []

# TimeSeriesDataSet time-varying unknown reals (전처리 완료 기준 컬럼명)

TIME_VARYING_UNKNOWN_REALS = [
    "AUM_log",
    "usd_krw",
    "vix_close",
    "domestic_mean",
    "domestic_std",
    "domestic_count_log",
    "global_mean",
    "global_std",
    "global_count_log",
]


# ── 모델 하이퍼파라미터 (수정)────────────────────────────────────────────────────────
MAX_ENCODER_LENGTH    = 60
MAX_PREDICTION_LENGTH = 1
BATCH_SIZE            = 64
LEARNING_RATE         = 1e-4

# ── TFT 아키텍처 ──────────────────────────────────────────────────────────────
HIDDEN_SIZE            = 64
ATTENTION_HEAD_SIZE    = 2
DROPOUT                = 0.2
HIDDEN_CONTINUOUS_SIZE = 32

# ── Trainer ───────────────────────────────────────────────────────────────────
MAX_EPOCHS        = 50
PATIENCE          = 5
gradient_clip_val = 0.5

# ── 포트폴리오 설정 ───────────────────────────────────────────────────────────
import pathlib as _pathlib

DATE_COL             = "date"
NUM_WORKERS          = 0
TEST_START_DATE      = "2025-01-02"
PORTFOLIO_RESULT_DIR = _pathlib.Path(_OUT_DIR) / "portfolio"

# DB GAPS 전체 제약
MAX_SINGLE_TICKER = 0.20   # 개별 종목 상한 20%
MAX_RISK_ASSET    = 0.70   # 위험자산 합산 상한 70%

# 섹터 분류 (포트폴리오 점수 계산용)
DEFENSIVE_SECTORS   = {"defensive", "rate_cash", "sovereign_kr", "sovereign_us"}
GROWTH_SECTORS      = {"global_macro", "domestic_core", "Semiconductor",
                       "SecondaryBattery", "BigTech", "IndiaEquity", "ChinaEquity"}
ALTERNATIVE_SECTORS = {"real_assets"}

# 포트폴리오 전략 파라미터
CONSERVATIVE = dict(
    max_single_weight  = 0.15,
    defensive_floor    = 0.50,
    top_n              = 10,
    return_threshold   = 0.0,
    risk_free_weight   = 0.10,
    rebalance_freq     = "weekly",
)
AGGRESSIVE = dict(
    max_single_weight  = 0.20,
    defensive_floor    = 0.10,
    top_n              = 8,
    return_threshold   = 0.005,
    risk_free_weight   = 0.0,
    rebalance_freq     = "weekly",
)

# 동일 종목 중복 ticker (A 접두사 없는 버전) — 포트폴리오 후보에서 제외
DUPLICATE_TICKERS = {"157450"}

# DB GAPS 자산군 매핑 — ticker 단위 (정확)
# (자산구분, 세부자산명, 세부자산별 상한)
TICKER_ASSET_MAP = {
    # BigTech
    "A381170": ("위험자산", "해외주식_섹터",         0.10),

    # ChinaEquity
    "A192090": ("위험자산", "해외주식_지수",         0.30),

    # IndiaEquity
    "A453870": ("위험자산", "해외주식_지수",         0.30),

    # SecondaryBattery
    "A305720": ("위험자산", "국내주식_섹터",         0.15),

    # Semiconductor
    "A381180": ("위험자산", "해외주식_섹터",         0.10),  # 미국 반도체
    "A091160": ("위험자산", "국내주식_섹터",         0.15),  # 국내 반도체

    # defensive (섹터 내 혼재 → ticker별 분리)
    "161510":  ("위험자산", "국내주식_섹터",         0.15),  # PLUS 고배당주
    "174350":  ("위험자산", "국내주식_섹터",         0.15),  # TIGER 로우볼
    "182480":  ("위험자산", "해외주식_섹터",         0.10),  # TIGER 미국MSCI리츠
    "402970":  ("위험자산", "해외주식_지수",         0.30),  # ACE 미국배당다우존스

    # domestic_core
    "229200":  ("위험자산", "국내주식_지수",         0.30),  # KODEX 코스닥150
    "69500":   ("위험자산", "국내주식_지수",         0.30),  # KODEX 200

    # global_macro
    "133690":  ("위험자산", "해외주식_지수",         0.30),  # TIGER 미국나스닥100
    "360750":  ("위험자산", "해외주식_지수",         0.30),  # TIGER 미국S&P500
    "251350":  ("위험자산", "해외주식_지수",         0.30),  # KODEX 선진국MSCI
    "241180":  ("위험자산", "해외주식_지수",         0.30),  # TIGER 일본니케이225

    # rate_cash
    "423160":  ("안전자산", "금리연계형_초단기채권", 0.50),  # KODEX KOFR금리액티브
    #"157450":  ("안전자산", "금리연계형_초단기채권", 0.50),  # TIGER 단기통안채
    "459580":  ("안전자산", "금리연계형_초단기채권", 0.50),  # KODEX CD금리액티브

    # real_assets
    "A261220": ("위험자산", "FX및원자재",            0.20),  # KODEX WTI원유선물
    "A261240": ("위험자산", "FX및원자재",            0.20),  # KODEX 미국달러선물
    "A411060": ("위험자산", "FX및원자재",            0.20),  # ACE KRX금현물
    "A144600": ("위험자산", "FX및원자재",            0.20),  # KODEX 은선물

    # sovereign_kr
    "A148070": ("안전자산", "국내채권_종합",         0.50),  # KIWOOM 국고채10년
    "A157450": ("안전자산", "국내채권_종합", 0.50),  # TIGER 단기통안채
    "A439870": ("안전자산", "국내채권_종합",         0.50),  # KODEX 국고채30년액티브
    "157450":  ("안전자산", "국내채권_종합", 0.50),  # TIGER 단기통안채

    # sovereign_us
    "A453850": ("안전자산", "해외채권_종합",         0.50),  # ACE 미국30년국채
    "A305080": ("안전자산", "해외채권_종합",         0.50),  # TIGER 미국채10년선물
    "A329750": ("안전자산", "해외채권_종합",         0.50),  # TIGER 미국달러단기채권
}