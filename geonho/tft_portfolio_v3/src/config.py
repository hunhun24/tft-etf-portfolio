"""
config.py — 전역 상수 관리
"""

from pathlib import Path

# ──────────────────────────────────────────
# 경로
# ──────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "FINAL_DATA" / "tft_data"
RAW_CSV  = DATA_DIR / "final_dataset_tft_ready.csv"

SRC_DIR     = Path(__file__).resolve().parent
OUTPUTS_DIR = SRC_DIR.parent / "outputs"

PREPROCESSED_CSV     = OUTPUTS_DIR / "data"      / "final_dataset_preprocessed.csv"
SCALER_DIR           = OUTPUTS_DIR / "scalers"
CHECKPOINT_DIR       = OUTPUTS_DIR / "checkpoints"
LOG_DIR              = OUTPUTS_DIR / "logs"
PORTFOLIO_RESULT_DIR = OUTPUTS_DIR / "portfolio"

for d in [PREPROCESSED_CSV.parent, SCALER_DIR, CHECKPOINT_DIR, LOG_DIR, PORTFOLIO_RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────
# Train / Val Split
# ──────────────────────────────────────────
TRAIN_END_DATE  = "2024-07-23"
VAL_START_DATE  = "2024-07-30"
TEST_START_DATE = "2025-01-02"

# ──────────────────────────────────────────
# 컬럼명
# ──────────────────────────────────────────
TARGET_COL   = "target_5d"
TIME_IDX_COL = "time_idx"
GROUP_COL    = "ticker"
DATE_COL     = "date"
SECTOR_COL   = "sector"

LOG1P_COLS          = ["AUM", "domestic_count", "global_count"]
SENTIMENT_COLS      = ["domestic_mean", "domestic_std", "global_mean", "global_std"]
STANDARD_SCALE_COLS = ["usd_krw", "vix_close"]

TIME_VARYING_KNOWN_CATS    = []
TIME_VARYING_UNKNOWN_REALS = [
    TARGET_COL, "end",
    "domestic_mean", "domestic_std", "domestic_count",
    "global_mean",   "global_std",   "global_count",
    "AUM_log",
]
TIME_VARYING_KNOWN_REALS = ["usd_krw", "vix_close"]
STATIC_CATS              = [SECTOR_COL]
STATIC_REALS             = []

# ──────────────────────────────────────────
# TFT 파라미터
# ──────────────────────────────────────────
ENCODER_LENGTH         = 60
MAX_PREDICTION_LENGTH  = 1
HIDDEN_SIZE            = 64
ATTENTION_HEAD_SIZE    = 4
DROPOUT                = 0.1
HIDDEN_CONTINUOUS_SIZE = 32
LEARNING_RATE          = 3e-4
QUANTILES              = [0.1, 0.5, 0.9]
MAX_EPOCHS             = 100
PATIENCE               = 10
BATCH_SIZE             = 128
NUM_WORKERS            = 0
GRADIENT_CLIP          = 0.1

# ──────────────────────────────────────────
# 섹터 분류 (포트폴리오 점수 계산용)
# ──────────────────────────────────────────
DEFENSIVE_SECTORS   = {"defensive", "rate_cash", "sovereign_kr", "sovereign_us"}
GROWTH_SECTORS      = {"global_macro", "domestic_core", "Semiconductor",
                       "SecondaryBattery", "BigTech", "IndiaEquity", "ChinaEquity"}
ALTERNATIVE_SECTORS = {"real_assets"}

# ──────────────────────────────────────────
# DB GAPS 자산군 매핑 — ticker 단위 (정확)
# (자산구분, 세부자산명, 세부자산별 상한)
#
# [매핑 근거]
# BigTech
#   A381170 TIGER 미국테크TOP10  → 미국 테크 섹터 ETF    → 해외주식_섹터
# ChinaEquity
#   A192090 TIGER 차이나CSI300  → CSI300 지수 추종       → 해외주식_지수
# IndiaEquity
#   A453870 TIGER 인도니프티50  → Nifty50 지수 추종      → 해외주식_지수
# SecondaryBattery
#   A305720 KODEX 2차전지산업   → 국내 2차전지 섹터       → 국내주식_섹터
# Semiconductor
#   A381180 TIGER 미국필라델피아반도체  → 미국 반도체 섹터 → 해외주식_섹터
#   A091160 KODEX 반도체              → 국내 반도체 섹터  → 국내주식_섹터
# defensive (섹터 내 혼재 — ticker별 분리)
#   161510  PLUS 고배당주         → 국내 배당주 섹터      → 국내주식_섹터
#   174350  TIGER 로우볼          → 국내 저변동성 섹터    → 국내주식_섹터
#   182480  TIGER 미국MSCI리츠   → 미국 리츠 섹터        → 해외주식_섹터
#   402970  ACE 미국배당다우존스  → 미국 배당 지수        → 해외주식_지수
# domestic_core
#   229200  KODEX 코스닥150       → 국내 주식 지수        → 국내주식_지수
#   69500   KODEX 200            → 국내 주식 지수        → 국내주식_지수
# global_macro
#   133690  TIGER 미국나스닥100   → 미국 주식 지수        → 해외주식_지수
#   360750  TIGER 미국S&P500     → 미국 주식 지수        → 해외주식_지수
#   251350  KODEX 선진국MSCI     → 선진국 주식 지수      → 해외주식_지수
#   241180  TIGER 일본니케이225  → 일본 주식 지수        → 해외주식_지수
# rate_cash
#   423160  KODEX KOFR금리액티브  → 초단기 금리연계       → 금리연계형_초단기채권
#   157450  TIGER 단기통안채      → 초단기 국내채권       → 금리연계형_초단기채권
#   459580  KODEX CD금리액티브    → 초단기 금리연계       → 금리연계형_초단기채권
# real_assets
#   A261220 KODEX WTI원유선물     → 원자재 선물          → FX및원자재
#   A261240 KODEX 미국달러선물    → FX 선물              → FX및원자재
#   A411060 ACE KRX금현물        → 금 현물              → FX및원자재
#   A144600 KODEX 은선물         → 귀금속 선물          → FX및원자재
# sovereign_kr
#   A148070 KIWOOM 국고채10년    → 국내 장기 국채        → 국내채권_종합
#   A157450 TIGER 단기통안채     → 초단기 국내채권       → 금리연계형_초단기채권
#   A439870 KODEX 국고채30년액티브 → 국내 초장기 국채    → 국내채권_종합
# sovereign_us
#   A453850 ACE 미국30년국채     → 미국 장기 국채        → 해외채권_종합
#   A305080 TIGER 미국채10년선물 → 미국 중기 국채        → 해외채권_종합
#   A329750 TIGER 미국달러단기채권 → 미국 단기채권        → 해외채권_종합
# ──────────────────────────────────────────────────────────────
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
    "157450":  ("안전자산", "금리연계형_초단기채권", 0.50),  # TIGER 단기통안채
    "459580":  ("안전자산", "금리연계형_초단기채권", 0.50),  # KODEX CD금리액티브

    # real_assets
    "A261220": ("위험자산", "FX및원자재",            0.20),  # KODEX WTI원유선물
    "A261240": ("위험자산", "FX및원자재",            0.20),  # KODEX 미국달러선물
    "A411060": ("위험자산", "FX및원자재",            0.20),  # ACE KRX금현물
    "A144600": ("위험자산", "FX및원자재",            0.20),  # KODEX 은선물

    # sovereign_kr
    "A148070": ("안전자산", "국내채권_종합",         0.50),  # KIWOOM 국고채10년
    "A157450": ("안전자산", "금리연계형_초단기채권", 0.50),  # TIGER 단기통안채
    "A439870": ("안전자산", "국내채권_종합",         0.50),  # KODEX 국고채30년액티브

    # sovereign_us
    "A453850": ("안전자산", "해외채권_종합",         0.50),  # ACE 미국30년국채
    "A305080": ("안전자산", "해외채권_종합",         0.50),  # TIGER 미국채10년선물
    "A329750": ("안전자산", "해외채권_종합",         0.50),  # TIGER 미국달러단기채권
}

# DB GAPS 전체 제약
MAX_SINGLE_TICKER = 0.20   # 개별 종목 상한 20%
MAX_RISK_ASSET    = 0.70   # 위험자산 합산 상한 70%

# ──────────────────────────────────────────
# 포트폴리오 전략 파라미터
# ──────────────────────────────────────────
CONSERVATIVE = dict(
    max_single_weight  = 0.15,
    defensive_floor    = 0.50,
    top_n              = 10,
    return_threshold   = 0.0,
    risk_free_weight   = 0.10,
    rebalance_freq     = "weekly",
)

AGGRESSIVE = dict(
    max_single_weight  = 0.20,   # DB GAPS 상한 20%
    defensive_floor    = 0.10,
    top_n              = 8,
    return_threshold   = 0.005,
    risk_free_weight   = 0.0,
    rebalance_freq     = "weekly",
)
