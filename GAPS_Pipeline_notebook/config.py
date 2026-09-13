"""
config.py  ─  GAPS Pipeline 전체 설정
★ 여기만 수정하면 됩니다
"""
from pathlib import Path
from datetime import datetime, timedelta

# ══════════════════════════════════════════════════════════════
# 경로
# ══════════════════════════════════════════════════════════════
ROOT       = Path(__file__).resolve().parent
SRC_TFT    = ROOT / "src_tft"          # TFT 학습/예측 코드
DATA_DIR   = ROOT / "data"
NEWS_DIR   = DATA_DIR / "news"
RAW_DIR    = NEWS_DIR / "raw"
MERGED_DIR = NEWS_DIR / "merged"
LOG_DIR    = DATA_DIR / "outputs" / "logs"

# 메인 데이터셋 (파이프라인이 매주 append)
FINAL_DATASET = DATA_DIR / "final_dataset.csv"

for d in [RAW_DIR, MERGED_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# API 키 ###보안 상 지움
# ══════════════════════════════════════════════════════════════
BOK_API_KEY = ""   # 한국은행 ECOS (국고채 10년)
NYT_API_KEY = ""         # NYT (IP 차단 시 폴백)

# ══════════════════════════════════════════════════════════════
# 크롤링 기준일 ("N시간 전" 날짜 변환용 — 매주 자동 계산)
# ══════════════════════════════════════════════════════════════
def _last_biz():
    t = datetime.today()
    offset = (t.weekday() - 4) % 7
    d = t - timedelta(days=offset)
    return d if offset >= 0 else t
REF_DATE = _last_biz()

# ══════════════════════════════════════════════════════════════
# 뉴스 키워드 (섹터별)  ← 자유롭게 수정
# ══════════════════════════════════════════════════════════════
MK_KEYWORDS = {
    "rate_cash":     ["부동자금","금리","현금확보","유동성","예금","mmf","CD금리","cp"],
    "defensive":     ["밸류업","PBR","배당","리츠","안전자산","자사주","지배구조",
                      "현금흐름","저평가","주주환원","자산가치","필수소비재"],
    "domestic_core": ["코스피","코스닥","MSCI_Korea"],
    "global_macro":  ["S&P500","나스닥","다우존스","러셀","DAX","유로스탁스",
                      "니케이","TOPIX","MSCI_ACWI","MSCI_World","MSCI_EAFE"],
    "tech_future":   ["2차전지","반도체","빅테크","AI 인프라","전기차"],
    "emerging_asia": ["인도 증시","중국 증시","베트남 증시"],
    "sovereign_debt":["채권금리","국채","기준금리"],
    "real_assets":   ["달러강세","환율","국제유가","원유","금값","원달러","WTI"],
}

NYT_KEYWORDS = {
    "rate_cash":     ["Money Market","Cash Management","Treasury bill","Benchmark rate",
                      "T-bill","Fixed Income","Interest rate policy","CD Rate",
                      "Short-term bond","Liquidity","SOFR"],
    "defensive":     ["Property Market","dividend","Real Estate Investment","Value Stock",
                      "Medical Device","Pharmaceutical","Low Volatility","Defensive Stock",
                      "Healthcare","Biotech","Equity Income","REITs","SCHD"],
    "domestic_core": ["KOSPI","KOSDAQ"],
    "global_macro":  ["Nikkei 225","Euro Stoxx","global stock market","equity market",
                      "wall street stocks","stock market rally","stock market selloff",
                      "Federal Reserve rates","global recession","trade war tariffs",
                      "NASDAQ100","DOW JONES","CSI 300","S&P500"],
    "tech_future":   ["Secondary Battery","Semiconductor","Electric Vehicle",
                      "Big Tech","AI Infrastructure"],
    "emerging_asia": ["Indian Stock Market","Indian Equities","Chinese Stock Market",
                      "Chinese Equities","Vietnamese Stock Market","Vietnamese Equities"],
    "sovereign_debt":["south korea bond market","south korea government bond",
                      "treasury yields","u.s. treasury","bank of korea rate",
                      "korea treasury bond","junk bond","high yield bond"],
    "real_assets":   ["u.s. dollar index","crude oil prices","gold prices",
                      "grain prices","silver futures","gold futures"],
}

# ══════════════════════════════════════════════════════════════
# 거시지표
# ══════════════════════════════════════════════════════════════
YF_TICKERS = {"vix_close": "^VIX", "usd_krw": "KRW=X", "us_10y": "^TNX"}
BOK_STAT   = "817Y002"
BOK_ITEM   = "010210000"   # 국고채 10년

# ══════════════════════════════════════════════════════════════
# 크롤링 옵션
# ══════════════════════════════════════════════════════════════
MK_SLEEP          = 2.0
NYT_SLEEP         = 8.0
NYT_STABLE_ROUNDS = 3
HEADLESS          = False   # True: 창 없이 실행

from pathlib import Path
_ROOT = Path(r"C:\Users\LG\Desktop\Quant\GAPS_Pipeline_final\GAPS_Pipeline_notebook")

TFT_SRC_DIR   = _ROOT / "src_tft"
FINAL_DATASET = _ROOT / "data" / "final_dataset.csv"
RAW_DIR       = _ROOT / "data" / "news" / "raw"
MERGED_DIR    = _ROOT / "data" / "news" / "merged"