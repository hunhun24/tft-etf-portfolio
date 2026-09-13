"""
run_weekly.py — GAPS ETF 주간 파이프라인
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
전체 흐름:
  1. 가격 + 거시지표 수집  (pykrx + Yahoo Finance + BOK)
  2. 뉴스 크롤링           (매일경제 + NYT)
  3. 감성 점수 산출        (KR-FinBERT / FinBERT)
  4. final_dataset.csv 갱신
  5. TFT 전처리            (src_tft/data/preprocess.py)
  6. TFT 학습              (src_tft/train.py)  ← --retrain 시만
  7. TFT 예측              (src_tft/predict.py)
  8. 포트폴리오 CSV 출력   (src_tft/outputs/portfolio/)

실행:
  python run_weekly.py                  # 기본 (수집+감성+예측)
  python run_weekly.py --retrain        # 모델 재학습 포함
  python run_weekly.py --skip-news      # 뉴스 크롤링 스킵
  python run_weekly.py --step tft       # 예측만 다시
  python run_weekly.py --date 2026-06-02

대시보드:
  streamlit run dashboard/app.py
"""
import sys
import argparse
import logging
import subprocess
import time
import traceback
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import LOG_DIR, FINAL_DATASET, SRC_TFT

# ── 로그 ───────────────────────────────────────────────────────
log_file = LOG_DIR / f"weekly_{datetime.today().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
)
logger = logging.getLogger("run_weekly")


# ── ETF 리스트 ─────────────────────────────────────────────────
def load_etf_meta():
    import pandas as pd
    df = pd.read_csv(FINAL_DATASET)
    meta = df[["ticker","name","sector","category1","category2"]].drop_duplicates("ticker").copy()
    meta["code"] = meta["ticker"].str.replace("A","",1).str.zfill(6)
    meta["aum"]  = 0
    return meta


# ── Step 1: 가격 + 거시 ────────────────────────────────────────
def step_price(full=False):
    logger.info("━"*50 + "\nSTEP 1. 가격 + 거시지표 수집\n" + "━"*50)
    from collectors.collect_macro import collect_prices, collect_macro
    etf = load_etf_meta()
    collect_prices(etf, incremental=not full)
    collect_macro(incremental=not full)
    logger.info("✅ Step 1 완료")


# ── Step 2: 뉴스 크롤링 ────────────────────────────────────────
def step_news(start, end, sectors=None):
    logger.info("━"*50 + "\nSTEP 2. 뉴스 크롤링\n" + "━"*50)
    from collectors.crawl_mk  import crawl_mk
    from collectors.crawl_nyt import crawl_nyt
    crawl_mk(start, end, sectors)
    crawl_nyt(start, end, sectors)
    logger.info("✅ Step 2 완료")


# ── Step 3: 감성 점수 ──────────────────────────────────────────
def step_sentiment(sectors=None):
    logger.info("━"*50 + "\nSTEP 3. 감성 점수 산출\n" + "━"*50)
    from processors.sentiment_score import score_all_sectors
    score_all_sectors(sectors)
    logger.info("✅ Step 3 완료")


# ── Step 4: 데이터셋 갱신 ─────────────────────────────────────
def step_dataset():
    logger.info("━"*50 + "\nSTEP 4. final_dataset.csv 갱신\n" + "━"*50)
    from processors.build_dataset import append_new_data
    append_new_data()
    logger.info("✅ Step 4 완료")


# ── Step 5: TFT 전처리 ────────────────────────────────────────
def step_preprocess():
    logger.info("━"*50 + "\nSTEP 5. TFT 전처리\n" + "━"*50)
    _run_src("data/preprocess.py", "preprocess")
    logger.info("✅ Step 5 완료")


# ── Step 6: TFT 학습 ──────────────────────────────────────────
def step_train():
    logger.info("━"*50 + "\nSTEP 6. TFT 모델 학습\n" + "━"*50)
    _run_src("train.py", "train")
    logger.info("✅ Step 6 완료")


# ── Step 7: TFT 예측 ──────────────────────────────────────────
def step_predict(date=None):
    logger.info("━"*50 + "\nSTEP 7. TFT 예측 + 포트폴리오\n" + "━"*50)
    cmd = ["predict.py"]
    if date:
        cmd += ["--date", date]
    _run_src(cmd if isinstance(cmd, list) else [cmd], "predict")

    # predict_regime 있으면 같이
    if (SRC_TFT / "predict_regime.py").exists():
        cmd2 = ["predict_regime.py"]
        if date:
            cmd2 += ["--date", date]
        try:
            _run_src(cmd2, "predict_regime")
        except:
            pass

    logger.info(f"✅ Step 7 완료 → {SRC_TFT}/outputs/portfolio/")


def _run_src(script, step_name):
    """src_tft 폴더에서 스크립트 실행"""
    cmd = [sys.executable] + (script if isinstance(script, list) else [script])
    log_path = LOG_DIR / f"{step_name}_{datetime.today().strftime('%Y%m%d_%H%M%S')}.log"

    result = subprocess.run(
        cmd, cwd=str(SRC_TFT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)

    if result.returncode != 0:
        logger.error(f"❌ [{step_name}] 실패\n{result.stdout[-1500:]}")
        raise RuntimeError(f"{step_name} 실패 (로그: {log_path})")


# ── 메인 ───────────────────────────────────────────────────────
def main(args):
    t0    = time.time()
    today = datetime.today().strftime("%Y-%m-%d")
    date  = args.date or today
    step  = args.step

    news_start = args.news_start or (
        datetime.today() - timedelta(days=7)
    ).strftime("%Y-%m-%d")

    logger.info(f"""
{'━'*50}
🚀 GAPS ETF 주간 파이프라인
   기준일: {date}  |  뉴스: {news_start} ~ {date}
   재학습: {args.retrain}
   로그: {log_file}
{'━'*50}
""")

    errors = []

    def run(name, fn, *a, **kw):
        try:
            fn(*a, **kw)
        except Exception as e:
            errors.append(f"{name}: {e}")
            logger.error(f"❌ {name}\n{traceback.format_exc()}")
            if not args.continue_on_error:
                sys.exit(1)

    # 단계 실행
    if step in (None, "price") and not args.skip_price:
        run("Step1 가격", step_price, args.full)

    if step in (None, "news") and not args.skip_news:
        run("Step2 뉴스", step_news, news_start, date, args.sectors)

    if step in (None, "sentiment"):
        run("Step3 감성", step_sentiment, args.sectors)

    if step in (None, "dataset"):
        run("Step4 데이터셋", step_dataset)

    if step in (None, "preprocess", "train", "tft"):
        run("Step5 전처리", step_preprocess)

    if step in (None, "train") or (step in (None, "tft") and args.retrain):
        run("Step6 학습", step_train)

    if step in (None, "predict", "tft"):
        run("Step7 예측", step_predict, date)

    elapsed = time.time() - t0
    logger.info(f"""
{'━'*50}
🎉 완료! ({elapsed/60:.1f}분)
   포트폴리오: {SRC_TFT}/outputs/portfolio/
   오류: {len(errors)}건
   대시보드: streamlit run dashboard/app.py
{'━'*50}
""")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GAPS ETF 주간 파이프라인")
    p.add_argument("--retrain",     action="store_true", help="TFT 재학습")
    p.add_argument("--skip-news",   action="store_true", dest="skip_news")
    p.add_argument("--skip-price",  action="store_true", dest="skip_price")
    p.add_argument("--full",        action="store_true", help="전체 재수집")
    p.add_argument("--sectors",     nargs="*", default=None)
    p.add_argument("--date",        default=None, help="예측 기준일 YYYY-MM-DD")
    p.add_argument("--news-start",  default=None, dest="news_start")
    p.add_argument("--step", default=None,
                   choices=["price","news","sentiment","dataset",
                            "preprocess","train","predict","tft"])
    p.add_argument("--continue-on-error", action="store_true", dest="continue_on_error")
    main(p.parse_args())
