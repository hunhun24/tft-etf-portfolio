"""
collectors/crawl_mk.py
매일경제 뉴스 크롤러 (섹터별 키워드)
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import time
import pandas as pd
import signal
from pathlib import Path
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager


from config import MK_KEYWORDS, RAW_DIR, MK_SLEEP, HEADLESS, REF_DATE
import re


def _driver():
    opts = Options()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--start-maximized")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    driver.set_page_load_timeout(30)   # ← 추가
    driver.set_script_timeout(10)      # ← 추가
    return driver


def _parse_date(raw: str) -> str | None:
    if not raw or pd.isna(raw):
        return None
    s = str(raw).strip()
    if "전" in s:
        if any(x in s for x in ["분","시간","초"]):
            return REF_DATE.strftime("%Y-%m-%d")
        m = re.search(r"(\d+)일", s)
        if m:
            return (REF_DATE - __import__("datetime").timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d")
        m = re.search(r"(\d+)주", s)
        if m:
            return (REF_DATE - __import__("datetime").timedelta(weeks=int(m.group(1)))).strftime("%Y-%m-%d")
        return REF_DATE.strftime("%Y-%m-%d")
    try:
        return pd.to_datetime(s).strftime("%Y-%m-%d")
    except:
        return None


def crawl_mk_period(driver, keyword: str, sector: str, start: str, end: str) -> list:
    import time as _time
    HARD_TIMEOUT = 180  # ← 키워드당 최대 2분
    url = (
        f"https://www.mk.co.kr/search/news?"
        f"word={quote(keyword)}&sort=desc&dateType=direct"
        f"&startDate={start}&endDate={end}&searchField=all&newsType=all"
    )
    try:
        driver.get(url)
    except Exception:
        print(f"    MK [{sector}/{keyword}] 페이지 로드 실패 → 스킵")
        return []
    
    time.sleep(MK_SLEEP + 3)

    results, seen = [], set()
    prev, no_grow = 0, 0
    MAX_NO_GROW  = 3    # ← 추가: 3회 연속 변화 없으면 종료 (기존 4)
    MAX_SCROLLS  = 35   # ← 추가: 최대 스크롤 횟수 제한
    scroll_count = 0    # ← 추가
    t_start = _time.time() 

    while True:
         # 하드 타임아웃
        if _time.time() - t_start > HARD_TIMEOUT:
            print(f"    [TIMEOUT] {keyword} {HARD_TIMEOUT}초 초과 → 강제 종료")
            break

        # ← 추가: 최대 스크롤 횟수 초과 시 강제 종료
        if scroll_count >= MAX_SCROLLS:
            print(f"    [TIMEOUT] {keyword} 최대 스크롤 도달 → 강제 종료")
            break

        for _ in range(3):
            try:
                driver.execute_script("window.scrollBy(0, 1000);")
            except Exception:
                break
            time.sleep(0.5)
        scroll_count += 1

        for art in driver.find_elements(By.XPATH, "//li[descendant::h3]"):
            try:
                title = art.find_element(By.TAG_NAME, "h3").text.strip()
                if not title or title in seen:
                    continue
                date_els = art.find_elements(
                    By.XPATH, ".//*[contains(@class,'date') or contains(@class,'time')]"
                )
                date_raw = date_els[0].text.replace("\n"," ").strip() if date_els else ""
                results.append({"keyword":keyword,"제목":title,"날짜":date_raw,"sector":sector})
                seen.add(title)
            except:
                continue

        cur = len(results)
        no_grow = 0 if cur != prev else no_grow + 1
        prev = cur

        if no_grow >= MAX_NO_GROW:
            try:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            except:
                pass
            time.sleep(2)
            if len(driver.find_elements(By.XPATH, "//li[descendant::h3]")) <= len(seen):
                break
            no_grow = 0

        try:
            btn = driver.find_element(By.XPATH, "//button[contains(.,'더보기')]")
            if btn.is_displayed():
                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
                time.sleep(0.5)
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(2)
        except:
            time.sleep(1)

    print(f"    MK [{sector}/{keyword}] {start}~{end}: {len(results)}건")
    return results


def _split_periods(start: str, end: str, months=6) -> list:
    from datetime import datetime, timedelta
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")
    periods, cur = [], s
    while cur <= e:
        m   = cur.month - 1 + months
        nxt = cur.replace(year=cur.year + m//12, month=m%12+1, day=1)
        pe  = min(nxt - timedelta(days=1), e)
        periods.append((cur.strftime("%Y-%m-%d"), pe.strftime("%Y-%m-%d")))
        cur = nxt
    return periods


def crawl_mk(start: str, end: str, sectors: list = None):
    """
    sectors: None이면 전체, 또는 ['rate_cash','defensive'] 등 일부만
    """
    target = {k:v for k,v in MK_KEYWORDS.items() if sectors is None or k in sectors}

    driver = _driver()
    try:
        for sector, keywords in target.items():
            out_path = RAW_DIR / f"mk_{sector}.csv"

            existing, exist_titles = pd.DataFrame(), set()
            if out_path.exists():
                existing    = pd.read_csv(out_path, encoding="utf-8-sig")
                exist_titles = set(existing["제목"].dropna())
                print(f"  [{sector}] 기존 {len(existing)}건 로드")

            all_data = existing.to_dict("records") if not existing.empty else []
            periods  = _split_periods(start, end)

            for kw in keywords:
                for ps, pe in periods:
                    batch = crawl_mk_period(driver, kw, sector, ps, pe)
                    new   = [r for r in batch if r["제목"] not in exist_titles]
                    all_data.extend(new)
                    exist_titles.update(r["제목"] for r in new)
                    pd.DataFrame(all_data).to_csv(out_path, index=False, encoding="utf-8-sig")
                    time.sleep(1)

            df = pd.DataFrame(all_data).drop_duplicates("제목")
            df["date"] = df["날짜"].apply(_parse_date)
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  ✅ MK [{sector}] 완료: {len(df)}건 → {out_path}")
    finally:
        driver.quit()


if __name__ == "__main__":
    import argparse, datetime
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=None)
    p.add_argument("--end",   default=datetime.datetime.today().strftime("%Y-%m-%d"))
    p.add_argument("--sectors", nargs="*", default=None)
    args = p.parse_args()

    start = args.start or (datetime.datetime.today() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    crawl_mk(start, args.end, args.sectors)
