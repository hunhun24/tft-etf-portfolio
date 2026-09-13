"""
collectors/crawl_nyt.py
NYT 뉴스 크롤러 (Selenium 우선, IP 차단 시 NYT API 폴백)
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import time
import requests
import pandas as pd
from pathlib import Path
from urllib.parse import quote_plus

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

from config import NYT_KEYWORDS, RAW_DIR, NYT_SLEEP, NYT_STABLE_ROUNDS, HEADLESS, NYT_API_KEY

EXCLUDE_SECTION = [
    "THE DAILY","MUSIC","MOVIES","TELEVISION","SPORTS","ARTS","LETTERS",
    "SOCCER","BOOKS","THEATER","OBITUARIES","FASHION","STYLE","MAGAZINE",
    "TRAVEL","FOOD","HEALTH","REAL ESTATE","HOME & GARDEN",
]

MAX_CLICKS = 50


def _driver():
    opts = Options()
    if HEADLESS:
        opts.add_argument("--headless=new")
        opts.add_argument("--window-size=1440,1200")
    opts.add_argument("--start-maximized")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })
    driver.set_page_load_timeout(180)
    return driver


def _normalize_date(text: str) -> pd.Timestamp:
    import re
    text = str(text).strip().title()
    text = text.replace("Sept.", "Sep")
    text = re.sub(r"([A-Za-z]{3})\.", r"\1", text)
    if not any(str(y) in text for y in range(2019, 2028)):
        text = f"{text}, 2026"
    return pd.to_datetime(text, errors="coerce", format="mixed")


# ── Selenium 크롤링 ───────────────────────────────────────────
def _crawl_selenium(driver, sector: str, keyword: str, start: str, end: str) -> list:
    url = (
        f"https://www.nytimes.com/search?"
        f"dropmab=false&endDate={end}&lang=en"
        f"&query={quote_plus(keyword)}"
        f"&sort=newest&startDate={start}&types=article"
    )
    try:
        driver.get(url)
        time.sleep(NYT_SLEEP)
    except WebDriverException as e:
        print(f"    [NYT/Selenium] 로드 실패: {e}")
        return []

    prev_count, stable, clicks = -1, 0, 0
    while stable < NYT_STABLE_ROUNDS and clicks < MAX_CLICKS:
        items = driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="search-bodega-result"]')
        cur   = len(items)
        if cur == prev_count:
            stable += 1
        else:
            stable, prev_count = 0, cur

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        try:
            btn = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'button[data-testid="search-show-more-button"]')
                )
            )
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
            time.sleep(0.7)
            if btn.is_displayed() and btn.is_enabled():
                try:
                    btn.click()
                except:
                    driver.execute_script("arguments[0].click();", btn)
                time.sleep(2)
                clicks += 1
        except TimeoutException:
            stable += 1

    rows = []
    for item in driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="search-bodega-result"]'):
        try:
            a_el    = item.find_element(By.CSS_SELECTOR, 'div[data-tpl="h"] a')
            title   = a_el.text.strip()
            href    = a_el.get_attribute("href") or ""
            if href.startswith("/"):
                href = "https://www.nytimes.com" + href
            url_    = href.split("?")[0]
            date_t  = item.find_element(By.CSS_SELECTOR,'span[data-testid="todays-date"]').text.strip()
            sec_els = item.find_elements(By.CSS_SELECTOR,'div[data-tpl="la"]')
            section = sec_els[0].text.strip() if sec_els else ""

            if not title or not date_t:
                continue
            date = _normalize_date(date_t)
            if pd.isna(date) or section.upper() in EXCLUDE_SECTION:
                continue
            rows.append({"keyword":keyword,"title":title,"date":date,
                         "section":sector,"url":url_})
        except:
            continue
    print(f"    NYT/Selenium [{sector}/{keyword}]: {len(rows)}건")
    return rows


# ── NYT API 폴백 ─────────────────────────────────────────────
NYT_API_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

def _crawl_api(sector: str, keyword: str, start: str, end: str) -> list:
    if not NYT_API_KEY or NYT_API_KEY == "YOUR_NYT_API_KEY":
        print("    [NYT/API] API 키 없음")
        return []

    rows, page = [], 0
    s = start.replace("-","")
    e = end.replace("-","")

    while True:
        try:
            resp = requests.get(NYT_API_URL, params={
                "q": keyword, "begin_date": s, "end_date": e,
                "sort": "newest", "page": page,
                "fl": "headline,abstract,pub_date",
                "api-key": NYT_API_KEY,
            }, timeout=15)
            if resp.status_code == 429:
                time.sleep(12)
                continue
            if resp.status_code != 200:
                break
            docs = resp.json().get("response",{}).get("docs",[])
            if not docs:
                break
            for d in docs:
                headline = d.get("headline",{}).get("main","")
                abstract = d.get("abstract","")
                text     = f"{headline}. {abstract}".strip(". ")
                date_raw = d.get("pub_date","")
                try:
                    date = pd.to_datetime(date_raw)
                except:
                    continue
                rows.append({"keyword":keyword,"title":text,
                             "date":date,"section":sector,"url":""})
            page += 1
            time.sleep(0.5)
            if page >= 10:
                break
        except Exception as e:
            print(f"    [NYT/API] 오류: {e}")
            break

    print(f"    NYT/API [{sector}/{keyword}]: {len(rows)}건")
    return rows


# ── 메인 수집 ─────────────────────────────────────────────────
def crawl_nyt(start: str, end: str, sectors: list = None):
    target = {k:v for k,v in NYT_KEYWORDS.items() if sectors is None or k in sectors}

    driver = _driver()
    ip_blocked = False

    try:
        for sector, keywords in target.items():
            out_path = RAW_DIR / f"nyt_{sector}.csv"

            existing, exist_urls = pd.DataFrame(), set()
            if out_path.exists():
                existing   = pd.read_csv(out_path, encoding="utf-8-sig")
                exist_urls = set(existing["url"].dropna())
                print(f"  [{sector}] 기존 {len(existing)}건 로드")

            all_data = existing.to_dict("records") if not existing.empty else []

            for kw in keywords:
                if not ip_blocked:
                    batch = _crawl_selenium(driver, sector, kw, start, end)
                    # IP 차단 감지: 수집량이 너무 적으면 API로 전환
                    if len(batch) == 0:
                        print(f"    ⚠️  Selenium 수집 0건 → API 폴백 시도")
                        batch = _crawl_api(sector, kw, start, end)
                        if len(batch) == 0:
                            ip_blocked = True
                else:
                    batch = _crawl_api(sector, kw, start, end)

                new = [r for r in batch if r.get("url","") not in exist_urls]
                all_data.extend(new)
                exist_urls.update(r.get("url","") for r in new)
                pd.DataFrame(all_data).to_csv(out_path, index=False, encoding="utf-8-sig")
                time.sleep(1)

            df = pd.DataFrame(all_data)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
                df = df.dropna(subset=["date"])
                df["date"] = df["date"].dt.strftime("%Y-%m-%d")
                df = df.drop_duplicates(subset=["title","date"])
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  ✅ NYT [{sector}] 완료: {len(df)}건 → {out_path}")
    finally:
        driver.quit()


if __name__ == "__main__":
    import argparse, datetime
    p = argparse.ArgumentParser()
    p.add_argument("--start",   default=None)
    p.add_argument("--end",     default=datetime.datetime.today().strftime("%Y-%m-%d"))
    p.add_argument("--sectors", nargs="*", default=None)
    args = p.parse_args()
    start = args.start or (datetime.datetime.today() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    crawl_nyt(start, args.end, args.sectors)
