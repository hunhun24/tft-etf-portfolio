"""
NYT news append pipeline for 2026 data.

기존 geonho/nyt_code 파일과 기존 산출물은 수정하지 않고, 지정 기간의
NYT 뉴스만 새로 수집해 append용 raw/filtered/clean 파일을 만든다.

Example:
python geonho/data_pipeline/03_collect_nyt_news_append.py \
  --start-date 2026-01-01 \
  --end-date 2026-05-24
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

try:
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
except Exception:
    Service = None
    ChromeDriverManager = None


GEONHO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = GEONHO_ROOT / "data_pipeline" / "outputs" / "nyt"
RAW_DIR = OUTPUT_DIR / "raw"

CATEGORY_KEYWORDS = {
    "real_assets": [
        "gold prices",
        "gold futures",
        "silver futures",
        "crude oil prices",
        "wti crude",
        "u.s. dollar index",
        "japanese yen",
        "grain prices",
    ],
    "overseas_bond": [
        "u.s. treasury",
        "treasury yields",
        "investment grade corporate bond",
        "high yield bond",
        "credit spreads",
        "junk bond",
    ],
    "domestic_bond": [
        "south korea bond market",
        "south korea government bond",
        "korea treasury bond",
        "korea corporate bond",
        "bank of korea rate",
        "korean credit market",
    ],
}

CATEGORY_TO_SECTION = {
    "domestic_bond": "domestic_bond",
    "overseas_bond": "overseas_bond",
    "real_assets": "real_assets",
}

RAW_COLUMNS = ["keyword", "title", "date", "section", "url", "source_file"]
CLEAN_COLUMNS = ["keyword", "title", "date", "section"]

EXCLUDE_SECTION = [
    "THE DAILY",
    "MUSIC",
    "MOVIES",
    "THE LEARNING NETWORK",
    "TELEVISION",
    "SPORTS",
    "ARTS",
    "LETTERS",
    "SOCCER",
    "BOOKS",
    "THEATER",
    "OBITUARIES",
    "OLYMPICS",
    "BASEBALL",
    "FASHION",
    "STYLE",
    "MAGAZINE",
    "TRAVEL",
    "T MAGAZINE",
    "FOOD",
    "BOOK REVIEW",
    "PODCASTS",
    "LOVE",
    "ART & DESIGN",
    "FAMILY",
    "EAT",
    "SELF-CARE",
    "AT HOME",
    "WELL",
    "HEALTH",
    "COLLEGE BASKETBALL",
    "SKIING",
    "TENNIS",
    "GOLF",
    "N.F.L.",
    "DANCE",
    "WINE, BEER & COCKTAILS",
    "REAL ESTATE",
    "NEW YORK",
    "U.S.",
    "WORLD",
    "HOME & GARDEN",
    "MEN’S STYLE",
    "PREGNANCY",
    "WIRECUTTER",
    "HORSE RACING",
    "AUTO RACING",
    "HOCKEY",
    "COLLEGE FOOTBALL",
    "GAMEPLAY",
    "BASKETBALL",
    "ENTREPRENEURSHIP",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect and clean NYT news append data into geonho/data_pipeline/outputs/nyt."
    )
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(CATEGORY_KEYWORDS.keys()),
        choices=list(CATEGORY_KEYWORDS.keys()),
    )
    parser.add_argument("--request-sleep", type=float, default=5.0)
    parser.add_argument("--stable-rounds", type=int, default=3)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def build_driver(headless: bool = False) -> webdriver.Chrome:
    options = Options()
    options.add_argument("--start-maximized")
    if headless:
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1440,1200")

    if ChromeDriverManager is not None and Service is not None:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    else:
        driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(180)
    return driver


def normalize_nyt_date(value: str) -> pd.Timestamp | pd.NaT:
    text = str(value).strip()
    text = text.replace("Sept.", "Sep")
    text = pd.Series([text]).str.replace(r"([A-Za-z]{3})\.", r"\1", regex=True).iloc[0]
    return pd.to_datetime(text, errors="coerce", format="mixed")


def search_url(keyword: str, start_date: str, end_date: str) -> str:
    encoded_keyword = quote_plus(keyword)
    return (
        "https://www.nytimes.com/search?"
        f"dropmab=false&endDate={end_date}&lang=en&query={encoded_keyword}"
        f"&sort=newest&startDate={start_date}&types=article"
    )


def crawl_keyword(
    driver: webdriver.Chrome,
    category: str,
    keyword: str,
    start_date: str,
    end_date: str,
    request_sleep: float,
    stable_rounds_required: int,
) -> pd.DataFrame:
    print(f"\n[CRAWL] {category} / {keyword}: {start_date} ~ {end_date}")
    source_file = f"nyt_{category}_2026_append"

    try:
        driver.get(search_url(keyword, start_date, end_date))
        time.sleep(request_sleep)
    except WebDriverException as e:
        print(f"  [WARN] page load failed: {e}")
        return pd.DataFrame(columns=RAW_COLUMNS)

    prev_item_count = -1
    stable_rounds = 0

    while True:
        items = driver.find_elements(By.CSS_SELECTOR, 'li[data-testid="search-bodega-result"]')
        current_count = len(items)

        if current_count == prev_item_count:
            stable_rounds += 1
        else:
            stable_rounds = 0
            prev_item_count = current_count

        if stable_rounds >= stable_rounds_required:
            print(f"  list expanded: {current_count} items")
            break

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        try:
            button = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'button[data-testid="search-show-more-button"]')
                )
            )
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
            time.sleep(0.7)
            if button.is_displayed() and button.is_enabled():
                try:
                    button.click()
                except Exception:
                    driver.execute_script("arguments[0].click();", button)
                time.sleep(2)
            else:
                stable_rounds += 1
        except TimeoutException:
            stable_rounds += 1

    rows = []
    for item in driver.find_elements(By.CSS_SELECTOR, 'li[data-testid="search-bodega-result"]'):
        try:
            date_text = item.find_element(By.CSS_SELECTOR, 'span[data-testid="todays-date"]').text
            title = item.find_element(By.CSS_SELECTOR, "h4").text
            url = item.find_element(By.CSS_SELECTOR, "a").get_attribute("href").split("?")[0]
            try:
                section = item.find_element(By.CSS_SELECTOR, "p").text
            except Exception:
                section = ""

            date = normalize_nyt_date(date_text)
            if pd.isna(date):
                continue

            rows.append({
                "keyword": keyword,
                "title": title,
                "date": date,
                "section": str(section).strip(),
                "url": url,
                "source_file": source_file,
            })
        except Exception:
            continue

    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    if not df.empty:
        df = df.drop_duplicates(subset=["title", "date", "url"]).reset_index(drop=True)
        df = df.sort_values(["date", "keyword", "title"]).reset_index(drop=True)
    print(f"  collected={len(df)}")
    return df


def collect_category(
    driver: webdriver.Chrome,
    category: str,
    start_date: str,
    end_date: str,
    request_sleep: float,
    stable_rounds: int,
) -> pd.DataFrame:
    frames = []
    for keyword in CATEGORY_KEYWORDS[category]:
        try:
            frames.append(
                crawl_keyword(
                    driver=driver,
                    category=category,
                    keyword=keyword,
                    start_date=start_date,
                    end_date=end_date,
                    request_sleep=request_sleep,
                    stable_rounds_required=stable_rounds,
                )
            )
        except Exception as e:
            print(f"  [WARN] keyword failed, continue: {category} / {keyword}: {e}")
            continue

    if not frames:
        return pd.DataFrame(columns=RAW_COLUMNS)

    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)
    df = df.drop_duplicates(subset=["title", "date", "url"]).reset_index(drop=True)
    df = df.sort_values(["date", "keyword", "title"]).reset_index(drop=True)
    return df[RAW_COLUMNS]


def filter_nyt_news(raw_df: pd.DataFrame) -> pd.DataFrame:
    """기존 nyt_filtering_final.py의 section/title 필터를 영어 컬럼명 기준으로 적용한다."""
    if raw_df.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    df = raw_df[RAW_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    for col in ["keyword", "title", "section", "url", "source_file"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df = df.dropna(subset=["date"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["title", "date"]).reset_index(drop=True)

    df_filtered = df[~df["section"].isin(EXCLUDE_SECTION)].reset_index(drop=True)

    briefing_df = df_filtered[df_filtered["section"] == "BRIEFING"].copy()
    if not briefing_df.empty:
        title_lc = briefing_df["title"].str.lower().str.strip()
        briefing_pattern = (
            r"(^your\s.+briefing$)|"
            r"(^monday\sbriefing)|(^tuesday\sbriefing)|(^wednesday\sbriefing)|"
            r"(^thursday\sbriefing)|(^friday\sbriefing)|(^saturday\sbriefing)|(^sunday\sbriefing)|"
            r"(^the\sevening$)|(^your\sevening\sbriefing$)|(^evening\sbriefing$)|"
            r"(\bbriefing\b)"
        )
        market_keep_pattern = (
            r"(inflation|oil|economy|economic|market|markets|treasury|treasuries|yield|yields|"
            r"bond|bonds|fed|rate|rates|tariff|tariffs|trade|gas|stock|stocks|dollar|currency|"
            r"crude|grain|recession|sanction|sanctions|energy|prices?)"
        )
        drop_mask = (
            title_lc.str.contains(briefing_pattern, regex=True, na=False)
            & ~title_lc.str.contains(market_keep_pattern, regex=True, na=False)
        )
        df_filtered = df_filtered.drop(index=briefing_df.loc[drop_mask].index).reset_index(drop=True)

    asia_df = df_filtered[df_filtered["section"] == "ASIA PACIFIC"].copy()
    if not asia_df.empty:
        title_lc = asia_df["title"].str.lower().str.strip()
        asia_drop_pattern = (
            r"(covid|coronavirus|vaccine|vaccination|outbreak|quarantine|lockdown|"
            r"protest|protests|abortion|criminal|crime|murder|death|deaths|"
            r"lost weight|silence|first coronavirus death|booking system|"
            r"most notorious|health)"
        )
        asia_small_drop_pattern = (
            r"(apologizes|photographer|dies\b|bombing|kills\b|earthquake|"
            r"latest developments|same-sex|buddhist statue|"
            r"tell me who my mother is|families demand answers|construction site|"
            r"talent agency)"
        )
        drop_mask = (
            title_lc.str.contains(asia_drop_pattern, regex=True, na=False)
            | title_lc.str.contains(asia_small_drop_pattern, regex=True, na=False)
        )
        df_filtered = df_filtered.drop(index=asia_df.loc[drop_mask].index).reset_index(drop=True)

    opinion_df = df_filtered[df_filtered["section"].isin(["OPINION", "SUNDAY OPINION"])].copy()
    if not opinion_df.empty:
        title_lc = opinion_df["title"].str.lower().str.strip()
        opinion_keep_pattern = (
            r"(inflation|fed|federal reserve|interest rate|rates|bond|bonds|yield|yields|"
            r"treasury|treasuries|trade|tariff|tariffs|economy|economic|market|markets|"
            r"tax|taxes|recession|fracking|oil|gas|energy|pipeline|housing|home|homes|"
            r"rent|rents|mortgage|mortgages|china|sanction|sanctions|dollar|currency|"
            r"billionaire|billionaires|invest|investing|investment|supply chain|epa|"
            r"nuclear power|population)"
        )
        keep_mask = title_lc.str.contains(opinion_keep_pattern, regex=True, na=False)
        df_filtered = df_filtered.drop(index=opinion_df.loc[~keep_mask].index).reset_index(drop=True)

    return df_filtered.sort_values(["date", "source_file", "keyword", "title"]).reset_index(drop=True)[RAW_COLUMNS]


def clean_nyt_news(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """기존 nyt_clean.py의 최종 컬럼 정리 로직을 append 파일명 기준으로 적용한다."""
    if filtered_df.empty:
        return pd.DataFrame(columns=CLEAN_COLUMNS)

    df = filtered_df[RAW_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    df["keyword"] = df["keyword"].fillna("").astype(str).str.strip()
    df["title"] = df["title"].fillna("").astype(str).str.strip()

    def _section_from_source(source_file: str) -> str | None:
        for category, section in CATEGORY_TO_SECTION.items():
            if source_file == f"nyt_{category}_2026_append":
                return section
        return None

    df["section"] = df["source_file"].map(_section_from_source)
    if df["section"].isna().any():
        unknown = df.loc[df["section"].isna(), "source_file"].drop_duplicates().tolist()
        raise ValueError(f"section 매핑 안 된 source_file 있음: {unknown}")

    df = df[CLEAN_COLUMNS].copy()
    df = df[df["title"].str.strip() != ""].reset_index(drop=True)
    df = df.drop_duplicates(subset=["title", "date"]).reset_index(drop=True)
    df = df.sort_values(["date", "section", "keyword", "title"]).reset_index(drop=True)
    return df[CLEAN_COLUMNS]


def print_summary(
    raw_counts: dict[str, int],
    filtered_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    print("\n" + "=" * 80)
    print("NYT append pipeline summary")
    print("=" * 80)
    print("Raw rows by category:")
    for category, count in raw_counts.items():
        print(f"  {category}: {count:,}")
    print(f"\nfiltered rows: {len(filtered_df):,}")
    print(f"clean rows   : {len(clean_df):,}")

    if clean_df.empty:
        print("final date range: -")
        print("final section distribution: -")
    else:
        print(f"final date range: {clean_df['date'].min().date()} ~ {clean_df['date'].max().date()}")
        print("final section distribution:")
        print(clean_df["section"].value_counts().to_string())

    print("\nSaved files:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


def main() -> None:
    args = parse_args()
    start_date = pd.Timestamp(args.start_date).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(args.end_date).strftime("%Y-%m-%d")
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise ValueError(f"start-date가 end-date보다 늦음: {start_date} > {end_date}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    raw_counts = {}
    raw_paths = {}
    raw_frames = []

    driver = build_driver(headless=args.headless)
    try:
        for category in args.categories:
            raw_path = RAW_DIR / f"nyt_{category}_2026_append.csv"
            if raw_path.exists():
                print(f"[SKIP] {category}: raw file already exists -> {raw_path}")
                df_category = pd.read_csv(raw_path, encoding="utf-8-sig")
                raw_counts[category] = len(df_category)
                raw_paths[category] = raw_path
                raw_frames.append(df_category)
                continue

            print("\n" + "=" * 80)
            print(f"Category: {category}")
            print("=" * 80)

            df_category = collect_category(
                driver=driver,
                category=category,
                start_date=start_date,
                end_date=end_date,
                request_sleep=args.request_sleep,
                stable_rounds=args.stable_rounds,
            )
            raw_path = RAW_DIR / f"nyt_{category}_2026_append.csv"
            df_category.to_csv(raw_path, index=False, encoding="utf-8-sig")
            raw_counts[category] = len(df_category)
            raw_paths[category] = raw_path
            raw_frames.append(df_category)
            print(f"[SAVE] raw {category}: {raw_path} ({len(df_category):,} rows)")
            time.sleep(max(args.request_sleep, 3.0))

    finally:
        driver.quit()

    if raw_frames:
        raw_all = pd.concat(raw_frames, ignore_index=True)
    else:
        raw_all = pd.DataFrame(columns=RAW_COLUMNS)

    filtered_df = filter_nyt_news(raw_all)
    filtered_path = OUTPUT_DIR / "nyt_filtered_2026_append.csv"
    filtered_df.to_csv(filtered_path, index=False, encoding="utf-8-sig")

    clean_df = clean_nyt_news(filtered_df)
    clean_path = OUTPUT_DIR / "nyt_news_geonho_2026_append.csv"
    clean_df.to_csv(clean_path, index=False, encoding="utf-8-sig")

    clean_copy_path = OUTPUT_DIR / "nyt_news_geonho_2026_append_clean.csv"
    clean_df.to_csv(clean_copy_path, index=False, encoding="utf-8-sig")

    paths = {
        **{f"raw/{category}": path for category, path in raw_paths.items()},
        "filtered": filtered_path,
        "clean": clean_path,
        "clean_copy": clean_copy_path,
    }
    print_summary(raw_counts, filtered_df, clean_df, paths)


if __name__ == "__main__":
    main()
