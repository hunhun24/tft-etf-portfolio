"""
MK news append pipeline for 2026 data.

기존 geonho/mk_code 파일과 기존 산출물은 수정하지 않고, 지정 기간의
매일경제 뉴스만 새로 수집해 append용 concat/clean/filter 파일을 만든다.

Example:
python data_pipeline/02_collect_mk_news_append.py \
  --start-date 2026-01-01 \
  --end-date 2026-05-24
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


GEONHO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = GEONHO_ROOT / "data_pipeline" / "outputs" / "mk"
RAW_DIR = OUTPUT_DIR / "raw"

DEFAULT_KEYWORDS = [
    "기준금리",
    "국채",
    "채권금리",
    "금값",
    "국제유가",
    "원유",
    "WTI",
    "환율",
    "원달러",
    "달러강세",
]

SECTION_MAP = {
    "국채": "domestic_bond",
    "기준금리": "domestic_bond",
    "채권금리": "domestic_bond",
    "달러강세": "real_assets",
    "원달러": "real_assets",
    "환율": "real_assets",
    "국제유가": "real_assets",
    "금값": "real_assets",
    "원유": "real_assets",
    "WTI": "real_assets",
}

RAW_COLUMNS = ["keyword", "title", "date", "section"]
FINAL_COLUMNS = ["keyword", "title", "date", "section"]

NOISE_PATTERNS = [
    r"매.세.지",
    r"\[?포토\]?",
    r"카드뉴스",
    r"\[광고\]|\[PR\]|advertorial",
    r"동영상|영상|유튜브|[Vv]ideo",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect and clean MK news append data into geonho/data_pipeline/outputs/mk."
    )
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=DEFAULT_KEYWORDS,
        help="Optional keyword override. Defaults to the historical MK keyword list.",
    )
    parser.add_argument("--months-per-range", type=int, default=6)
    parser.add_argument("--request-sleep", type=float, default=3.0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def split_date_range(start_date: str, end_date: str, months: int) -> list[tuple[str, str]]:
    """기존 crawl_news2.py처럼 next_date + 1일로 구간 경계 중복을 피한다."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    ranges = []
    current = start
    while current <= end:
        next_date = current + timedelta(days=30 * months)
        if next_date > end:
            next_date = end
        ranges.append((current.strftime("%Y-%m-%d"), next_date.strftime("%Y-%m-%d")))
        current = next_date + timedelta(days=1)
    return ranges


def mk_search_url(keyword: str, start_date: str, end_date: str) -> str:
    return (
        "https://www.mk.co.kr/search?"
        f"word={keyword}&sort=asc&dateType=direct"
        f"&startDate={start_date}&endDate={end_date}"
        "&searchField=all&newsType=all"
    )


def build_driver(headless: bool = False) -> webdriver.Chrome:
    options = Options()
    options.add_argument("--start-maximized")
    if headless:
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1440,1200")
    return webdriver.Chrome(options=options)


def safe_get(driver: webdriver.Chrome, url: str, request_sleep: float, retries: int = 5) -> bool:
    for attempt in range(retries):
        try:
            driver.get(url)
            time.sleep(request_sleep)
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
            )
            return True
        except Exception as e:
            print(f"  [WARN] page load retry {attempt + 1}/{retries}: {e}")
            time.sleep(max(request_sleep, 5.0))
    return False


def get_total_count(
    driver: webdriver.Chrome,
    keyword: str,
    start_date: str,
    end_date: str,
    request_sleep: float,
) -> int:
    url = mk_search_url(keyword, start_date, end_date)
    if not safe_get(driver, url, request_sleep=request_sleep):
        print(f"  [WARN] [{keyword}] totalCount page load failed: {start_date} ~ {end_date}")
        return 0

    try:
        total_elem = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "totalCount"))
        )
        total = int(total_elem.text.replace(",", "").strip())
    except Exception as e:
        print(f"  [WARN] [{keyword}] totalCount 못 찾음 -> 0 처리: {e}")
        total = 0

    print(f"  [{keyword}] {start_date} ~ {end_date} totalCount={total}")
    return total


def crawl_range(
    driver: webdriver.Chrome,
    keyword: str,
    start_date: str,
    end_date: str,
    request_sleep: float,
) -> pd.DataFrame:
    print(f"\n[CRAWL] {keyword}: {start_date} ~ {end_date}")
    url = mk_search_url(keyword, start_date, end_date)
    if not safe_get(driver, url, request_sleep=request_sleep):
        print(f"  [WARN] [{keyword}] page load failed, skip range")
        return pd.DataFrame(columns=RAW_COLUMNS)

    click_count = 0
    while True:
        articles = driver.find_elements(By.CSS_SELECTOR, "li.news_node")
        current_count = len(articles)
        print(f"  [{keyword}] visible articles={current_count}")

        try:
            button = driver.find_element(By.CSS_SELECTOR, "button[data-btn-more]")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(max(request_sleep, 4.0))
            driver.execute_script("arguments[0].click();", button)
            click_count += 1

            if click_count % 50 == 0:
                print(f"  [{keyword}] cooldown after {click_count} clicks")
                time.sleep(10)

            for _ in range(10):
                time.sleep(max(request_sleep, 4.0))
                new_count = len(driver.find_elements(By.CSS_SELECTOR, "li.news_node"))
                if new_count > current_count:
                    break
            else:
                print(f"  [{keyword}] no more articles loaded")
                break

        except Exception:
            print(f"  [{keyword}] no more button")
            break

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(max(request_sleep, 4.0))

    results = []
    for article in driver.find_elements(By.CSS_SELECTOR, "li.news_node"):
        try:
            title = article.find_element(By.CSS_SELECTOR, "h3.news_ttl").text
            date = article.find_element(By.CSS_SELECTOR, "p.time_info").text
            results.append({
                "keyword": keyword,
                "title": title,
                "date": date,
                "section": SECTION_MAP.get(keyword, "unknown"),
            })
        except Exception:
            continue

    df = pd.DataFrame(results, columns=RAW_COLUMNS)
    if not df.empty:
        df = df.drop_duplicates(subset=["title", "date"]).reset_index(drop=True)
    print(f"  [{keyword}] collected={len(df)}")
    return df


def collect_keyword(
    driver: webdriver.Chrome,
    keyword: str,
    start_date: str,
    end_date: str,
    months_per_range: int,
    request_sleep: float,
) -> pd.DataFrame:
    ranges = split_date_range(start_date, end_date, months_per_range)
    frames = []

    for range_start, range_end in ranges:
        try:
            total = get_total_count(
                driver=driver,
                keyword=keyword,
                start_date=range_start,
                end_date=range_end,
                request_sleep=request_sleep,
            )
            if total == 0:
                print(f"  [SKIP] [{keyword}] totalCount=0: {range_start} ~ {range_end}")
                continue

            if total < 1800:
                frames.append(crawl_range(driver, keyword, range_start, range_end, request_sleep))
            else:
                print(f"  [SPLIT] [{keyword}] large range total={total}, split to 3-month ranges")
                for sub_start, sub_end in split_date_range(range_start, range_end, 3):
                    sub_total = get_total_count(driver, keyword, sub_start, sub_end, request_sleep)
                    if sub_total == 0:
                        continue
                    if sub_total < 1800:
                        frames.append(crawl_range(driver, keyword, sub_start, sub_end, request_sleep))
                    else:
                        print(f"  [SPLIT] [{keyword}] still large total={sub_total}, split to 1-month ranges")
                        for ss, ee in split_date_range(sub_start, sub_end, 1):
                            if get_total_count(driver, keyword, ss, ee, request_sleep) == 0:
                                continue
                            frames.append(crawl_range(driver, keyword, ss, ee, request_sleep))

        except (WebDriverException, Exception) as e:
            print(f"  [WARN] [{keyword}] range failed {range_start} ~ {range_end}: {e}")
            continue

    if not frames:
        return pd.DataFrame(columns=RAW_COLUMNS)

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["title", "date"]).reset_index(drop=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df[RAW_COLUMNS]


def save_raw_keyword(df: pd.DataFrame, keyword: str) -> Path:
    path = RAW_DIR / f"mk_news_{keyword}_2026_append.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def build_concat_from_raw(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(raw_dir.glob("mk_news_*_2026_append.csv")):
        keyword = path.name.replace("mk_news_", "").replace("_2026_append.csv", "")
        if keyword not in SECTION_MAP:
            raise ValueError(f"section mapping missing for keyword: {keyword}")

        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = [str(col).strip() for col in df.columns]
        missing = [col for col in RAW_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"{path.name} missing columns: {missing}")

        df = df[RAW_COLUMNS].copy()
        df["keyword"] = keyword
        df["section"] = SECTION_MAP[keyword]
        df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
        df["title"] = df["title"].fillna("").astype(str).str.strip()
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=FINAL_COLUMNS)

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.dropna(subset=["date"]).reset_index(drop=True)
    df_all = df_all[df_all["title"].str.strip() != ""].reset_index(drop=True)
    df_all = df_all.drop_duplicates(subset=["title", "date"]).reset_index(drop=True)
    df_all = df_all.sort_values(["date", "section", "keyword", "title"]).reset_index(drop=True)
    return df_all[FINAL_COLUMNS]


def clean_title(title: str) -> str:
    if pd.isna(title):
        return ""

    text = str(title).strip()
    if not text:
        return ""

    text = re.sub(r"[\r\n\t]+", " ", text)
    text = (
        text.replace("“", '"')
            .replace("”", '"')
            .replace("‘", '"')
            .replace("’", '"')
            .replace("`", '"')
    )
    text = re.sub(r'"{2,}', '"', text)
    text = re.sub(r"'{2,}", "'", text)
    text = re.sub(r'^["\']+', "", text)
    text = re.sub(r'["\']+$', "", text)
    text = re.sub(r"\s*\[[^\]]+\]\s*", " ", text)
    text = text.replace("…", "...")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = text.strip("\"'").strip()

    if text in {'"', "'"}:
        return ""
    return text


def clean_mk_news(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in FINAL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"clean input missing columns: {missing}")

    cleaned = df[FINAL_COLUMNS].copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce", format="mixed")
    cleaned["title"] = cleaned["title"].apply(clean_title)
    cleaned["keyword"] = cleaned["keyword"].fillna("").astype(str).str.strip()
    cleaned["section"] = cleaned["section"].fillna("").astype(str).str.strip()

    cleaned = cleaned.dropna(subset=["date"]).reset_index(drop=True)
    cleaned = cleaned[cleaned["title"].str.strip() != ""].reset_index(drop=True)
    cleaned = cleaned.drop_duplicates(subset=["title", "date"]).reset_index(drop=True)
    cleaned = cleaned.sort_values(["date", "section", "keyword", "title"]).reset_index(drop=True)
    return cleaned[FINAL_COLUMNS]


def filter_mk_news(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[FINAL_COLUMNS].copy()
    noise_mask = filtered["title"].str.contains(
        "|".join(NOISE_PATTERNS),
        regex=True,
        na=False,
    )
    return filtered[~noise_mask].copy().reset_index(drop=True)


def print_summary(
    raw_counts: dict[str, int],
    concat_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    print("\n" + "=" * 80)
    print("MK append pipeline summary")
    print("=" * 80)
    print("Raw rows by keyword:")
    for keyword, count in raw_counts.items():
        print(f"  {keyword}: {count:,}")

    print(f"\nconcat rows  : {len(concat_df):,}")
    print(f"clean rows   : {len(clean_df):,}")
    print(f"filtered rows: {len(filtered_df):,}")

    if filtered_df.empty:
        print("final date range: -")
        print("final section distribution: -")
    else:
        print(
            "final date range: "
            f"{filtered_df['date'].min().date()} ~ {filtered_df['date'].max().date()}"
        )
        print("final section distribution:")
        print(filtered_df["section"].value_counts().to_string())

    print("\nSaved files:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


def main() -> None:
    args = parse_args()
    start_date = (pd.Timestamp(args.start_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(args.end_date).strftime("%Y-%m-%d")
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise ValueError(f"start-date가 end-date보다 늦음: {start_date} > {end_date}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    raw_counts = {}
    raw_paths = {}

    driver = build_driver(headless=args.headless)
    try:
        for keyword in args.keywords:
            if keyword not in SECTION_MAP:
                print(f"[WARN] section mapping missing, skip keyword: {keyword}")
                raw_counts[keyword] = 0
                continue

            raw_path = RAW_DIR / f"mk_news_{keyword}_2026_append.csv"
            if raw_path.exists():
                print(f"[SKIP] {keyword}: raw file already exists -> {raw_path}")
                df_keyword = pd.read_csv(raw_path, encoding="utf-8-sig")
                raw_counts[keyword] = len(df_keyword)
                raw_paths[keyword] = raw_path
                continue

            print("\n" + "=" * 80)
            print(f"Keyword: {keyword}")
            print("=" * 80)
            df_keyword = collect_keyword(
                driver=driver,
                keyword=keyword,
                start_date=start_date,
                end_date=end_date,
                months_per_range=args.months_per_range,
                request_sleep=args.request_sleep,
            )
            raw_path = save_raw_keyword(df_keyword, keyword)
            raw_counts[keyword] = len(df_keyword)
            raw_paths[keyword] = raw_path
            print(f"[SAVE] raw {keyword}: {raw_path} ({len(df_keyword):,} rows)")

            # 기존 코드처럼 키워드 단위 작업 후 잠깐 쉰다.
            time.sleep(max(args.request_sleep, 3.0))

    finally:
        driver.quit()

    concat_df = build_concat_from_raw(RAW_DIR)
    concat_path = OUTPUT_DIR / "mk_news_geonho_2026_append.csv"
    concat_df.to_csv(concat_path, index=False, encoding="utf-8-sig")

    clean_df = clean_mk_news(concat_df)
    clean_path = OUTPUT_DIR / "mk_news_geonho_2026_append_clean.csv"
    clean_df.to_csv(clean_path, index=False, encoding="utf-8-sig")

    filtered_df = filter_mk_news(clean_df)
    filtered_path = OUTPUT_DIR / "mk_news_geonho_2026_append_filtered.csv"
    filtered_df.to_csv(filtered_path, index=False, encoding="utf-8-sig")

    paths = {
        **{f"raw/{keyword}": path for keyword, path in raw_paths.items()},
        "concat": concat_path,
        "clean": clean_path,
        "filtered": filtered_path,
    }
    print_summary(raw_counts, concat_df, clean_df, filtered_df, paths)


if __name__ == "__main__":
    main()
